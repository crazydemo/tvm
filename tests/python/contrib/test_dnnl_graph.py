# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
import pytest
import itertools
import numpy as np
import sys
import subprocess
import math
import collections

import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import dnnl
import tvm.testing


has_dnnl_codegen = pytest.mark.skipif(
    not tvm.get_global_func("relay.ext.dnnl", True), reason="DNNL codegen not available"
)

run_module = tvm.testing.parameter(
    pytest.param(False, marks=[has_dnnl_codegen, *tvm.testing.requires_llvm.marks()]),
    pytest.param(True, marks=[has_dnnl_codegen, *tvm.testing.requires_llvm.marks()]),
    ids=["compile", "run"],
)

_bf16_supported = None


def bf16_supported():
    global _bf16_supported
    if _bf16_supported is None:
        _bf16_supported = False
        if sys.platform.startswith("darwin"):
            cpu_info = subprocess.check_output("sysctl -a", shell=True).strip().decode()
            for line in cpu_info.split("\n"):
                if line.startswith("hw.optional.avx512f"):
                    _bf16_supported = bool(int(line.split(":", 1)[1]))
        elif sys.platform.startswith("linux"):
            _bf16_supported = "avx512" in open("/proc/cpuinfo", "r").read()
    return _bf16_supported


def partition_for_dnnl(mod, params=None, alter_layout=True, prune_subgraphs=True):
    """Partition the graph greedily offloading supported operators to DNNL.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    Returns
    -------
    mod : Module
        Annotated and partitioned module.
    """
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)

    seq = tvm.transform.Sequential(
        [
            transform.CanonicalizeOps(),
            transform.InferType(),
            transform.SimplifyInference(),
            transform.FoldConstant(),
            transform.FoldScaleAxis(),
            # fold consecutive add ops to simplify pattern `conv2d-bias_add-bn-relu`
            transform.SimplifyExpr(),
            transform.FoldConstant(),
            # alter group conv /conv_transpose layout to `GOIHW` / `GIOHW`
            transform.Legalize(),
            transform.FoldConstant(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)

    with TempOpAttr("add", "FTVMLegalize", dnnl.legalize_add):
        leg_seq = tvm.transform.Sequential(
            [
                transform.Legalize(),
                transform.FoldConstant(),
            ]
        )
        with tvm.transform.PassContext(opt_level=3):
            mod = leg_seq(mod)


    byoc_seq = tvm.transform.Sequential(
        [
            transform.MergeDNNLGraph(),
            # transform.MergeComposite(dnnl.pattern_table()),
            tvm.transform.PrintIR(),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )

    with tvm.transform.PassContext(opt_level=3):
        mod = byoc_seq(mod)
        if prune_subgraphs:
            mod = dnnl.prune_dnnl_subgraphs(mod)
    return mod


def vmobj_to_list(o):
    if isinstance(o, tvm.nd.NDArray):
        o_np = o.numpy()
        if o_np.dtype == np.uint16:
            o_np = np.left_shift(o_np.astype("uint32"), 16).view("<f4")
        return [o_np]
    elif isinstance(o, tvm.runtime.container.ADT) or isinstance(o, list):
        return [vmobj_to_list(f) for f in o]
    else:
        raise RuntimeError("Unknown object type: %s" % type(o))


def assert_result_dict_holds(result_dict):
    for k1, k2 in itertools.combinations(result_dict, 2):
        res1 = vmobj_to_list(result_dict[k1])
        res2 = vmobj_to_list(result_dict[k2])
        for r1, r2 in zip(res1, res2):
            if "bf16" in k1 or "bf16" in k2:
                np.testing.assert_array_almost_equal(r1, r2, decimal=1)
            else:
                tvm.testing.assert_allclose(r1, r2, rtol=1e-3, atol=1e-3)


def check_dnnl_used(mod, subgraph_num=None):
    num_dnnl_subgraphs = sum([1 if "dnnl" in gv.name_hint else 0 for gv in mod.get_global_vars()])
    if subgraph_num:
        assert num_dnnl_subgraphs == subgraph_num
    else:
        assert num_dnnl_subgraphs >= 1


def run_and_verify(mod, input, params, target, run_module, subgraph_num=None, test_bf16=True):
    dev = tvm.cpu()
    result_dict = dict()
    for mode in ["graph"]:#, "vm"
        configs = [
            (False, False, False),
            (True, False, False),
        ]
        # if test_bf16 and bf16_supported():
        #     configs += [(True, False, True), (True, False, True)]

        for use_dnnl, alter_layout, use_bf16 in configs:
            result_key = (
                mode
                + ("_dnnl" if use_dnnl else "")
                + ("_layout" if alter_layout else "")
                + ("_bf16" if use_bf16 else "_fp32")
            )
            processed_mod = mod
            if use_bf16:
                processed_mod = relay.transform.ToMixedPrecision("bfloat16")(processed_mod)
                if tvm.ir.structural_equal(processed_mod, mod):
                    print("can not convert to bfloat16, skipping...")
                    continue
            if use_dnnl:
                processed_mod = partition_for_dnnl(processed_mod, params, alter_layout)
                check_dnnl_used(processed_mod)
            print(processed_mod)

            with tvm.transform.PassContext(opt_level=3):
                func = relay.create_executor(
                    mode, mod=processed_mod, device=dev, target=target
                ).evaluate()
            if run_module:
                if isinstance(input, dict):
                    result_dict[result_key] = func(**input, **params)
                else:
                    result_dict[result_key] = func(input, **params)

    if run_module:
        assert_result_dict_holds(result_dict)


def run_and_verify_func(
    config, run_module, subgraph_num=None, target="llvm", dtype="float32", test_bf16=True
):
    """Test a Relay func by compiling, running, and comparing TVM and DNNL outputs.
    Parameters
    ----------
    config : Tuple[relay.Function, Dict[str, NDArray], List[str]]
        A tuple containing 1) The function to test, 2) A dictionary of var names to input shapes and
        3) A list of which vars should be considered params.
    run_module: bool
        If True, the built module will be run after being compiled.
    """
    f, input_shapes, is_param = config
    params = {x: np.random.uniform(-1, 1, input_shapes[x]).astype(dtype) for x in is_param}
    input_dict = {
        k: np.random.uniform(-1, 1, v).astype(dtype)
        for k, v in input_shapes.items()
        if k not in is_param
    }
    run_and_verify(
        f,
        input_dict,
        params,
        subgraph_num=subgraph_num,
        target=target,
        run_module=run_module,
        test_bf16=test_bf16,
    )


def get_conv2d(
    x_shape=(1, 32, 8, 8),
    k_shape=(16, 32, 3, 3),
    groups=1,
    padding=(0, 0),
    strides=(1, 1),
    dilation=(1, 1),
    activation=None,
    dtype="float32",
):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    kernel = relay.var("kernel", shape=(k_shape), dtype=dtype)
    out = relay.nn.conv2d(
        x,
        kernel,
        kernel_size=k_shape[2:4],
        groups=groups,
        padding=padding,
        strides=strides,
        dilation=dilation,
        channels=k_shape[0],
    )
    dic = {"x": x_shape, "kernel": k_shape}
    param_lst = ["kernel"]

    if activation == "relu":
        return relay.nn.relu(out), dic, param_lst
    elif activation == "tanh":
        return relay.tanh(out), dic, param_lst
    elif activation == "sigmoid":
        return relay.sigmoid(out), dic, param_lst
    else:
        return out, dic, param_lst


def get_conv2d_weights_const(
    x_shape=(1, 32, 8, 8),
    k_shape=(16, 32, 3, 3),
    groups=1,
    padding=(0, 0),
    strides=(1, 1),
    dilation=(1, 1),
    dtype="float32",
):
    x = relay.var("x", shape=(x_shape), dtype=dtype)
    kernel = relay.const(np.random.random(k_shape).astype(dtype))
    out = relay.nn.conv2d(
        x,
        kernel,
        channels=k_shape[0],
        kernel_size=k_shape[2:4],
        groups=groups,
        padding=padding,
        strides=strides,
        dilation=dilation,
    )
    dic = {"x": x_shape}
    param_lst = []
    return out, dic, param_lst


def get_conv2d_bias(
    x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), activation=None, dtype="float32"
):
    conv, dic, param_lst = get_conv2d_weights_const(x_shape=x_shape, k_shape=k_shape, dtype=dtype)
    bias = relay.var("bias", shape=(k_shape[0],), dtype=dtype)
    out = relay.nn.bias_add(conv, bias)
    dic["bias"] = (k_shape[0],)
    param_lst += ["bias"]

    if activation == "relu":
        return relay.nn.relu(out), dic, param_lst
    elif activation == "tanh":
        return relay.tanh(out), dic, param_lst
    elif activation == "sigmoid":
        return relay.sigmoid(out), dic, param_lst
    else:
        return out, dic, param_lst


def get_conv2d_bias_bn_relu(x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), dtype="float32"):
    conv2d_bias, dic, param_lst = get_conv2d_bias(x_shape, k_shape, dtype=dtype)
    beta = relay.const(np.zeros(k_shape[0]).astype(dtype))
    gamma = relay.const(np.ones(k_shape[0]).astype(dtype))
    moving_mean = relay.const(np.zeros(k_shape[0]).astype(dtype))
    moving_var = relay.const(np.ones(k_shape[0]).astype(dtype))
    conv2d_bias_bn, _, _ = relay.nn.batch_norm(
        conv2d_bias,
        gamma=gamma,
        beta=beta,
        moving_mean=moving_mean,
        moving_var=moving_var,
        axis=1,
        center=True,
        scale=True,
        epsilon=1e-5,
    )
    return relay.nn.relu(conv2d_bias_bn), dic, param_lst


def get_conv2d_bias_sum_relu(x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), dtype="float32"):
    conv2d_bias, dic, param_lst = get_conv2d_bias(x_shape, k_shape, dtype=dtype)
    sum_data = relay.const(np.random.randint(x_shape).astype(dtype))
    conv2d_bias_sum = relay.add(sum_data, conv2d_bias)
    return relay.nn.relu(conv2d_bias_sum), dic, param_lst


def test_conv2d(run_module, dtype="float32"):
    x_shape = (1, 32, 8, 8)
    for k_shape, groups in [((16, 32, 3, 3), 1), ((32, 1, 3, 3), 32), ((32, 2, 3, 3), 16)]:
        for padding in [(0, 0), (1, 1)]:
            for strides in [(1, 1), (2, 2)]:
                for dilation in [(1, 1), (2, 2)]:
                    conv2d, dic, param_lst = get_conv2d(
                        x_shape=x_shape,
                        k_shape=k_shape,
                        groups=groups,
                        padding=padding,
                        strides=strides,
                        dilation=dilation,
                        dtype=dtype,
                    )
                    conv2d = tvm.IRModule.from_expr(conv2d)
                    config = conv2d, dic, param_lst
                    run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_conv2d_weights_const(run_module, dtype="float32"):
    x_shape = (1, 32, 8, 8)
    k_shape = (16, 32, 3, 3)
    conv2d, dic, param_lst = get_conv2d_weights_const(x_shape, k_shape, dtype=dtype)
    conv2d = tvm.IRModule.from_expr(conv2d)
    config = conv2d, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)

    x_shape = (1, 3, 8, 8)
    k_shape = (16, 3, 3, 3)
    conv2d, dic, param_lst = get_conv2d_weights_const(x_shape, k_shape, dtype=dtype)
    conv2d = tvm.IRModule.from_expr(conv2d)
    config = conv2d, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_conv2d_pattern(run_module, dtype="float32"):
    x_shape = (1, 32, 8, 8)
    k_shape = (16, 32, 3, 3)
    activation_lst = ["relu",]#None, "relu"
    for a in activation_lst:
        # conv2d, dic, param_lst = get_conv2d(x_shape, k_shape, activation=a, dtype=dtype)
        # conv2d = tvm.IRModule.from_expr(conv2d)
        # config = conv2d, dic, param_lst
        # run_and_verify_func(config, run_module=run_module, dtype=dtype)

        conv2d_bias, dic, param_lst = get_conv2d_bias(x_shape, k_shape, activation=a, dtype=dtype)
        conv2d_bias = tvm.IRModule.from_expr(conv2d_bias)
        config = conv2d_bias, dic, param_lst
        run_and_verify_func(config, run_module=run_module, dtype=dtype)
    
    # conv2d_bias_bn_relu, dic, param_lst = get_conv2d_bias_bn_relu(x_shape, k_shape, dtype=dtype)
    # conv2d_bias_bn_relu = tvm.IRModule.from_expr(conv2d_bias_bn_relu)
    # config = conv2d_bias_bn_relu, dic, param_lst
    # run_and_verify_func(config, run_module=run_module, dtype=dtype)

    # conv2d_bias_bn_relu, dic, param_lst = get_conv2d_bias_bn_relu(x_shape, k_shape, dtype=dtype)
    # conv2d_bias_bn_relu = tvm.IRModule.from_expr(conv2d_bias_bn_relu)
    # config = conv2d_bias_bn_relu, dic, param_lst
    # run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_conv2d_bias_sum_relu(run_module, dtype="float32"):
    x_shape=(1, 32, 8, 8)
    k_shape=(16, 32, 3, 3)
    def get_conv2d_bn_sum_relu(x_shape=(1, 32, 8, 8), k_shape=(16, 32, 3, 3), dtype="float32"):
        out, dic, param_lst = get_conv2d_bias(x_shape=x_shape, k_shape=k_shape, dtype=dtype)
        beta = relay.const(np.zeros(k_shape[0]).astype(dtype))
        gamma = relay.const(np.ones(k_shape[0]).astype(dtype))
        moving_mean = relay.const(np.zeros(k_shape[0]).astype(dtype))
        moving_var = relay.const(np.ones(k_shape[0]).astype(dtype))
        out, _, _ = relay.nn.batch_norm(
            out,
            gamma=gamma,
            beta=beta,
            moving_mean=moving_mean,
            moving_var=moving_var,
            axis=1,
            center=True,
            scale=True,
            epsilon=1e-5,
        )
        sum_data = relay.var("data1", shape=(1, 16, 6, 6), dtype=dtype)
        out = relay.add(out, sum_data)
        dic["data1"] = (1, 16, 6, 6)
        param_lst += ["data1"]
        return relay.nn.relu(out), dic, param_lst
    conv2d_bn_sum_relu, dic, param_lst = get_conv2d_bn_sum_relu(x_shape, k_shape, dtype=dtype)
    conv2d_bn_sum_relu = tvm.IRModule.from_expr(conv2d_bn_sum_relu)
    config = conv2d_bn_sum_relu, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)


def test_invalid_graph_pattern(run_module, dtype="float32"):
    def get_graph():
        x = relay.var("x", shape=(1, 3), dtype=dtype)
        y = relay.var("y", shape=(1, 3), dtype=dtype)
        z = relay.add(x, y)
        # w = relay.add(z, y)
        # out = relay.Tuple((z, w))
        f = tvm.IRModule.from_expr(z)
        return f, {"x": (1, 3), "y": (1, 3)}, []

    run_and_verify_func(get_graph(), run_module=run_module, dtype=dtype)


def test_pool2d(run_module, dtype="float32"):
    def get_graph(
        op,
        x_shape=(1, 3, 32, 32),
        pool_size=(2, 2),
        strides=(2, 2),
        padding=(0, 0),
        ceil_mode=False,
        count_include_pad=None,
    ):
        x = relay.var("x", shape=(x_shape), dtype=dtype)
        if count_include_pad is not None:
            out = op(
                x,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                ceil_mode=ceil_mode,
                count_include_pad=count_include_pad,
            )
        else:
            out = op(
                x,
                pool_size=pool_size,
                strides=strides,
                padding=padding,
                ceil_mode=ceil_mode,
            )
        out = tvm.IRModule.from_expr(out)
        return out, {"x": x_shape}, []

    for pool_size in [(2, 2), (3, 3)]:
        for strides in [(1, 1), (2, 2)]:
            for padding in [(0, 0), (1, 1), (0, 0, 1, 1)]:
                    run_and_verify_func(
                        get_graph(
                            relay.nn.max_pool2d,
                            pool_size=pool_size,
                            strides=strides,
                            padding=padding,
                        ),
                        run_module=run_module,
                    )

    
def test_resnetv1_rewrite(run_module, dtype="float32"):
    def get_graph():
        data_shape = (1, 256, 56, 56)
        w_shapes = [(64, 256, 1, 1),
                    (64, 64, 3, 3),
                    (256, 64, 1, 1),
                    (128, 256, 1, 1),
                    (128, 128, 3, 3),
                    (512, 128, 1, 1),
                    (512, 256, 1, 1),
                    ]
        x = relay.var("x", shape=data_shape, dtype=dtype)
        wights = [relay.const(np.random.randint(0, 1, w).astype(dtype)) for w in w_shapes]
        biases = [relay.const(np.random.randint(0, 1, w[0]).astype(dtype)) for w in w_shapes]

        conv1 = relay.nn.conv2d(x, wights[0], channels=w_shapes[0][0], kernel_size=w_shapes[0][2:4],
                                padding=(w_shapes[0][2]//2, w_shapes[0][3]//2))
        conv1 = relay.nn.bias_add(conv1, biases[0])
        conv1 = relay.nn.relu(conv1)

        conv2 = relay.nn.conv2d(conv1, wights[1], channels=w_shapes[1][0], kernel_size=w_shapes[1][2:4],
                                padding=(w_shapes[1][2]//2, w_shapes[1][3]//2))
        conv2 = relay.nn.bias_add(conv2, biases[1])
        conv2 = relay.nn.relu(conv2)

        conv3 = relay.nn.conv2d(conv2, wights[2], channels=w_shapes[2][0], kernel_size=w_shapes[2][2:4],
                                padding=(w_shapes[2][2]//2, w_shapes[2][3]//2))
        conv3 = relay.nn.bias_add(conv3, biases[2])
        conv3 = relay.add(conv3, x)
        conv3 = relay.nn.relu(conv3)

        left_conv4 = relay.nn.conv2d(conv3, wights[3], channels=w_shapes[3][0],
                                     strides=(2, 2), kernel_size=w_shapes[3][2:4],
                                     padding=(w_shapes[3][2]//2, w_shapes[3][3]//2))
        left_conv4 = relay.nn.bias_add(left_conv4, biases[3])
        left_conv4 = relay.nn.relu(left_conv4)

        left_conv5 = relay.nn.conv2d(left_conv4, wights[4], channels=w_shapes[4][0], kernel_size=w_shapes[4][2:4],
                                     padding=(w_shapes[4][2]//2, w_shapes[4][3]//2))
        left_conv5 = relay.nn.bias_add(left_conv5, biases[4])
        left_conv5 = relay.nn.relu(left_conv5)

        left_conv6 = relay.nn.conv2d(left_conv5, wights[5], channels=w_shapes[5][0], kernel_size=w_shapes[5][2:4],
                                     padding=(w_shapes[5][2]//2, w_shapes[5][3]//2))
        left_conv6 = relay.nn.bias_add(left_conv6, biases[5])

        right_conv7 = relay.nn.conv2d(conv3, wights[6], channels=w_shapes[6][0],
                                      strides=(2, 2), kernel_size=w_shapes[6][2:4],
                                      padding=(w_shapes[6][2]//2, w_shapes[6][3]//2))
        right_conv7 = relay.nn.bias_add(right_conv7, biases[6])

        out = relay.add(left_conv6, right_conv7)
        out = relay.nn.relu(out)

        dic = {"x": data_shape}
        param_lst = []
        return out, dic, param_lst

    net, dic, param_lst = get_graph()
    net = tvm.IRModule.from_expr(net)
    config = net, dic, param_lst
    run_and_verify_func(config, run_module=run_module, dtype=dtype)


if __name__ == "__main__":
    # # tvm.testing.main()
    # test_conv2d_weights_const(True)
    test_conv2d_pattern(True)
    # test_conv2d_bias_sum_relu(True)
    # test_invalid_graph_pattern(True)
    # test_pool2d(True)
    # test_resnetv1_rewrite(True)
