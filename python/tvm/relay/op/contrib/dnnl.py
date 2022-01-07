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
# pylint: disable=invalid-name, unused-argument
"""DNNL library supported operators.
There are two ways to registering a function for an op to indicate if it is
supported by DNNL.

- The first and simplest way is to use the helper so that
users only need to provide the operator name and a boolean value to indicate if
it is supported. For example:

    .. code-block:: python

      add = _register_external_op_helper("add")
      add = _register_external_op_helper("add", True)
      add = _register_external_op_helper("add", False)

- The other way is to implement the function by themselves to
check the attributes of the op and decide if it should be offloaded to DNNL.
"""
import logging

import tvm.ir
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name

from ... import _ffi_api
from ...dataflow_pattern import wildcard, is_op
from .register import register_pattern_table

logger = logging.getLogger("DNNL")


def _register_external_op_helper(op_name, supported=True):
    """The helper function to indicate that a given operator can be supported
    by DNNL.

    Paramters
    ---------
    op_name : Str
        The name of operator that will be registered.

    Returns
    -------
    f : callable
        A function that returns if the operator is supported by DNNL.
    """

    @tvm.ir.register_op_attr(op_name, "target.dnnl")
    def _func_wrapper(expr):
        return supported

    return _func_wrapper


_register_external_op_helper("nn.batch_norm")
_register_external_op_helper("nn.conv1d")
_register_external_op_helper("nn.conv2d")
_register_external_op_helper("nn.conv3d")
_register_external_op_helper("nn.conv2d_transpose")
_register_external_op_helper("nn.conv3d_transpose")
_register_external_op_helper("nn.dense")
"""Pooling"""
_register_external_op_helper("nn.max_pool2d")
_register_external_op_helper("nn.avg_pool2d")
_register_external_op_helper("nn.max_pool3d")
_register_external_op_helper("nn.avg_pool3d")
"""Activation"""
_register_external_op_helper("abs")
_register_external_op_helper("clip")
_register_external_op_helper("exp")
_register_external_op_helper("log")
_register_external_op_helper("sqrt")
_register_external_op_helper("round")
_register_external_op_helper("logsumexp")
_register_external_op_helper("nn.relu")
_register_external_op_helper("nn.leaky_relu")
_register_external_op_helper("tanh")
_register_external_op_helper("sigmoid")
_register_external_op_helper("nn.softmax")
"""Binary"""
_register_external_op_helper("add")
_register_external_op_helper("multiply")


def make_conv_pattern(conv_name, with_bias=True, with_eltwise=None):
    """Create patterns related to nn.conv2d.

    Parameters
    ----------
    with_bias : bool
        Whether attach `bias_add` to `nn.conv2d`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    conv_out : CallPattern
        Call node sequence.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    conv = is_op(conv_name)(data, weight)
    if with_bias:
        conv_out = is_op("add")(conv, bias)
    else:
        conv_out = conv
    if with_eltwise:
        return is_op(with_eltwise)(conv_out)
    return conv_out


def make_dense_pattern(with_bias=True, with_eltwise=None):
    """Create patterns related to nn.dense.

    Parameters
    ----------
    with_bias : bool
        Whether attach `bias_add` to `nn.dense`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    dense_out : CallPattern
        Call node sequence.
    """
    data = wildcard()
    weight = wildcard()
    bias = wildcard()
    dense = is_op("nn.dense")(data, weight)
    if with_bias:
        dense_out = is_op("add")(dense, bias)
    else:
        dense_out = dense
    if with_eltwise:
        dense_out = is_op(with_eltwise)(dense_out)
    return dense_out


def make_dnnl_pattern(op, with_bias, with_eltwise):
    """Create dnnl patterns.

    Parameters
    ----------
    op : str
        The first call node's op name.
    with_bias : bool
        Whether attach `bias_add` to `nn.dense`.
    with_eltwise : str
        The attached elementwise post-op name.
    Returns
    -------
    pattern : Tuple(pattern_name, CallPattern)
        Created pattern name, along with its CallPattern.
    """
    pat_name = op.replace("nn", "dnnl")
    pat_name += "_bias" if with_bias else ""
    pat_name += ("_" + with_eltwise.split(".")[-1]) if with_eltwise else ""
    if op == "nn.conv1d":
        dnnl_pattern = (pat_name, make_conv_pattern(op, with_bias, with_eltwise))
    elif op == "nn.conv2d":
        dnnl_pattern = (pat_name, make_conv_pattern(op, with_bias, with_eltwise))
    elif op == "nn.conv3d":
        dnnl_pattern = (pat_name, make_conv_pattern(op, with_bias, with_eltwise))
    elif op == "nn.conv2d_transpose":
        dnnl_pattern = (pat_name, make_conv_pattern(op, with_bias, with_eltwise))
    elif op == "nn.conv3d_transpose":
        dnnl_pattern = (pat_name, make_conv_pattern(op, with_bias, with_eltwise))
    elif op == "nn.dense":
        dnnl_pattern = (pat_name, make_dense_pattern(with_bias, with_eltwise))
    else:
        logger.warning(
            "Currently, only conv1d, conv2d, conv2d_transpose, conv3d_transpose and dense op are supported, but got %s.",
            op,
        )
        dnnl_pattern = ()
    return dnnl_pattern


@register_pattern_table("dnnl")
def pattern_table():
    """Create dnnl patterns.

    Returns
    -------
    dnnl_patterns : List[dnnl_pattern]
        Created patterns.
    """
    elt_list = ["nn.relu", "tanh", "sigmoid", None]
    dnnl_patterns = []
    for with_bias in [True, False]:
        for elt in elt_list:
            if not with_bias and not elt:
                return dnnl_patterns
            for conv_name in [
                "nn.conv1d",
                "nn.conv2d",
                "nn.conv3d",
                "nn.conv2d_transpose",
                "nn.conv3d_transpose",
            ]:
                dnnl_patterns.append(make_dnnl_pattern(conv_name, with_bias, elt))
            dnnl_patterns.append(make_dnnl_pattern("nn.dense", with_bias, elt))
    return dnnl_patterns


def get_optimal_layout_for_conv(input_size, weight_shape, out_shape, paddings, strides, dilates, G):
    """Get the optimal layout of dnnl, given shape of conv2d.

    Parameters
    ----------
    N,IC,KH,KW,OC,SH,SW,PH_L,PH_R,PW_L,PW_R,OH,OW,G : Int
                                                     Input argument.

    Returns
    -------
    layouts : string
              The result.
    """
    return _ffi_api.get_optimal_layout_for_conv(
        input_size,
        weight_shape,
        out_shape,
        paddings,
        strides,
        dilates,
        G,
    )


def get_optimal_layout_for_deconv(
    input_size, weight_shape, out_shape, paddings, output_paddings, strides, dilates, G
):
    """Get the optimal layout of dnnl, given shape of tranpose conv2d.

    Parameters
    ----------
    N,IC,KH,KW,OC,SH,SW,PH_L,PH_R,PW_L,PW_R,OH,OW,G : Int
                                                     Input argument.

    Returns
    -------
    layouts : string
              The result.
    """
    return _ffi_api.get_optimal_layout_for_deconv(
        input_size,
        weight_shape,
        out_shape,
        paddings,
        output_paddings,
        strides,
        dilates,
        G,
    )


def alter_conv(attrs, inputs, tinfos, out_type):
    """The convolution's layout auto-query func for dnnl."""

    def get_shape(tensor):
        if isinstance(tensor, relay.expr.Var):
            return tensor.type_annotation.concrete_shape
        elif isinstance(tensor, relay.expr.Constant):
            return tensor.data.shape
        elif isinstance(tensor, tvm.ir.tensor_type.TensorType):
            return tensor.concrete_shape
        else:
            raise TypeError("Unsupport data type: %s" % type(tensor))

    def trans_data(input_data, is_weight=False, conv_type=1):
        if conv_type == 1:
            data_dic = {"a": "N", "b": "C", "c": "W"}
            weight_dic = {"a": "O", "b": "I", "c": "W", "d": "G"}
        elif conv_type == 2:
            data_dic = {"a": "N", "b": "C", "c": "H", "d": "W"}
            weight_dic = {"a": "O", "b": "I", "c": "H", "d": "W", "e": "G"}
        elif conv_type == 3:
            data_dic = {"a": "N", "b": "C", "c": "D", "d": "H", "e": "W"}
            weight_dic = {"a": "O", "b": "I", "c": "D", "d": "H", "e": "W", "f": "G"}

        dic = weight_dic if is_weight else data_dic
        res = ""

        for i in input_data:
            if i.isupper():
                i = i.lower()
                res += dic[i]
                dic[i] = dic[i].lower()
            elif i.islower():
                res += dic[i]
            elif i.isdigit():
                res += i
            else:
                raise ValueError("Unsupport layout format: %s" % input_data)
        return res

    data, weight = inputs
    weight_shape = ",".join([str(x) for x in get_shape(weight)])
    out_shape = ",".join([str(x) for x in get_shape(out_type)])
    paddings = ",".join([str(x) for x in attrs.get_int_tuple("padding")])
    strides = ",".join([str(x) for x in attrs.get_int_tuple("strides")])
    dilates = ",".join([str(x) for x in attrs.get_int_tuple("dilation")])
    G = str(attrs.groups)
    new_attrs = dict(attrs)
    conv_type = len(get_shape(weight)) - 2

    # To do optimal layout transform for group convolution.
    # Set group convolution as plain format currently.
    if int(G) > 1:
        if conv_type == 1:
            return relay.nn.conv1d(data, weight, **attrs)
        elif conv_type == 2:
            return relay.nn.conv2d(data, weight, **attrs)
        elif conv_type == 3:
            return relay.nn.conv3d(data, weight, **attrs)

    res = get_optimal_layout_for_conv(
        len(get_shape(weight)), weight_shape, out_shape, paddings, strides, dilates, G
    )
    src_df, weight_df, dst_df = res.split(",")
    new_attrs["data_layout"] = trans_data(src_df, is_weight=False, conv_type=conv_type)
    new_attrs["kernel_layout"] = trans_data(weight_df, is_weight=True, conv_type=conv_type)
    new_attrs["out_layout"] = trans_data(dst_df, is_weight=False, conv_type=conv_type)

    if conv_type == 1:
        return relay.nn.conv1d(data, weight, **new_attrs)
    elif conv_type == 2:
        return relay.nn.conv2d(data, weight, **new_attrs)
    elif conv_type == 3:
        return relay.nn.conv3d(data, weight, **new_attrs)


def alter_deconv(attrs, inputs, tinfos, out_type):
    """The transpose convolution's layout auto-query func for dnnl."""

    def get_shape(tensor):
        if isinstance(tensor, relay.expr.Var):
            return tensor.type_annotation.concrete_shape
        elif isinstance(tensor, relay.expr.Constant):
            return tensor.data.shape
        elif isinstance(tensor, tvm.ir.tensor_type.TensorType):
            return tensor.concrete_shape
        else:
            raise TypeError("Unsupport data type: %s" % type(tensor))

    def trans_data(input_data, is_weight=False, conv_type=1):
        if conv_type == 1:
            data_dic = {"a": "N", "b": "C", "c": "W"}
            weight_dic = {"a": "O", "b": "I", "c": "W", "d": "G"}
        elif conv_type == 2:
            data_dic = {"a": "N", "b": "C", "c": "H", "d": "W"}
            weight_dic = {"a": "O", "b": "I", "c": "H", "d": "W", "e": "G"}
        elif conv_type == 3:
            data_dic = {"a": "N", "b": "C", "c": "D", "d": "H", "e": "W"}
            weight_dic = {"a": "O", "b": "I", "c": "D", "d": "H", "e": "W", "f": "G"}

        dic = weight_dic if is_weight else data_dic
        res = ""

        for i in input_data:
            if i.isupper():
                i = i.lower()
                res += dic[i]
                dic[i] = dic[i].lower()
            elif i.islower():
                res += dic[i]
            elif i.isdigit():
                res += i
            else:
                raise ValueError("Unsupport layout format: %s" % input_data)
        return res

    data, weight = inputs
    weight_shape = ",".join([str(x) for x in get_shape(weight)])
    out_shape = ",".join([str(x) for x in get_shape(out_type)])
    paddings = ",".join([str(x) for x in attrs.get_int_tuple("padding")])
    output_paddings = ",".join([str(x) for x in attrs.get_int_tuple("output_padding")])
    strides = ",".join([str(x) for x in attrs.get_int_tuple("strides")])
    dilates = ",".join([str(x) for x in attrs.get_int_tuple("dilation")])
    G = str(attrs.groups)
    new_attrs = dict(attrs)
    conv_type = len(get_shape(weight)) - 2

    # To do optimal layout transform for group convolution.
    # Set group convolution as plain format currently.
    if int(G) > 1:
        if conv_type == 1:
            return relay.nn.conv1d_transpose(data, weight, **attrs)
        elif conv_type == 2:
            return relay.nn.conv2d_transpose(data, weight, **attrs)
        elif conv_type == 3:
            return relay.nn.conv3d_transpose(data, weight, **attrs)

    res = get_optimal_layout_for_deconv(
        len(get_shape(weight)),
        weight_shape,
        out_shape,
        paddings,
        output_paddings,
        strides,
        dilates,
        G,
    )
    src_df, weight_df, dst_df = res.split(",")
    new_attrs["data_layout"] = trans_data(src_df, is_weight=False, conv_type=conv_type)
    new_attrs["kernel_layout"] = trans_data(weight_df, is_weight=True, conv_type=conv_type)
    new_attrs["out_layout"] = trans_data(dst_df, is_weight=False, conv_type=conv_type)

    if conv_type == 1:
        return relay.nn.conv1d_transpose(data, weight, **new_attrs)
    elif conv_type == 2:
        return relay.nn.conv2d_transpose(data, weight, **new_attrs)
    elif conv_type == 3:
        return relay.nn.conv3d_transpose(data, weight, **new_attrs)


def partition_for_dnnl(mod, params=None, alter_layout=True):
    """Partition the graph greedily offloading supported operators to DNNL.

    Parameters
    ----------
    mod : Module
        The module to run passes on.
    params : Optional[Dict[str, NDArray]]
        Constant input parameters.
    alter_layout : bool
        Whether alter conv2d's layout to the optimal one of dnnl.
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
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = seq(mod)
    if alter_layout:
        from tvm.relay.testing.temp_op_attr import TempOpAttr

        with TempOpAttr("nn.conv1d", "FTVMAlterOpLayout", alter_conv):
            with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", alter_conv):
                with TempOpAttr("nn.conv3d", "FTVMAlterOpLayout", alter_conv):
                    with TempOpAttr("nn.conv2d_transpose", "FTVMAlterOpLayout", alter_deconv):
                        with TempOpAttr("nn.conv3d_transpose", "FTVMAlterOpLayout", alter_deconv):
                            alter_layout_seq = tvm.transform.Sequential(
                                [
                                    transform.AlterOpLayout(),
                                    transform.FoldConstant(),
                                ]
                            )
                            with tvm.transform.PassContext(opt_level=3):
                                mod = alter_layout_seq(mod)

    byoc_seq = tvm.transform.Sequential(
        [
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = byoc_seq(mod)
    return mod
