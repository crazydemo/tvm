'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import time
import mxnet as mx
import warnings

# from torch._C import T
warnings.filterwarnings("ignore")
from mxnet.gluon.model_zoo.vision import *
import tvm
from tvm.relay.op.contrib.dnnl import *
from tvm import relay
import tvm.contrib.graph_executor as runtime
import numpy as np
from tvm.relay.testing import *
import os
from tvm.contrib import utils

# model_dict = {'resnet50_v1': resnet50_v1}#{'mobilenet_v2_1_0': mobilenet_v2_1_0}
model_dict = {'resnet50_v1': resnet}
# model_dict = {'resnet50_v1': resnet, 'mobilenet_v2_1_0': mobilenet}

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

@relay.op.register_alter_op_layout("nn.conv2d", level=114)
def alter_conv2d(attrs, inputs, tinfos, out_type):
    data_layout = attrs['data_layout']
    kernel_layout = attrs['kernel_layout']
    data, weight = inputs
    new_attrs = dict(attrs)
    new_attrs['data_layout'] = 'NCHW'
    new_attrs['kernel_layout'] = 'OIHW16o'
    try:
        if weight.type_annotation.shape[1]>=8:
            new_attrs = dict(attrs)
            new_attrs['data_layout'] = 'NCHW8c'
            new_attrs['kernel_layout'] = 'OIHW8o8i'
            return relay.nn.conv2d(data, weight, **new_attrs)
    except:
        return relay.nn.conv2d(data, weight, **new_attrs)
    return relay.nn.conv2d(data, weight, **new_attrs)

def update_lib(lib):
    # Include the path of src/runtime/contrib/dnnl/dnnl.cc
    test_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    # source_dir = os.path.join(test_dir, "..", "..", "..")
    # contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")
    source_dir = os.path.join(test_dir, "..", "tvm")
    contrib_path = os.path.join(source_dir, "src", "runtime", "contrib")

    # Setup the gcc flag to compile DNNL code.
    kwargs = {}
    kwargs["options"] = ["-O2", "-std=c++14", "-I" + contrib_path]
    tmp_path = utils.tempdir()
    lib_name = 'lib.so'
    lib_path = tmp_path.relpath(lib_name)

    # The generated C code with DNNL APIs is compiled to a binary lib.so.
    lib.export_library(lib_path, fcompile=False, **kwargs)

    # Load the lib.so back to a runtime module.
    lib = tvm.runtime.load_module(lib_path)
    return lib


def benchmark(batch_size=1, batches=10, warmup=2):
    mx.random.seed(0)
    sample = mx.nd.random.uniform(-1.0, 1.0, shape=(batch_size,3,224,224))
    target = "llvm -model=platinum-8124m -mcpu=skylake-avx512"
    ctx = tvm.cpu()

    input_shape = (batch_size, 3, 224, 224)
    for model_name in model_dict.keys():
        # net = model_dict[model_name](pretrained=True)
        # net.hybridize(static_alloc=True, static_shape=True)
        # mod, params = relay.frontend.from_mxnet(net, shape={"data": input_shape}, dtype="float32")#port the Gluon model to a portable computational graph
        mod, params = model_dict[model_name].get_workload(batch_size=batch_size, dtype="float32")
        # print(mod)
        # mod =relay.transform.AlterOpLayout()(mod)
        # print('==================1 relayed model ==================')
        # print(mod["main"].astext(show_meta_data=False))
        # mod2 = relay.transform.MergeComposite(pattern_table())(mod)
        # # print('==================2 MergeComposite ==================')
        # # print(mod2["main"].astext(show_meta_data=False))
        # mod3 = relay.transform.AnnotateTarget(["dnnl"])(mod2)
        # print('==================3 AnnotateTarget ==================')
        # # print(mod3["main"].astext(show_meta_data=False))
        # # print(mod3)

        # mod4 = relay.transform.MergeCompilerRegions()(mod3)
        # print('==================4 MergeCompilerRegions ==================')
        # # print(mod4["main"].astext(show_meta_data=False))
        # # print(mod4)

        # mod5 = relay.transform.PartitionGraph()(mod4)
        # print('==================5 PartitionGraph ==================')
        # # print(mod5["main"].astext(show_meta_data=False))
        # # print(mod5)

        # mod6 =relay.transform.AlterOpLayout()(mod5)

    #     seq = tvm.transform.Sequential(
    #     [
    #         # transform.InferType(),
    #         relay.transform.MergeComposite(pattern_table()),
    #         relay.transform.AnnotateTarget(["dnnl"]),
    #         relay.transform.MergeCompilerRegions(),
    #         relay.transform.PartitionGraph(),
    #         relay.transform.RemoveUnusedFunctions(),
    #         relay.transform.AlterOpLayout()
    #         # relay.transform.ConvertLayout(
    #         #     {
    #         #         "nn.conv2d": ["NCHW", "OIHW"],
    #         #     }
    #         # ),
    #         # relay.transform.FoldConstant(),
    #     ]
    # )
        desired_layouts = {"nn.conv2d": ["NCHW8c", "OIHW8o8i"], "nn.batch_norm": ["NCHW8c", "OIHW8o8i"]}
        seq = tvm.transform.Sequential(
            [
                # transform.InferType(),
                # relay.transform.RemoveUnusedFunctions(),
                relay.transform.CanonicalizeOps(),
                relay.transform.AlterOpLayout(),
                # relay.transform.ConvertLayout(desired_layouts),
                # relay.transform.FoldConstant(),
                relay.transform.MergeComposite(pattern_table()),
                relay.transform.AnnotateTarget("dnnl"),
                relay.transform.MergeCompilerRegions(),
                relay.transform.PartitionGraph(),
                # transform.InferType(),
            ]
        )
        print(seq(mod))
        with tvm.transform.PassContext(opt_level=3):#, instruments=[PrintIR()]):# 
            json, lib, params = relay.build(seq(mod), "llvm", params=params)
            # with tvm.target.Target("llvm"):
            #     mod = seq(mod)

        # with tvm.transform.PassContext(opt_level=3):#compile the graph , instruments=[PrintIR()]
        #     json, lib, param = tvm.relay.build(mod, target="llvm", params=params)
        # with relay.build_config(opt_level=3):
        #     json, lib, params = relay.build(mod, "llvm", params=params)
        # lib = update_lib(lib)
        rt_mod = tvm.contrib.graph_executor.create(json, lib, ctx)#Create a runtime executor module given a graph and module.

        data = np.random.uniform(size=input_shape)
        # rt_mod.set_input("data", sample)
        rt_mod.set_input("data", tvm.nd.array(data.astype("float32")))
        for i in range(batches+warmup):
            if i == warmup:
                tic = time.time()
            out = rt_mod.run()
        with_fuse_ms = (time.time() - tic) / (batches) * 1000
        print("{}: with_fuse_ms: {:.4f} ms".format(model_name, with_fuse_ms))

benchmark(batch_size=1)