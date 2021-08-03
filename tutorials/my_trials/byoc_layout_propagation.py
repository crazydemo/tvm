# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
import time
import mxnet as mx
import warnings

from tvm.relay.build_module import GraphExecutor
from tvm.relay.transform.transform import AlterOpLayout

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
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay import transform, analysis

import mxnet as mx
from mxnet.gluon import HybridBlock, nn

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
    new_attrs['kernel_layout'] = 'OIHW'
    try:
        if weight.type_annotation.shape[1]>=8:
            new_attrs = dict(attrs)
            new_attrs['data_layout'] = 'NCHW8c'
            new_attrs['kernel_layout'] = 'OIHW8o8i'
            return relay.nn.conv2d(data, weight, **new_attrs)
    except:
        return relay.nn.conv2d(data, weight, **new_attrs)
    return relay.nn.conv2d(data, weight, **new_attrs)

# @relay.op.register_alter_op_layout("nn.conv2d", level=101)
# def alter_conv2d(attrs, inputs, tinfos, out_type):
#     data, weight = inputs
#     new_attrs = dict(attrs)
#     new_attrs["data_layout"] = "NCHW16c"
#     return relay.nn.conv2d(data, weight, **new_attrs)

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

class Model(HybridBlock):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        # use name_scope to give child Blocks appropriate names.
        # with self.name_scope():
        self.bn1 = nn.BatchNorm()
        self.bn2 = nn.BatchNorm()
        self.conv0 = nn.Conv2D(16, 3, use_bias=True)# + mx.nd.random.uniform(-1.0, 1.0, shape=(256))
        self.conv1 = nn.Conv2D(64, 3, use_bias=True)# + mx.nd.random.uniform(-1.0, 1.0, shape=(512))
        self.conv2 = nn.Conv2D(64, 3, use_bias=True)# + mx.nd.random.uniform(-1.0, 1.0, shape=(512))
        self.conv3 = nn.Conv2D(64, 3, use_bias=True)
        self.relu = nn.Activation('relu')

    def hybrid_forward(self, F, x):
        x = self.bn1(x)
        x = self.conv0(x)
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.bn2(self.conv3(x))
        return x1+x2+x3

def benchmark(batch_size=1, batches=10, warmup=2, cin=8):
    
    mx.random.seed(0)
    sample = np.ones((batch_size,cin,32, 32), np.float32)#mx.nd.random.uniform(-1.0, 1.0, shape=(batch_size,cin,8,8))
    sample_for_mxnet = mx.ndarray.array(sample)
    target = "llvm -model=platinum-8124m -mcpu=skylake-avx512"
    ctx = mx.cpu()
    # print("input:{}".format(sample_for_mxnet))

    input_shape = (batch_size, cin, 32, 32)
    
    model = Model()
    mx.random.seed(0)
    model.initialize(ctx=ctx)
    output = model(sample_for_mxnet)
    # print("mxnet output:{}".format(output))


    mod, params = relay.frontend.from_mxnet(model, shape={"data": input_shape}, dtype="float32")#port the Gluon model to a portable computational graph
    # print(mod)
    desired_layouts = {"nn.conv2d": ["NCHW8c", "OIHW8o8i"], "nn.batch_norm": ["NCHW8c", "OIHW8o8i"]}#, "nn.bias_add": ["NCHW8c", "OIHW8o8i"]}
    seq = tvm.transform.Sequential(
        [
            relay.transform.CanonicalizeOps(),
            relay.transform.SimplifyInference(),
            relay.transform.FoldScaleAxis(),

            # transform.AlterOpLayout(),
            transform.ConvertLayout(desired_layouts),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            
        ]
    )

    with tvm.transform.PassContext(opt_level=3):#, instruments=[PrintIR()]):#compile the graph x, instruments=[PrintIR()]
        graph, lib, param = tvm.relay.build(seq(mod), target="llvm", params=params)
    # lib = update_lib(lib)
    rt_mod = tvm.contrib.graph_executor.create(graph, lib, tvm.cpu())#Create a runtime executor module given a graph and module.

    # print("tvm input{}".format(tvm.nd.array(sample)))
    rt_mod.set_input("data", tvm.nd.array(sample), **param)
    rt_mod.run()
    tvm_output = rt_mod.get_output(0)
    # print(tvm_output.shape)
    # print("tvm output:{}".format(tvm_output))
    # for i in range(batches+warmup):
    #     if i == warmup:
    #         tic = time.time()
    #     out = rt_mod.run()
    # with_fuse_ms = (time.time() - tic) / (batches) * 1000
    # print("{}: with_fuse_ms: {:.4f} ms".format("net_with_branches", with_fuse_ms))

benchmark(batch_size=1)