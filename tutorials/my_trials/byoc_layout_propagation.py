# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
from hashlib import new
import time
import mxnet as mx
import warnings
from tvm._ffi._ctypes.ndarray import TVMPyCapsuleDestructor

from tvm.relay.build_module import GraphExecutor
from tvm.relay.expr import Tuple
from tvm.relay.op.contrib.arm_compute_lib import conv2d
from tvm.relay.transform.transform import AlterOpLayout, CanonicalizeCast

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
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib.download import download_testdata
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
from matplotlib import pyplot as plt

import mxnet as mx
from mxnet.gluon import HybridBlock, nn

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image


@relay.op.register_alter_op_layout("nn.conv2d", level=114)
def alter_conv2d(attrs, inputs, tinfos, out_type):
    data_layout = attrs['data_layout']
    kernel_layout = attrs['kernel_layout']
    data, weight = inputs
    new_attrs = dict(attrs)
    new_attrs['data_layout'] = 'NCHW'
    new_attrs['kernel_layout'] = 'OHWI8o'
    new_attrs['out_layout'] = 'NCHW8c'
    try:
        if weight.type_annotation.shape[1]>=8:
            new_attrs = dict(attrs)
            new_attrs['data_layout'] = 'NCHW8c'
            new_attrs['kernel_layout'] = 'OIHW8i8o'
            new_attrs['out_layout'] = 'NCHW8c'
            return relay.nn.conv2d(data, weight, **new_attrs)
    except:
        if weight.data.shape[1]>=8:
            new_attrs = dict(attrs)
            new_attrs['data_layout'] = 'NCHW8c'
            new_attrs['kernel_layout'] = 'OIHW8i8o'
            new_attrs['out_layout'] = 'NCHW8c'
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


@relay.transform.function_pass(opt_level=1)
class CustomPipeline:
    """Simple test function to replace one argument to another."""

    def __init__(self):
        # self.multiplier = multiplier
        self.cnt = 0
        self.merge_dict = {}
        self.branch_dict = {}
        self.block_dict = {}
        self.op_lst = []
        self.tmp_block = []

    # This function can define a pass.
    def transform_function(self, func, mod, ctx):
        self.merge_consecutive_add(func.body)
        res = self.rewrite_graph()
        res = relay.Function([self.op_lst[-1]], res)
        return res

    def rewrite_graph(self):
        start = max(self.block_dict.keys())
        new_node = self.op_lst[start]
        cur_node = self.op_lst[start]
        for i in range(start-1, -1, -1):
            
            cur_node = self.op_lst[i]

            if i+1 in self.block_dict.keys():
                node_for_next_block = new_node
                # print(node_for_next_block.op.name)
                # if i+1 in self.branch_dict.keys():
                #     self.branch_dict[i+1][-1] = node_for_next_block
            
            if i in self.branch_dict.keys():
                branch_lst = self.branch_dict[i]
                tmp_new_node = node_for_next_block
                for j in range(len(branch_lst)-2, -1, -1):
                    tmp_cur_node = branch_lst[j]
                    tmp_new_node = self.get_op(tmp_cur_node, tmp_new_node, tmp_cur_node.args[1])
                new_node = self.get_op(cur_node, new_node, tmp_new_node)

            elif i in self.merge_dict.keys():
                new_node = self.get_op(cur_node, new_node, self.merge_dict[i])
            
            elif cur_node.op.name=="add":
                new_node = self.get_op(cur_node, new_node, cur_node.args[1])
            elif cur_node.op.name=="nn.conv2d":
                new_node = self.get_op(cur_node, new_node, cur_node.args[1], cur_node.attrs)
            else:
                new_node = self.get_op(cur_node, new_node)
            print(new_node.op.name)
        return new_node

    def merge_consecutive_add(self, node):
        while node:
            try:
                if self.check_block(node):
                    self.block_dict[self.cnt] = node
                
                if self.check_consecutive_add(node):
                    a1 = node
                    a2 = a1.args[0]
                    data = relay.add(a1.args[1], a2.args[1])
                    self.merge_dict[self.cnt] = data
                    # print(obj.cnt)
                    node = a2

                if self.check_branch(node):
                    tmp_node = node.args[1]
                    self.branch_lst = [tmp_node]
                    while (not self.check_block(tmp_node)):
                        tmp_node = tmp_node.args[0]
                        self.branch_lst.append(tmp_node)
                    self.branch_dict[self.cnt] = self.branch_lst
                
                # print(node.op.name)
                self.op_lst.append(node)
                node = node.args[0]
                self.cnt += 1
            except:
                self.cnt = 0
                break

    def check_consecutive_add(self, node):
        try:
            return node.op.name=='add' and len(node.type_args[1].shape)==3 and node.args[0].op.name=='add' and len(node.args[0].type_args[1].shape)==3
        except:
            return False
    
    def check_block(self, node):
        try:
            return (node.op.name=='nn.relu' and not self.check_constant(node.args[0].args[1])) or node.op.name=='nn.max_pool2d'
        except:
            return False

    def check_branch(self, node):
        try:
            return node.op.name=='add' and not self.check_constant(node.args[1])
        except:
            return False

    def check_constant(self, node):
        try:
            return 'Constant' in str(type(node))
        except:
            return False
    
    def get_op(self, node, *args):
        if node.op.name=='nn.conv2d':
            if node.args[1].data.shape[-1]==3:
                return relay.nn.conv2d(args[0], args[1], padding=(1, 1))
            return relay.nn.conv2d(args[0], args[1])
        elif node.op.name=='nn.relu':
            return relay.nn.relu(args[0])
        elif node.op.name=='add':
            return relay.add(args[0], args[1])
        elif node.op.name=='nn.max_pool2d':
            return relay.nn.max_pool2d(args[0], (3, 3), (2, 2), padding=(1, 1))
        else:
            return False


class Model(HybridBlock):
    def __init__(self, **kwargs):
        super(Model, self).__init__(**kwargs)
        self.relu = nn.Activation('relu')
        self.conv0 = nn.Conv2D(64, 7, use_bias=False, strides=(2, 2), padding=(3,3))
        self.bn0 = nn.BatchNorm()
        self.maxpool = nn.MaxPool2D((3,3), (2,2), (1,1))
        self.conv1 = nn.Conv2D(64, 1, use_bias=True)
        self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(64, 3, use_bias=False, padding=(1, 1))
        self.bn2 = nn.BatchNorm()
        self.conv3 = nn.Conv2D(256, 1, use_bias=True)
        self.bn3 = nn.BatchNorm()

        self.conv11 = nn.Conv2D(256, 1, use_bias=False)
        self.bn11 = nn.BatchNorm()

        self.conv21 = nn.Conv2D(64, 1, use_bias=True)
        self.bn21 = nn.BatchNorm()
        self.conv22 = nn.Conv2D(64, 3, use_bias=False, padding=(1, 1))
        self.bn22 = nn.BatchNorm()
        self.conv23 = nn.Conv2D(256, 1, use_bias=True)
        self.bn23 = nn.BatchNorm()


    def hybrid_forward(self, F, x):
        x = self.relu(self.bn0(self.conv0(x)))
        x_ = self.maxpool(x)
        x = self.relu(self.bn1(self.conv1(x_)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x_ = self.relu(x + self.bn11(self.conv11(x_)))

        x = self.relu(self.bn21(self.conv21(x_)))
        x = self.relu(self.bn22(self.conv22(x)))
        x = self.bn23(self.conv23(x))

        x = self.relu(x + x_)
        return x

def benchmark(batch_size=1, batches=10, warmup=2, cin=3):
    
    mx.random.seed(0)
    # sample = sample = np.ones((batch_size, cin, 8, 8))#np.random.rand(batch_size, cin, 8, 8)
    sample = np.random.rand(batch_size, cin, 224, 224)
    # sample_for_mxnet = mx.ndarray.array(sample)
    # img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    # img_name = "cat.png"
    # img_path = download_testdata(img_url, img_name, module="data")
    # image = Image.open(img_path).resize((224, 224))
    # sample = transform_image(image)
    target = "llvm -model=platinum-8124m -mcpu=skylake-avx512"
    ctx = mx.cpu()
    # print("input:{}".format(sample_for_mxnet))

    input_shape = (batch_size, cin, 224, 224)
    
    model = Model()
    mx.random.seed(0)
    model.initialize(ctx=ctx)
    sample_for_mxnet = mx.ndarray.array(sample)
    output = model(sample_for_mxnet)
    # print("mxnet output:{}".format(output))


    mod, params = relay.frontend.from_mxnet(model, shape={"data": input_shape}, dtype="float32")#port the Gluon model to a portable computational graph
    # print(mod)
    # desired_layouts = {"nn.conv2d": ["NCHW8c", "OIHW8o8i"], "nn.batch_norm": ["NCHW8c", "OIHW8o8i"]}#, "nn.bias_add": ["NCHW8c", "OIHW8o8i"]}
    seq = tvm.transform.Sequential(
        [
            relay.transform.CanonicalizeOps(),
            # relay.transform.SimplifyInference(),
            # relay.transform.FoldScaleAxis(),
            # relay.transform.SimplifyExpr(),
            relay.transform.InferType(),
            relay.transform.SimplifyInference(),
            relay.transform.FoldConstant(),
            relay.transform.FoldScaleAxis(),
            # tvm.transform.PrintIR(),

            CustomPipeline(),
            relay.transform.FoldConstant(),
            # tvm.transform.PrintIR(),

            transform.AlterOpLayout(),
            # tvm.transform.PrintIR(),
            # transform.ConvertLayout(desired_layouts),
            transform.MergeComposite(pattern_table()),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
            tvm.transform.PrintIR(),
            
        ]
    )
    
    if params:
        mod["main"] = bind_params_by_name(mod["main"], params)
    with tvm.transform.PassContext(opt_level=3):#, instruments=[PrintIR()]):#compile the graph x, instruments=[PrintIR()]
        graph, lib, param = tvm.relay.build(seq(mod), target="llvm", params=params)
    lib = update_lib(lib)
    rt_mod = tvm.contrib.graph_executor.create(graph, lib, tvm.cpu())#Create a runtime executor module given a graph and module.

    # print("tvm input{}".format(tvm.nd.array(sample)))
    rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")), **param)
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