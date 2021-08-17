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
from tvm.relay.build_module import bind_params_by_name
from tvm.contrib.download import download_testdata
from mxnet.gluon.model_zoo.vision import get_model
from PIL import Image
from matplotlib import pyplot as plt
import tvm.contrib.graph_executor as graph_executor
# from tvm.contrib.debugger import debug_executor as graph_executor
# 
# model_dict = {'resnet50_v1': resnet50_v1}#{'mobilenet_v2_1_0': mobilenet_v2_1_0}
model_dict = {'resnet50_v1': resnet}

@tvm.instrument.pass_instrument
class PrintIR:
    """Print the name of the pass, the IR, only before passes execute."""

    def run_before_pass(self, mod, info):
        print("Running pass: {}", info)
        print(mod)

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
            
            if i in self.branch_dict.keys():
                branch_lst = self.branch_dict[i]
                tmp_new_node = node_for_next_block
                for j in range(len(branch_lst)-2, -1, -1):
                    tmp_cur_node = branch_lst[j]
                    tmp_new_node = self.get_op(tmp_cur_node, tmp_new_node, tmp_cur_node.args[1])
                new_node = self.get_op(cur_node, new_node, tmp_new_node)

            elif cur_node.op.name=="add":
                new_node = self.get_op(cur_node, new_node, cur_node.args[1])
            elif cur_node.op.name=="nn.conv2d":
                new_node = self.get_op(cur_node, new_node, cur_node.args[1], cur_node.attrs)
            elif cur_node.op.name=="nn.dense":
                new_node = self.get_op(cur_node, new_node, cur_node.args[1])
            elif cur_node.op.name=="nn.batch_flatten":
                new_node = self.get_op(cur_node, new_node)
            else:
                new_node = self.get_op(cur_node, new_node)
                # return new_node
                # if i==4:
                #     return new_node
        # print(new_node.op.name)
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
            # print("check ...")
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
            # print(node.args[1].data.shape)
            if node.args[1].data.shape[-1]==3:
                return relay.nn.conv2d(args[0], args[1], padding=(1, 1))
            elif node.args[1].data.shape[-1]==1 and node.args[1].data.shape[1]/node.args[1].data.shape[0]==2:
                return relay.nn.conv2d(args[0], args[1], strides=(2, 2))
            elif node.args[1].data.shape[-1]==1 and node.args[1].data.shape[0]/node.args[1].data.shape[1]==2:
                return relay.nn.conv2d(args[0], args[1], strides=(2, 2))
            return relay.nn.conv2d(args[0], args[1])
        elif node.op.name=='nn.relu':
            return relay.nn.relu(args[0])
        elif node.op.name=='add':
            return relay.add(args[0], args[1])
        elif node.op.name=='nn.max_pool2d':
            return relay.nn.max_pool2d(args[0])
        elif node.op.name=='nn.global_avg_pool2d':
            return relay.nn.global_avg_pool2d(args[0])
        elif node.op.name=='nn.batch_flatten':
            return relay.nn.batch_flatten(args[0])
        elif node.op.name=='nn.dense':
            return relay.nn.dense(args[0], args[1])
        else:
            return False

@relay.op.register_alter_op_layout("nn.conv2d", level=114)
def alter_conv2d(attrs, inputs, tinfos, out_type):
    data, weight = inputs
    new_attrs = dict(attrs)
    new_attrs['data_layout'] = 'NCHW'
    new_attrs['kernel_layout'] = 'OHWI8o'
    new_attrs['out_layout'] = 'NCHW8c'
    try:
        if weight.type_annotation.shape[1]>=8:
            new_attrs = dict(attrs)
            new_attrs['data_layout'] = 'NCHW8c'
            new_attrs['kernel_layout'] = 'OIHW8i8o'#'OIHW'
            new_attrs['out_layout'] = 'NCHW8c'
            return relay.nn.conv2d(data, weight, **new_attrs)
    except:
        if weight.data.shape[1]>=8:
            new_attrs = dict(attrs)
            new_attrs['data_layout'] = 'NCHW8c'
            new_attrs['kernel_layout'] = 'OIHW8i8o'#'OIHW'
            new_attrs['out_layout'] = 'NCHW8c'
            return relay.nn.conv2d(data, weight, **new_attrs)
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

def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def benchmark(batch_size=1, batches=10, warmup=2):
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_name = "cat.png"
    img_path = download_testdata(img_url, img_name, module="data")
    image = Image.open(img_path).resize((224, 224))
    sample = transform_image(image)
    # print("x", sample.shape)
    # np.random.seed(0)
    # sample = np.random.rand(batch_size, 3, 224, 224)#np.ones((batch_size, 3, 224, 224))#

    target = "llvm -model=platinum-8124m -mcpu=skylake-avx512"
    ctx = tvm.cpu()

    input_shape = (batch_size, 3, 224, 224)
    for model_name in model_dict.keys():
        block = mx.gluon.model_zoo.vision.get_resnet(1, 50, pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype="float32"
        )
        # mod, params = model_dict[model_name].get_workload(batch_size=batch_size, dtype="float32")
        # print(mod)
        # sample_for_mxnet = mx.ndarray.array(sample)
        # output = block(sample_for_mxnet)
        # print("mxnet output:{}".format(output))
        # print(params)
        desired_layouts = {"nn.conv2d": ["NCHW16c", "OIHW16o16i"],"nn.batch_norm": ["NCHW16c", "OIHW16o16i"]}#
        seq = tvm.transform.Sequential(
            [
                relay.transform.CanonicalizeOps(),
                relay.transform.SimplifyInference(),
                relay.transform.FoldScaleAxis(),
                relay.transform.SimplifyExpr(),

                CustomPipeline(),
                relay.transform.FoldConstant(),
                # tvm.transform.PrintIR(),
                # relay.transform.FuseOps(),
                # tvm.transform.PrintIR(),
                relay.transform.AlterOpLayout(),
                # tvm.transform.PrintIR(),
                # relay.transform.ConvertLayout(desired_layouts),
                relay.transform.MergeComposite(pattern_table()),
                relay.transform.AnnotateTarget("dnnl"),
                relay.transform.MergeCompilerRegions(),
                relay.transform.PartitionGraph(),
                # tvm.transform.PrintIR(),
            ]
        )


        if params:
            mod["main"] = bind_params_by_name(mod["main"], params)
        with tvm.transform.PassContext(opt_level=3, instruments=[PrintIR()]):# 
            json, lib, params = relay.build(seq(mod), "llvm", params=params)
        lib = update_lib(lib)
        # print(json)
        rt_mod = graph_executor.create(json, lib, ctx)#, dump_root="/home/zy/tvm/tutorials/experiment_res/")#Create a runtime executor module given a graph and module.
        
        rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
        rt_mod.set_input(**params)
        rt_mod.run()

        # out= rt_mod.debug_get_output("tvmgen_default_dnnl_0", out=tvm.nd.empty((1, 64, 112, 112), dtype="float32"))
        # print(out)
        # tvm_out = rt_mod.get_output(1, tvm.nd.empty((1, 1000), "float32")).numpy()
        # print(tvm_out)
        # for i in range(batches+warmup):
        #     if i == warmup:
        #         tic = time.time()
        #     out = rt_mod.run()
        #     # out.wait_to_read()
        # with_fuse_fps = batches * batch_size / (time.time() - tic)
        # print("{}: with_fuse_ms: {:.4f} ms".format(model_name, with_fuse_fps))
        tvm_output = rt_mod.get_output(0)
        # print(tvm_output)
        

benchmark(batch_size=1)