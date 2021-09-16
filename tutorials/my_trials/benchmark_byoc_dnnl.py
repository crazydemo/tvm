'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import time
import mxnet as mx
import warnings
import gluoncv

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
#from tvm.runtime.vm import VirtualMachine
#from tvm.contrib.debugger import debug_executor as graph_executor
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

            elif i in self.merge_dict.keys():
                new_node = self.get_op(cur_node, new_node, self.merge_dict[i])

            elif cur_node.op.name=="add":
                new_node = self.get_op(cur_node, new_node, cur_node.args[1])
            elif cur_node.op.name=="nn.conv2d":
                new_node = self.get_op(cur_node, new_node, cur_node.args[1], cur_node.attrs)
                # return new_node
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
                
                elif self.check_consecutive_add(node):
                    a1 = node
                    a2 = a1.args[0]
                    data = relay.add(a1.args[1], a2.args[1])
                    self.merge_dict[self.cnt] = data
                    node = a2

                elif self.check_branch(node):
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
            return relay.nn.conv2d(args[0], args[1], **node.attrs)
        elif node.op.name=='nn.relu':
            return relay.nn.relu(args[0])
        elif node.op.name=='add':
            return relay.add(args[0], args[1])
        elif node.op.name=='nn.max_pool2d':
            return relay.nn.max_pool2d(args[0], **node.attrs)
        elif node.op.name=='nn.global_avg_pool2d':
            return relay.nn.global_avg_pool2d(args[0], **node.attrs)
        elif node.op.name=='nn.batch_flatten':
            return relay.nn.batch_flatten(args[0])
        elif node.op.name=='nn.dense':
            return relay.nn.dense(args[0], args[1], **node.attrs)
        else:
            return False

@relay.op.register_alter_op_layout("nn.conv2d", level=114)
def alter_conv2d(attrs, inputs, tinfos, out_type):
    data, weight = inputs
    new_attrs = dict(attrs)
    new_attrs['data_layout'] = 'NCHW'
    new_attrs['kernel_layout'] = 'OHWI16o'
    new_attrs['out_layout'] = 'NCHW16c'
    try:
        if weight.type_annotation.shape[1]>=16:
            new_attrs = dict(attrs)
            new_attrs['data_layout'] = 'NCHW16c'
            new_attrs['kernel_layout'] = 'OIHW16i16o'#'OIHW'
            new_attrs['out_layout'] = 'NCHW16c'
            return relay.nn.conv2d(data, weight, **new_attrs)
    except:
        if weight.data.shape[1]>=16:
            new_attrs = dict(attrs)
            new_attrs['data_layout'] = 'NCHW16c'
            new_attrs['kernel_layout'] = 'OIHW16i16o'#'OIHW'
            new_attrs['out_layout'] = 'NCHW16c'
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

def benchmark(batch_size=1, batches=400, warmup=100):
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_name = "cat.png"
    img_path = download_testdata(img_url, img_name, module="data")
    image = Image.open(img_path).resize((224, 224))
    sample = transform_image(image)
    # print("x", sample.shape)
    # np.random.seed(0)
    #sample = np.random.rand(batch_size, 3, 224, 224)#np.ones((batch_size, 3, 224, 224))#

    target = "llvm -model=platinum-8124m -mcpu=skylake-avx512"
    ctx = tvm.cpu()

    input_shape = (batch_size, 3, 224, 224)
    for model_name in model_dict.keys():
        block = gluoncv.model_zoo.get_model('ResNet50_v1b', pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype="float32"
        )
        # mod, params = model_dict[model_name].get_workload(batch_size=batch_size, dtype="float32")
        # print(mod)
        #sample_for_mxnet = mx.ndarray.array(sample)
        #output = block(sample_for_mxnet)
        #print("mxnet output:{}".format(output))
        # print(params)
        # desired_layouts = {"nn.conv2d": ["NCHW16c", "OIHW16o16i"],"nn.batch_norm": ["NCHW16c", "OIHW16o16i"]}#
        seq = tvm.transform.Sequential(
            [ 
               relay.transform.CanonicalizeOps(),
                relay.transform.InferType(),
                relay.transform.SimplifyInference(),
                relay.transform.FoldConstant(),
                relay.transform.FoldScaleAxis(),

                CustomPipeline(),
                relay.transform.FoldConstant(),
                
                relay.transform.AlterOpLayout(),
                
                relay.transform.MergeComposite(pattern_table()),
                relay.transform.AnnotateTarget("dnnl"),
                relay.transform.MergeCompilerRegions(),
                relay.transform.PartitionGraph(),
            ]
        )


        if params:
            mod["main"] = bind_params_by_name(mod["main"], params)
        with tvm.transform.PassContext(opt_level=3):#, instruments=[PrintIR()]):# 
            json, lib, params = relay.build(seq(mod), "llvm", params=params)
        #exe =  relay.vm.compile(mod, target="llvm", params=params)
        #lib = update_lib(lib)
        # print(json)
        rt_mod = graph_executor.create(json, lib, ctx)#, dump_root="/home/zy/tvm/tutorials/experiment_res/")#Create a runtime executor module given a graph and module.
        
        rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
        rt_mod.set_input(**params)
        rt_mod.run()
        

        # out= rt_mod.debug_get_output("tvmgen_default_dnnl_0", out=tvm.nd.empty((1, 64, 112, 112), dtype="float32"))
        #vm = VirtualMachine(exe, ctx)
        #input_list = [tvm.nd.array(sample.astype("float32"))]
        #results = vm.invoke("main", input_list)# print(out)
        # tvm_out = rt_mod.get_output(1, tvm.nd.empty((1, 1000), "float32")).numpy()
        # print(tvm_out)
        
        for i in range(batches+warmup):
            if i == warmup:
                tic = time.time()
            out = rt_mod.run()
            #rt_mod.profile()
            #vm.invoke("main", input_list)
            # out.wait_to_read()
        with_fuse_fps = batches * batch_size / (time.time() - tic)
        print("{}: with_fuse_fps: {:.4f} fps".format(model_name, with_fuse_fps))
        #tvm_output = rt_mod.get_output(0)
        #print(tvm_output)
        
        #ftimer = rt_mod.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=10)
        #print(np.array(ftimer().results))
benchmark(batch_size=1)
