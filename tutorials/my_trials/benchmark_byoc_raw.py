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
from tvm.runtime.vm import VirtualMachine
#from tvm.contrib.debugger import debug_executor as graph_executor
# 
# model_dict = {'resnet50_v1': resnet50_v1}#{'mobilenet_v2_1_0': mobilenet_v2_1_0}
model_dict = {'resnet50_v1': resnet}

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

def benchmark(batch_size=1, batches=100, warmup=20):
    img_url = "https://github.com/dmlc/mxnet.js/blob/main/data/cat.png?raw=true"
    img_name = "cat.png"
    img_path = download_testdata(img_url, img_name, module="data")
    image = Image.open(img_path).resize((224, 224))
    sample = transform_image(image)
    # print("x", sample.shape)
    # np.random.seed(0)
    sample = np.random.rand(batch_size, 3, 224, 224)#np.ones((batch_size, 3, 224, 224))#

    target = "llvm -model=platinum-8124m -mcpu=skylake-avx512"
    ctx = tvm.cpu()

    input_shape = (batch_size, 3, 224, 224)
    for model_name in model_dict.keys():
        block = mx.gluon.model_zoo.vision.get_resnet(1, 50, pretrained=True)
        mod, params = relay.frontend.from_mxnet(
            block, shape={"data": input_shape}, dtype="float32"
        )
        seq = tvm.transform.Sequential(
            [
                relay.transform.MergeComposite(pattern_table()),
                relay.transform.AnnotateTarget("dnnl"),
                relay.transform.MergeCompilerRegions(),
                relay.transform.PartitionGraph(),
            ]
        )


        #if params:
        #    mod["main"] = bind_params_by_name(mod["main"], params)
        with tvm.transform.PassContext(opt_level=3):#, instruments=[PrintIR()]):# 
            json, lib, params = relay.build(seq(mod), "llvm", params=params)
        #exe =  relay.vm.compile(mod, target="llvm", params=params)
        #lib = update_lib(lib)
        # print(json)
        rt_mod = graph_executor.create(json, lib, ctx)#, dump_root="/home/zy/tvm/tutorials/experiment_res/")#Create a runtime executor module given a graph and module.
        
        rt_mod.set_input("data", tvm.nd.array(sample.astype("float32")))
        rt_mod.set_input(**params)
        #print("##############################################################")
        #rt_mod.run()
        

        # out= rt_mod.debug_get_output("tvmgen_default_dnnl_0", out=tvm.nd.empty((1, 64, 112, 112), dtype="float32"))
        #vm = VirtualMachine(exe, ctx)
        #input_list = [tvm.nd.array(sample.astype("float32"))]
        #results = vm.invoke("main", input_list)# print(out)
        # tvm_out = rt_mod.get_output(1, tvm.nd.empty((1, 1000), "float32")).numpy()
        # print(tvm_out)
        
        for i in range(batches+warmup):
            if i == warmup:
                tic = time.time()
            #print("##############################################################")
            out = rt_mod.run()
            #vm.invoke("main", input_list)
            # out.wait_to_read()
        with_fuse_fps = batches * batch_size / (time.time() - tic)
        print("{}: with_fuse_fps: {:.4f} fps".format(model_name, with_fuse_fps))
        
        #tvm_output = rt_mod.get_output(0)
        #print(tvm_output)
        
        #ftimer = rt_mod.module.time_evaluator("run", ctx, min_repeat_ms=500, repeat=10)
        #print(np.array(ftimer().results))
benchmark(batch_size=1)
