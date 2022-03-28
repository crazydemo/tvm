'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import argparse

import time
from typing import Tuple
import gluoncv
from mxnet.gluon.model_zoo import vision
from gluoncv.utils import export_block
from mxnet.contrib import onnx as onnx_mxnet
import mxnet as mx
import onnx
import warnings
warnings.filterwarnings("ignore")
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import dnnl

# Tensorflow imports
import tensorflow as tf

try:
    tf_compat_v1 = tf.compat.v1
except ImportError:
    tf_compat_v1 = tf

# Tensorflow utility functions
import tvm.relay.testing.tf as tf_testing


import numpy as np


network_dic = {"googlenet-v1-tf": "/home2/zhangya9/open_model_zoo/public/googlenet-v1-tf/inception_v1.frozen.pb"}

def transform_image(image):
    image = np.array(image) - np.array([123.0, 117.0, 104.0])
    image /= np.array([58.395, 57.12, 57.375])
    image = image.transpose((2, 0, 1))
    image = image[np.newaxis, :]
    return image

def partition_for_dnnl(mod, params=None, alter_layout=True):
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

    with TempOpAttr("nn.conv2d", "FTVMLegalize", dnnl.legalize_group_conv):
        with TempOpAttr("nn.conv2d_transpose", "FTVMLegalize", dnnl.legalize_group_conv):
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
    if alter_layout:
        with TempOpAttr("nn.conv1d", "FTVMAlterOpLayout", dnnl.alter_conv):
            with TempOpAttr("nn.conv2d", "FTVMAlterOpLayout", dnnl.alter_conv):
                with TempOpAttr("nn.conv3d", "FTVMAlterOpLayout", dnnl.alter_conv):
                    with TempOpAttr(
                        "nn.conv2d_transpose", "FTVMAlterOpLayout", dnnl.alter_conv_transpose
                    ):
                        with TempOpAttr(
                            "nn.conv3d_transpose", "FTVMAlterOpLayout", dnnl.alter_conv_transpose
                        ):
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
            transform.MergeComposite(dnnl.pattern_table()),
            transform.AnnotateTarget("dnnl"),
            transform.MergeCompilerRegions(),
            transform.PartitionGraph(),
        ]
    )
    with tvm.transform.PassContext(opt_level=3):
        mod = byoc_seq(mod)
        mod = dnnl.prune_dnnl_subgraphs(mod)
    return mod

def benchmark(network, batch_size, profiling=False, check_acc=False, warmup=100, batches=400, dtype="float32", target="llvm"):
    ctx = tvm.cpu()
    layout = "NHWC"
    input_shape = (batch_size, 224, 224, 3)
    shape_dict = {"input": input_shape}
    model_path = network_dic[network]
    sample = np.random.randint(0, 1, input_shape)

    with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
        graph_def = tf_compat_v1.GraphDef()
        graph_def.ParseFromString(f.read())
        graph = tf.import_graph_def(graph_def, name="")
        # Call the utility to import the graph definition into default graph.
        graph_def = tf_testing.ProcessGraphDefParam(graph_def)
    # Add shapes to the graph.
    # with tf_compat_v1.Session() as sess:
    #     graph_def = tf_testing.AddShapesToGraphDef(sess, "InceptionV1/Logits/Predictions/Softmax")

    mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)

    processed_mod = partition_for_dnnl(mod, params, alter_layout=True)
    print(processed_mod)    
    with tvm.transform.PassContext(opt_level=3):
        json, lib, params = relay.build(processed_mod, target=target, params=params)
    # print(json)
    import tvm.contrib.graph_executor as graph_executor
    # from tvm.contrib.debugger import debug_executor as graph_executor
    rt_mod = graph_executor.create(json, lib, ctx)
    
    rt_mod.set_input("input", tvm.nd.array(sample.astype("float32")))
    rt_mod.set_input(**params)
    # out = rt_mod.run()
    
    for i in range(batches+warmup):
        if i == warmup:
            tic = time.time()
        # print("================start run=========================")
        rt_mod.run()
        # print(rt_mod.profile())
    with_fuse_fps = batches * batch_size / (time.time() - tic)
    print("{}: with_fuse_fps: {:.4f} fps".format(network, with_fuse_fps))

if __name__ == "__main__":
    # os.environ["TVM_LOG_DEBUG"]="DEFAULT=1;ir/transform.cc=1;relay/ir/transform.cc=1"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network",
        type=str,
        default="googlenet-v1-tf",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The batch size")
    parser.add_argument(
        "--target",
        type=str,
        default="llvm -mcpu=cascadelake -model=platinum-8280",
        help="The compilation target.",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="The data type.")
    
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--batches", type=int, default=100)
    parser.add_argument("--profiling", type=bool, default=False)
    parser.add_argument("--check_acc", type=bool, default=False)
    args = parser.parse_args()

    target = tvm.target.Target(args.target)

    benchmark(args.network, args.batch_size, profiling=args.profiling,check_acc=args.check_acc,\
            warmup=args.warmup, batches=args.batches, dtype=args.dtype, target=args.target)