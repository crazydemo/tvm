'''
for resnet50 and mobilenet
# BENCHMARKING SCRIPT FOR GLUON MXNET 2.0
'''
import argparse

import time
from typing import Tuple
import os
# import gluoncv
# from mxnet.gluon.model_zoo import vision
# from gluoncv.utils import export_block
# from mxnet.contrib import onnx as onnx_mxnet
# import mxnet as mx
# import onnx
import warnings
warnings.filterwarnings("ignore")
import tvm
from tvm import relay
from tvm.relay import transform
from tvm.relay.build_module import bind_params_by_name
from tvm.relay.testing.temp_op_attr import TempOpAttr
from tvm.relay.op.contrib import dnnl
from onnxruntime.backend.backend import OnnxRuntimeBackend as backend

import numpy as np

network_root_path = "/home2/zhangya9/onnx_models"
network_dic = { "MobileNet-v2-1.0": "MobileNet/torch_model/mobilenetv2_torch.onnx",
                "resnet50-v1": "ResNet/v1/resnet50v1/resnet50-v1-7.onnx",
                "resnet50-v2": "ResNet/v2/resnet50v2/resnet50-v2-7.onnx",
                "squeezenet1.0": "SqueezeNet/squeezenet/model.onnx",
                "squeezenet1.1": "SqueezeNet/squeezenet1.1/squeezenet1.1.onnx",
                "vgg16": "VGG/vgg16/vgg16.onnx",
                "vgg16-bn": "VGG/vgg16-bn/vgg16-bn.onnx",
                "googlenet": "GoogleNet/bvlc_googlenet/model.onnx",
                "rcnn": "RCNN_ILSVRC13/bvlc_reference_rcnn_ilsvrc13/model_N.onnx",
                "densenet121": "DenseNet-121/densenet121/model.onnx",
                "inception_v1": "Inception_V1/inception_v1/model.onnx",
                "inception_v2": "Inception_V2/inception_v2/model.onnx",
                "inception_v3": "Inception_V3/torch_model/inception_v3.onnx",
                "shufflenet_v1": "ShuffleNet_V1/shufflenet/model.onnx",
                "shufflenet_v2": "ShuffleNet_V2/torch_model/shufflenet_v2_x1_0.onnx",
                "zfnet512": "ZFNet-512/zfnet512/model.onnx",
                "efficientnet-lite4": "EfficientNet-Lite4/efficientnet-lite4/efficientnet-lite4.onnx",
                "efficientnet-b0-pytorch": "efficientnet-b0-pytorch/efficientnet-b0.onnx",
                "resnext50_32x4d": "ResNext/torch_model/resnext50_32x4d.onnx",
                "wide_resnet50_2": "Wide_ResNet/torch_model/wide_resnet50_2.onnx",
                "resnest50": "ResNeSt/torch_model/resnest50.onnx",
                "googlenet-v2-tf": "googlenet-v2-tf/inception_v2.frozen.pb",
                "googlenet-v3": "googlenet-v3/inception_v3_2016_08_28_frozen.pb",
                "googlenet-v3-pytorch": "googlenet-v3-pytorch/googlenet-v3.onnx",
                "googlenet-v4-tf": "googlenet-v4-tf/inception_v4.frozen.pb",
                "inception-resnet-v2-tf": "inception-resnet-v2-tf/inception_resnet_v2.pb",
                "anti-spoof-mn3": "anti-spoof-mn3/anti-spoof-mn3.onnx",
                "densenet-121-caffe2": "densenet-121-caffe2/densenet-121-caffe2.onnx",
                "efficientnet-b5-pytorch": "efficientnet-b5-pytorch/efficientnet-b5.onnx",
                "efficientnet-b7-pytorch": "efficientnet-b7-pytorch/efficientnet-b7.onnx",
                "hbonet-1.0": "hbonet-1.0/hbonet_1_0.onnx",
                "hbonet-0.25": "hbonet-0.25/hbonet_0_25.onnx",
                "mobilenet-v1-0.25-128": "mobilenet-v1-0.25-128/mobilenet_v1_0.25_128_frozen.pb",
                "mobilenet-v1-1.0-224-tf": "mobilenet-v1-1.0-224-tf/mobilenet_v1_1.0_224_frozen.pb",
                "mobilenet-v2-1.0-224": "mobilenet-v2-1.0-224/mobilenet_v2_1.0_224_frozen.pb",
                "mobilenet-v2-1.4-224": "mobilenet-v2-1.4-224/mobilenet_v2_1.4_224_frozen.pb",
                "mobilenet-v3-small-1.0-224-tf": "mobilenet-v3-small-1.0-224-tf/v3-small_224_1.0_float/v3-small_224_1.0_float.pb",
                "mobilenet-v3-large-1.0-224-tf": "mobilenet-v3-large-1.0-224-tf/v3-large_224_1.0_float/v3-large_224_1.0_float.pb",
                "octave-resnet-26-0.25": "octave-resnet-26-0.25/a02_resnet-26_alpha-0.250/checkpoint-0-symbol.json",
                "open-closed-eye-0001": "open-closed-eye-0001/open-closed-eye.onnx",
                "resnet-50-tf": "resnet-50-tf/resnet_v1-50.pb",
                "deeplabv3": "deeplabv3/deeplabv3_mnv2_pascal_train_aug/frozen_inference_graph.pb",
                "fastseg-large": "fastseg-large/fastseg-large.onnx",
                "fastseg-small": "fastseg-small/fastseg-small.onnx",
                "mask_rcnn_inception_resnet_v2_atrous_coco": "mask_rcnn_inception_resnet_v2_atrous_coco/mask_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb",
                "mask_rcnn_resnet50_atrous_coco": "mask_rcnn_resnet50_atrous_coco/mask_rcnn_resnet50_atrous_coco_2018_01_28/frozen_inference_graph.pb",
                "brain-tumor-segmentation-0002": "brain-tumor-segmentation-0002/brain-tumor-segmentation-0002.onnx",
                "ctpn": "ctpn/ctpn.pb",
                "faceboxes-pytorch": "faceboxes-pytorch/faceboxes-pytorch.onnx",
                "faster_rcnn_inception_resnet_v2_atrous_coco": "faster_rcnn_inception_resnet_v2_atrous_coco/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28/frozen_inference_graph.pb",
                "faster_rcnn_resnet50_coco": "faster_rcnn_resnet50_coco/faster_rcnn_resnet50_coco_2018_01_28/frozen_inference_graph.pb",
                "retinanet-tf": "retinanet-tf/retinanet_resnet50_coco_best_v2.1.0.pb",
                "rfcn-resnet101-coco-tf": "rfcn-resnet101-coco-tf/rfcn_resnet101_coco_2018_01_28/frozen_inference_graph.pb",
                "ssd_mobilenet_v1_coco": "ssd_mobilenet_v1_coco/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb",
                "ssd_mobilenet_v1_fpn_coco": "ssd_mobilenet_v1_fpn_coco/ssd_mobilenet_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03/frozen_inference_graph.pb",
                "ssdlite_mobilenet_v2": "ssdlite_mobilenet_v2/ssdlite_mobilenet_v2_coco_2018_05_09/frozen_inference_graph.pb",
                "ssd-resnet34-1200-onnx": "ssd-resnet34-1200-onnx/resnet34-ssd1200.onnx",
                "ultra-lightweight-face-detection-rfb-320": "ultra-lightweight-face-detection-rfb-320/ultra-lightweight-face-detection-rfb-320.onnx",
                "ultra-lightweight-face-detection-slim-320": "ultra-lightweight-face-detection-slim-320/ultra-lightweight-face-detection-slim-320.onnx",
                "yolo-v1-tiny-tf": "yolo-v1-tiny-tf/yolo-v1-tiny-tf.pb",
                "face-recognition-resnet100-arcface-onnx": "face-recognition-resnet100-arcface-onnx/arcfaceresnet100-8.onnx",
                "human-pose-estimation-3d-0001": "human-pose-estimation-3d-0001/human-pose-estimation-3d-0001.onnx",
                "gmcnn-places2-tf": "gmcnn-places2-tf/gmcnn-places2-tf/frozen_model.pb",
                "fast-neural-style-mosaic-onnx": "fast-neural-style-mosaic-onnx/fast-neural-style-mosaic-onnx.onnx",
                "i3d-rgb-tf": "i3d-rgb-tf/i3d-rgb.frozen.pb",
                "common-sign-language-0001": "common-sign-language-0001/s3d-rgb-mobilenet-v3-large-stream-jester.onnx",
                "colorization-v2": "colorization-v2/colorization-v2-eccv16.onnx",
                "colorization-siggraph": "colorization-siggraph/colorization-siggraph.onnx",
                "deblurgan-v2": "deblurgan-v2.onnx",
                "f3net": "f3net/f3net.onnx",
                "vehicle-reid-0001": "vehicle-reid-0001/osnet_ain_x1_0_vehicle_reid.onnx",
              }

input_dic = { "MobileNet-v2-1.0": "input",
              "resnet50-v1": "data",
              "resnet50-v2": "data",
              "squeezenet1.0": "data_0",
              "squeezenet1.1": "data",
              "vgg16": "data",
              "vgg16-bn": "data",
              "googlenet": "data_0",
              "rcnn": "data_0",
              "densenet121": "data_0",
              "inception_v1": "data_0",
              "inception_v2": "data_0",
              "inception_v3": "input",
              "shufflenet_v1": "gpu_0/data_0",
              "shufflenet_v2": "input",
              "zfnet512": "gpu_0/data_0",
              "efficientnet-lite4": "images:0",
              "resnext50_32x4d": "input",
              "wide_resnet50_2": "input",
              "resnest50": "input",
              "googlenet-v2-tf": "input",
              "googlenet-v3": "input",
              "googlenet-v3-pytorch": "data",
              "googlenet-v4-tf": "input",
              "inception-resnet-v2-tf": "input",
              "anti-spoof-mn3": "actual_input_1",
              "densenet-121-caffe2": "data",
              "efficientnet-b0-pytorch": "data",
              "efficientnet-b5-pytorch": "data",
              "efficientnet-b7-pytorch": "data",
              "octave-resnet-26-0.25": "data",
              "hbonet-1.0": "data",
              "hbonet-0.25": "data",
              "mobilenet-v1-0.25-128": "input",
              "mobilenet-v1-1.0-224-tf": "input",
              "mobilenet-v2-1.0-224": "input",
              "mobilenet-v2-1.4-224": "input",
              "mobilenet-v3-small-1.0-224-tf": "input_1",
              "mobilenet-v3-large-1.0-224-tf": "input_1",
              "open-closed-eye-0001": "input.1",
              "resnet-50-tf": "map/TensorArrayStack/TensorArrayGatherV3",
              "deeplabv3": "ImageTensor",
              "fastseg-large": "input0",
              "fastseg-small": "input0",
              "mask_rcnn_inception_resnet_v2_atrous_coco": "image_tensor",
              "mask_rcnn_resnet50_atrous_coco": "image_tensor",
              "brain-tumor-segmentation-0002": "0",
              "ctpn": "image_tensor",
              "faceboxes-pytorch": "input.1",
              "faster_rcnn_inception_resnet_v2_atrous_coco": "image_tensor",
              "faster_rcnn_resnet50_coco": "image_tensor",
              "retinanet-tf": "input_1",
              "rfcn-resnet101-coco-tf": "image_tensor",
              "ssd_mobilenet_v1_coco": "image_tensor",
              "ssd_mobilenet_v1_fpn_coco": "image_tensor",
              "ssdlite_mobilenet_v2": "image_tensor",
              "ssd-resnet34-1200-onnx": "data",
              "ultra-lightweight-face-detection-rfb-320": "input",
              "ultra-lightweight-face-detection-slim-320": "input",
              "yolo-v1-tiny-tf": "input_1",
              "face-recognition-resnet100-arcface-onnx": "data",
              "human-pose-estimation-3d-0001": "data",
              "gmcnn-places2-tf": "Placeholder",
              "fast-neural-style-mosaic-onnx": "input1",
              "i3d-rgb-tf": "Placeholder",
              "common-sign-language-0001": "data",
              "colorization-v2": "data_l",
              "colorization-siggraph": "data_l",
              "deblurgan-v2": "blur_image",
              "f3net": "input.1",
              "vehicle-reid-0001": "data",
            }

shape_dic = { "MobileNet-v2-1.0": [1, 3, 224, 224],
              "resnet50-v1": [1, 3, 224, 224],
              "resnet50-v2": [1, 3, 224, 224],
              "squeezenet1.0": [1, 3, 224, 224],
              "squeezenet1.1": [1, 3, 224, 224],
              "vgg16": [1, 3, 224, 224],
              "vgg16-bn": [1, 3, 224, 224],
              "googlenet": [1, 3, 224, 224],
              "rcnn": [1, 3, 224, 224],
              "densenet121": [1, 3, 224, 224],
              "inception_v1": [1, 3, 224, 224],
              "inception_v2": [1, 3, 224, 224],
              "inception_v3": [1, 3, 224, 224],
              "shufflenet_v1": [1, 3, 224, 224],
              "shufflenet_v2": [1, 3, 224, 224],
              "zfnet512": [1, 3, 224, 224],
              "efficientnet-lite4": [1, 224, 224, 3],
              "resnext50_32x4d": [1, 3, 224, 224],
              "wide_resnet50_2": [1, 3, 224, 224],
              "googlenet-v2-tf": [1, 224, 224, 3],
              "googlenet-v3": [1, 299, 299, 3],
              "googlenet-v3-pytorch": [1, 3, 299, 299],
              "googlenet-v4-tf": [1, 299, 299, 3],
              "inception-resnet-v2-tf": [1, 299, 299, 3],
              "anti-spoof-mn3": [1, 3, 128, 128],
              "densenet-121-caffe2": [1, 3, 224, 224],
              "efficientnet-b0-pytorch": [1, 3, 224, 224],
              "efficientnet-b5-pytorch": [1, 3, 456, 456],
              "efficientnet-b7-pytorch": [1, 3, 600, 600],
              "hbonet-1.0": [1, 3, 224, 224],
              "hbonet-0.25": [1, 3, 224, 224],
              "mobilenet-v1-0.25-128": [1, 224, 224, 3],
              "mobilenet-v1-1.0-224-tf": [1, 224, 224, 3],
              "mobilenet-v2-1.0-224": [1, 224, 224, 3],
              "mobilenet-v2-1.4-224": [1, 224, 224, 3],
              "mobilenet-v3-small-1.0-224-tf": [1, 224, 224, 3],
              "mobilenet-v3-large-1.0-224-tf": [1, 224, 224, 3],
              "octave-resnet-26-0.25": [1, 3, 224, 224],
              "open-closed-eye-0001": [1, 3, 32, 32],
              "resnest50": [1, 3, 224, 224],
              "resnet-50-tf": [1, 224, 224, 3],
              "deeplabv3": [1, 513, 513, 3],
              "fastseg-large": [1, 3, 1024, 2048],
              "fastseg-small": [1, 3, 1024, 2048],
              "mask_rcnn_inception_resnet_v2_atrous_coco": [1, 800, 1365, 3],
              "mask_rcnn_resnet50_atrous_coco": [1, 800, 1365, 3],
              "brain-tumor-segmentation-0002": [1, 4, 128, 128, 128],
              "ctpn": [1, 600, 600, 3],
              "faceboxes-pytorch": [1, 3, 1024, 1024],
              "faster_rcnn_inception_resnet_v2_atrous_coco": [1, 600, 1024, 3],
              "faster_rcnn_resnet50_coco": [1, 600, 1024, 3],
              "retinanet-tf": [1, 1333, 1333, 3],
              "rfcn-resnet101-coco-tf": [1, 600, 600, 3],
              "ssd_mobilenet_v1_coco": [1, 300, 300, 3],
              "ssd_mobilenet_v1_fpn_coco": [1, 640, 640, 3],
              "ssdlite_mobilenet_v2": [1, 300, 300, 3],
              "ssd-resnet34-1200-onnx": [1, 3, 1200, 1200],
              "ultra-lightweight-face-detection-rfb-320": [1, 3, 240, 320],
              "ultra-lightweight-face-detection-slim-320": [1, 3, 240, 320],
              "yolo-v1-tiny-tf": [1, 416, 416, 3],
              "face-recognition-resnet100-arcface-onnx": [1, 3, 112, 112],
              "human-pose-estimation-3d-0001": [1, 3, 256, 448],
              "gmcnn-places2-tf": [1, 512, 680, 3],
              "fast-neural-style-mosaic-onnx": [1, 3, 224, 224],
              "i3d-rgb-tf": [1, 79, 224, 224, 3],
              "common-sign-language-0001": [1, 3, 8, 224, 224],
              "colorization-v2": [1, 1, 256, 256],
              "colorization-siggraph": [1, 1, 256, 256],
              "deblurgan-v2": [1, 3, 736, 1312],
              "f3net": [1, 3, 352, 352],
              "vehicle-reid-0001": [1, 3, 208, 208],
            }

layout_dic = { "googlenet-v1-tf": "NHWC",
               "googlenet-v2-tf": "NHWC",
               "googlenet-v3": "NHWC",
               "googlenet-v3-pytorch": "NCHW",
               "googlenet-v4-tf": "NHWC",
               "inception-resnet-v2-tf": "NHWC",
               "anti-spoof-mn3": "NCHW",
               "densenet-121-caffe2": "NCHW",
               "efficientnet-b0-pytorch": "NCHW",
               "efficientnet-b5-pytorch": "NCHW",
               "efficientnet-b7-pytorch": "NCHW",
               "hbonet-1.0": "NCHW",
               "hbonet-0.25": "NCHW",
               "mobilenet-v1-0.25-128": "NHWC",
               "mobilenet-v1-1.0-224-tf": "NHWC",
               "mobilenet-v2-1.0-224": "NHWC",
               "mobilenet-v2-1.4-224": "NHWC",
               "mobilenet-v3-small-1.0-224-tf": "NHWC",
               "mobilenet-v3-large-1.0-224-tf": "NHWC",
               "octave-resnet-26-0.25": "NCHW",
               "open-closed-eye-0001": "NCHW",
               "resnest50": "NCHW",
               "resnet-50-tf": "NHWC",
               "deeplabv3": "NHWC",
               "fastseg-large": "NCHW",
               "fastseg-small": "NCHW",
               "mask_rcnn_inception_resnet_v2_atrous_coco": "NHWC",
               "mask_rcnn_resnet50_atrous_coco": "NHWC",
               "brain-tumor-segmentation-0002": "NCDHW",
               "ctpn": "NHWC",
               "faceboxes-pytorch": "NCHW",
               "faster_rcnn_inception_resnet_v2_atrous_coco": "NHWC",
               "faster_rcnn_resnet50_coco": "NHWC",
               "retinanet-tf": "NHWC",
               "rfcn-resnet101-coco-tf": "NHWC",
               "ssd_mobilenet_v1_coco": "NHWC",
               "ssd_mobilenet_v1_fpn_coco": "NHWC",
               "ssdlite_mobilenet_v2": "NHWC",
               "ssd-resnet34-1200-onnx": "NCHW",
               "ultra-lightweight-face-detection-rfb-320": "NCHW",
               "ultra-lightweight-face-detection-slim-320": "NCHW",
               "yolo-v1-tiny-tf": "NHWC",
               "face-recognition-resnet100-arcface-onnx": "NCHW",
               "human-pose-estimation-3d-0001": "NCHW",
               "gmcnn-places2-tf": "NHWC",
               "fast-neural-style-mosaic-onnx": "NCHW",
               "i3d-rgb-tf": "NCDHW",
               "common-sign-language-0001": "NCDHW",
               "colorization-v2": "NCHW",
               "colorization-siggraph": "NCHW",
               "deblurgan-v2": "NCHW",
               "f3net": "NCHW",
               "vehicle-reid-0001": "NCHW",
             }

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
                    # tvm.transform.PrintIR(),
                    transform.CanonicalizeOps(),
                    transform.InferType(),
                    transform.SimplifyInference(),
                    transform.FoldConstant(),
                    transform.FoldScaleAxis(),
                    # tvm.transform.PrintIR(),
                    # fold consecutive add ops to simplify pattern `conv2d-bias_add-bn-relu`
                    transform.SimplifyExpr(),
                    transform.FoldConstant(),
                    # tvm.transform.PrintIR(),
                    # alter group conv /conv_transpose layout to `GOIHW` / `GIOHW`
                    transform.Legalize(),
                    transform.FoldConstant(),
                    # tvm.transform.PrintIR(),
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
                                    # tvm.transform.PrintIR(),
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
    input_name = input_dic[network]
    input_shape = shape_dic[network]
    input_shape[0] = batch_size
    # print(input_shape)
    shape_dict = {input_name: input_shape}
    model_path = os.path.join(network_root_path, network_dic[network])
    sample = np.random.randint(0, 1, input_shape).astype(dtype)
    if network_dic[network] == "":
        print("=============converting torch model===============")
        import torch
        import torchvision

        model_name = network
        # get list of models
        torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        # load pretrained models, using ResNeSt-50 as an example
        model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
        model.eval()
        # model = getattr(torchvision.models, model_name)(pretrained=True)
        # model = model.eval()

        input_data = torch.randn(input_shape)
        scripted_model = torch.jit.trace(model, input_data).eval()
        shape_list = [(input_name, input_shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)
    elif model_path.split(".")[-1] == "pb":
        print("=============converting TF model===============")
        import tensorflow as tf
        try:
            tf_compat_v1 = tf.compat.v1
        except ImportError:
            tf_compat_v1 = tf
        import tvm.relay.testing.tf as tf_testing
        layout = layout_dic[network]
        with tf_compat_v1.gfile.GFile(model_path, "rb") as f:
            graph_def = tf_compat_v1.GraphDef()
            graph_def.ParseFromString(f.read())
            graph_def = tf_testing.ProcessGraphDefParam(graph_def)
        mod, params = relay.frontend.from_tensorflow(graph_def, layout=layout, shape=shape_dict)
    elif model_path.split(".")[-1] == "onnx":
        print("=============converting ONNX model===============")
        import onnx
        onnx_model = onnx.load(model_path)
        # print(onnx_model.graph.input)
        mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)
        # print(mod)
    elif model_path.split(".")[-1] == "caffemodel":
        print("=============converting caffe model===============")
        import caffe.proto.caffe_pb2 as caffe_pb2
        model = caffe_pb2.NetParameter()
        f = open(model_path, 'rb')
        model.ParseFromString(f.read())
        mod, params = relay.frontend.from_caffe2(
                        model, shape_dict)
    elif model_path.split(".")[-1] == "json":
        print("=============converting MXNet model===============")
        import mxnet as mx
        json_file = model_path
        params_file = model_path.replace("symbol.json", "0000.params")
        mod = mx.gluon.nn.SymbolBlock(outputs=mx.sym.load(json_file), inputs=mx.sym.var('data'))
        mod.load_params(params_file, ctx=ctx)
        mod, params = relay.frontend.from_mxnet(mod, shape_dict)
    else:
        print("Unsupported model type!")

    print("=============Optimizing===============")
    processed_mod = partition_for_dnnl(mod, params, alter_layout=True)
    print(processed_mod)

    print("=============Building===============")
    with tvm.transform.PassContext(opt_level=3):
        json, lib, params = relay.build(processed_mod, target=target, params=params)
    # print(json)
    import tvm.contrib.graph_executor as graph_executor
    # from tvm.contrib.debugger import debug_executor as graph_executor
    rt_mod = graph_executor.create(json, lib, ctx)
    
    rt_mod.set_input(input_name, tvm.nd.array(sample.astype(dtype)))
    rt_mod.set_input(**params)

    print("=============Checking accuracy===============")
    if check_acc:
        onnx_output = list(backend.run_model(onnx_model, sample))
        rt_mod.run()
        tvm_output = rt_mod.get_output(0)
        np.testing.assert_almost_equal(onnx_output, [tvm_output.asnumpy()], 5)
    # out = rt_mod.run()
    print("=============Running===============")
    for i in range(batches+warmup):
        if i == warmup:
            tic = time.time()
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
        default="shufflenet_v2",
        help="The name of the neural network.",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="The batch size")
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