import torch
# model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'wide_resnet50_2', pretrained=True)
torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True)
model.eval()

# pthfile = "/home2/zhangya9/open_model_zoo/public/nfnet-f0/dm_nfnet_f0-604f9c3a.pth"
# mod = torch.load(pthfile)
print(model)

# Let's create a dummy input tensor  
dummy_input = torch.randn(1,3,224,224, requires_grad=True)  

# Export the model   
torch.onnx.export(model,         # model being run 
        dummy_input,       # model input (or a tuple for multiple inputs) 
        "/home2/zhangya9/onnx_models/ResNeSt/torch_model/resnest50.onnx",       # where to save the model  
        export_params=True,  # store the trained parameter weights inside the model file 
        opset_version=7,    # the ONNX version to export the model to 
        do_constant_folding=True,  # whether to execute constant folding for optimization 
        input_names = ['input'],   # the model's input names 
        output_names = ['output'], # the model's output names 
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes 
                            'output' : {0 : 'batch_size'}}) 
print(" ") 
print('Model has been converted to ONNX') 
# net.load_state_dict({k.replace(k, 'CCN.' + k): v for k, v in model_test.items()})
# net.cuda()
# net.eval()
# image = cv2.imread('test1.jpg')
# img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# img = img_transform(img)
# img = img.view(1, img.shape[0], img.shape[1], img.shape[2])
# with torch.no_grad():  #不需要降低梯度
#     img = Variable(img).cuda()
# input_names = ["feature4"]  #只代表输入节点名称
# output_names = ["de_pred"]  #只代表输出节点名称
# #只能输入卷积网络的内容，类似回归损失之类的除外
# torch.onnx.export(net.CCN, img, "res50.onnx", verbose=True, input_names=input_names,
#                     output_names=output_names)
                