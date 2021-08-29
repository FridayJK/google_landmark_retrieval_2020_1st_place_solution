import torch
import onnx

def eff_conv2onnx(eff_model, onnx_name, op_version):
    input_name = ['input']
    output_name = ['output']
    input = torch.randn(1, 3, 512, 512, device='cuda')
    eff_model.set_swish(memory_efficient=False)
    torch.onnx.export(eff_model, input, onnx_name, input_names=input_name, output_names=output_name, opset_version=op_version, verbose=True)
    # torch.onnx.export(eff_model, input, onnx_name, input_names=input_name, output_names=output_name, opset_version=11, verbose=True)
    test = onnx.load(onnx_name)
    onnx.checker.check_model(test)
    print("==> Passed")