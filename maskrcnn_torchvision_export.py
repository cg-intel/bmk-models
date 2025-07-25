import torch
import torchvision

backbone = torchvision.models.detection.maskrcnn_resnet50_fpn(weights="DEFAULT").backbone.eval()
dummy_input = torch.randn(1, 3, 800, 800)

torch.onnx.export(
    backbone,
    (dummy_input, ),
    "models/maskrcnn-resnet50-fpn-backbone.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11,
    do_constant_folding=True,
    dynamic_axes=None
)
