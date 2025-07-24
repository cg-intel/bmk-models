import torch
import torchvision.models as models
from torchvision.models import (
    ResNet50_Weights,
    MobileNet_V2_Weights,
    MobileNet_V3_Small_Weights,
    Wide_ResNet50_2_Weights,
)

def export_torchvision_onnx(model_name, weights_enum, output_path, input_height=224, input_width=224, batch_size=1):
    model_fn = getattr(models, model_name)
    model = model_fn(weights=weights_enum).eval()

    dummy_input = torch.randn(batch_size, 3, input_height, input_width)

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None
    )

    print(f"ONNX model exported to: {output_path}")
    print(f"Input shape: [{batch_size}, 3, {input_height}, {input_width}]")

if __name__ == "__main__":
    export_torchvision_onnx("resnet50", ResNet50_Weights.DEFAULT, "models/resnet50.onnx")
    export_torchvision_onnx("mobilenet_v2", MobileNet_V2_Weights.DEFAULT, "models/mobilenetv2.onnx")
    export_torchvision_onnx("mobilenet_v3_small", MobileNet_V3_Small_Weights.DEFAULT, "models/mobilenetv3-small.onnx")
    export_torchvision_onnx("wide_resnet50_2", Wide_ResNet50_2_Weights.DEFAULT, "models/wide-resnet50.onnx")
