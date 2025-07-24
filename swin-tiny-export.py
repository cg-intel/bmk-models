import torch
import timm
import os

def export_timm_onnx(model_name, output_path, input_height=224, input_width=224, batch_size=1):
    model = timm.create_model(model_name, pretrained=True)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, input_height, input_width)

    torch.onnx.export(
        model,
        (dummy_input, ),
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
    os.makedirs("models", exist_ok=True)
    export_timm_onnx("swin_tiny_patch4_window7_224", "models/swin-tiny.onnx")
