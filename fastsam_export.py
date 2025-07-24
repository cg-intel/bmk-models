import torch
from ultralytics import FastSAM

def export_fastsam_onnx(model_name, output_path, input_height=1024, input_width=1024, batch_size=1):
    model = FastSAM(f"{model_name}.pt").model
    model.eval()

    dummy_input = torch.randn(batch_size, 3, input_height, input_width)

    torch.onnx.export(
        model,
        (dummy_input, ),
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None
    )

    print(f"ONNX model exported to: {output_path}")
    print(f"Input shape: [{batch_size}, 3, {input_height}, {input_width}]")

if __name__ == '__main__':
    export_fastsam_onnx("FastSAM-s", "models/FastSAM-s.onnx")
    export_fastsam_onnx("FastSAM-x", "models/FastSAM-x.onnx")