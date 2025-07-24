import torch
import timm

def export_efficientnetv2_b0_onnx(output_path="models/efficientnetv2-b0.onnx", input_height=224, input_width=224, batch_size=1):
    model = timm.create_model("tf_efficientnetv2_b0", pretrained=True)
    model.eval()

    dummy_input = torch.randn(batch_size, 3, input_height, input_width)

    torch.onnx.export(
        model,
        (dummy_input, ),
        output_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=11,
        do_constant_folding=True
    )

    print(f"Exported ONNX to: {output_path}")
    print(f"Input shape: [{batch_size}, 3, {input_height}, {input_width}]")

if __name__ == "__main__":
    export_efficientnetv2_b0_onnx()