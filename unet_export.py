import torch
import segmentation_models_pytorch as smp

def export_unet_onnx(output_path="models/unet-resnet18.onnx", input_height=512, input_width=512, batch_size=1):
    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=12,
        decoder_channels=(256, 128, 64, 32, 16)
    )
    model.eval()

    dummy_input = torch.randn(batch_size, 3, input_height, input_width)

    torch.onnx.export(
        model,
        (dummy_input,),
        output_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None
    )

    print(f"Exported to {output_path}")
    print(f"Input shape: [{batch_size}, 3, {input_height}, {input_width}]")

if __name__ == "__main__":
    export_unet_onnx()