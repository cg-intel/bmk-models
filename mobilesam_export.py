import os
import torch
from mobile_sam import sam_model_registry
from mobile_sam.utils.onnx import SamOnnxModel

# image encoder
def export_image_encoder_onnx(checkpoint, model_type, mode_path, input_height=1024, input_width=1024, batch_size=1):
    sam = sam_model_registry[model_type](checkpoint=checkpoint).eval()

    image_encoder = sam.image_encoder

    dummy_image = torch.randn(batch_size, 3, input_height, input_width)

    torch.onnx.export(
        image_encoder,
        (dummy_image, ),
        mode_path,
        input_names=["image"],
        output_names=["image_embeddings"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None,
    )

    print(f"ONNX model exported to: {mode_path}")
    print(f"Input shape: [{batch_size}, 3, {input_height}, {input_width}]")

# prompt encoder + decoder
def export_prompt_decoder_onnx(checkpoint, model_type, mode_path):
    sam = sam_model_registry[model_type](checkpoint=checkpoint).eval()

    onnx_model = SamOnnxModel(sam, return_single_mask=True)

    dummy_inputs = {
        "image_embeddings": torch.randn(1, 256, 64, 64),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, 256, 256),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
        "orig_im_size": torch.tensor([1024, 1024], dtype=torch.int64),
    }

    torch.onnx.export(
        onnx_model,
        tuple(dummy_inputs.values()),
        mode_path,
        input_names=list(dummy_inputs.keys()),
        output_names=["masks", "iou_predictions", "low_res_masks"],
        opset_version=11,
        do_constant_folding=True,
        dynamic_axes=None,
    )

    print(f"ONNX model exported to: {mode_path}")


if __name__ == "__main__":
    checkpoint = "mobile_sam.pt"
    model_type = "vit_t"

    if not os.path.exists(checkpoint):
        print("Download from https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt.")
    else:
        print("Checkpoint already exists.")

    export_image_encoder_onnx(checkpoint, model_type, "models/mobilesam-image-encoder.onnx")
    export_prompt_decoder_onnx(checkpoint, model_type, "models/mobilesam-prompt-decoder.onnx")
