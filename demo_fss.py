import argparse
import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from detectron2.structures import Instances, BitMasks
import torch.nn.functional as F
from torchvision.transforms.functional import pil_to_tensor
from tqdm import tqdm

# Import build_model from the inference_fss module
from inference_fss.model.model import build_model


def pad_img(x, pad_size_h, pad_size_w):
    """Pad an image or mask tensor to the specified height and width."""
    assert isinstance(x, torch.Tensor)
    h, w = x.shape[-2:]
    padh = pad_size_h - h
    padw = pad_size_w - w
    return F.pad(x, (0, padw, 0, padh))


def preprocess_image_and_mask(
    img_path, mask_path, args, device, class_id=0, instance_id=0
):
    """Load and preprocess an image and optionally its mask, returning a dict for model input."""
    # Define image transformation
    encoder_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

    # Load image
    img = Image.open(img_path).convert("RGB")
    img_tensor = encoder_transform(img).to(device)
    original_shape = img_tensor.shape[-2:]  # (212, 1152)
    img_tensor = pad_img(img_tensor, args.pad_size_h, args.pad_size_w)

    # Base dictionary
    data_dict = {
        "image": img_tensor,
        "height": original_shape[0],
        "width": original_shape[1],
        "original_img": img,  # Store original PIL image for overlay
    }

    # Process mask if provided
    if mask_path:
        mask = pil_to_tensor(Image.open(mask_path).convert("L")).long().to(device)
        mask = pad_img(mask, args.pad_size_h, args.pad_size_w)  # (1, H, W)
    else:
        mask = torch.zeros_like(img_tensor[:1])  # (1, H, W)

    # Create Instances object
    instances = Instances(original_shape)
    instances.gt_classes = torch.tensor([class_id], device=device)
    mask = BitMasks(mask)
    instances.gt_masks = mask.tensor
    instances.gt_boxes = mask.get_bounding_boxes()
    instances.ins_ids = torch.tensor([instance_id], device=device)
    data_dict["instances"] = instances

    return data_dict


def load_support_data(support_img_paths, support_mask_paths, args, device):
    """Load and preprocess all support images and masks."""
    return [
        preprocess_image_and_mask(
            img_path, mask_path, args, device, class_id=0, instance_id=0
        )
        for img_path, mask_path in zip(support_img_paths, support_mask_paths)
    ]


def parse_data(support_imgs, support_masks, query_imgs):
    """Parse and examine file paths for support and query sets."""

    for paths, name in [
        (support_imgs, "support images"),
        (support_masks, "support masks"),
        (query_imgs, "query images"),
    ]:
        assert all(os.path.exists(p) for p in paths), f"Some {name} are missing"

    return support_imgs, support_masks, query_imgs


def overlay_mask_on_image(original_img, pred_mask, color=(173, 216, 230), alpha=0.5):
    """Overlay a binary mask on the original image with specified color and transparency."""
    # Convert to PIL images
    image = original_img.convert("RGBA")
    mask = pred_mask.convert("L")

    # Create a color overlay with the given color and transparency
    overlay = Image.new("RGBA", image.size, color + (0,))
    overlay.putalpha(mask.point(lambda p: p * alpha))

    # Composite the overlay onto the original image
    overlayed = Image.alpha_composite(image, overlay)

    return overlayed


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Batch Inference with Overlay for Few-Shot Segmentation"
    )
    parser.add_argument(
        "--support_img_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to support images",
    )
    parser.add_argument(
        "--support_mask_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to support masks corresponding to support images",
    )
    parser.add_argument(
        "--query_img_path",
        type=str,
        nargs="+",
        required=True,
        help="Path to the query image to segment",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save overlaid images",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default="checkpoints/pytorch_model.bin",
        help="Path to the pre-trained model weights",
    )
    parser.add_argument(
        "--pad_size_h",
        type=int,
        default=896,
        help="Height to pad images to",
    )
    parser.add_argument(
        "--pad_size_w",
        type=int,
        default=896,
        help="Width to pad images to",
    )
    parser.add_argument(
        "--device", default="cuda", help="Device to run inference on (cuda or cpu)"
    )
    parser.add_argument(
        "--score_threshold",
        type=float,
        default=0.5,
        help="Threshold for binary segmentation",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Transparency of the mask overlay (0.0 to 1.0)",
    )
    # Model-specific arguments
    parser.add_argument("--feat_chans", type=int, default=256)
    parser.add_argument("--image_enc_use_fc", action="store_true")
    parser.add_argument("--transformer_depth", type=int, default=6)
    parser.add_argument("--transformer_nheads", type=int, default=8)
    parser.add_argument("--transformer_mlp_dim", type=int, default=2048)
    parser.add_argument("--transformer_mask_dim", type=int, default=256)
    parser.add_argument("--transformer_fusion_layer_depth", type=int, default=1)
    parser.add_argument("--transformer_num_queries", type=int, default=200)
    parser.add_argument("--transformer_pre_norm", action="store_true", default=True)
    parser.add_argument("--pt_model", type=str, default="dinov2")
    parser.add_argument("--dinov2-size", type=str, default="vit_large")
    parser.add_argument(
        "--dinov2-weights", type=str, default="checkpoints/dinov2_vitl14_pretrain.pth"
    )

    args = parser.parse_args()

    # Validate input
    assert len(args.support_img_paths) == len(args.support_mask_paths), (
        "Number of support images must match number of support masks"
    )

    # Set up device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load the model
    model = build_model(args)
    state_dict = torch.load(args.model_weights, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Load support and query data
    support_imgs, support_masks, query_imgs = parse_data(
        args.support_img_paths,
        args.support_mask_paths,
        args.query_img_path,
    )
    ref_list = load_support_data(support_imgs, support_masks, args, device)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Process query images
    print("Running inference and generating overlaid images...")
    for query_img_path in tqdm(query_imgs, total=len(query_imgs)):
        # Preprocess query image and mask
        tar_dict = preprocess_image_and_mask(
            query_img_path, None, args, device, class_id=0, instance_id=1
        )

        # Prepare input data
        data = [
            {"ref_dict": ref, "tar_dict": tar_dict if i == 0 else None}
            for i, ref in enumerate(ref_list)
        ]

        # Perform inference
        with torch.no_grad():
            output = model(data)
            pred = output["sem_seg"].squeeze()  # Shape: (H, W)
            pred = pred > args.score_threshold  # Binary mask
            pred = pred.float().cpu().numpy()
            pred_mask = (pred * 255).astype(np.uint8)

        # Save the predicted mask
        pred_mask = Image.fromarray(pred_mask)
        pred_mask.save(
            os.path.join(
                args.output_dir, f"{os.path.basename(query_img_path)[:-4]}_mask.png"
            )
        )

        # Save the overlaid image
        overlaid_img = overlay_mask_on_image(tar_dict["original_img"], pred_mask)
        overlaid_img.save(
            os.path.join(
                args.output_dir, f"{os.path.basename(query_img_path)[:-4]}_overlay.png"
            )
        )

    print(f"Overlaid images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
