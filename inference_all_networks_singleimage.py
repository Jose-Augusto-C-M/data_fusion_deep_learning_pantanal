import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import torch.nn as nn
from segmentation_models_pytorch import Unet, DeepLabV3Plus, PSPNet
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# ------------------------------
# Helper: Replace BatchNorm with GroupNorm (as used during training)
# ------------------------------
def replace_bn_with_gn(module, num_groups=16):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            groups = num_groups if num_features >= num_groups else num_features
            new_layer = nn.GroupNorm(num_groups=groups, num_channels=num_features)
            setattr(module, name, new_layer)
        else:
            replace_bn_with_gn(child, num_groups)

# ------------------------------
# Model Definitions with modifications for GroupNorm
# ------------------------------
def get_unet_modified():
    model = Unet(encoder_name="resnet34", encoder_weights=None, in_channels=16, classes=1)
    replace_bn_with_gn(model, num_groups=16)
    return model

def get_deeplabv3plus_modified():
    model = DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=16,
        classes=1,
        decoder_norm_layer=lambda num_features: nn.GroupNorm(
            num_groups=16 if num_features >= 16 else num_features,
            num_channels=num_features
        )
    )
    replace_bn_with_gn(model, num_groups=16)
    return model

def get_pspnet_modified():
    model = PSPNet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=16,
        classes=1,
        decoder_norm_layer=lambda num_features: nn.GroupNorm(
            num_groups=16 if num_features >= 16 else num_features,
            num_channels=num_features
        )
    )
    replace_bn_with_gn(model, num_groups=16)
    return model

# ------------------------------
# Helper: Pad image so dimensions are divisible by a given factor.
# ------------------------------
def pad_to_divisible(image, div=32, mode='constant'):
    """
    Pads a 3D numpy array (C, H, W) so that H and W are divisible by `div`.
    Returns the padded image and a tuple (pad_h, pad_w) indicating the added padding along the bottom and right.
    """
    C, H, W = image.shape
    pad_h = (div - H % div) if H % div != 0 else 0
    pad_w = (div - W % div) if W % div != 0 else 0
    padded_image = np.pad(image, ((0, 0), (0, pad_h), (0, pad_w)), mode=mode)
    return padded_image, (pad_h, pad_w)

# ------------------------------
# Inference Pipeline Functions
# ------------------------------
def load_image(image_path):
    """
    Loads a 16-band image from a TIFF file.
    Returns the image as a numpy array of shape (16, H, W) in float32.
    """
    with rasterio.open(image_path) as src:
        image = src.read()
    return image.astype(np.float32)

def preprocess_image(image):
    """
    Normalizes the image to the [0,1] range.
    """
    img_min = image.min()
    img_max = image.max()
    image = (image - img_min) / (img_max - img_min + 1e-8)
    return image

def predict(model, image_tensor, device, original_shape, pad_info):
    """
    Runs inference on the given image tensor and crops the predicted mask back to the original size.
    """
    model.eval()
    image_tensor = image_tensor.to(device)
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # add batch dimension
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type):
            logits = model(image_tensor)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
    pred_mask = preds.cpu().numpy()[0, 0]
    # Crop back to original dimensions (assumes padding was added at the bottom and right)
    pad_h, pad_w = pad_info
    H_orig, W_orig = original_shape
    pred_mask = pred_mask[:H_orig, :W_orig]
    return pred_mask

def plot_results(image, pred_mask, model_name, alpha=0.5, title_suffix=""):
    """
    Plots a pseudo-RGB image and an overlay of the segmentation mask on the pseudo-RGB image,
    with a specified transparency (alpha) for the overlay. A custom legend shows the classes.
    """
    # Create pseudo-RGB using the first three bands.
    if image.shape[0] >= 3:
        pseudo_rgb = np.stack([image[0], image[1], image[2]], axis=-1)
    else:
        pseudo_rgb = np.stack([image[0]] * 3, axis=-1)
    
    pseudo_rgb = (pseudo_rgb - pseudo_rgb.min()) / (pseudo_rgb.max() - pseudo_rgb.min() + 1e-8)
    
    # Create a discrete colormap for the segmentation mask:
    # "lightblue" for Non-Forest (0) and "forestgreen" for Forest (1).
    cmap_seg = ListedColormap(["lightblue", "forestgreen"])
    
    # Create the figure with two subplots.
    fig, ax = plt.subplots(1, 2, figsize=(14, 7), dpi=150)
    
    # Left subplot: the original pseudo-RGB image.
    ax[0].imshow(pseudo_rgb)
    ax[0].set_title(f"{model_name} Input (Pseudo-RGB)", fontsize=16)
    ax[0].axis("off")
    
    # Right subplot: overlay the segmentation mask.
    ax[1].imshow(pseudo_rgb)
    # Overlay the segmentation mask with transparency (alpha).
    ax[1].imshow(pred_mask, cmap=cmap_seg, vmin=0, vmax=1, alpha=alpha)
    ax[1].set_title(f"{model_name} Overlay Prediction", fontsize=16)
    ax[1].axis("off")
    
    # Create custom legend.
    patch0 = mpatches.Patch(color="lightblue", label="Non-Forest")
    patch1 = mpatches.Patch(color="forestgreen", label="Forest")
    ax[1].legend(handles=[patch0, patch1], loc="upper right", fontsize=12)
    
    plt.tight_layout()
    plt.show()

# ------------------------------
# Main inference loop for all models
# ------------------------------
def main():
    # ------------------------------
    # Variables and Paths (update these as needed)
    # ------------------------------
    # Sample image path:
    sample_image_path = "D:/AGRICULTURA_MS/SOJA_SHP/MERGED_1/output/period_20191001_20191231/fused_v2/fused_period_20191001_20191231_78.tif"
    
    # Checkpoint paths for each model:
    checkpoint_unet = "//storage-ume.slu.se/home$/joms0005/Desktop/UFMS/PANTANAL/UNet_fold3_best_16band.h5"
    checkpoint_deeplab = "//storage-ume.slu.se/home$/joms0005/Desktop/UFMS/PANTANAL/DeepLabV3+_fold3_best_16band.h5"
    checkpoint_pspnet = "//storage-ume.slu.se/home$/joms0005/Desktop/UFMS/PANTANAL/PSPNet_fold3_best_16band.h5"
    
    # Transparency factor for overlay (0: fully transparent, 1: fully opaque)
    overlay_alpha = 0.2
    
    # ------------------------------
    # Mapping models with their checkpoint paths.
    # ------------------------------
    model_defs = {
        "UNet": {
            "get_model": get_unet_modified,
            "checkpoint": checkpoint_unet
        },
        "DeepLabV3+": {
            "get_model": get_deeplabv3plus_modified,
            "checkpoint": checkpoint_deeplab
        },
        "PSPNet": {
            "get_model": get_pspnet_modified,
            "checkpoint": checkpoint_pspnet
        }
    }
    
    # Device configuration.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load and preprocess the image.
    image = load_image(sample_image_path)
    image = preprocess_image(image)
    # Get original image dimensions.
    _, H_orig, W_orig = image.shape
    # Pad the image so that height and width are divisible by 32.
    padded_image, pad_info = pad_to_divisible(image, div=32, mode='constant')
    image_tensor = torch.from_numpy(padded_image)
    
    # Loop over each model.
    for model_name, info in model_defs.items():
        print(f"Processing model: {model_name}")
        model = info["get_model"]().to(device)
        checkpoint_path = info["checkpoint"]
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint, strict=False)
            print(f"Loaded checkpoint for {model_name} from {checkpoint_path}")
        else:
            print(f"Checkpoint file {checkpoint_path} not found. Skipping {model_name}.")
            continue
        
        # Run prediction and crop the predicted mask to the original dimensions.
        pred_mask = predict(model, image_tensor, device, (H_orig, W_orig), pad_info)
        # Plot the results with the overlay transparency defined by overlay_alpha.
        plot_results(image, pred_mask, model_name, alpha=overlay_alpha)
        
if __name__ == "__main__":
    main()
