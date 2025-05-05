import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import torch.nn as nn
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

# segmentation_models_pytorch models
from segmentation_models_pytorch import Unet, DeepLabV3Plus, PSPNet
# your custom AttentionUNet implementation
from attention_unet import AttentionUNet

# ------------------------------
# 1) Replace BatchNorm with GroupNorm for SMP models
# ------------------------------
def replace_bn_with_gn(module, num_groups=16):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            nf = child.num_features
            grp = min(num_groups, nf)
            setattr(module, name, nn.GroupNorm(num_groups=grp, num_channels=nf))
        else:
            replace_bn_with_gn(child, num_groups)

# ------------------------------
# 2) Model factory functions
# ------------------------------
def get_unet_modified():
    m = Unet("resnet34", encoder_weights=None, in_channels=16, classes=1)
    replace_bn_with_gn(m)
    return m

def get_deeplabv3plus_modified():
    m = DeepLabV3Plus(
        "resnet34", encoder_weights=None, in_channels=16, classes=1,
        decoder_norm_layer=lambda nf: nn.GroupNorm(num_groups=min(16,nf), num_channels=nf)
    )
    replace_bn_with_gn(m)
    return m

def get_pspnet_modified():
    m = PSPNet(
        "resnet34", encoder_weights=None, in_channels=16, classes=1,
        decoder_norm_layer=lambda nf: nn.GroupNorm(num_groups=min(16,nf), num_channels=nf)
    )
    replace_bn_with_gn(m)
    return m

def get_attentionunet_modified():
    # instantiate your AttentionUNet
    m = AttentionUNet(n_channels=16, n_classes=1, bilinear=True)
    # optionally replace its BatchNorm with GroupNorm if desired:
    replace_bn_with_gn(m)
    return m

# ------------------------------
# 3) Image helpers
# ------------------------------
def load_image(path):
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)

def preprocess_image(img):
    mn, mx = img.min(), img.max()
    return (img - mn) / (mx - mn + 1e-8)

def pad_to_divisible(img, div=32):
    C, H, W = img.shape
    ph = (div - H % div) % div
    pw = (div - W % div) % div
    padded = np.pad(img, ((0,0),(0,ph),(0,pw)), mode='constant')
    return padded, (ph, pw)

# ------------------------------
# 4) Inference helper
# ------------------------------
def predict(model, tensor, device, orig_shape, pad):
    model.eval()
    t = tensor.to(device)
    if t.dim()==3:
        t = t.unsqueeze(0)
    with torch.no_grad(), torch.amp.autocast(device_type=device.type):
        logits = model(t)
        prob = torch.sigmoid(logits)[0,0].cpu().numpy()
    h, w = orig_shape
    ph, pw = pad
    return prob[:h, :w] > 0.5

# ------------------------------
# 5) Plotting summary
# ------------------------------
def plot_summary(img, mask, output_dir, sample_name, model_name,
                 alpha=0.3, brightness_factor=1.2):
    os.makedirs(output_dir, exist_ok=True)

    # SAR composite: replicate band 13
    sar = np.stack([img[13]]*3, axis=-1)
    p2,p98 = np.percentile(sar, (2,98))
    sar = np.clip((sar - p2)/(p98-p2+1e-8),0,1)

    # RGB composite: bands 3,2,1 (adjust order if needed)
    rgb = np.stack([img[3], img[2], img[1]], axis=-1)
    p2,p98 = np.percentile(rgb, (2,98))
    rgb = np.clip((rgb - p2)/(p98-p2+1e-8),0,1)
    rgb = np.clip(rgb*brightness_factor,0,1)

    # prepare masks
    nf_mask = ~mask
    cmap = ListedColormap(["red","blue"])
    patch_nf = mpatches.Patch(color="red", label="Non-Forest")
    patch_f  = mpatches.Patch(color="blue", label="Forest")

    fig, axes = plt.subplots(1,4, figsize=(24,6), dpi=150)
    axes[0].imshow(sar); axes[0].set_title("SAR (band 13)", fontsize=20); axes[0].axis('off')
    axes[1].imshow(sar)
    axes[1].imshow(nf_mask, cmap=cmap, alpha=alpha*0.4)
    axes[1].imshow(mask,    cmap=cmap, alpha=alpha)
    axes[1].set_title(f"{model_name} on SAR", fontsize=20); axes[1].axis('off')
    axes[1].legend(handles=[patch_nf,patch_f], loc='upper right', fontsize=20)
    axes[2].imshow(rgb); axes[2].set_title("True Color", fontsize=20); axes[2].axis('off')
    axes[3].imshow(rgb)
    axes[3].imshow(nf_mask, cmap=cmap, alpha=alpha*0.4)
    axes[3].imshow(mask,    cmap=cmap, alpha=alpha)
    axes[3].set_title(f"{model_name} on RGB", fontsize=20); axes[3].axis('off')
    axes[3].legend(handles=[patch_nf,patch_f], loc='upper right', fontsize=20)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"{sample_name}_{model_name}.png")
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"→ Saved {out_path}")

# ------------------------------
# 6) Main: sample & infer with all four models
# ------------------------------
def main():
    base_folder    = r"D:/AGRICULTURA_MS/SOJA_SHP/MERGED_1/output"
    input_folder   = os.path.join(base_folder, "period_20191001_20191231", "fused_v2")
    output_folder  = os.path.join(input_folder, "plots")
    sample_paths   = [os.path.join(input_folder,f) for f in os.listdir(input_folder) if f.endswith(".tif")]
    sample_paths   = random.sample(sample_paths, min(20, len(sample_paths)))

    # checkpoints for each model
    checkpoints = {
        #"UNet":        os.path.join(base_folder, "UNet_fold1_best_16band.h5"),
        "DeepLabV3+":  os.path.join(base_folder, "DeepLabV3+_fold1_best_16band.h5"),
        "PSPNet":      os.path.join(base_folder, "PSPNet_fold1_best_16band.h5"),
        "AttentionUNet": os.path.join(base_folder, "AttentionUNet_fold1_best_16band.h5"),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_getters = {
        #"UNet":         get_unet_modified,
        "DeepLabV3+":   get_deeplabv3plus_modified,
        "PSPNet":       get_pspnet_modified,
        "AttentionUNet":get_attentionunet_modified,
    }

    for path in sample_paths:
        sample_name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n=== Sample: {sample_name} ===")
        img = preprocess_image(load_image(path))
        padded, pad = pad_to_divisible(img, div=32)
        tensor = torch.from_numpy(padded)
        h,w    = img.shape[1], img.shape[2]

        for name, ckpt in checkpoints.items():
            print(f"• Model: {name}")
            model = model_getters[name]().to(device)
            if os.path.exists(ckpt):
                state = torch.load(ckpt, map_location=device)
                model.load_state_dict(state, strict=False)
            else:
                print(f"  ! checkpoint not found: {ckpt}")
                continue

            mask = predict(model, tensor, device, (h,w), pad)
            plot_summary(img, mask, output_folder, sample_name, name)

if __name__ == "__main__":
    main()
