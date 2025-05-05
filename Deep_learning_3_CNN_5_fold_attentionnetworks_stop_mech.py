# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 11:49:32 2025

@author: joms0005
"""

"""
5-Fold Cross Validation for 16-Band Semantic Segmentation (Forest vs. Background)
using AttentionUNet, DeepLabV3+, and PSPNet.

Key improvements to reduce overfitting:
 - Simple data augmentation (random horizontal and vertical flips).
 - Weight decay (L2 regularization) added to the optimizer.
 - Learning rate scheduler (ReduceLROnPlateau) to adjust the learning rate on plateau.
 - Intermediate checkpoints: every checkpoint_interval epochs, save prediction figures and loss plots.
 - Early stopping: training stops if no validation loss improvement is seen for early_stop_patience epochs.
 - Saves the best model (by validation IoU) in H5 format.
 - Saves a log file with training metrics.
 - Saves a figure of the confusion matrix with percentage annotations.
 - Uses mixed precision training (AMP) and cuDNN benchmark for improved GPU utilization.
 - Replaces BatchNorm with GroupNorm in the decoders for DeepLabV3+ and PSPNet.
 - Incorporates a state-of-the-art attention network (AttentionUNet) for segmentation.
 - Improved figure text quality.
 
Author: ChatGPT
"""

import os
import glob
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns  # For enhanced heatmap visualization

from segmentation_models_pytorch import DeepLabV3Plus, PSPNet

# Enable cuDNN benchmark mode for optimized GPU performance.
torch.backends.cudnn.benchmark = True

# ------------------------------
# Device Setup: Force GPU usage
# ------------------------------
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Please ensure that you have a GPU and CUDA installed.")
device = torch.device("cuda")
print(f"Using GPU: {torch.cuda.get_device_name(device)}")

# ------------------------------
# Base Folder for Saving Files
# ------------------------------
base_folder = r"D:/AGRICULTURA_MS/SOJA_SHP/MERGED_1/output/"
# All subfolders will be created relative to this base_folder.

# ------------------------------
# Helper: Replace BatchNorm with GroupNorm
# ------------------------------
def replace_bn_with_gn(module, num_groups=16):
    """
    Recursively replace all BatchNorm2d layers in the given module with GroupNorm layers.
    """
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_features = child.num_features
            groups = num_groups if num_features >= num_groups else num_features
            new_layer = nn.GroupNorm(num_groups=groups, num_channels=num_features)
            setattr(module, name, new_layer)
        else:
            replace_bn_with_gn(child, num_groups)

# ------------------------------
# Helper: Simple Data Augmentation Transform
# ------------------------------
def simple_transform(image, mask):
    """
    Applies random horizontal and vertical flips to the image and mask.
    Assumes image shape is (channels, H, W) and mask shape is (1, H, W).
    """
    if random.random() > 0.5:
        image = np.flip(image, axis=2).copy()  # Flip horizontally
        mask = np.flip(mask, axis=2).copy()
    if random.random() > 0.5:
        image = np.flip(image, axis=1).copy()  # Flip vertically
        mask = np.flip(mask, axis=1).copy()
    return image, mask

# ------------------------------
# 1) Dataset Definition
# ------------------------------
class CropDataset(Dataset):
    """
    Dataset for 16-band images and corresponding single-band masks.
    Binarizes masks at threshold=0.5 by default.
    """
    def __init__(self, image_dir, mask_dir, threshold=0.5, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.tif")))
        self.threshold = threshold
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        with rasterio.open(img_path) as src:
            image = src.read()  # shape: (16, H, W)
        image = image.astype(np.float32)
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # shape: (H, W)
        mask = mask.astype(np.float32)
        mask = (mask > self.threshold).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)  # shape: (1, H, W)
        if self.transform:
            image, mask = self.transform(image, mask)
        return torch.from_numpy(image), torch.from_numpy(mask)

# ------------------------------
# 2) Loss & Metric Functions
# ------------------------------
class DiceLoss(nn.Module):
    """Binary Dice Loss."""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    def forward(self, preds, targets):
        preds = preds.view(-1)
        targets = targets.view(-1)
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        return 1 - dice

class BCEDiceLoss(nn.Module):
    """Combination of BCEWithLogitsLoss and DiceLoss."""
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        dice_loss = self.dice(probs, targets)
        return bce_loss + dice_loss

def compute_metrics(preds, targets):
    """
    Computes IoU, Accuracy, F1-score and returns the confusion matrix.
    preds, targets: (N, 1, H, W) in {0,1}
    """
    preds = preds.view(-1).cpu().numpy()
    targets = targets.view(-1).cpu().numpy()
    cm = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    iou = tp / (tp + fp + fn + 1e-7)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-7)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-7)
    return {"IoU": iou, "Accuracy": accuracy, "F1": f1, "ConfusionMatrix": cm}

# ------------------------------
# 3) Training & Evaluation Functions
# ------------------------------
def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler):
    model.train()
    total_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            logits = model(images)
            loss = loss_fn(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * images.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_fn(logits, masks)
            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            all_preds.append(preds.cpu())
            all_targets.append(masks.cpu())
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)
    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, metrics

# ------------------------------
# 4) AttentionUNet Implementation
# ------------------------------
class DoubleConv(nn.Module):
    """(convolution => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv."""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.mpconv(x)

class AttentionGate(nn.Module):
    """
    Attention Gate to filter features from the skip connection.
    F_g: number of channels in the gating signal.
    F_l: number of channels in the skip connection.
    F_int: number of intermediate channels.
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Up(nn.Module):
    """
    Upscaling then double conv with attention gate on the skip connection.
    - in_channels: channels of the upsampled tensor (gating signal).
    - skip_channels: channels in the skip connection.
    - out_channels: desired output channels after convolution.
    """
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.att = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=skip_channels // 2)
        self.conv = DoubleConv(in_channels + skip_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x2 = self.att(g=x1, x=x2)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class AttentionUNet(nn.Module):
    """
    Attention U-Net architecture for segmentation.
    """
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024 // factor, skip_channels=512, out_channels=512 // factor, bilinear=bilinear)
        self.up2 = Up(512 // factor, skip_channels=256, out_channels=256 // factor, bilinear=bilinear)
        self.up3 = Up(256 // factor, skip_channels=128, out_channels=128 // factor, bilinear=bilinear)
        self.up4 = Up(128 // factor, skip_channels=64, out_channels=64, bilinear=bilinear)
        self.outc = OutConv(64, n_classes)
    def forward(self, x):
        x1 = self.inc(x)       # [B, 64, H, W]
        x2 = self.down1(x1)    # [B, 128, H/2, W/2]
        x3 = self.down2(x2)    # [B, 256, H/4, W/4]
        x4 = self.down3(x3)    # [B, 512, H/8, W/8]
        x5 = self.down4(x4)    # [B, 1024//factor, H/16, W/16]
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

# ------------------------------
# 4) Plotting and Logging Helpers
# ------------------------------
def plot_confusion_matrix(cm, classes, title="Confusion Matrix", cmap=plt.cm.Blues, save_path=None):
    total = np.sum(cm)
    cm_perc = np.round((cm / total) * 100, 2)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_perc, annot=True, fmt=".2f", cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=16)
    plt.ylabel("True label", fontsize=14)
    plt.xlabel("Predicted label", fontsize=14)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    plt.show()

def plot_predictions(model, loader, device, save_path, num_samples=3):
    model.eval()
    images_list, gt_list, pred_list = [], [], []
    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            images_list.append(images.cpu())
            gt_list.append(masks.cpu())
            pred_list.append(preds.cpu())
            if len(torch.cat(images_list, dim=0)) >= num_samples:
                break
    images = torch.cat(images_list, dim=0)[:num_samples]
    gt_masks = torch.cat(gt_list, dim=0)[:num_samples]
    pred_masks = torch.cat(pred_list, dim=0)[:num_samples]
    os.makedirs(save_path, exist_ok=True)
    for idx in range(num_samples):
        img = images[idx].numpy()  # shape: (16, H, W)
        if img.shape[0] >= 3:
            rgb = np.stack([img[0], img[1], img[2]], axis=-1)
        else:
            rgb = np.stack([img[0]]*3, axis=-1)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        gt = gt_masks[idx].squeeze().numpy()
        pred = pred_masks[idx].squeeze().numpy()
        fig, ax = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
        ax[0].imshow(rgb)
        ax[0].set_title("Input (Pseudo-RGB)", fontsize=16)
        ax[1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        ax[1].set_title("Ground Truth", fontsize=16)
        ax[2].imshow(pred, cmap='gray', vmin=0, vmax=1)
        ax[2].set_title("Prediction", fontsize=16)
        for a in ax:
            a.tick_params(labelsize=14)
            a.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"prediction_sample_{idx}.png"), bbox_inches='tight')
        plt.close()

def plot_loss_curve(train_losses, val_losses, model_name, fold_num, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(dpi=150)
    epochs = range(1, len(train_losses)+1)
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title(f"{model_name} - Fold {fold_num} Loss Curve", fontsize=18)
    plt.legend(fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"{model_name}_fold{fold_num}_loss.png")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

# ------------------------------
# 5) Main Script (5-Fold CV) with Logging, Confusion Matrix, Plotting, and Early Stopping
# ------------------------------
def main():
    # Set plot_only to True to generate plots without training.
    plot_only = False  # Change to True to only generate plots from saved checkpoints.
    
    # Checkpoint interval (in epochs) for saving intermediate plots.
    checkpoint_interval = 50
    
    # Early stopping: if no improvement over early_stop_patience epochs, training for the fold stops.
    early_stop_patience = 50
    
    # ------------------------------
    # PARAMETERS (Update as needed)
    # ------------------------------
    image_dir = os.path.join(base_folder, "merged_fused_centralized_patches_raw_128")
    mask_dir = os.path.join(base_folder, "annotations_patches_128")
    base_fig_dir = os.path.join(base_folder, "prediction_figures")
    os.makedirs(base_fig_dir, exist_ok=True)
    
    num_epochs = 500
    lr = 1e-6
    batch_size = 8
    n_splits = 3
    # ------------------------------
    
    full_dataset = CropDataset(image_dir, mask_dir, threshold=0.5, transform=simple_transform)
    num_samples = len(full_dataset)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Define model architectures.
    # Remove UNet; use our state-of-the-art attention network (AttentionUNet) along with DeepLabV3+ and PSPNet.
    def create_attention_unet():
        return AttentionUNet(n_channels=16, n_classes=1, bilinear=True)
    
    model_defs = {
        "AttentionUNet": create_attention_unet,
        "DeepLabV3+": lambda: DeepLabV3Plus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=16,
            classes=1,
            decoder_norm_layer=lambda num_features: nn.GroupNorm(num_groups=16 if num_features >= 16 else num_features, num_channels=num_features)
        ),
        "PSPNet": lambda: PSPNet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=16,
            classes=1,
            decoder_norm_layer=lambda num_features: nn.GroupNorm(num_groups=16 if num_features >= 16 else num_features, num_channels=num_features)
        )
    }
    
    loss_fn = BCEDiceLoss()
    fold_num = 1
    for train_idx, val_idx in kf.split(np.arange(num_samples)):
        print(f"\n===== Fold {fold_num} / {n_splits} =====")
        train_dataset = Subset(full_dataset, train_idx)
        val_dataset = Subset(full_dataset, val_idx)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        
        for model_name, model_fn in model_defs.items():
            print(f"\n--- {model_name} on Fold {fold_num} ---")
            model = model_fn()
            replace_bn_with_gn(model, num_groups=16)
            model = model.to("cuda")
            
            checkpoint_path = os.path.join(base_folder, f"{model_name}_fold{fold_num}_best_16band.h5")
            log_path = os.path.join(base_fig_dir, f"{model_name}_fold{fold_num}_log.txt")
            log_lines = []
            
            best_val_iou = 0.0
            best_epoch = 0
            train_losses = []
            val_losses = []
            
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            scaler = torch.amp.GradScaler()
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
            
            for epoch in range(num_epochs):
                train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, "cuda", scaler)
                val_loss, val_metrics = evaluate(model, val_loader, loss_fn, "cuda")
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                scheduler.step(val_loss)
                
                iou = val_metrics["IoU"]
                acc = val_metrics["Accuracy"]
                f1  = val_metrics["F1"]
                log_line = (f"[{model_name}][Fold {fold_num}][Epoch {epoch+1}/{num_epochs}] "
                            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                            f"IoU: {iou:.4f}, Acc: {acc:.4f}, F1: {f1:.4f}")
                print(log_line)
                log_lines.append(log_line)
                
                # Update best model if current IoU improves.
                if iou > best_val_iou:
                    best_val_iou = iou
                    best_epoch = epoch+1
                    torch.save(model.state_dict(), checkpoint_path)
                
                # Early stopping check.
                if epoch+1 - best_epoch >= early_stop_patience:
                    print(f"No improvement over {early_stop_patience} epochs. Early stopping.")
                    break
                    
                if (epoch + 1) % checkpoint_interval == 0:
                    interim_fig_path = os.path.join(base_fig_dir, f"{model_name}_fold{fold_num}_epoch{epoch+1}")
                    plot_predictions(model, val_loader, "cuda", save_path=interim_fig_path, num_samples=3)
                    interim_loss_dir = os.path.join(base_fig_dir, "loss_plots", f"{model_name}_fold{fold_num}_epoch{epoch+1}")
                    plot_loss_curve(train_losses, val_losses, model_name, fold_num, interim_loss_dir)
                    print(f"Saved intermediate plots for {model_name} on Fold {fold_num} at epoch {epoch+1}")
                    
            print(f"Best {model_name} IoU on Fold {fold_num}: {best_val_iou:.4f} at epoch {best_epoch}")
            
            loss_plot_dir = os.path.join(base_fig_dir, "loss_plots")
            plot_loss_curve(train_losses, val_losses, model_name, fold_num, loss_plot_dir)
            fig_save_path = os.path.join(base_fig_dir, f"{model_name}_fold{fold_num}")
            plot_predictions(model, val_loader, "cuda", save_path=fig_save_path, num_samples=3)
            print(f"Prediction figures saved in {fig_save_path}")
            
            # Final evaluation and confusion matrix plot.
            val_loss, val_metrics = evaluate(model, val_loader, loss_fn, "cuda")
            cm = val_metrics["ConfusionMatrix"]
            cm_save_path = os.path.join(base_fig_dir, f"{model_name}_fold{fold_num}_confusion_matrix.png")
            plot_confusion_matrix(cm, classes=["Non-Forest", "Forest"],
                                  title=f"{model_name} Fold {fold_num} Confusion Matrix",
                                  save_path=cm_save_path)
            
            with open(log_path, "w") as log_file:
                for line in log_lines:
                    log_file.write(line + "\n")
            print(f"Log saved to {log_path}")
        fold_num += 1

if __name__ == "__main__":
    main()
