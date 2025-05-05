import os
import glob
import random
import numpy as np
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt

# ------------------------------
# PARAMETERS (Update as needed)
# ------------------------------
raw_folder = r"D:/AGRICULTURA_MS/SOJA_SHP/MERGED_1/output/merged_fused_centralized"
ann_folder = r"D:/AGRICULTURA_MS/SOJA_SHP/MERGED_1/output/prob_export"

patch_raw_folder = r"D:/AGRICULTURA_MS/SOJA_SHP/MERGED_1/output/merged_fused_centralized_patches_raw_128"
patch_ann_folder = r"D:/AGRICULTURA_MS/SOJA_SHP/MERGED_1/output/annotations_patches_128"
patch_size = 128

# Annotation threshold parameters:
min_class1_ratio = 0.1  # Only keep patches with >10% pixels of class 1.
class_threshold = 0.8   # Pixels above this value are considered class 1.

# Zero-value filtering parameters:
skip_if_any_zero = True   # Approach A: skip if any pixel is zero in any band.
max_zero_fraction = 0.0    # Approach B: skip if > X% of a patch is zero (set to 0.0 to disable).

# Plotting parameters:
plot_interval = 50  # Every 50 patches, update the plot.
num_samples_to_plot = 4  # Number of patch pairs to plot.

# ------------------------------
# Create patch folders if they don't exist.
# ------------------------------
os.makedirs(patch_raw_folder, exist_ok=True)
os.makedirs(patch_ann_folder, exist_ok=True)

# Lists to store file paths for plotting.
extracted_raw_files = []
extracted_ann_files = []

# ------------------------------
# Patch Extraction
# ------------------------------
raw_files = sorted(glob.glob(os.path.join(raw_folder, "*.tif")))
ann_files = sorted(glob.glob(os.path.join(ann_folder, "*.tif")))

print("Starting patch extraction...")
patch_counter = 0

for raw_file, ann_file in zip(raw_files, ann_files):
    raw_name = os.path.splitext(os.path.basename(raw_file))[0]
    ann_name = os.path.splitext(os.path.basename(ann_file))[0]
    print(f"Processing: {raw_name} and {ann_name}")

    with rasterio.open(raw_file) as raw_src, rasterio.open(ann_file) as ann_src:
        width, height = raw_src.width, raw_src.height

        for top in range(0, height, patch_size):
            for left in range(0, width, patch_size):
                # Skip if the window exceeds image dimensions
                if (left + patch_size > width) or (top + patch_size > height):
                    continue

                window = Window(left, top, patch_size, patch_size)
                raw_patch = raw_src.read(window=window)  # shape: (bands, patch_size, patch_size)
                ann_patch = ann_src.read(1, window=window)  # shape: (patch_size, patch_size)

                # Option A: Skip if any pixel is zero in any band.
                if skip_if_any_zero and np.any(raw_patch == 0):
                    continue

                # Option B: Skip if any band has more than max_zero_fraction zeros.
                if max_zero_fraction > 0:
                    skip_band = False
                    for b in range(raw_patch.shape[0]):
                        frac_zeros = np.count_nonzero(raw_patch[b] == 0) / raw_patch[b].size
                        if frac_zeros > max_zero_fraction:
                            skip_band = True
                            break
                    if skip_band:
                        continue

                # Check if annotation patch has enough class1 pixels.
                class1_pixels = np.count_nonzero(ann_patch > class_threshold)
                ratio = class1_pixels / ann_patch.size
                if ratio < min_class1_ratio:
                    continue

                # Threshold the annotation patch to be binary.
                ann_patch_binary = (ann_patch > class_threshold).astype(np.float32)

                # Save the raw patch.
                patch_raw_filename = os.path.join(patch_raw_folder, f"{raw_name}_patch_{patch_counter}.tif")
                raw_meta = raw_src.meta.copy()
                raw_meta.update({
                    "height": patch_size,
                    "width": patch_size,
                    "transform": rasterio.windows.transform(window, raw_src.transform)
                })
                with rasterio.open(patch_raw_filename, "w", **raw_meta) as dst:
                    dst.write(raw_patch)
                extracted_raw_files.append(patch_raw_filename)

                # Save the binary annotation patch.
                patch_ann_filename = os.path.join(patch_ann_folder, f"{ann_name}_patch_{patch_counter}.tif")
                ann_meta = ann_src.meta.copy()
                ann_meta.update({
                    "count": 1,
                    "height": patch_size,
                    "width": patch_size,
                    "transform": rasterio.windows.transform(window, ann_src.transform)
                })
                with rasterio.open(patch_ann_filename, "w", **ann_meta) as dst:
                    dst.write(ann_patch_binary, 1)
                extracted_ann_files.append(patch_ann_filename)

                print(f"Saved patch {patch_counter} | Class1 ratio: {ratio:.2f}")
                patch_counter += 1

                # Every 'plot_interval' patches, update the visualization.
                if patch_counter % plot_interval == 0 and len(extracted_raw_files) >= num_samples_to_plot:
                    sample_indices = random.sample(range(len(extracted_raw_files)), num_samples_to_plot)
                    fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(10, 4 * num_samples_to_plot))
                    
                    # If only one sample, adjust axes shape.
                    if num_samples_to_plot == 1:
                        axes = np.expand_dims(axes, axis=0)
                    
                    for i, idx in enumerate(sample_indices):
                        with rasterio.open(extracted_raw_files[idx]) as src:
                            raw_patch_sample = src.read()  # shape: (bands, patch_size, patch_size)
                        if raw_patch_sample.shape[0] >= 3:
                            pseudo_rgb = np.stack([raw_patch_sample[0], raw_patch_sample[1], raw_patch_sample[2]], axis=-1)
                        else:
                            pseudo_rgb = np.stack([raw_patch_sample[0]]*3, axis=-1)
                        pseudo_rgb = (pseudo_rgb - np.min(pseudo_rgb)) / (np.max(pseudo_rgb) - np.min(pseudo_rgb) + 1e-8)
                        
                        with rasterio.open(extracted_ann_files[idx]) as src:
                            ann_patch_sample = src.read(1)
                        
                        axes[i, 0].imshow(pseudo_rgb)
                        axes[i, 0].set_title("Raw Patch (Pseudo-RGB)", fontsize=14)
                        axes[i, 0].axis("off")
                        
                        axes[i, 1].imshow(ann_patch_sample, cmap="gray")
                        axes[i, 1].set_title("Annotation Patch (Binary)", fontsize=14)
                        axes[i, 1].axis("off")
                    
                    plt.tight_layout()
                    plt.show()
                    # Optionally, pause before continuing.
                    plt.pause(0.5)
                    plt.close(fig)

print("Patch extraction complete.")
