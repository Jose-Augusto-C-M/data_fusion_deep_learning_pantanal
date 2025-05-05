#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script performs two main tasks:
  1. Merges adjacent tile images (clusters) for Sentinel-2 and Sentinel-1 separately.
  2. Fuses the merged images by stacking their bands.
  
If data for one sensor is missing for a cluster, the script will create a blank (nodata) array for that sensor's portion.

Additionally, you have the option to drop Sentinel-2 bands 14 to 16 (assumed to be indices 13, 14, and 15 in a 0-indexed array) if they contain no useful information.

Activate each part by setting the corresponding flags (DO_MERGE, DO_FUSE, and DROP_S2_NOINFO_BANDS) to True or False.
"""

import os
import rasterio
from rasterio.merge import merge
from shapely.geometry import box
import networkx as nx
import numpy as np

# -------------------------------
# ACTIVATION FLAGS
# -------------------------------
DO_MERGE = False         # Set to True to run the merging step for each sensor.
DO_FUSE  = True         # Set to True to run the fusion step (which expects merged images).
DROP_S2_NOINFO_BANDS = True  # Set to True to drop Sentinel-2 bands 14 to 16 (0-indexed indices 13 to 15).

# -------------------------------
# CONFIGURATION: Folder Paths
# -------------------------------v
# Input folders for the individual tile images
SENTINEL2_TILES_FOLDER = r"D:/AGRICULTURA_MS/SOJA_SHP/MERGED_1/output/period_20200101_20200331/Sentinel2_tiles"
SENTINEL1_TILES_FOLDER = r"D:/AGRICULTURA_MS/SOJA_SHP/MERGED_1/output/period_20200101_20200331/Sentinel1_tiles"

# Output folders for merged clusters (created separately for each sensor)
MERGED_S2_FOLDER = r"D:/AGRICULTURA_MS/SOJA_SHP/MERGED_1/output/period_20200101_20200331/Sentinel2_merged_v2"
MERGED_S1_FOLDER = r"D:/AGRICULTURA_MS/SOJA_SHP/MERGED_1/output/period_20200101_20200331/Sentinel1_merged_v2"

# Output folder for final fused images
FUSED_OUTPUT_FOLDER = r"D:/AGRICULTURA_MS/SOJA_SHP/MERGED_1/output/period_20200101_20200331/merged_fused"

# Create output folders if they do not exist.
for folder in [MERGED_S2_FOLDER, MERGED_S1_FOLDER, FUSED_OUTPUT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# Buffer to allow slight gaps between tiles (in CRS units)
BUFFER = 0.0

# -------------------------------
# MERGING FUNCTIONS
# -------------------------------
def get_raster_bounds(raster_path):
    """Return the bounding box of the raster as a Shapely geometry."""
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
    return box(bounds.left, bounds.bottom, bounds.right, bounds.top)

def build_connectivity_graph(raster_files, buffer=0.0):
    """
    Build a graph where each node is a raster file and an edge is created if 
    the (buffered) bounding boxes of two files touch or intersect.
    """
    G = nx.Graph()
    geom_dict = {}
    for fp in raster_files:
        geom = get_raster_bounds(fp).buffer(buffer)
        geom_dict[fp] = geom
        G.add_node(fp)
    raster_list = list(raster_files)
    for i in range(len(raster_list)):
        for j in range(i + 1, len(raster_list)):
            fp1 = raster_list[i]
            fp2 = raster_list[j]
            if geom_dict[fp1].intersects(geom_dict[fp2]):
                G.add_edge(fp1, fp2)
    return G

def merge_cluster(raster_files, output_filename):
    """
    Merge the given raster files into a single output file.
    Returns the output filename if successful, otherwise None.
    """
    src_files = []
    for fp in raster_files:
        try:
            src = rasterio.open(fp)
            src_files.append(src)
        except Exception as e:
            print(f"Error opening {fp}: {e}")
    if not src_files:
        print("No files to merge for this cluster.")
        return None

    mosaic, out_trans = merge(src_files)
    out_meta = src_files[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
        "count": mosaic.shape[0]
    })
    with rasterio.open(output_filename, "w", **out_meta) as dst:
        dst.write(mosaic)
    print(f"Merged cluster saved to {output_filename}")

    for src in src_files:
        src.close()
    return output_filename

def merge_tiles_in_folder(input_folder, merged_output_folder, buffer=0.0):
    """
    For all GeoTIFF files in input_folder, group those that touch or border each other,
    merge them into clusters, and save each cluster as a separate GeoTIFF in merged_output_folder.
    Returns a dictionary mapping the merged filename (basename) to its full file path.
    """
    raster_files = [os.path.join(input_folder, f)
                    for f in os.listdir(input_folder)
                    if f.lower().endswith(('.tif', '.tiff'))]
    if not raster_files:
        print(f"No raster files found in {input_folder}")
        return {}
    
    G = build_connectivity_graph(raster_files, buffer=buffer)
    clusters = list(nx.connected_components(G))
    print(f"Found {len(clusters)} clusters in {input_folder}.")
    
    output_files = {}
    for idx, cluster in enumerate(clusters, start=1):
        cluster_list = list(cluster)
        out_filename = os.path.join(merged_output_folder, f"merged_cluster_{idx}.tif")
        result = merge_cluster(cluster_list, out_filename)
        if result:
            basename = os.path.basename(out_filename)
            output_files[basename] = out_filename
    return output_files

# -------------------------------
# FUSION FUNCTIONS
# -------------------------------
def read_raster_as_array(path):
    """Read a raster and return its data array and metadata."""
    with rasterio.open(path) as src:
        data = src.read()  # shape: (bands, height, width)
        meta = src.meta.copy()
    return data, meta

def create_blank_array(shape, dtype, nodata=0):
    """Create a blank (nodata) array with given shape and dtype."""
    return np.full(shape, nodata, dtype=dtype)

def fuse_merged_images(s2_path, s1_path, output_path):
    """
    Fuse two merged images by stacking their bands.
    If one sensor's image is missing, a blank array is created for that sensor.
    Also, if DROP_S2_NOINFO_BANDS is True, drop Sentinel-2 bands 14-16 (indices 13 to 15).
    Assumes at least one sensor image is available and defines the spatial dimensions.
    """
    # Read Sentinel-2 data if available.
    if s2_path and os.path.exists(s2_path):
        s2_data, s2_meta = read_raster_as_array(s2_path)
    else:
        s2_data, s2_meta = None, None
    # Read Sentinel-1 data if available.
    if s1_path and os.path.exists(s1_path):
        s1_data, s1_meta = read_raster_as_array(s1_path)
    else:
        s1_data, s1_meta = None, None

    # Use available image as reference.
    if s2_data is not None:
        ref_data = s2_data
        ref_meta = s2_meta
    elif s1_data is not None:
        ref_data = s1_data
        ref_meta = s1_meta
    else:
        print("No data available to fuse.")
        return

    bands, height, width = ref_data.shape
    dtype = ref_data.dtype

    # Debug prints for sensor data ranges.
    if s2_data is not None:
        print("Sentinel-2 data range:", s2_data.min(), s2_data.max())
    if s1_data is not None:
        print("Sentinel-1 data range:", s1_data.min(), s1_data.max())

    # If enabled, drop Sentinel-2 bands 14 to 16 (assumed indices 13, 14, 15)
    if DROP_S2_NOINFO_BANDS and s2_data is not None:
        if s2_data.shape[0] >= 16:
            print("Dropping Sentinel-2 bands 14 to 16 (indices 13-15).")
            s2_data = np.delete(s2_data, np.s_[13:16], axis=0)
        else:
            print("Warning: Sentinel-2 data has fewer than 16 bands; skipping band drop.")

    # Create blank arrays if one sensor's data is missing.
    if s2_data is None:
        print("Sentinel-2 image missing; creating blank array for S2 bands.")
        # Define number of S2 bands (adjust as needed; here we assume 3 bands).
        s2_data = create_blank_array((3, height, width), dtype, nodata=0)
    if s1_data is None:
        print("Sentinel-1 image missing; creating blank array for S1 bands.")
        # Define number of S1 bands (adjust as needed; here we assume 1 band).
        s1_data = create_blank_array((1, height, width), dtype, nodata=0)

    # Fuse by stacking bands: S2 bands first, then S1 bands.
    fused_data = np.vstack([s2_data, s1_data])
    fused_meta = ref_meta.copy()
    fused_meta.update({"count": fused_data.shape[0]})
    
    with rasterio.open(output_path, "w", **fused_meta) as dst:
        dst.write(fused_data)
    print(f"Fused image saved to {output_path}")

# -------------------------------
# MAIN PROCESSING
# -------------------------------
def main():
    # ---------------------------
    # Step 1: Merging
    # ---------------------------
    merged_s2_dict = {}
    merged_s1_dict = {}
    if DO_MERGE:
        print("Merging Sentinel-2 tiles...")
        merged_s2_dict = merge_tiles_in_folder(SENTINEL2_TILES_FOLDER, MERGED_S2_FOLDER, buffer=BUFFER)
        print("Merging Sentinel-1 tiles...")
        merged_s1_dict = merge_tiles_in_folder(SENTINEL1_TILES_FOLDER, MERGED_S1_FOLDER, buffer=BUFFER)
    else:
        # If merging is not activated, assume merged images already exist.
        merged_s2_dict = {f: os.path.join(MERGED_S2_FOLDER, f)
                          for f in os.listdir(MERGED_S2_FOLDER)
                          if f.lower().endswith(('.tif', '.tiff'))}
        merged_s1_dict = {f: os.path.join(MERGED_S1_FOLDER, f)
                          for f in os.listdir(MERGED_S1_FOLDER)
                          if f.lower().endswith(('.tif', '.tiff'))}

    # ---------------------------
    # Step 2: Fusion
    # ---------------------------
    if DO_FUSE:
        # Use the union of keys from both merged dictionaries.
        all_keys = set(merged_s2_dict.keys()).union(set(merged_s1_dict.keys()))
        print(f"Found {len(all_keys)} potential clusters to fuse.")
        if len(all_keys) == 0:
            print("No potential clusters (or tile images) found to fuse based on filename matching.")
            return
        total = len(all_keys)
        incomplete = 0
        for key in all_keys:
            s2_path = merged_s2_dict.get(key, None)
            s1_path = merged_s1_dict.get(key, None)
            fused_filename = f"fused_{key}"
            fused_output_path = os.path.join(FUSED_OUTPUT_FOLDER, fused_filename)
            try:
                fuse_merged_images(s2_path, s1_path, fused_output_path)
                # If one sensor's data was missing, the corresponding print statement is shown.
                # (You can add additional counters here if needed.)
            except Exception as e:
                print(f"Error fusing {key}: {e}")
                incomplete += 1
        print(f"Fusion complete. Total items processed: {total}")
        print(f"Incomplete fusions (errors): {incomplete} ({(incomplete/total)*100:.1f}%)")
    else:
        print("Fusion step not activated (DO_FUSE is False).")

if __name__ == '__main__':
    main()
