#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script loads an AOI from a shapefile and exports per-tile composites
for Sentinel-2 and Sentinel-1. Tiles already on disk can be skipped by toggling SKIP_EXISTING.
Merging logic has been removed.
"""

import os
import sys
import time
import ee
import geemap
import geopandas as gpd
from tqdm import tqdm

# ------------- CONFIGURATION -------------
# If True, any tile TIFF that already exists on disk will be skipped.
SKIP_EXISTING = True

# Path to your AOI shapefile and base output folder
SHP_PATH        = r"D:\AGRICULTURA_MS\SOJA_SHP\MERGED_1.shp"
BASE_OUTPUT     = r"D:\AGRICULTURA_MS\SOJA_SHP\MERGED_2\output"

# Period definitions
PERIODS = [
    ("2024-01-01","2024-03-31"),
    ("2024-04-01","2024-06-30"),
    ("2024-07-01","2024-09-30"),
    ("2024-10-01","2024-12-31"),
]

# Export parameters
S2_SCALE, S1_SCALE, CRS = 10, 10, "EPSG:4326"
H_INTERVAL, V_INTERVAL   = 0.05, 0.05
# -----------------------------------------

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception:
    print("Run `earthengine authenticate` first.")
    sys.exit(1)

def load_geometry_from_shp(shp_path):
    gdf = gpd.read_file(shp_path)
    if gdf.crs.to_string() != 'EPSG:4326':
        gdf = gdf.to_crs(epsg=4326)
    union = gdf.union_all()
    return ee.Geometry(union.__geo_interface__)

def export_tile(image, region, out_filename, scale, crs):
    if SKIP_EXISTING and os.path.exists(out_filename):
        print(f"  → Skipping (exists): {os.path.basename(out_filename)}")
        return
    print(f"  → Exporting: {os.path.basename(out_filename)}")
    task = geemap.ee_export_image(
        image, filename=out_filename, scale=scale, region=region,
        crs=crs, file_per_band=False
    )
    if task is None:
        return
    pbar = tqdm(total=100, desc="    export", unit="%", leave=False)
    while True:
        st = task.status().get('state', 'UNKNOWN')
        if st == 'COMPLETED':
            pbar.update(100 - pbar.n); pbar.close()
            break
        if st == 'FAILED':
            pbar.close(); print("    ❌ failed"); break
        if pbar.n < 95: pbar.update(5)
        time.sleep(5)

def export_tiles_for_dataset(composite, aoi, fishnet, out_folder,
                             name, scale, crs):
    os.makedirs(out_folder, exist_ok=True)

    for i, feat in enumerate(tqdm(fishnet, desc=f"Tiles {name}", unit="tile")):
        tile_fname = os.path.join(out_folder, f"{name}_tile_{i}.tif")

        # 1) SKIP check up front:
        if SKIP_EXISTING and os.path.exists(tile_fname):
            continue

        # 2) Now compute the intersection (only when we actually need to export):
        region = ee.Geometry(feat['geometry']).intersection(aoi, 1)
        try:
            if region.area().getInfo() == 0:
                continue
        except Exception:
            continue

        # 3) Export
        export_tile(composite.clip(region), region, tile_fname, scale, crs)

def process_period(start, end):
    period = f"{start.replace('-','')}_{end.replace('-','')}"
    out_root = os.path.join(BASE_OUTPUT, f"period_{period}")
    os.makedirs(out_root, exist_ok=True)
    print(f"\n⏳ Period {start} → {end}")

    aoi = load_geometry_from_shp(SHP_PATH)
    fishnet = geemap.fishnet(aoi,
                             h_interval=H_INTERVAL,
                             v_interval=V_INTERVAL,
                             delta=0).getInfo().get('features', [])
    print(f"  • {len(fishnet)} tiles generated")

    # Sentinel-2
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
          .filterBounds(aoi)
          .filterDate(start, end)
          .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20))
          .select(['B1','B2','B3','B4','B5','B6','B7','B8','B8A','B9','B11','B12'])
          .median().clip(aoi))
    s2_folder = os.path.join(out_root, "Sentinel2_tiles")
    print("Exporting Sentinel-2…")
    export_tiles_for_dataset(s2, aoi, fishnet, s2_folder,
                             "Sentinel2", S2_SCALE, CRS)

    # Sentinel-1
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD")
          .filterBounds(aoi)
          .filterDate(start, end)
          .filter(ee.Filter.eq("instrumentMode","IW"))
          .filter(ee.Filter.Or(
              ee.Filter.listContains('transmitterReceiverPolarisation','VV'),
              ee.Filter.listContains('transmitterReceiverPolarisation','VH'),
             ))
          .select(['VV','VH'])
          .median().clip(aoi))
    s1_folder = os.path.join(out_root, "Sentinel1_tiles")
    print("Exporting Sentinel-1…")
    export_tiles_for_dataset(s1, aoi, fishnet, s1_folder,
                             "Sentinel1", S1_SCALE, CRS)

if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT, exist_ok=True)
    for sd, ed in PERIODS:
        process_period(sd, ed)
