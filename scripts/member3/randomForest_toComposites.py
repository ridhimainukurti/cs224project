import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml import PipelineModel

import builtins

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_PATH = DATA_DIR / "urban_rf_model"
OUTPUT_DIR = DATA_DIR / "classified"

# composite TIFFs to classify
COMPOSITE_FILES = [
    "riverside_1990_composite.tif",
    "riverside_2000_composite.tif",
    "riverside_2010_composite.tif",
    "riverside_2020_composite.tif",
    "phoenix_1990_composite.tif",
    "phoenix_2000_composite.tif",
    "phoenix_2010_composite.tif",
    "phoenix_2020_composite.tif",
    "austin_1990_composite.tif",
    "austin_2000_composite.tif",
    "austin_2010_composite.tif",
    "austin_2020_composite.tif",
    "las_vegas_1990_composite.tif",
    "las_vegas_2000_composite.tif",
    "las_vegas_2010_composite.tif",
    "las_vegas_2020_composite.tif",
]

# value used for nodata pixels in the output classification raster
OUTPUT_NODATA = 255

# thresholds by city and time period 
THRESHOLDS = {
    "riverside": {
        "legacy": 0.80,
        "2020": 0.63,
    },
    "phoenix": {
        "legacy": 0.80,
        "2020": 0.68,
    },
    "austin": {
        "legacy": 0.35,
        "2020": 0.45,
    },
    "las_vegas": {
        "1990": 0.99,
        "2000": 0.97,
        "2010": 0.95,
        "2020": 0.85,
    },
}

# model paths 
LEGACY_MODEL_PATH = f"{DATA_DIR}/urban_rf_model_legacy"
MODEL_2020_PATH = f"{DATA_DIR}/urban_rf_model_2020"

def parse_city_and_era(tif_name):
    tif_name = tif_name.lower()

    city = None
    for c in ["riverside", "phoenix", "austin", "las_vegas"]:
        if tif_name.startswith(c + "_"):
            city = c
            break

    if city is None:
        raise ValueError(f"Could not determine city from file: {tif_name}")

    if "2020" in tif_name:
        era = "2020"
    elif any(year in tif_name for year in ["1990", "2000", "2010"]):
        era = "legacy"
    else:
        raise ValueError(f"Could not determine era from file: {tif_name}")

    return city, era

def get_model_path_from_filename(tif_name):
    _, era = parse_city_and_era(tif_name)

    if era == "2020":
        return MODEL_2020_PATH
    elif era == "legacy":
        return LEGACY_MODEL_PATH
    else:
        raise ValueError(f"Could not determine model for file: {tif_name}")

def get_threshold_from_filename(tif_name):
    tif_name = tif_name.lower()
    city, era = parse_city_and_era(tif_name)

    if city == "las_vegas":
        if "1990" in tif_name:
            return THRESHOLDS["las_vegas"]["1990"]
        elif "2000" in tif_name:
            return THRESHOLDS["las_vegas"]["2000"]
        elif "2010" in tif_name:
            return THRESHOLDS["las_vegas"]["2010"]
        elif "2020" in tif_name:
            return THRESHOLDS["las_vegas"]["2020"]

    return THRESHOLDS[city][era]

# compute NVDI (formula)
def compute_ndvi(nir, red):
    denom = nir + red
    ndvi = np.zeros_like(denom, dtype=np.float32)

    valid = np.abs(denom) > 1e-6
    ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]
    ndvi = np.clip(ndvi, -1.0, 1.0)

    return ndvi

# print summary statistics for a raster band 
def print_band_stats(name, arr, valid_mask):
    vals = arr[valid_mask]
    if len(vals) == 0:
        print(f"{name}: no valid pixels")
        return

    print(
        f"{name}: min={np.nanmin(vals):.4f}, "
        f"max={np.nanmax(vals):.4f}, "
        f"mean={np.nanmean(vals):.4f}, "
        f"std={np.nanstd(vals):.4f}"
    )

# detect whether reflectance values needs to be rescaled or clipped 
def maybe_rescale_reflectance(arr, valid_mask):
    vals = arr[valid_mask]
    if len(vals) == 0:
        return arr

    max_val = np.nanmax(vals)

    # if values look like 0-10000 scaled reflectance
    if max_val > 100:
        print("Detected very large reflectance values; rescaling by 10000.")
        arr = arr / 10000.0

    # if values are only somewhat above 1, clip them
    elif max_val > 1.0:
        print("Detected reflectance values above 1.0; clipping to [0, 1].")
        arr = np.clip(arr, 0.0, 1.0)

    return arr

# classify one composite (tiff) into urban/non-urban classes 
def classify_composite(tif_name, chunk_size=200000):
    tif_path = f"{DATA_DIR}/{tif_name}"
    output_name = tif_name.replace("_composite.tif", "_classification.tif")
    output_path = f"{output_dir}/{output_name}"

    print(f"\n=== Classifying: {tif_name} ===")

    # load correct model and threshold 
    city, era = parse_city_and_era(tif_name)
    model_path = get_model_path_from_filename(tif_name)
    urban_threshold = get_threshold_from_filename(tif_name)

    print(f"City: {city}")
    print(f"Era: {era}")
    print(f"Using model: {model_path}")
    print(f"Using threshold: {urban_threshold:.2f}")

    model = PipelineModel.load(model_path)

    with rasterio.open(tif_path) as src:
        # assumes band order: red, green, blue, nir
        red = src.read(1, masked=True).astype("float32")
        green = src.read(2, masked=True).astype("float32")
        blue = src.read(3, masked=True).astype("float32")
        nir = src.read(4, masked=True).astype("float32")

        height, width = red.shape

        valid_mask = (
            ~red.mask & ~green.mask & ~blue.mask & ~nir.mask &
            np.isfinite(red.filled(np.nan)) &
            np.isfinite(green.filled(np.nan)) &
            np.isfinite(blue.filled(np.nan)) &
            np.isfinite(nir.filled(np.nan))
        )

        valid_count = int(valid_mask.sum())
        total_count = height * width
        valid_pct = 100.0 * valid_count / total_count

        print(f"Raster shape: {height} x {width}")
        print(f"Valid pixels: {valid_count} / {total_count} ({valid_pct:.2f}%)")

        if valid_count == 0:
            print("No valid pixels. Skipping.")
            return

        red_data = maybe_rescale_reflectance(red.filled(np.nan), valid_mask)
        green_data = maybe_rescale_reflectance(green.filled(np.nan), valid_mask)
        blue_data = maybe_rescale_reflectance(blue.filled(np.nan), valid_mask)
        nir_data = maybe_rescale_reflectance(nir.filled(np.nan), valid_mask)

        # clip negatives
        red_data = np.maximum(red_data, 0.0)
        green_data = np.maximum(green_data, 0.0)
        blue_data = np.maximum(blue_data, 0.0)
        nir_data = np.maximum(nir_data, 0.0)

        print_band_stats("Red", red_data, valid_mask)
        print_band_stats("Green", green_data, valid_mask)
        print_band_stats("Blue", blue_data, valid_mask)
        print_band_stats("NIR", nir_data, valid_mask)

        ndvi_data = compute_ndvi(nir_data, red_data)
        print_band_stats("NDVI", ndvi_data, valid_mask)

        # flatten valid pixels
        valid_idx = np.where(valid_mask.ravel())[0]

        red_vals = red_data.ravel()[valid_idx]
        green_vals = green_data.ravel()[valid_idx]
        blue_vals = blue_data.ravel()[valid_idx]
        nir_vals = nir_data.ravel()[valid_idx]
        ndvi_vals = ndvi_data.ravel()[valid_idx]

        classified_flat = np.full(total_count, OUTPUT_NODATA, dtype=np.uint8)

        schema = StructType([
            StructField("red", DoubleType(), False),
            StructField("green", DoubleType(), False),
            StructField("blue", DoubleType(), False),
            StructField("nir", DoubleType(), False),
            StructField("ndvi", DoubleType(), False),
        ])

        all_probs = []

        # process pixels in chunks to reduce memory pressure 
        num_chunks = (valid_count + chunk_size - 1) // chunk_size
        print(f"Processing in {num_chunks} chunk(s) of up to {chunk_size} pixels each...")

        for chunk_num, start in enumerate(range(0, valid_count, chunk_size), start=1):
            end = builtins.min(start + chunk_size, valid_count)

            rows = [
                (
                    float(r),
                    float(g),
                    float(b),
                    float(n),
                    float(nd)
                )
                for r, g, b, n, nd in zip(
                    red_vals[start:end],
                    green_vals[start:end],
                    blue_vals[start:end],
                    nir_vals[start:end],
                    ndvi_vals[start:end]
                )
            ]

            pixel_df = spark.createDataFrame(rows, schema=schema)

            predictions = model.transform(pixel_df).select("probability")
            prediction_rows = predictions.collect()

            urban_probs = np.array(
                [float(r["probability"][1]) for r in prediction_rows],
                dtype=np.float32
            )
            # covert probabilities to binary class labels using threshold 
            prediction_values = (urban_probs >= urban_threshold).astype(np.uint8)

            classified_flat[valid_idx[start:end]] = prediction_values
            all_probs.append(urban_probs)

            print(
                f"  Chunk {chunk_num}/{num_chunks}: "
                f"pixels {start:,} to {end-1:,} processed"
            )

        all_probs = np.concatenate(all_probs)

        print(
            f"Urban probability stats: "
            f"min={all_probs.min():.4f}, "
            f"max={all_probs.max():.4f}, "
            f"mean={all_probs.mean():.4f}, "
            f"std={all_probs.std():.4f}"
        )
        print(f"Applied urban threshold: {urban_threshold:.2f}")
        # reshape flat classification back into original raster dimensions 
        classified = classified_flat.reshape((height, width))

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=OUTPUT_NODATA,
            compress="lzw"
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(classified, 1)
        # compute and print final class counts 
        urban_pixels = int((classified == 1).sum())
        nonurban_pixels = int((classified == 0).sum())
        nodata_pixels = int((classified == OUTPUT_NODATA).sum())
        nodata_pct = 100.0 * nodata_pixels / total_count
        urban_pct_valid = 100.0 * urban_pixels / valid_count
        nonurban_pct_valid = 100.0 * nonurban_pixels / valid_count

        print(f"Saved: {output_path}")
        print(f"Urban pixels     : {urban_pixels} ({urban_pct_valid:.2f}% of valid)")
        print(f"Non-urban pixels : {nonurban_pixels} ({nonurban_pct_valid:.2f}% of valid)")
        print(f"Nodata pixels    : {nodata_pixels}")
        print(f"Nodata percent   : {nodata_pct:.2f}%")

for tif_name in COMPOSITE_FILES:
    classify_composite(tif_name)