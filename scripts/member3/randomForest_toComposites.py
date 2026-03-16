import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml import PipelineModel

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
URBAN_THRESHOLD = 0.25

# these are model paths 
# seperate models for different landsat data
LEGACY_MODEL_PATH = f"{DATA_DIR}/urban_rf_model_legacy"
MODEL_2020_PATH = f"{DATA_DIR}/urban_rf_model_2020"

def get_model_path_from_filename(tif_name):
    tif_name = tif_name.lower()

    if "2020" in tif_name:
        return MODEL_2020_PATH
    elif any(year in tif_name for year in ["1990", "2000", "2010"]):
        return LEGACY_MODEL_PATH
    else:
        raise ValueError(f"Could not determine model for file: {tif_name}")

# compute NVDI (formula)
def compute_ndvi(nir, red):
    denom = nir + red
    ndvi = np.zeros_like(denom, dtype=np.float32)

    # only compute NDVI where denominator is safely away from zero
    valid = np.abs(denom) > 1e-6
    ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]

    # clip to valid NDVI range
    ndvi = np.clip(ndvi, -1.0, 1.0)

    return ndvi

# helps debug band values before classification 
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

# rescale reflectance if tiff values look to large 
def maybe_rescale_reflectance(arr, valid_mask):
    vals = arr[valid_mask]
    if len(vals) == 0:
        return arr

    # crude safeguard in case a TIFF is scaled 0-10000 instead of ~0-1
    if np.nanmax(vals) > 2.0:
        print("Detected large reflectance values; rescaling by 10000.")
        return arr / 10000.0

    return arr

# classify one compositve tf 
# laods the correct model for the year, reads the 4-band, computes NVDI, predicts urban proability, writes classification
def classify_composite(tif_name):
    tif_path = f"{DATA_DIR}/{tif_name}"
    output_name = tif_name.replace("_composite.tif", "_classification.tif")
    output_path = f"{output_dir}/{output_name}"

    print(f"\n=== Classifying: {tif_name} ===")

    # load correct model for this file 
    model_path = get_model_path_from_filename(tif_name)
    print(f"Using model: {model_path}")
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

        red_data = np.maximum(red_data, 0.0)
        green_data = np.maximum(green_data, 0.0)
        blue_data = np.maximum(blue_data, 0.0)
        nir_data = np.maximum(nir_data, 0.0)

        print_band_stats("Red", red_data, valid_mask)
        print_band_stats("Green", green_data, valid_mask)
        print_band_stats("Blue", blue_data, valid_mask)
        print_band_stats("NIR", nir_data, valid_mask)

        # compute NVDI and print its stats 
        ndvi_data = compute_ndvi(nir_data, red_data)
        print_band_stats("NDVI", ndvi_data, valid_mask)

        # extract only valid pixel values from each vand 
        red_vals = red_data[valid_mask]
        green_vals = green_data[valid_mask]
        blue_vals = blue_data[valid_mask]
        nir_vals = nir_data[valid_mask]
        ndvi_vals = ndvi_data[valid_mask]

        rows = [
            (
                float(r),
                float(g),
                float(b),
                float(n),
                float(nd)
            )
            for r, g, b, n, nd in zip(red_vals, green_vals, blue_vals, nir_vals, ndvi_vals)
        ]

        schema = StructType([
            StructField("red", DoubleType(), False),
            StructField("green", DoubleType(), False),
            StructField("blue", DoubleType(), False),
            StructField("nir", DoubleType(), False),
            StructField("ndvi", DoubleType(), False),
        ])

        pixel_df = spark.createDataFrame(rows, schema=schema)

        # run model prediction 
        # use probabilities instead of default hard prediction
        predictions = model.transform(pixel_df).select("probability")
        prediction_rows = predictions.collect()

        urban_probs = np.array(
            [float(r["probability"][1]) for r in prediction_rows],
            dtype=np.float32
        )

        prediction_values = (urban_probs >= URBAN_THRESHOLD).astype(np.uint8)

        print(
            f"Urban probability stats: "
            f"min={urban_probs.min():.4f}, "
            f"max={urban_probs.max():.4f}, "
            f"mean={urban_probs.mean():.4f}, "
            f"std={urban_probs.std():.4f}"
        )
        print(f"Using urban threshold: {URBAN_THRESHOLD}")

        # start output raster + write predictions 
        classified = np.full((height, width), OUTPUT_NODATA, dtype=np.uint8)
        classified[valid_mask] = prediction_values

        profile = src.profile.copy()
        profile.update(
            count=1,
            dtype=rasterio.uint8,
            nodata=OUTPUT_NODATA,
            compress="lzw"
        )
        # save the classified raster to disk 
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(classified, 1)

        urban_pixels = int((classified == 1).sum())
        nonurban_pixels = int((classified == 0).sum())
        nodata_pixels = int((classified == OUTPUT_NODATA).sum())
        nodata_pct = 100.0 * nodata_pixels / total_count

        print(f"Saved: {output_path}")
        print(f"Urban pixels     : {urban_pixels}")
        print(f"Non-urban pixels : {nonurban_pixels}")
        print(f"Nodata pixels    : {nodata_pixels}")
        print(f"Nodata percent   : {nodata_pct:.2f}%")

# classify all composite files 
for tif_name in COMPOSITE_FILES:
    classify_composite(tif_name)