# Urban Expansion Mapping (CS224 Project)

This repository contains a full pipeline for urban expansion analysis across four cities and four decades:

- Cities: Riverside, Phoenix, Las Vegas, Austin
- Years: 1990, 2000, 2010, 2020
- Main outputs:
	- Random Forest urban/non-urban classification GeoTIFFs
	- Urban growth metrics and charts
	- Interactive Streamlit dashboard

## Project Pipeline

1. Build Landsat composites in Google Earth Engine (GEE)
2. Generate training samples in GEE
3. Train Spark Random Forest model
4. Apply model to all composites and export classification rasters
5. Compute growth metrics and charts
6. Explore results in dashboard

## Repository Layout

- `data/`
	- Input composites and training CSVs
	- `classified/` contains predicted classification rasters
	- `charts/` contains generated static plots
- `scripts/member1/`
	- Composite export utilities from GEE
- `scripts/member2/`
	- GEE JavaScript examples for map inspection + training sample export
- `scripts/member3/`
	- Spark model training and composite classification
- `scripts/member4/`
	- Urban growth metrics and analysis charts
- `scripts/member5/`
	- Streamlit dashboard

## Prerequisites

- Python 3.9+
- Java installed (required by PySpark)
- Google Earth Engine account (only needed if regenerating raw composites/samples)

## Setup

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you plan to run GEE export scripts (`scripts/member1`), authenticate first:

```bash
earthengine authenticate
```

## Input Data Requirements

At minimum, the project expects these files in `data/`:

- `riverside_1990_training_samples.csv`
- `riverside_2000_training_samples.csv`
- `riverside_2010_training_samples.csv`
- `riverside_2020_training_samples.csv`
- `{city}_{year}_composite.tif` for all city/year combinations

If these are already present, you can skip the GEE generation steps and go directly to model training.

## Run Everything (Recommended Order)

### 1) (Optional) Quick Spark sanity check

```bash
python spark_test.py
```

### 2) Train Random Forest model

Use the member script version:

```bash
python scripts/member3/randomForest_training.py
```

This writes model artifacts to:

- `data/urban_rf_model`

### 3) Classify all composites

```bash
python scripts/member3/randomForest_toComposites.py
```

This writes outputs to:

- `data/classified/{city}_{year}_classification.tif`

### 4) Generate urban growth metrics and static charts

```bash
python scripts/member4/urban_growth_metrics.py
```

This writes:

- `data/urban_growth_metrics.csv`
- `data/charts/urban_area_over_time.png`
- `data/charts/city_comparison_growth.png`
- `data/charts/urban_area_by_city_year.png`

### 5) Launch dashboard

```bash
streamlit run scripts/member5/dashboard.py
```

## Regenerating Data from GEE (Optional)

Use this only if you need to rebuild composites/samples from scratch.

### Member 1 composites

Single test export:

```bash
python scripts/member1/exportRegionYear.py
```

Export all configured cities/years:

```bash
python scripts/member1/exportAllComposites.py
```

Exports are sent to Google Drive folder `urban_expansion_exports`. Download resulting `.tif` files and place them into `data/`.

### Member 2 training samples (GEE Code Editor)

Run scripts manually in GEE Code Editor:

- `scripts/member2/gee_map.js`
- `scripts/member2/gee_trainingsample.js`

Exported sample CSVs should be downloaded from Drive and copied into `data/`.

## Optional Visualization Check

Overlay one classification on composite image:

```bash
python scripts/member3/randomForest_testing_composites.py
```

## Notes

- There is also a root-level `randomForest_training.py`; prefer `scripts/member3/randomForest_training.py` for the maintained workflow.
- Run all commands from project root so relative paths resolve correctly.

