# Project Title: Urban Expansion Mapping
## Group Name: Potniac Bandits 

- **Deepthi Dayanand, University of California, Riverside, Group: Pontiac Bandits, Student ID: 862637806**
- **Ashish Kulkarni, University of California, Riverside, Group: Pontiac Bandits, Student ID: 862637808**
- **Abhijith A Nadig, University of California, Riverside, Group: Pontiac Bandits, Student ID: 862546804**
- **Ananya Sood, University of California, Riverside, Group: Pontiac Bandits, Student ID: 862359197**
- **Ridhima Inukurti, University of California, Riverside, Group: Pontiac Bandits, Student ID: 862355715**

## Author Contributions

- **Build Landsat composites in Google Earth Engine (GEE)** — Ridhima Inukurti  
- **Generate Training Samples in GEE** — Ananya Sood  
- **Train Spark Random Forest Model** — Ridhima Inukurti & Ananya Sood  
- **Apply Model to all Composites and Export Classification Rasters** — Ridhima Inukurti & Ananya Sood  
- **Compute Growth Metrics and Charts** — Deepthi Dayanand, Ashish Kulkarni & Abhijith A Nadig  
- **Build The Interactive Website** — Deepthi Dayanand, Ashish Kulkarni & Abhijith A Nadig

## Repo Description 
This repository contains a full pipeline for urban expansion analysis across four cities and four decades:

- Cities: Riverside, Phoenix, Las Vegas, Austin
- Years: 1990, 2000, 2010, 2020
- Main outputs:
	- Random Forest urban/non-urban classification GeoTIFFs
	- Urban growth metrics and charts
	- Interactive Streamlit dashboard

## Project Pipeline

1. Build Landsat composites in Google Earth Engine (GEE)
2. Generate Training Samples in GEE
3. Train Spark Random Forest Model
4. Apply Model to all Composites and Export Classification Rasters
5. Compute Growth Metrics and Charts
6. Build The Interactive Website

 
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

## Troubleshooting

- `ModuleNotFoundError`: ensure virtual environment is activated and dependencies installed.
- Spark startup issues: verify Java is installed and available on `PATH`.
- Missing input files warnings: confirm expected CSV/TIFF files exist under `data/`.
- Empty dashboard charts: run Member 3 and Member 4 steps first.
