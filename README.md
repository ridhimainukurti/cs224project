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
- **Build The Interactive Website** — Deepthi Dayanand, Ashish Kulkarni, Abhijith A Nadig, Ananya Sood & Ridhima Inukurti

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


## Setup

From project root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Input Data Requirements

At minimum, the project expects these files in `data/`:

- `riverside_1990_training_samples.csv`
- `riverside_2000_training_samples.csv`
- `riverside_2010_training_samples.csv`
- `riverside_2020_training_samples.csv`
- `{city}_{year}_composite.tif` for all city/year combinations


## Run Everything (Recommended Order)

### Launch interactive website 

```bash
streamlit run scripts/member5/dashboard.py
```
This will launch the interactive website which leads to the visualizations of urban expansion. Please use the localhost link provided in the terminal to access the interface. 


### (Optional) Train Random Forest model

Use the member script version:

```bash
python scripts/member3/randomForest_training.py
```

This writes model artifacts to:

- `data/urban_rf_model`

### (Optional) Classify all composites

```bash
python scripts/member3/randomForest_toComposites.py
```

This writes outputs to:

- `data/classified/{city}_{year}_classification.tif`

### (Optional) Generate urban growth metrics and static charts

```bash
python scripts/member4/urban_growth_metrics.py
```

This writes:

- `data/urban_growth_metrics.csv`
- `data/charts/urban_area_over_time.png`
- `data/charts/city_comparison_growth.png`
- `data/charts/urban_area_by_city_year.png`


### (Optional) Visualization Check

Overlay one classification on composite image:

```bash
python scripts/member3/randomForest_testing_composites.py
```

## Notes

- Run all commands from project root so relative paths resolve correctly.

## Troubleshooting

- `ModuleNotFoundError`: ensure virtual environment is activated and dependencies installed.
- Spark startup issues: verify Java is installed and available on `PATH`.
- Missing input files warnings: confirm expected CSV/TIFF files exist under `data/`.
- Empty dashboard charts: run Member 3 and Member 4 steps first.
