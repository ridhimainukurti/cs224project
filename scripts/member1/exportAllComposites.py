from landsatHelpers import (
    initialize_earth_engine,
    get_composite,
    get_image_count,
    create_export_task,
)

"""
Member 1 - Export all city-year composites

This script loops through:
- 4 cities
- 4 years

and starts an export task for each composite.

Expected outputs:
riverside_1990_composite.tif
riverside_2000_composite.tif
riverside_2010_composite.tif
riverside_2020_composite.tif
phoenix_1990_composite.tif
...
"""

PROJECT_ID = "cs224project-490217"
YEARS = [1990, 2000, 2010, 2020]


def main():
    # Initialize Earth Engine once.
    initialize_earth_engine(PROJECT_ID, authenticate=False)

    from regions import REGIONS

    for city_name, region in REGIONS.items():
        for year in YEARS:
            print("=" * 60)
            print(f"Preparing export for {city_name} {year}")

            # Count scenes for sanity checking.
            image_count = get_image_count(region, year)
            print(f"Image count: {image_count}")

            if image_count == 0:
                print(f"Skipping {city_name} {year}: no images found")
                continue

            # Build composite image.
            composite = get_composite(region, year)

            # Confirm output bands.
            bands = composite.bandNames().getInfo()
            print(f"Bands: {bands}")

            # Create and start export task.
            task = create_export_task(composite, region, city_name, year)
            task.start()

            print(f"Started export task: {city_name}_{year}_composite")

    print("=" * 60)
    print("All export tasks have been started.")
    print("Check Google Drive folder: urban_expansion_exports")


if __name__ == "__main__":
    main()