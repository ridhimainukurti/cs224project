from landsatHelpers import (
    initialize_earth_engine,
    get_composite,
    get_image_count,
    create_export_task,
)

"""
Member 1 - Test export script

This script exports one test composite:
Riverside, 1990

Use this first to verify:
- Earth Engine is working
- the region is valid
- the export starts correctly
- the band names are correct
"""

PROJECT_ID = "cs224project-490217"
CITY_NAME = "riverside"
YEAR = 1990


def main():
    # Initialize Earth Engine.
    # Set authenticate=True only if you need to log in again.
    initialize_earth_engine(PROJECT_ID, authenticate=False)

    from regions import REGIONS

    region = REGIONS[CITY_NAME]

    # Quick sanity check: how many scenes are being pulled?
    image_count = get_image_count(region, YEAR)
    print(f"Image count for {CITY_NAME} {YEAR}: {image_count}")

    # Build the composite image.
    composite = get_composite(region, YEAR)

    # Confirm output band names.
    print("Band names:", composite.bandNames().getInfo())

    # Create and start the export task.
    task = create_export_task(composite, region, CITY_NAME, YEAR)
    task.start()

    print(f"Export task started for {CITY_NAME}_{YEAR}_composite")
    print(task.status())
    print("Check Google Drive folder: urban_expansion_exports")


if __name__ == "__main__":
    main()