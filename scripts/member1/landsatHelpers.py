import ee

"""
Member 1 - Reusable Landsat helper functions

This file contains all reusable logic for:
- Earth Engine initialization
- Landsat scaling
- cloud masking
- choosing the right Landsat collection by year
- renaming bands into a consistent output format
- creating city-year composites
- creating export tasks
"""


def initialize_earth_engine(project_id: str, authenticate: bool = False) -> None:
    """
    Initialize Earth Engine.

    Parameters
    ----------
    project_id : str
        Google Cloud project ID registered for Earth Engine.
    authenticate : bool
        Set to True only if you need to authenticate again.
    """
    if authenticate:
        ee.Authenticate()

    ee.Initialize(project=project_id)


def apply_scale_factors(image: ee.Image) -> ee.Image:
    """
    Apply Collection 2 Level 2 optical reflectance scale factors.

    Landsat Collection 2 Level 2 surface reflectance uses a scale factor
    and offset for the SR_B* optical bands.
    """
    optical = image.select("SR_B.*").multiply(2.75e-05).add(-0.2)
    return image.addBands(optical, overwrite=True)


def mask_landsat_clouds(image: ee.Image) -> ee.Image:
    """
    Mask clouds and cloud shadows using the QA_PIXEL band.

    This is a simple cloud mask suitable for a first project version.
    """
    qa = image.select("QA_PIXEL")

    cloud_shadow_bit = 1 << 4
    cloud_bit = 1 << 3

    mask = qa.bitwiseAnd(cloud_shadow_bit).eq(0).And(
        qa.bitwiseAnd(cloud_bit).eq(0)
    )

    return image.updateMask(mask)


def get_collection_id(year: int) -> str:
    """
    Choose the correct Landsat collection based on target year.

    Year mapping:
    - 1990, 2000 -> Landsat 5
    - 2010 -> Landsat 7
    - 2020 -> Landsat 8
    """
    if year in [1990, 2000]:
        return "LANDSAT/LT05/C02/T1_L2"

    if year == 2010:
        return "LANDSAT/LE07/C02/T1_L2"

    if year == 2020:
        return "LANDSAT/LC08/C02/T1_L2"

    raise ValueError(
        f"Unsupported year: {year}. Use one of 1990, 2000, 2010, 2020."
    )


def get_date_window(year: int) -> tuple[str, str]:
    """
    Use a 3-year window centered on the target year.

    Example:
    1990 -> 1989-01-01 to 1991-12-31
    """
    start = f"{year - 1}-01-01"
    end = f"{year + 1}-12-31"
    return start, end


def rename_bands(image: ee.Image, year: int) -> ee.Image:
    """
    Rename Landsat bands into a common feature schema:
    red, green, blue, nir

    Landsat 5 and Landsat 7:
    - SR_B1 = blue
    - SR_B2 = green
    - SR_B3 = red
    - SR_B4 = nir

    Landsat 8:
    - SR_B2 = blue
    - SR_B3 = green
    - SR_B4 = red
    - SR_B5 = nir
    """
    if year in [1990, 2000, 2010]:
        return image.select(
            ["SR_B3", "SR_B2", "SR_B1", "SR_B4"],
            ["red", "green", "blue", "nir"],
        )

    if year == 2020:
        return image.select(
            ["SR_B4", "SR_B3", "SR_B2", "SR_B5"],
            ["red", "green", "blue", "nir"],
        )

    raise ValueError(f"Unsupported year for renaming: {year}")


def get_composite(region: ee.Geometry, year: int) -> ee.Image:
    """
    Build a median Landsat composite for a given region and year.

    Steps:
    1. choose the collection
    2. filter by region
    3. filter by date window
    4. scale reflectance values
    5. mask clouds
    6. create median composite
    7. rename bands to red, green, blue, nir
    """
    collection_id = get_collection_id(year)
    start_date, end_date = get_date_window(year)

    composite = (
        ee.ImageCollection(collection_id)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .map(apply_scale_factors)
        .map(mask_landsat_clouds)
        .median()
    )

    composite = rename_bands(composite, year)

    return composite


def get_image_count(region: ee.Geometry, year: int) -> int:
    """
    Count how many images are available before median compositing.
    Useful as a quick sanity check.
    """
    collection_id = get_collection_id(year)
    start_date, end_date = get_date_window(year)

    count = (
        ee.ImageCollection(collection_id)
        .filterBounds(region)
        .filterDate(start_date, end_date)
        .size()
    )

    return count.getInfo()


def create_export_task(
    image: ee.Image,
    region: ee.Geometry,
    city_name: str,
    year: int,
    folder: str = "urban_expansion_exports",
) -> ee.batch.Task:
    """
    Create a Google Drive export task for a composite image.
    """
    file_prefix = f"{city_name}_{year}_composite"

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=file_prefix,
        folder=folder,
        fileNamePrefix=file_prefix,
        region=region,
        scale=30,
        crs="EPSG:4326",
        maxPixels=1e13,
        fileFormat="GeoTIFF",
    )

    return task