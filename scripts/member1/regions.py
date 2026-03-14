import ee

"""
Member 1 - Study region definitions

This file stores the city study regions as simple bounding boxes.
These are not meant to be perfect city boundaries yet.
They are practical starting regions for exporting Landsat composites.
"""

REGIONS = {
    "riverside": ee.Geometry.Rectangle([-117.55, 33.85, -117.20, 34.10]),
    "phoenix": ee.Geometry.Rectangle([-112.30, 33.25, -111.90, 33.65]),
    "las_vegas": ee.Geometry.Rectangle([-115.35, 35.95, -114.95, 36.35]),
    "austin": ee.Geometry.Rectangle([-97.95, 30.10, -97.55, 30.50]),
}