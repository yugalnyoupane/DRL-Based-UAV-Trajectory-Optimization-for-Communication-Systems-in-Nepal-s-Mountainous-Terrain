"""
clip_population.py

Clips the full-Nepal WorldPop raster down to the exact geographic extent
covered by your terrain (lower_part.tif + upper_part.tif merged).

Why this approach:
- lower_part.tif and upper_part.tif are two 1x1 degree SRTM tiles that,
  once merged, form the area you rendered as combined_heightmap.png.
- Rather than hardcoding the bounding box, this script re-derives it
  directly from the two DEM tiles, so if you ever swap in different
  tiles (e.g. for a different region), you don't need to touch any
  numbers by hand.

Output:
- population_clipped.tif   -> population raster clipped to the DEM extent
                               (still in original 100m resolution, still a
                               real GeoTIFF you can open in QGIS)

Requires: rasterio, numpy
    pip install rasterio numpy --break-system-packages
"""

import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box
import numpy as np

# ---- INPUT PATHS (edit if your filenames differ) ----
LOWER_DEM_PATH = "lower_part.tif"
UPPER_DEM_PATH = "upper_part.tif"
POP_PATH = "npl_pop_2025_CN_100m_R2025A_v1.tif"

# ---- OUTPUT PATHS ----
POP_CLIPPED_PATH = "population_clipped.tif"


def get_dem_extent(lower_path, upper_path):
    """Merge the two DEM tiles just to read off their combined bounding box.
    We don't keep the merged array here -- Unreal already has your
    combined_heightmap.png for that. We only need the extent (in degrees)
    so we know exactly what area to clip the population data to."""
    srcs = [rasterio.open(lower_path), rasterio.open(upper_path)]
    merged_array, merged_transform = merge(srcs)
    for s in srcs:
        s.close()

    # bounding box of the merged tiles
    height, width = merged_array.shape[1], merged_array.shape[2]
    left = merged_transform.c
    top = merged_transform.f
    right = left + width * merged_transform.a
    bottom = top + height * merged_transform.e  # e is negative

    return left, bottom, right, top


def clip_population(pop_path, bounds, out_path):
    left, bottom, right, top = bounds
    aoi_geom = [box(left, bottom, right, top)]

    with rasterio.open(pop_path) as src:
        src_crs = src.crs
        clipped_array, clipped_transform = mask(
            src, aoi_geom, crop=True, nodata=src.nodata
        )
        out_meta = src.meta.copy()

    out_meta.update({
        "driver": "GTiff",
        "height": clipped_array.shape[1],
        "width": clipped_array.shape[2],
        "transform": clipped_transform,
        "crs": src_crs,
    })

    with rasterio.open(out_path, "w", **out_meta) as dst:
        dst.write(clipped_array)

    return clipped_array, clipped_transform


def main():
    print("Reading DEM extent from lower_part.tif + upper_part.tif ...")
    bounds = get_dem_extent(LOWER_DEM_PATH, UPPER_DEM_PATH)
    print(f"  DEM extent (lon/lat): left={bounds[0]:.6f}, bottom={bounds[1]:.6f}, "
          f"right={bounds[2]:.6f}, top={bounds[3]:.6f}")

    print(f"\nClipping {POP_PATH} to that extent ...")
    clipped_array, transform = clip_population(POP_PATH, bounds, POP_CLIPPED_PATH)

    valid = clipped_array[clipped_array != -99999.0]
    print(f"  Clipped raster shape: {clipped_array.shape}")
    print(f"  Valid pixels: {valid.size}")
    print(f"  Estimated total population in region: {np.nansum(valid):,.0f}")
    print(f"\nSaved -> {POP_CLIPPED_PATH}")


if __name__ == "__main__":
    main()