"""
generate_nodes.py

Turns population_clipped.tif into nodes.csv: village/settlement clusters
with coordinates already converted into your Unreal Landscape's local
coordinate system (since you imported terrain via the Landscape heightmap
tool, Unreal has no built-in notion of lon/lat -- we have to do this
mapping ourselves).

Pipeline:
    population_clipped.tif
        |
        v
    populated pixels (lon, lat, population)
        |
        v
    reproject to meters (UTM 45N) for accurate clustering
        |
        v
    DBSCAN clustering (population-weighted)
        |
        v
    cluster centroids -> back to lon/lat
        |
        v
    lon/lat -> normalized position within DEM extent
        |
        v
    normalized position -> Unreal local X/Y (+ Z sampled from heightmap)
        |
        v
    nodes.csv

Requires:
    pip install rasterio numpy scikit-learn pyproj pillow --break-system-packages
"""

import numpy as np
import rasterio
from rasterio.merge import merge
from sklearn.cluster import DBSCAN
from pyproj import Transformer
from PIL import Image
import csv
import math

# =====================================================================
# CONFIG -- edit these to match your project
# =====================================================================

POP_CLIPPED_PATH = "population_clipped.tif"
LOWER_DEM_PATH = "lower_part.tif"
UPPER_DEM_PATH = "upper_part.tif"
HEIGHTMAP_PNG_PATH = "combined_heightmap.png"   # the 16-bit heightmap you imported
OUT_CSV_PATH = "nodes.csv"

# --- Clustering parameters ---
MIN_POP_PER_PIXEL = 1          # ignore pixels with fewer people than this
CLUSTER_RADIUS_M = 1500        # pixels within this distance (meters) join the same village
MIN_SAMPLES = 3                # minimum weighted "mass" to seed a cluster core (see DBSCAN sample_weight)
USERS_PER_NODE = 50            # how many real people one BP_UserNode actor represents

# This project is about rural/mountain coverage, not dense urban areas like
# Kathmandu Valley. Any cluster with more population than this is treated as
# a city, not a village, and is dropped entirely rather than spawned as nodes.
MAX_CLUSTER_POPULATION = 20000

# --- Landscape transform (READ THESE FROM YOUR LANDSCAPE ACTOR IN UNREAL) ---
# Select your Landscape actor -> Details panel -> Transform.
# These are the DEFAULT values Unreal uses when you don't touch them during import.
# If you changed the scale or moved the landscape after import, update these.
LANDSCAPE_LOCATION_X = 0.0      # Actor Location X (cm)
LANDSCAPE_LOCATION_Y = 0.0      # Actor Location Y (cm)
LANDSCAPE_LOCATION_Z = 0.0      # Actor Location Z (cm)
LANDSCAPE_SCALE_X = 100.0       # Actor Scale X (100 = 1m between heightmap columns)
LANDSCAPE_SCALE_Y = 100.0       # Actor Scale Y (100 = 1m between heightmap rows)
LANDSCAPE_SCALE_Z = 100.0       # Actor Scale Z (affects vertical exaggeration)

NODE_HOVER_HEIGHT_CM = 300.0    # lift nodes slightly above ground so they don't clip into terrain

# Nepal sits in UTM zone 45N
UTM_CRS = "EPSG:32645"
WGS84_CRS = "EPSG:4326"


def get_dem_extent_and_size(lower_path, upper_path):
    """Re-derive the merged DEM's geographic bounds and pixel dimensions,
    exactly matching what combined_heightmap.png represents."""
    srcs = [rasterio.open(lower_path), rasterio.open(upper_path)]
    merged_array, merged_transform = merge(srcs)
    for s in srcs:
        s.close()

    height, width = merged_array.shape[1], merged_array.shape[2]
    left = merged_transform.c
    top = merged_transform.f
    right = left + width * merged_transform.a
    bottom = top + height * merged_transform.e

    return (left, bottom, right, top), (width, height)


def load_population_points(pop_path, min_pop):
    """Read every populated pixel as a (lon, lat, population) point."""
    with rasterio.open(pop_path) as src:
        arr = src.read(1)
        transform = src.transform
        nodata = src.nodata

    rows, cols = np.where((arr != nodata) & (arr >= min_pop))
    pops = arr[rows, cols]

    lons, lats = rasterio.transform.xy(transform, rows, cols)
    lons = np.array(lons)
    lats = np.array(lats)

    return lons, lats, pops


def cluster_population(lons, lats, pops, eps_m, min_samples):
    """Reproject to meters, then run population-weighted DBSCAN."""
    to_utm = Transformer.from_crs(WGS84_CRS, UTM_CRS, always_xy=True)
    xs, ys = to_utm.transform(lons, lats)
    coords = np.column_stack([xs, ys])

    db = DBSCAN(eps=eps_m, min_samples=min_samples)
    labels = db.fit_predict(coords, sample_weight=pops)

    to_wgs84 = Transformer.from_crs(UTM_CRS, WGS84_CRS, always_xy=True)

    clusters = []
    for label in sorted(set(labels)):
        if label == -1:
            continue  # noise, not a real settlement
        mask = labels == label
        cluster_pop = pops[mask].sum()
        # population-weighted centroid
        cx = np.average(xs[mask], weights=pops[mask])
        cy = np.average(ys[mask], weights=pops[mask])
        clon, clat = to_wgs84.transform(cx, cy)
        clusters.append({
            "id": label,
            "lon": clon,
            "lat": clat,
            "population": int(cluster_pop),
            "pixel_count": int(mask.sum()),
        })

    return clusters


def lonlat_to_unreal(clon, clat, dem_bounds, dem_size, heightmap_img):
    """Convert lon/lat -> Unreal local X/Y/Z, matching how the Landscape
    tool laid out combined_heightmap.png (col 0/row 0 = top-left = north-west)."""
    left, bottom, right, top = dem_bounds
    width, height = dem_size

    norm_x = (clon - left) / (right - left)          # 0 (west) -> 1 (east)
    norm_y = (top - clat) / (top - bottom)            # 0 (north) -> 1 (south)

    quads_x = width - 1
    quads_y = height - 1

    local_x = norm_x * quads_x * LANDSCAPE_SCALE_X
    local_y = norm_y * quads_y * LANDSCAPE_SCALE_Y

    world_x = LANDSCAPE_LOCATION_X + local_x
    world_y = LANDSCAPE_LOCATION_Y + local_y

    # sample height from the heightmap PNG at the matching pixel
    px = min(int(norm_x * width), width - 1)
    py = min(int(norm_y * height), height - 1)
    raw_height = heightmap_img[py, px]

    world_z = LANDSCAPE_LOCATION_Z + (float(raw_height) - 32768.0) * LANDSCAPE_SCALE_Z / 128.0
    world_z += NODE_HOVER_HEIGHT_CM

    return world_x, world_y, world_z


def main():
    print("1) Deriving DEM extent from lower_part.tif + upper_part.tif ...")
    dem_bounds, dem_size = get_dem_extent_and_size(LOWER_DEM_PATH, UPPER_DEM_PATH)
    print(f"   bounds={dem_bounds}, size={dem_size}")

    print("\n2) Loading heightmap PNG for Z sampling ...")
    heightmap_img = np.array(Image.open(HEIGHTMAP_PNG_PATH))
    print(f"   heightmap shape: {heightmap_img.shape}")

    print("\n3) Reading populated pixels from clipped population raster ...")
    lons, lats, pops = load_population_points(POP_CLIPPED_PATH, MIN_POP_PER_PIXEL)
    print(f"   {len(pops)} populated pixels found, total population ~{pops.sum():,.0f}")

    print(f"\n4) Clustering (eps={CLUSTER_RADIUS_M}m, min_samples={MIN_SAMPLES}) ...")
    clusters = cluster_population(lons, lats, pops, CLUSTER_RADIUS_M, MIN_SAMPLES)
    print(f"   {len(clusters)} settlement clusters found")

    urban = [c for c in clusters if c["population"] > MAX_CLUSTER_POPULATION]
    rural = [c for c in clusters if c["population"] <= MAX_CLUSTER_POPULATION]
    if urban:
        print(f"   Dropping {len(urban)} urban cluster(s) above {MAX_CLUSTER_POPULATION:,} people "
              f"(e.g. Kathmandu Valley) -> not relevant to rural coverage study:")
        for c in urban:
            print(f"     - cluster {c['id']}: {c['population']:,} people at "
                  f"({c['lat']:.4f}, {c['lon']:.4f})")
    clusters = rural
    print(f"   {len(clusters)} rural/village clusters remain")

    print("\n5) Converting cluster centroids to Unreal coordinates + writing CSV ...")
    with open(OUT_CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cluster_id", "lon", "lat", "population",
            "unreal_x", "unreal_y", "unreal_z", "num_user_nodes"
        ])
        for c in clusters:
            wx, wy, wz = lonlat_to_unreal(c["lon"], c["lat"], dem_bounds, dem_size, heightmap_img)
            num_nodes = max(1, math.ceil(c["population"] / USERS_PER_NODE))
            writer.writerow([
                c["id"], f"{c['lon']:.6f}", f"{c['lat']:.6f}", c["population"],
                f"{wx:.2f}", f"{wy:.2f}", f"{wz:.2f}", num_nodes
            ])

    print(f"\nDone. Saved -> {OUT_CSV_PATH}")


if __name__ == "__main__":
    main()