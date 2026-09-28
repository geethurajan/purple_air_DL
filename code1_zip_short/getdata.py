import json
import struct
import zipfile
import numpy as np
import pandas as pd

# =============================================================================
# EDIT THESE PATHS
# =============================================================================
POP_JSON_PATH = "la_tract_population.json"
TRACT_SHAPEFILE_ZIP_PATH = "tl_2023_06_tract.zip"
CITY_BOUNDARY_GEOJSON_PATH = "City_Boundary.geojson"

EPA_LAT = 34.06659
EPA_LON = -118.22688

# =============================================================================
# 1. LOAD POPULATION
# =============================================================================

with open(POP_JSON_PATH) as f:
    raw = json.load(f)

pop_df = pd.DataFrame(raw[1:], columns=raw[0])
pop_df = pop_df.rename(columns={"B01003_001E": "population"})
pop_df["population"] = pop_df["population"].astype(int)
pop_df["GEOID"] = pop_df["state"] + pop_df["county"] + pop_df["tract"]
print(f"Loaded population for {len(pop_df)} tracts (all of LA County).")

# =============================================================================
# 2. HAND-WRITTEN .dbf READER (attribute table: GEOID, NAME, COUNTYFP, etc.)
# =============================================================================

def read_dbf(dbf_bytes):
    num_records = struct.unpack('<I', dbf_bytes[4:8])[0]
    header_size = struct.unpack('<H', dbf_bytes[8:10])[0]
    record_size = struct.unpack('<H', dbf_bytes[10:12])[0]

    fields = []
    offset = 32
    while dbf_bytes[offset] != 0x0D:  # 0x0D marks end of field descriptors
        desc = dbf_bytes[offset:offset + 32]
        name = desc[0:11].split(b'\x00')[0].decode('ascii', errors='replace')
        length = desc[16]
        fields.append((name, length))
        offset += 32

    records = []
    pos = header_size
    for _ in range(num_records):
        rec = dbf_bytes[pos:pos + record_size]
        pos += record_size
        if rec[0:1] == b'*':  # deleted record marker
            continue
        values, field_pos = {}, 1  # skip deletion-flag byte
        for name, length in fields:
            values[name] = rec[field_pos:field_pos + length].decode('ascii', errors='replace').strip()
            field_pos += length
        records.append(values)
    return records


# =============================================================================
# 3. HAND-WRITTEN .shp READER (polygon geometry -- parts + point coordinates)
# =============================================================================

def read_shp(shp_bytes):
    shapes = []
    offset = 100  # fixed-size file header
    n = len(shp_bytes)
    while offset < n:
        content_len_words = struct.unpack('>i', shp_bytes[offset + 4:offset + 8])[0]
        offset += 8
        content = shp_bytes[offset:offset + content_len_words * 2]
        offset += content_len_words * 2

        shape_type = struct.unpack('<i', content[0:4])[0]
        if shape_type == 0:  # null shape
            shapes.append({'parts': [0], 'points': []})
            continue

        num_parts = struct.unpack('<i', content[36:40])[0]
        num_points = struct.unpack('<i', content[40:44])[0]
        parts = list(struct.unpack(f'<{num_parts}i', content[44:44 + 4 * num_parts]))
        pts_offset = 44 + 4 * num_parts
        points = [struct.unpack('<dd', content[pts_offset + i * 16: pts_offset + i * 16 + 16])
                  for i in range(num_points)]
        shapes.append({'parts': parts, 'points': points})
    return shapes


# =============================================================================
# 4. EXTRACT .shp/.dbf FROM THE ZIP (read straight from memory, no disk unzip)
# =============================================================================

with zipfile.ZipFile(TRACT_SHAPEFILE_ZIP_PATH) as z:
    shp_name = next(n for n in z.namelist() if n.endswith('.shp'))
    dbf_name = next(n for n in z.namelist() if n.endswith('.dbf'))
    shp_bytes = z.read(shp_name)
    dbf_bytes = z.read(dbf_name)

shapes = read_shp(shp_bytes)
attrs_list = read_dbf(dbf_bytes)
print(f"Loaded {len(shapes)} shapes / {len(attrs_list)} attribute records from the shapefile.")

# =============================================================================
# 5. FILTER TO LA COUNTY, COMPUTE CENTROIDS (shoelace formula)
# =============================================================================

records = []
for shape, attrs in zip(shapes, attrs_list):
    if attrs.get("COUNTYFP") != "037" or not shape["points"]:
        continue

    parts = list(shape["parts"]) + [len(shape["points"])]
    rings = [shape["points"][parts[i]:parts[i + 1]] for i in range(len(parts) - 1)]
    largest_ring = max(rings, key=len)

    x = np.array([p[0] for p in largest_ring])
    y = np.array([p[1] for p in largest_ring])
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y2 - x2 * y
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        cx, cy = x.mean(), y.mean()
    else:
        cx = ((x + x2) * cross).sum() / (6 * area)
        cy = ((y + y2) * cross).sum() / (6 * area)

    records.append(dict(GEOID=attrs.get("GEOID"), NAME=attrs.get("NAME"),
                         centroid_lon=cx, centroid_lat=cy))

tracts_df = pd.DataFrame(records)
print(f"Kept {len(tracts_df)} tracts in LA County.")

# =============================================================================
# 6. JOIN POPULATION
# =============================================================================

tracts_df = tracts_df.merge(pop_df[["GEOID", "population"]], on="GEOID", how="left")
missing_pop = tracts_df["population"].isna().sum()
if missing_pop:
    print(f"WARNING: {missing_pop} tracts have geometry but no matched population "
          f"-- check GEOID formats line up between the two files.")

# =============================================================================
# 7. POINT-IN-POLYGON TEST AGAINST THE LA CITY BOUNDARY (ray casting, no libraries)
# =============================================================================

def point_in_ring(x, y, ring):
    inside = False
    n = len(ring)
    j = n - 1
    for i in range(n):
        xi, yi = ring[i]
        xj, yj = ring[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-15) + xi):
            inside = not inside
        j = i
    return inside


def point_in_polygon_with_holes(x, y, rings):
    if not point_in_ring(x, y, rings[0]):
        return False
    return not any(point_in_ring(x, y, hole) for hole in rings[1:])


def point_in_city(x, y, geometry):
    gtype, coords = geometry["type"], geometry["coordinates"]
    if gtype == "Polygon":
        return point_in_polygon_with_holes(x, y, coords)
    elif gtype == "MultiPolygon":
        return any(point_in_polygon_with_holes(x, y, poly) for poly in coords)
    raise ValueError(f"Unexpected geometry type: {gtype}")


with open(CITY_BOUNDARY_GEOJSON_PATH) as f:
    city_geojson = json.load(f)

if city_geojson.get("type") == "FeatureCollection":
    city_geometry = city_geojson["features"][0]["geometry"]
elif city_geojson.get("type") == "Feature":
    city_geometry = city_geojson["geometry"]
else:
    city_geometry = city_geojson

tracts_df["in_la_city"] = tracts_df.apply(
    lambda r: point_in_city(r["centroid_lon"], r["centroid_lat"], city_geometry), axis=1
)
la_city_tracts = tracts_df[tracts_df["in_la_city"]].copy().reset_index(drop=True)
print(f"{len(la_city_tracts)} of {len(tracts_df)} LA County tracts have their "
      f"centroid inside the LA city boundary.")

# =============================================================================
# 8. LOCAL X/Y KM COORDINATES
# =============================================================================

def latlon_to_local_km(lat, lon, ref_lat, ref_lon):
    R = 6371.0
    lat_rad, ref_lat_rad = np.radians(lat), np.radians(ref_lat)
    dlat = np.radians(np.asarray(lat) - ref_lat)
    dlon = np.radians(np.asarray(lon) - ref_lon)
    x = dlon * np.cos((lat_rad + ref_lat_rad) / 2) * R
    y = dlat * R
    return x, y

la_city_tracts["x_km"], la_city_tracts["y_km"] = latlon_to_local_km(
    la_city_tracts["centroid_lat"], la_city_tracts["centroid_lon"], EPA_LAT, EPA_LON
)

# =============================================================================
# 9. SAVE OUTPUT
# =============================================================================

final_cols = ["GEOID", "NAME", "population", "centroid_lat", "centroid_lon", "x_km", "y_km"]
la_city_tracts[final_cols].to_csv("la_city_tracts_final.csv", index=False)
print("Saved la_city_tracts_final.csv -- ready to feed into the PM2.5/mortality pipeline.")