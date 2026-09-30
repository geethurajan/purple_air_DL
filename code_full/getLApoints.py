import json

with open("City_Boundary.geojson") as f:
    city_geojson = json.load(f)

if city_geojson.get("type") == "FeatureCollection":
    geometry = city_geojson["features"][0]["geometry"]
elif city_geojson.get("type") == "Feature":
    geometry = city_geojson["geometry"]
else:
    geometry = city_geojson

# Flatten every coordinate, regardless of Polygon vs MultiPolygon nesting
all_lons, all_lats = [], []
def collect_coords(coords):
    if isinstance(coords[0][0], (int, float)):  # this level is a ring of [lon, lat] pairs
        for lon, lat in coords:
            all_lons.append(lon)
            all_lats.append(lat)
    else:
        for sub in coords:
            collect_coords(sub)

collect_coords(geometry["coordinates"])

print(f"Northernmost (max lat): {max(all_lats):.5f}")
print(f"Southernmost (min lat): {min(all_lats):.5f}")
print(f"Easternmost (max lon):  {max(all_lons):.5f}")
print(f"Westernmost (min lon):  {min(all_lons):.5f}")
