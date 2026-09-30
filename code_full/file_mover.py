import os
import shutil

# Directory containing this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

for folder_name in os.listdir(BASE_DIR):
    # Only process PurpleAir download folders
    if not folder_name.startswith("PurpleAir Download Sensor"):
        continue

    folder_path = os.path.join(BASE_DIR, folder_name)

    if not os.path.isdir(folder_path):
        continue

    # Find the CSV file in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]

    if len(csv_files) != 1:
        print(f"Skipping '{folder_name}' (found {len(csv_files)} CSV files)")
        continue

    csv_name = csv_files[0]
    src = os.path.join(folder_path, csv_name)
    dst = os.path.join(BASE_DIR, csv_name)

    shutil.move(src, dst)
    os.rmdir(folder_path)

    print(f"Moved {csv_name}")

print("\nDone.")