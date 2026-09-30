import pandas as pd

input_file = "hourly_88101_2025.csv"

# Change this number each time you run the script
site_num_to_keep = 30

output_file = f"LA_Site_{site_num_to_keep}.csv"

# Get the name of column 17 (the 18th column)
column_names = pd.read_csv(input_file, nrows=0).columns
mixed_column_name = column_names[17]

first_write = True

for chunk in pd.read_csv(
    input_file,
    chunksize=1_000_000,
    dtype={
        "State Code": "Int64",
        "Site Num": "Int64",
        mixed_column_name: "string",
    },
):
    matching_rows = chunk[
        (chunk["State Code"] == 6)
        & (chunk["Site Num"] == site_num_to_keep)
    ]

    if not matching_rows.empty:
        matching_rows.to_csv(
            output_file,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False

print("Done.")