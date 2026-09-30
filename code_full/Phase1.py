import pandas as pd
import numpy as np
import os
import glob
import logging

# Configure logging to display real-time progress metrics in your terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_epa_sensor(file_path, date_col1='Date Local', date_col2='Time Local', pm_col='Sample Measurement'):
    """
    Ingests and cleans the individual EPA CSV file. 
    Combines independent local date/time fields, enforces data thresholds, 
    and applies temporal interpolation.
    """
    logging.info(f"Starting EPA processing on file: {file_path}")
    if not os.path.exists(file_path):
        logging.error(f"EPA file not found at: {file_path}")
        return pd.DataFrame()
        
    df = pd.read_csv(file_path)
    
    # 1. Feature Engineering: Combine local date and time strings into a datetime index
    df['date'] = pd.to_datetime(df[date_col1].astype(str) + ' ' + df[date_col2].astype(str))
    df = df.rename(columns={pm_col: 'pm25'})
    
    # Extract site coordinates dynamically from the dataset rows
    lat = df['Latitude'].iloc[0] if 'Latitude' in df.columns else 34.0522 
    lon = df['Longitude'].iloc[0] if 'Longitude' in df.columns else -118.2437
    
    # 2. Quality Control: 120-hour completeness filter
    valid_hours = df['pm25'].count()
    if valid_hours < 120:
        logging.warning(f"EPA Sensor EXCLUDED: Contains only {valid_hours} valid hours (Requires 120+).")
        return pd.DataFrame()
        
    # 3. Temporal Imputation: Close mini-gaps using linear interpolation
    df = df.sort_values('date')
    df['pm25'] = df['pm25'].interpolate(method='linear').bfill().ffill()
    
    # Standardize schema output
    df['station_id'] = 'EPA_1103'
    df['latitude'] = lat
    df['longitude'] = lon
    
    logging.info(f"EPA Sensor EPA_1103 successfully retained ({valid_hours} valid hours).")
    return df[['date', 'station_id', 'latitude', 'longitude', 'pm25']]


def process_purple_air_sensors(folder_path, metadata_path, time_col='time', pm_col='pm2.5_atm'):
    """
    Scans a directory for all PurpleAir historical tracking files, matches them 
    against a coordinate metadata file, cleans gaps, and enforces the 120-hour check.
    """
    logging.info(f"Starting PurpleAir directory scan within: {folder_path}")
    
    # Early trap for missing master metadata coordinate file
    if not os.path.exists(metadata_path):
        logging.error(f"Metadata coordinate mapping file not found at: {metadata_path}")
        return pd.DataFrame(), [] 
        
    meta_df = pd.read_csv(metadata_path)
    meta_df['sensor_id'] = meta_df['sensor_id'].astype(str).str.strip()
    meta_dict = meta_df.set_index('sensor_id')[['latitude', 'longitude']].to_dict(orient='index')
    
    # Collect all historical file profiles in the specified folder
    file_pattern = os.path.join(folder_path, "sensor_*_history.csv")
    pa_files = glob.glob(file_pattern)
    
    if not pa_files:
        logging.warning(f"No files matching 'sensor_*_history.csv' found in folder: {folder_path}")
        return pd.DataFrame(), []
        
    all_pa_data = []
    included_sensors = []
    excluded_insufficient_data = 0
    excluded_no_metadata = 0
    
    for file_path in pa_files:
        filename = os.path.basename(file_path)
        # Extract the sensor string segment from the name (e.g., 'sensor_12345_history.csv' -> '12345')
        sensor_id = filename.split('_')[1].strip()
        
        # Verify coordinates exist in metadata lookup
        if sensor_id not in meta_dict:
            excluded_no_metadata += 1
            continue
            
        lat = meta_dict[sensor_id]['latitude']
        lon = meta_dict[sensor_id]['longitude']
        
        # Process individual history files
        df = pd.read_csv(file_path)
        df['date'] = pd.to_datetime(df[time_col])
        df = df.rename(columns={pm_col: 'pm25'})
        
        # Enforce 120-hour filtering constraint
        valid_hours = df['pm25'].count()
        if valid_hours < 120:
            excluded_insufficient_data += 1
            continue
            
        # Linear imputation step
        df = df.sort_values('date')
        df['pm25'] = df['pm25'].interpolate(method='linear').bfill().ffill()
        
        df['station_id'] = f"PA_{sensor_id}"
        df['latitude'] = lat
        df['longitude'] = lon
        
        all_pa_data.append(df[['date', 'station_id', 'latitude', 'longitude', 'pm25']])
        included_sensors.append(f"PA_{sensor_id}")
        
    logging.info(f"PurpleAir Batch Run: Included {len(included_sensors)} sensors.")
    logging.info(f"Dropped: {excluded_insufficient_data} (<120 hrs), {excluded_no_metadata} (missing coordinates).")
    
    if all_pa_data:
        return pd.concat(all_pa_data, ignore_index=True), included_sensors
    return pd.DataFrame(), []


def aggregate_and_stage_scenarios(combined_df, output_dir="data_staged"):
    """
    Executes the split-scenario averaging logic required by Algorithm 1:
    Splits records into Factual (Jan 7-27) and Counterfactual (Jan 28+), 
    calculates localized averages, and exports the staged result.
    """
    logging.info("Splitting records into Factual (Wildfire) and Counterfactual (Baseline) means...")
    os.makedirs(output_dir, exist_ok=True)
    
    combined_df['month'] = combined_df['date'].dt.month
    combined_df['day'] = combined_df['date'].dt.day
    
    # Scenario A: Factual Window (Active Wildfire Impact: Jan 7 - Jan 27)
    is_factual = (combined_df['month'] == 1) & (combined_df['day'] >= 7) & (combined_df['day'] <= 27)
    pm_f = combined_df[is_factual].groupby('station_id')[['pm25']].mean().rename(columns={'pm25': 'PM_F'})
    
    # Scenario B: Counterfactual Window (Control Baseline: Jan 28 to End of Log)
    is_counterfactual = ((combined_df['month'] == 1) & (combined_df['day'] >= 28)) | (combined_df['month'] > 1)
    pm_nf = combined_df[is_counterfactual].groupby('station_id')[['pm25']].mean().rename(columns={'pm25': 'PM_NF'})
    
    # Capture unique geo-coordinates per station
    coords = combined_df.groupby('station_id')[['latitude', 'longitude']].first()
    
    # Build complete uniform spatial frame
    final_dataset = coords.join(pm_f).join(pm_nf).reset_index()
    final_dataset = final_dataset.dropna(subset=['PM_F', 'PM_NF'])
    
    output_path = os.path.join(output_dir, "sensor_scenarios_averaged.csv")
    final_dataset.to_csv(output_path, index=False)
    logging.info(f"Staged Master Sensor CSV written successfully to: {output_path}")
    
    return final_dataset


def process_population_and_baseline(population_path, baseline_mortality_rate, output_dir="data_staged"):
    """
    Cleans structural geographic data, binds the raw unrounded 
    daily baseline mortality rate (Bd), and stages the file for Phase 3.
    """
    logging.info("Staging demographic data profiles...")
    os.makedirs(output_dir, exist_ok=True)
    
    pop_df = pd.read_csv(population_path)
    pop_df['population'] = pd.to_numeric(pop_df['population'], errors='coerce').fillna(0)
    pop_df = pop_df[pop_df['population'] > 0]
    
    pop_df['baseline_mortality_rate'] = float(baseline_mortality_rate)
    
    output_path = os.path.join(output_dir, "clean_population_baseline.csv")
    pop_df.to_csv(output_path, index=False)
    logging.info(f"Staged Demographic CSV written successfully to: {output_path}")
    return pop_df


# =====================================================================
# Main Execution Entry Point
# =====================================================================
if __name__ == "__main__":
    # Change these absolute string paths to match the exact file locations on your machine
    EPA_FILE_PATH = "LA_Site_1103.csv"
    PURPLE_AIR_FOLDER = "." 
    METADATA_COORDINATE_FILE = "purple_air_metadata.csv" 
    RAW_POPULATION_FILE = "raw_la_tract_population.csv"
    
    # Calculate Bd dynamically using unrounded values from your annual target (780 per 100k)
    DAILY_BD_RATE = 780 / 365 / 100000 
    
    # Step 1: Clean and isolate EPA tracking stream
    epa_cleaned = process_epa_sensor(
        file_path=EPA_FILE_PATH,
        date_col1='Date Local', 
        date_col2='Time Local', 
        pm_col='Sample Measurement'
    )
    
    # Step 2: Clean and parse the full folder directory of PurpleAir streams
    pa_cleaned, included_pa_list = process_purple_air_sensors(
        folder_path=PURPLE_AIR_FOLDER,
        metadata_path=METADATA_COORDINATE_FILE,
        time_col='time', 
        pm_col='pm2.5_atm'
    )
    
    # Visibility Log Terminal Block
    print("\n" + "="*60)
    print("                ALGORITHM 1 RUNTIME SUMMARY             ")
    print("="*60)
    print(f"EPA Data Stream Operational:  {not epa_cleaned.empty}")
    print(f"PurpleAir Active Sensors (>=120 hrs): {len(included_pa_list)}")
    print(f"Retained Sensor Identifiers:\n{included_pa_list}")
    print("="*60 + "\n")
    
    # Step 3: Combine all active tracking streams and slice scenarios
    data_streams = [df for df in [epa_cleaned, pa_cleaned] if not df.empty]
    
    if data_streams:
        master_sensor_df = pd.concat(data_streams, ignore_index=True)
        aggregate_and_stage_scenarios(master_sensor_df)
        
        # Step 4: Clean and stage the demographic frames
        process_population_and_baseline(
            population_path=RAW_POPULATION_FILE, 
            baseline_mortality_rate=DAILY_BD_RATE
        )
        print("SUCCESS: Algorithm 1 finished cleanly. Check the 'data_staged' folder for outputs.")
    else:
        print("CRITICAL ERROR: Zero monitoring data rows survived filtering criteria. Run aborted.")