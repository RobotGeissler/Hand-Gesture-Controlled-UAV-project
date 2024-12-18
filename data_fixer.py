import os
import pandas as pd
# Simmillar to data_rearrange.py, this script fixes the column order in CSV files, was used to fix the column order in the data files
def swap_data_in_columns(folder_path):
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                # Load the CSV file
                try:
                    df = pd.read_csv(file_path)
                    
                    # Ensure the columns are as expected
                    expected_columns = ['acce_x', 'acce_y', 'acce_z', 'gyro_x', 'gyro_y', 'gyro_z', 'sensor']
                    if all(col in df.columns for col in expected_columns):
                        # Swap the data between corresponding accelerometer and gyroscope columns
                        df[['acce_x', 'gyro_x']] = df[['gyro_x', 'acce_x']].values
                        df[['acce_y', 'gyro_y']] = df[['gyro_y', 'acce_y']].values
                        df[['acce_z', 'gyro_z']] = df[['gyro_z', 'acce_z']].values
                        
                        # Save the updated CSV back to the same location
                        df.to_csv(file_path, index=False)
                        print(f"Processed: {file_path}")
                    else:
                        print(f"Skipped (unexpected columns): {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

data_folder = "data/idle-h"

swap_data_in_columns(data_folder)
