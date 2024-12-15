import os
import pandas as pd

def swap_data_in_columns(folder_path):
    """
    Recursively process all CSV files in the folder and its subfolders to swap
    accelerometer and gyroscope data for each dimension without overwriting data.
    """
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                
                # Load the CSV file
                try:
                    df = pd.read_csv(file_path)
                    
                    # Ensure the columns are as expected
                    expected_columns = ['sensor', 'gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']
                    if all(col in df.columns for col in expected_columns):
                        # Swap data between accelerometer and gyroscope columns
                        df['temp_gyro_x'] = df['gyro_x']
                        df['gyro_x'] = df['acce_x']
                        df['acce_x'] = df['temp_gyro_x']
                        del df['temp_gyro_x']

                        df['temp_gyro_y'] = df['gyro_y']
                        df['gyro_y'] = df['acce_y']
                        df['acce_y'] = df['temp_gyro_y']
                        del df['temp_gyro_y']

                        df['temp_gyro_z'] = df['gyro_z']
                        df['gyro_z'] = df['acce_z']
                        df['acce_z'] = df['temp_gyro_z']
                        del df['temp_gyro_z']
                        
                        # Save the updated CSV back to the same location
                        df.to_csv(file_path, index=False)
                        print(f"Processed: {file_path}")
                    else:
                        print(f"Skipped (unexpected columns): {file_path}")
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

# Replace this with the path to your "data" folder
data_folder = "data/down-h"

# Call the function to process all CSVs
swap_data_in_columns(data_folder)