import re
import os

# Patterns to match gyroscope and accelerometer data
gyro_pattern = re.compile(r'gyro_x:([-+]?\d*\.\d+|\d+), gyro_y:([-+]?\d*\.\d+|\d+), gyro_z:([-+]?\d*\.\d+|\d+)')
acce_pattern = re.compile(r'acce_x:([-+]?\d*\.\d+|\d+), acce_y:([-+]?\d*\.\d+|\d+), acce_z:([-+]?\d*\.\d+)')
sensor_pattern = re.compile(r'MPU\d')

# Gestures to process - run this script but be careful with names, idle is using number and other h files don't
gestures = ['up', 'down', 'left', 'idle']

# Paths for input and output
base_input_path = "."
base_output_path = "."

# Loop through all gestures and their corresponding files
for gesture in gestures:
    for i in range(0, 100):
        number = str(i).zfill(2)
        input_file = f"{base_input_path}/{gesture}-h/{gesture}_{i}.txt"
        output_file = f"{base_output_path}/{gesture}-h/{gesture}_{number}.csv"

        # Check if the input file exists
        if not os.path.exists(input_file):
            print(f"Input file not found: {input_file}")
            continue

        # Read the input file and write to output CSV
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            outfile.write("gyro_x,gyro_y,gyro_z,acce_x,acce_y,acce_z,sensor\n")  # Header row

            for line in infile:
                gyro_match = gyro_pattern.search(line)
                acce_match = acce_pattern.search(line)
                sensor_match = sensor_pattern.search(line)

                if gyro_match and acce_match and sensor_match:
                    gyro_x, gyro_y, gyro_z = gyro_match.groups()
                    acce_x, acce_y, acce_z = acce_match.groups()
                    sensor = sensor_match.group()

                    outfile.write(f"{gyro_x},{gyro_y},{gyro_z},{acce_x},{acce_y},{acce_z},{sensor}\n")

        print(f"Data extracted and saved to {output_file}")
