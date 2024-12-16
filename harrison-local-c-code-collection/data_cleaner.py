import re
import os
# Post processing of txt files for csv conversion
# Patterns to match MPU1 and MPU2 data
pattern = re.compile(r'(MPU\d) acce_x:([-+]?\d*\.\d+|\d+), acce_y:([-+]?\d*\.\d+|\d+), acce_z:([-+]?\d*\.\d+|\d+), gyro_x:([-+]?\d*\.\d+|\d+), gyro_y:([-+]?\d*\.\d+|\d+), gyro_z:([-+]?\d*\.\d+|\d+)')
# acce_pattern = re.compile(r'(MPU\d) ')

# Directories
input_dir = "new_data/down/"
output_dir = "."

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each text file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".txt"):
        input_file = os.path.join(input_dir, filename)
        output_file = os.path.join(output_dir, filename.replace(".txt", ".csv"))

        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            # Write header
            outfile.write("sensor,gyro_x,gyro_y,gyro_z,acce_x,acce_y,acce_z\n")

            # Parse each line in the input file
            for line in infile:
                # Check for accelerometer data
                match = pattern.search(line)
                if match:
                    sensor, acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z = match.groups()
                    
                    outfile.write(f"{sensor},{gyro_x},{gyro_y},{gyro_z},{acce_x},{acce_y},{acce_z}\n")

        print(f"Processed and saved: {output_file}")
