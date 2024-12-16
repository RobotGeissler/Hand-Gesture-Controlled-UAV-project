import os
import pandas as pd
import numpy as np
import random
from scipy.interpolate import interp1d
# basic data augmentation techniques for time-series data - credits to https://maddevs.io/writeups/basic-data-augmentation-method-applied-to-time-series/
# for code reference
def jitter(data, sigma=0.01):
    """Add random noise to the data."""
    noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
    return data + noise

def scaling(data, sigma=0.1):
    """Scale the data by a random factor."""
    factor = np.random.normal(loc=1.0, scale=sigma)
    return data * factor

def time_warp(data, sigma=0.2, knot=4):
    """Apply time warping to the data."""
    orig_steps = np.arange(data.shape[0])
    random_warp = np.random.normal(loc=1.0, scale=sigma, size=knot)
    warp_steps = np.linspace(0, data.shape[0] - 1, num=knot)
    warper = interp1d(warp_steps, random_warp, kind="linear", fill_value="extrapolate")
    new_steps = warper(orig_steps)
    interp = interp1d(orig_steps, data, axis=0, kind="linear", fill_value="extrapolate")
    return interp(new_steps)

def time_shift(data, shift_max=5):
    """Shift the data in time."""
    shift = np.random.randint(-shift_max, shift_max)
    if shift > 0:
        return np.pad(data, ((shift, 0), (0, 0)), mode="constant")[:-shift]
    elif shift < 0:
        return np.pad(data, ((0, -shift), (0, 0)), mode="constant")[-shift:]
    return data

def magnitude_warp(data, sigma=0.2, knot=4):
    """Warp the magnitude of the data."""
    orig_steps = np.arange(data.shape[0])
    random_warp = np.random.normal(loc=1.0, scale=sigma, size=knot)
    warp_steps = np.linspace(0, data.shape[0] - 1, num=knot)
    warper = interp1d(warp_steps, random_warp, kind="linear", fill_value="extrapolate")
    scale_factors = warper(orig_steps).reshape(-1, 1)
    return data * scale_factors

def augment_and_save_time_series(csv_file, output_dir, num_augments=3):
    # Load data
    df = pd.read_csv(csv_file)
    data = df.iloc[:, :-1].values  # Exclude the 'sensor' column for augmentation
    sensor_col = df.iloc[:, -1].values  # Preserve the 'sensor' column
    
    for i in range(num_augments):
        augmented_data = data.copy()
        
        # Apply random augmentations
        if random.random() < 0.5:
            augmented_data = jitter(augmented_data, sigma=0.01)
        if random.random() < 0.5:
            augmented_data = scaling(augmented_data, sigma=0.1)
        if random.random() < 0.5:
            augmented_data = time_warp(augmented_data, sigma=0.2)
        if random.random() < 0.5:
            augmented_data = time_shift(augmented_data, shift_max=5)
        if random.random() < 0.5:
            augmented_data = magnitude_warp(augmented_data, sigma=0.2)

        # Combine with sensor column
        augmented_df = pd.DataFrame(augmented_data)
        augmented_df['sensor'] = sensor_col

        # Save augmented data
        base_name = os.path.basename(csv_file).replace('.csv', f'_aug_{i}.csv')
        augmented_file = os.path.join(output_dir, base_name)
        augmented_df.to_csv(augmented_file, index=False)

def quadruple_time_series_data(input_dir, num_augments=3):
    for gesture_folder in os.listdir(input_dir):
        gesture_path = os.path.join(input_dir, gesture_folder)
        
        if os.path.isdir(gesture_path):  # Ensure it's a directory
        # if gesture_folder=='down':
            for csv_file in os.listdir(gesture_path):
                if csv_file.endswith('.csv'):
                    csv_path = os.path.join(gesture_path, csv_file)
                    
                    # Apply augmentations and save the results
                    augment_and_save_time_series(csv_path, gesture_path, num_augments=num_augments)

# Main execution
if __name__ == "__main__":
    input_directory = "data-ag/"  
    quadruple_time_series_data(input_directory, num_augments=3)
    print("Time-series data augmentation completed. Dataset has been quadrupled.")
