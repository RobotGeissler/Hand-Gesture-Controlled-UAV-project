import re
import os
import pandas as pd
import numpy as np
# These functions aren't used directly but were written by Harrison for final pipeline
# Patterns to match MPU1 and MPU2 data
pattern = re.compile(r'(MPU\d) acce_x:([-+]?\d*\.\d+|\d+), acce_y:([-+]?\d*\.\d+|\d+), acce_z:([-+]?\d*\.\d+|\d+), gyro_x:([-+]?\d*\.\d+|\d+), gyro_y:([-+]?\d*\.\d+|\d+), gyro_z:([-+]?\d*\.\d+|\d+)')
# acce_pattern = re.compile(r'(MPU\d) ')

def parse_data(line):
    match = pattern.search(line)
    if match:
        sensor, acce_x, acce_y, acce_z, gyro_x, gyro_y, gyro_z = match.groups()
        return gyro_x, gyro_y, gyro_z, acce_x, acce_y, acce_z, sensor
    return None

"""
MPU1 2D List: [[gyro_x, gyro_y, gyro_z, acce_x, acce_y, acce_z, sensor], ...]
MPU2 2D List: [[gyro_x, gyro_y, gyro_z, acce_x, acce_y, acce_z, sensor], ...]

Concat sensor data from MPU1 and MPU2 into a single DataFrame
then get the first 10 features of fft for each column
"""
def ml_parse_data(mpu1, mpu2):
    # Create a DataFrame from the MPU1 and MPU2 data and concatenate them without sensor column
    assert len(mpu1) == len(mpu2)
    assert len(mpu1) > 0 and len(mpu2) > 0
    assert len(mpu1[0]) == 7 and len(mpu2[0]) == 7
    mpu1_df = pd.DataFrame(mpu1, columns=['gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z', 'sensor'])
    mpu2_df = pd.DataFrame(mpu2, columns=['gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z', 'sensor'])
    mpu1_df = mpu1_df.drop(columns=['sensor']).reset_index(drop=True)
    mpu2_df = mpu2_df.drop(columns=['sensor']).reset_index(drop=True)
    df = pd.concat([mpu1_df, mpu2_df], axis=1)

    # Get the first 10 features of fft for each column
    fft_features = []
    for col in df.columns:
        fft_values = np.fft.fft(df[col].values)
        fft_mag = np.abs(fft_values)[:10]
        fft_features.extend(fft_mag)
    return fft_features

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# sklearn load model
import pickle

def predict_gesture(X):
    # Load the trained model
    with open('svm_model.pkl', 'rb') as file:
        model = pickle.load(file)
    # Predict the gesture
    y_pred = model.predict(X)
    return y_pred

import random
# Example Input 1x120
ex = [random.random() for _ in range(120)]
predict_gesture([ex])

import torch
import numpy as np

def parse_sensor_data(mpu1_data, mpu2_data, time_window=1):
    """
    Parse incoming MPU1 and MPU2 sensor data to match LSTM input format.
    
    Args:
        mpu1_data (list of lists): 2D list of MPU1 sensor readings (shape ~ 265x7).
        mpu2_data (list of lists): 2D list of MPU2 sensor readings (shape ~ 265x7).
        time_window (int): Time window used for the LSTM input format.

    Returns:
        input_tensor (torch.Tensor): Parsed and reshaped input tensor.
        sequence_length (int): The actual sequence length.
    """
    # Convert input lists to numpy arrays
    mpu1_data = np.array(mpu1_data)[:, :-1]  # Exclude 'sensor' column
    mpu2_data = np.array(mpu2_data)[:, :-1]  # Exclude 'sensor' column

    # Check dimensions
    assert mpu1_data.shape == mpu2_data.shape, "MPU1 and MPU2 data must have the same shape."

    # Combine MPU1 and MPU2 data interleaved: [MPU1_t1, MPU2_t1, MPU1_t2, MPU2_t2, ...]
    combined_data = np.empty((mpu1_data.shape[0] * 2, mpu1_data.shape[1]), dtype=np.float32)
    combined_data[0::2] = mpu1_data  # Insert MPU1 data at even indices
    combined_data[1::2] = mpu2_data  # Insert MPU2 data at odd indices

    # Reshape to align with LSTM input: [Batch, Time_Steps, Features]
    n_steps = combined_data.shape[0] // (time_window * 2)  # Adjust for the time window
    reshaped_data = combined_data.reshape(n_steps, -1)

    # Convert to torch tensor
    input_tensor = torch.from_numpy(reshaped_data)

    # Sequence length
    sequence_length = input_tensor.shape[0]

    return input_tensor, sequence_length

