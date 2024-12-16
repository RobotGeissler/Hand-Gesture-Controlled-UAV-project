import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

# Read the data
file_path = "down_0.csv" 
data = pd.read_csv(file_path)

# Split data by sensor
mpu1_data = data[data['sensor'] == 'MPU1']
mpu2_data = data[data['sensor'] == 'MPU2']

# Plot Gyroscope and Accelerometer data over time
def plot_time_series(sensor_data, sensor_name):
    time = np.arange(len(sensor_data))  # Simulated time axis

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Time Series Data for {sensor_name}", fontsize=16)

    # Gyroscope
    axs[0, 0].plot(time, sensor_data['gyro_x'], label='Gyro X')
    axs[0, 0].set_title('Gyroscope X')
    axs[0, 1].plot(time, sensor_data['gyro_y'], label='Gyro Y')
    axs[0, 1].set_title('Gyroscope Y')
    axs[0, 2].plot(time, sensor_data['gyro_z'], label='Gyro Z')
    axs[0, 2].set_title('Gyroscope Z')

    # Accelerometer
    axs[1, 0].plot(time, sensor_data['acce_x'], label='Accel X', color='orange')
    axs[1, 0].set_title('Accelerometer X')
    axs[1, 1].plot(time, sensor_data['acce_y'], label='Accel Y', color='orange')
    axs[1, 1].set_title('Accelerometer Y')
    axs[1, 2].plot(time, sensor_data['acce_z'], label='Accel Z', color='orange')
    axs[1, 2].set_title('Accelerometer Z')

    for ax in axs.flat:
        ax.set_xlabel('Time (arbitrary units)')
        ax.set_ylabel('Value')
        ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Generate spectrograms
def plot_spectrograms(sensor_data, sensor_name):
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Spectrograms for {sensor_name}", fontsize=16)

    # Gyroscope Spectrograms
    f, t, Sxx = spectrogram(sensor_data['gyro_x'], fs=1.0)
    axs[0, 0].pcolormesh(t, f, Sxx, shading='gouraud')
    axs[0, 0].set_title('Gyroscope X Spectrogram')

    f, t, Sxx = spectrogram(sensor_data['gyro_y'], fs=1.0)
    axs[0, 1].pcolormesh(t, f, Sxx, shading='gouraud')
    axs[0, 1].set_title('Gyroscope Y Spectrogram')

    f, t, Sxx = spectrogram(sensor_data['gyro_z'], fs=1.0)
    axs[0, 2].pcolormesh(t, f, Sxx, shading='gouraud')
    axs[0, 2].set_title('Gyroscope Z Spectrogram')

    # Accelerometer Spectrograms
    f, t, Sxx = spectrogram(sensor_data['acce_x'], fs=1.0)
    axs[1, 0].pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis')
    axs[1, 0].set_title('Accelerometer X Spectrogram')

    f, t, Sxx = spectrogram(sensor_data['acce_y'], fs=1.0)
    axs[1, 1].pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis')
    axs[1, 1].set_title('Accelerometer Y Spectrogram')

    f, t, Sxx = spectrogram(sensor_data['acce_z'], fs=1.0)
    axs[1, 2].pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis')
    axs[1, 2].set_title('Accelerometer Z Spectrogram')

    for ax in axs.flat:
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Plot for MPU1
plot_time_series(mpu1_data, 'MPU1')
# plot_spectrograms(mpu1_data, 'MPU1')

# Plot for MPU2
plot_time_series(mpu2_data, 'MPU2')
# plot_spectrograms(mpu2_data, 'MPU2')