import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from scipy.fft import fft
import pickle
import matplotlib.pyplot as plt

# Define data directory
data_dir = 'data/'
gesture_files = [
    os.path.join(root, f)
    for root, dirs, files in os.walk(data_dir)
    for f in files
    if f.endswith('.csv')
]

# Define sensor columns
sensor_columns = ['gyro_x', 'gyro_y', 'gyro_z', 'acce_x', 'acce_y', 'acce_z']

# Function to process each file
def process_file(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()  # Remove any spaces from column names

    # Split the data by sensor
    mpu1_data = df[df['sensor'] == 'MPU1']
    mpu2_data = df[df['sensor'] == 'MPU2']

    # Ensure both sensors have the same length
    min_length = min(len(mpu1_data), len(mpu2_data))
    mpu1_data = mpu1_data.iloc[:min_length]
    mpu2_data = mpu2_data.iloc[:min_length]
    
    # Rename mpu2 columns to avoid overlap with mpu1
    mpu2_data.columns = [f'{col}_2' for col in mpu2_data.columns]
    sensor_columns_2 = [f'{col}_2' for col in sensor_columns]

    # Concatenate sensor data row-wise
    combined_data = pd.concat([mpu1_data[sensor_columns].reset_index(drop=True),
                               mpu2_data[sensor_columns_2].reset_index(drop=True)], axis=1)

    return combined_data

# Function to compute FFT features for both sensors
def compute_fft_features(data):
    fft_features = []
    for col in data.columns:
        fft_values = fft(data[col].values)
        fft_magnitude = np.abs(fft_values)[:10]  # Use first 10 FFT components
        fft_features.extend(fft_magnitude)
    return fft_features

# Extract features and labels
X = []
y = []

for file in gesture_files:
    file_path = file
    gesture_label = file.split('_')[0]  # Extract gesture label from filename

    combined_data = process_file(file_path)
    features = compute_fft_features(combined_data)
    X.append(features)
    y.append(gesture_label)

X = np.array(X)
y = np.array(y)

# Encode labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "SVM": SVC(kernel='linear', probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# Train, evaluate, and save each model
for name, model in models.items():
    print(f"\nTraining {name} model...")
    model.fit(X_train, y_train)

    # Save model
    filename = f'{name.lower().replace(" ", "_")}_model.pkl'
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved as {filename}.")

    # Evaluate on the test set
    y_pred = model.predict(X_test)

    # Decode predictions and true labels to original string labels
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    y_test_decoded = label_encoder.inverse_transform(y_test)

    print(f"\n{name} Classification Report:\n")
    print(classification_report(y_test_decoded, y_pred_decoded))

    # Display confusion matrix
    cm = confusion_matrix(y_test_decoded, y_pred_decoded, labels=label_encoder.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix for {name}")
    plt.show()
