# Hand-Gesture-Controlled-UAV-project
The goal of this project is to develop a UAV drone control loop that can be controlled by 5-10 different hand gestures.

# Structure

The current working directory details the machine learning pipeline and setup for SVM, KNN, Random forest (svc_knn_rdf.py) and LSTM optimization (model.py). The data folder contains all original data, the data-ag folder generates more 'samples' by applying data augmentations. 'data_fixer.py' and 'data_rearrange.py' are just helper scripts we used to fix incorrectly formatted data.

'harrison-local-c-code-collection' details the onboard c code used by the esp32, similar to other assignments with slight output modifications for 2 IMUs.

'final-pipeline' contains the code to do online drone control with gestures.
