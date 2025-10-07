✋ Hand Gesture Recognition

A computer vision project for real-time hand gesture recognition using OpenCV, Mediapipe, and Scikit-learn.
This project allows you to collect your own hand gesture dataset, train a machine learning model, and recognize gestures live from your webcam.

🚀 Features

📸 Collects custom hand gesture images using your webcam.
✍️ Extracts hand landmarks with Mediapipe Hands.
🤖 Trains a Random Forest Classifier on gesture data.
🎥 Real-time gesture recognition with bounding boxes and labels.
🔟 Supports 10 different gesture classes (customizable).

📂 Project Structure
├── Collecting-data.py   # Capture raw gesture images with webcam
├── createdataset.py     # Process images → extract landmarks → save dataset
├── Traindata.py         # Train Random Forest model and save to model.p
├── outputcode.py        # Real-time gesture recognition
├── data/                # Captured dataset (images per class)
├── data.pickle          # Processed dataset (landmarks + labels)
├── model.p              # Trained model

⚙️ Installation

Clone this repo:

    git clone https://github.com/Lakshjagtap/HandGestureRecognition.git
    cd HandGestureRecognition


Install dependencies:

    pip install opencv-python mediapipe scikit-learn matplotlib numpy

🖐️ Usage
1. Collect Gesture Data

Run the script to capture images for each gesture class:

    python Collecting-data.py

Press Q to start capturing images for each class.
Images are saved under ./data/<class_number>/.

2. Create Dataset

Extract hand landmarks and save them as a dataset:

    python createdataset.py

3. Train Model

Train a Random Forest classifier on the dataset:

    python Traindata.py

4. Run Real-time Recognition

Start real-time hand gesture recognition:

    python outputcode.py

Press Q to quit the application.

🧠 Model

Features: Hand landmark (x, y) coordinates extracted via Mediapipe Hands.
Classifier: Random Forest.
Number of classes: 10 (customizable).
Accuracy is printed after training.

🎯 Customization

Change number_of_classes and dataset_size in Collecting-data.py to suit your project.
Update labels_dict in outputcode.py with your gesture names.
You can experiment with other models (e.g., SVM, Neural Networks) in Traindata.py.

🛠️ Requirements

Python 3.8+
OpenCV
Mediapipe
Scikit-lear
Matplotlib
Numpy

📜 License

This project is open-source under the MIT License.
