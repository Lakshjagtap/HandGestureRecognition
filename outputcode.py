import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the pre-trained model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Initialize Video Capture
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)

# Labels Dictionary
labels_dict = {0: 'ZERO', 1: 'ONE', 2: 'TWO', 3: 'THREE', 4:'FOUR',
               5:'FIVE', 6:'You are Beautiful', 7:'Rock', 8:'Thumbs-up', 9:'Thumbs-down'}

while cap.isOpened():
    # Read Frame
    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Collect landmark coordinates
            data_aux, x_, y_ = [], [], []
            for lm in hand_landmarks.landmark:
                x, y = lm.x, lm.y
                data_aux.extend([x, y])
                x_.append(x)
                y_.append(y)

            # Bounding Box Calculation
            x1, y1 = int(min(x_) * W) - 20, int(min(y_) * H) - 20
            x2, y2 = int(max(x_) * W) + 20, int(max(y_) * H) + 20

            # Gesture Prediction
            try:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
            except Exception as e:
                predicted_character = "Error"
                print(f"Prediction Error: {e}")

            # Draw Bounding Box & Prediction
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(frame, predicted_character, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display Frame
    cv2.imshow("Hand Gesture Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
