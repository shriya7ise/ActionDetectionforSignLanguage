import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import tensorflow as tf
tf.config.list_physical_devices('GPU')

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS) # Draw right hand connections
    
    
def draw_styled_landmarks(image, results):
    # Draw face connections
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 
    
    
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)

        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    
    results
    draw_landmarks(frame, results)
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    
lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)


def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([lh, rh])


result_test = extract_keypoints(results)
result_test

# Path for exported data, numpy arrays
DATA_PATH = os.path.join('/Users/shriya/Downloads/keypoints_sih/islrtc_shriya') 

# Actions that we try to detect
actions = np.array(['ajivit', 'anta_pranali', 'antagrahan','aterna','understory'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30


import os
import cv2

# Paths
DATASET_PATH = '/Users/shriya/Desktop/video_train'  # Replace with the path to your dataset containing folders with videos
OUTPUT_PATH = '/Users/shriya/Downloads/ex_data_frames/islrtc_shriya'
# Loop through each folder (each containing one video)
for action in os.listdir(DATASET_PATH):
    action_path = os.path.join(DATASET_PATH, action)
    
    # Check if the current item is a directory, and skip if it's not
    if not os.path.isdir(action_path):
        continue
    
    video_files = [f for f in os.listdir(action_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

    for video_file in video_files:
        video_path = os.path.join(action_path, video_file)
        cap = cv2.VideoCapture(video_path)
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Create a directory for frames if it doesn't exist
            frames_path = os.path.join(OUTPUT_PATH, action)
            os.makedirs(frames_path, exist_ok=True)

            # Save the frame
            frame_filename = os.path.join(frames_path, f'frame_{frame_count:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        cap.release()

print("Frames extraction complete.")


import os
import cv2

# Paths
FRAMES_PATH = '/Users/shriya/Downloads/ex_data_frames/islrtc_shriya'  # Path where the frames are saved
PROCESSED_FRAMES_PATH = '/Users/shriya/Downloads/processed_frames'  # Path where processed frames will be saved

# Loop through each action (class/folder)
for action in os.listdir(FRAMES_PATH):
    action_frames_path = os.path.join(FRAMES_PATH, action)
    
    if not os.path.isdir(action_frames_path):
        continue
    
    # Get list of all frames
    frames = sorted([f for f in os.listdir(action_frames_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    # Create directory for processed frames
    processed_action_path = os.path.join(PROCESSED_FRAMES_PATH, action)
    os.makedirs(processed_action_path, exist_ok=True)
    
    for frame in frames:
        frame_path = os.path.join(action_frames_path, frame)
        
        # Read the image
        img = cv2.imread(frame_path)
        if img is None:
            print(f"Warning: {frame_path} could not be read.")
            continue
        
        # Process the image (example: resize or extract keypoints)
        # For example, resize the image to 224x224
        processed_img = cv2.resize(img, (224, 224))
        
        # Save the processed image
        processed_frame_path = os.path.join(processed_action_path, frame)
        cv2.imwrite(processed_frame_path, processed_img)

print("Processing complete.")

import os
import numpy as np
import cv2

# Paths
PROCESSED_FRAMES_PATH = '/Users/shriya/Downloads/processed_frames'
actions = ['ajivit', 'anta_pranali', 'antagrahan', 'aterna', 'understory']
sequence_length = 30
img_height, img_width = 224, 224  # Resize images to 224x224

def load_images_from_folder(folder, sequence_length):
    images = []
    for filename in sorted(os.listdir(folder)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_width, img_height))  # Resize images
            images.append(img)
    return images

def create_sequences_from_images(images, sequence_length):
    sequences = []
    for i in range(0, len(images) - sequence_length + 1, sequence_length):
        sequence = images[i:i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

X, y = [], []

for action in actions:
    action_path = os.path.join(PROCESSED_FRAMES_PATH, action)
    images = load_images_from_folder(action_path, sequence_length)
    sequences = create_sequences_from_images(images, sequence_length)
    X.extend(sequences)
    y.extend([actions.index(action)] * len(sequences))

X = np.array(X)
y = np.array(y)
y = tf.keras.utils.to_categorical(y, num_classes=len(actions))

# Print shapes to verify
print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=7)

# Print shapes to verify
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, TimeDistributed, LSTM

model = Sequential()
model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(30, 224, 224, 3)))
model.add(TimeDistributed(MaxPooling2D((2, 2))))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(5, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=8)

