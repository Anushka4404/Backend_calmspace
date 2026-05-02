from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import random

import numpy as np
import cv2
from keras.models import load_model

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# =========================
# DATA DIRECTORIES
# =========================
train_data_dir='data/train/'

validation_data_dir='data/test/'

# =========================
# DATA AUGMENTATION
# =========================
train_datagen = ImageDataGenerator(
					rescale=1./255,
					rotation_range=30,
					shear_range=0.3,
					zoom_range=0.3,
					horizontal_flip=True,
					fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
					train_data_dir,
					color_mode='grayscale',
					target_size=(48, 48),
					batch_size=32,
					class_mode='categorical',
					shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
							validation_data_dir,
							color_mode='grayscale',
							target_size=(48, 48),
							batch_size=32,
							class_mode='categorical',
							shuffle=True)


# class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']

# img, label = train_generator.__next__()

# =========================
# MODEL
# =========================
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

model.compile(
    optimizer = 'adam', 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
    )
print(model.summary())

# =========================
# COUNT IMAGES SAFELY
# =========================
# train_path = "data/train/"
# test_path = "data/test"

# num_train_imgs = 0
# for root, dirs, files in os.walk(train_path):
#     num_train_imgs += len(files)
    
# num_test_imgs = 0
# for root, dirs, files in os.walk(test_path):
#     num_test_imgs += len(files)

num_train_imgs = sum(len(files) for _, _, files in os.walk(train_data_dir))
num_test_imgs = sum(len(files) for _, _, files in os.walk(validation_data_dir))

print("Train images:", num_train_imgs)
print("Test images:", num_test_imgs)


# =========================
# TRAINING
# =========================
# epochs=30
epochs=5

# history = model.fit(
#     train_generator,
#     steps_per_epoch=num_train_imgs//32,
#     epochs=epochs,
#     validation_data=validation_generator,
#     validation_steps=num_test_imgs//32)

history = model.fit(
    train_generator,
    steps_per_epoch=max(1, num_train_imgs // 32),
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=max(1, num_test_imgs // 32)
)

# SAVE MODEL
model.save('model_file.h5')
print("Model saved successfully!")

# =========================
# FLASK API (UNCHANGED STRUCTURE)
# =========================
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels_dict = {
    0: 'Angry',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Surprise'
}

# extra
def get_mood(emotion):
    if emotion == "Happy":
        return "Positive mood 😊"
    elif emotion in ["Sad", "Fear"]:
        return "Low mood 😔"
    else:
        return "Neutral mood 😐"
# extra

# @app.route('/predict_emotion', methods=['POST'])
# def predict_emotion():
#     """
#     Process image and return detected emotion.
#     This is a simple implementation that returns random moods for testing.
#     In a production environment, this would use the actual ML model.
#     """
#     try:
#         # Check if an image file was included in the request
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image file provided'}), 400
        
#         # Get the image file
#         image_file = request.files['image']
        
#         # Print some debugging information
#         print(f"Received image: {image_file.filename}, {image_file.content_type}, {image_file.content_length} bytes")
        
#         # Save the image for debugging (optional)
#         debug_dir = 'debug_images'
#         if not os.path.exists(debug_dir):
#             os.makedirs(debug_dir)
#         image_file.save(os.path.join(debug_dir, 'latest_capture.jpg'))
        
#         # In a real implementation, we would process the image with the ML model here
#         # For now, let's return a random mood
#         moods = [
#             {"mood": 0, "moodLabel": "Angry"},
#             {"mood": 1, "moodLabel": "Disgust"},
#             {"mood": 2, "moodLabel": "Fear"},
#             {"mood": 3, "moodLabel": "Happy"},
#             {"mood": 4, "moodLabel": "Neutral"},
#             {"mood": 5, "moodLabel": "Sad"},
#             {"mood": 6, "moodLabel": "Surprise"}
#         ]
        
#         # Choose a random mood (for testing purposes)
#         random_mood = random.choice(moods)
        
#         # Return the mood as JSON
#         return jsonify({
#             "mood": random_mood["mood"],
#             "moodLabel": random_mood["moodLabel"]
#         })
        
#     except Exception as e:
#         print(f"Error processing image: {str(e)}")
#         return jsonify({"error": str(e)}), 500

# if __name__ == '__main__':
#     # Run the Flask app on port 5000
#     print("Starting mood detection server on port 5000...")
#     app.run(host='0.0.0.0', port=5000, debug=True)



@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400

        image_file = request.files['image']
        if image_file is None:
            return jsonify({'error': 'No image provided'}), 400
        print(request.files)
        
        npimg = np.frombuffer(image_file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({'error': 'Invalid image'}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)

        if len(faces) == 0:
            return jsonify({'error': 'No face detected'}), 400

        for x, y, w, h in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face / 255.0
            face = np.reshape(face, (1, 48, 48, 1))

            result = model.predict(face)
            label = int(np.argmax(result))

            # return jsonify({
            #     "mood": label,
            #     "moodLabel": labels_dict[label]
            # })
            
            emotion = labels_dict[label]
            confidence = float(np.max(result))

            return jsonify({
                "emotion": emotion,
                "confidence": round(confidence, 2),
                "mood": get_mood(emotion)
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# RUN SERVER
# =========================
if __name__ == '__main__':
    print("Starting server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True)