import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up dataset path
dataset_path = "./data/test"
model_path = "emotion_model.h5"

if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
else:
    print("Training new model...")
    # Data augmentation and preprocessing
    datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

    train_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(48, 48),
        batch_size=32,
        color_mode="grayscale",
        class_mode="categorical",
        subset="training"
    )

    val_generator = datagen.flow_from_directory(
        dataset_path,
        target_size=(48, 48),
        batch_size=32,
        color_mode="grayscale",
        class_mode="categorical",
        subset="validation"
    )

    # Build the CNN model
    model = Sequential([
        Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(2,2),
        
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # Adjust output size to match dataset classes
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    # Training parameters
    epochs = 25

    # Train the model
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        verbose=1
    )

    # Save trained model
    model.save(model_path)

# Initialize webcam and face detector
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = np.expand_dims(face, axis=0) / 255.0
        
        predictions = model.predict(face)
        emotion = np.argmax(predictions)

        # Emotion labels
        labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        label = labels[emotion]

        cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Emotion Detection", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
