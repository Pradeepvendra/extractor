from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import numpy as np
import time

app = Flask(__name__)

# Load the Haar Cascade face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Set up the camera
camera = cv2.VideoCapture(0)
camera.set(cv2.CAP_PROP_FPS, 10)

# Set up the face extractor

# Define the max number of faces to extract per person
MAX_FACES_PER_PERSON = 50

# Define the directory to store the extracted faces
FACES_DIR = 'faces'

# Define the route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for processing the webcam image
@app.route('/submit', methods=['POST'])
def submit():
    # Get the name from the form
    name = request.form['name']
    
    # Create a directory for the person if it doesn't exist
    if not os.path.exists(os.path.join(FACES_DIR, name)):
        os.makedirs(os.path.join(FACES_DIR, name))
        
    # Extract faces from the webcam image
    face_count = 0
    while True:
        # Read the image from the camera
        ret, frame = camera.read()
        time.sleep(0.2)

        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        # Extract and save the faces
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face = frame[y:y+h, x:x+w]

            # Resize the face to a fixed size
            face = cv2.resize(face, (224, 244))

            # Save the face to the directory for the person
            filename = os.path.join(FACES_DIR, name, f'{name}_{face_count}.jpg')
            cv2.imwrite(filename, face)

            # Increment the face count
            face_count += 1

            # Break out of the loop if we've extracted the max number of faces
            if face_count == MAX_FACES_PER_PERSON:
                break

        # Break out of the loop if we've extracted the max number of faces or if the user hits the 'q' key
        if face_count == MAX_FACES_PER_PERSON or cv2.waitKey(1) == ord('q'):
            break
        
        # Display the image with bounding boxes around the detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow('image', frame)
        
    # Release the camera
    camera.release()
    cv2.destroyAllWindows()

    # Render the success page
    return render_template('success.html')

if __name__ == '__main__':
    app.run(debug=True)
