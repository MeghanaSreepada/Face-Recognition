import cv2
import dlib
import numpy as np
import os

# Load the detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Load the face recognition model
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Load the face embeddings
embeddings_dict = np.load('embeddings.npy', allow_pickle=True).item()

# Initialize the video feed
cap = cv2.VideoCapture(0)

# Define a font for displaying text on the image
font = cv2.FONT_HERSHEY_SIMPLEX

def recognize_face(face_embedding, embeddings_dict, threshold=0.6):
    # Loop through each person in the embeddings dictionary
    for name, embeddings in embeddings_dict.items():
        # Compute the distance between the face embedding and the embeddings for this person
        face_distances = np.linalg.norm(np.array(embeddings) - np.array(face_embedding), axis=0)

        # Get the minimum distance and index of the closest match
        min_distance = np.min(face_distances)
        min_distance_idx = np.argmin(face_distances)

        # If the minimum distance is less than the threshold, return the person name and distance
        if min_distance < threshold:
            return name, min_distance

    # If no match was found, return None
    return None, None

while True:
    # Capture the video frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop through each face in the frame
    for face in faces:
        # Get the face landmarks using the predictor
        landmarks = predictor(gray, face)

        # Get the face embedding using the landmarks
        embedding = np.array(face_recognition_model.compute_face_descriptor(frame, landmarks))

        # Recognize the face in the frame
        name, distance = recognize_face(embedding, embeddings_dict)

        # Draw a rectangle around the face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # If the face is recognized, add the person name below the bounding box
        if name is not None:
            cv2.putText(frame, name, (x, y+h+30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video feed
cap.release()

# Close all windows
cv2.destroyAllWindows()
