import cv2
import os
import dlib
import os
import numpy as np

# Set the input and output directories
input_dir = 'dataset'
output_file = 'embeddings.npy'

# Load the detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Initialize the list of face embeddings
face_embeddings = {}

# Loop through each image in the input directory
for filename in os.listdir(input_dir):
    if '.jpeg' not in filename and '.png' not in filename:
        continue
    # Read the image file    
    img_path = os.path.join(input_dir, filename)
    print('Processing image:', img_path)
    img = cv2.imread(img_path)

    # Convert the image from BGR (OpenCV default) to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces in the image
    faces = detector(img, 1)

    name, ext = os.path.splitext(filename)

    # Loop through each face in the image
    for face in faces:
        # Get the landmarks for the face
        landmarks = predictor(img, face)

        # Get the face embedding
        face_embedding = face_recognition_model.compute_face_descriptor(img, landmarks)
        
        # Add the face embedding to the list
        face_embeddings[name] = face_embedding

# Save the face embeddings to a file
np.save(output_file, face_embeddings)
print('Saved face embeddings to', output_file)
