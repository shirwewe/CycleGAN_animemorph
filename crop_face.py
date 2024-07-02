import cv2
import os
from pathlib import Path

# Paths
input_folder = '/home/shirwee/documents/aps360-project/data/dataSetA_10k'
output_folder = '/home/shirwee/documents/aps360-project/data/dataSetA_10k_cropped'
haar_cascade_path = '/home/shirwee/documents/aps360-project/haarcascade_frontalface_default.xml'

# Create the output folder if it doesn't exist
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Load the Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Function to crop faces
def crop_and_resize_faces(image_path, output_path, size=(128, 128)):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If at least one face is detected
    for (x, y, w, h) in faces:
        # Crop the face
        face = image[y:y+h, x:x+w]
        # Resize the face
        face_resized = cv2.resize(face, size)
        # Save the cropped and resized face
        cv2.imwrite(output_path, face_resized)
        break  # Save only the first detected face

# Process all images in the input folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        crop_and_resize_faces(input_path, output_path)
        print(f"Cropped, resized, and saved face from {filename}")

print("Processing complete.")