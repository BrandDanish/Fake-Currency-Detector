from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import cv2
import os
import requests
import numpy as np
from werkzeug.utils import secure_filename
app = Flask(__name__)

# Configure the upload folder and allowed file extensions
UPLOAD_FOLDER = 'uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'jfif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize ORB detector and load dataset
orb = cv2.ORB_create(nfeatures=1000)
path = 'ImagesQuery'
ClassNames = []
images = []

# Load dataset images and associate with note names
myList = os.listdir(path)
for cl in myList:
    imgCur = cv2.imread(f"{path}/{cl}", 0)
    if imgCur is not None:
        images.append(imgCur)
        ClassNames.append(os.path.splitext(cl)[0])  # Store the filename without extension

def extract_features(image):
    """Extract keypoints and descriptors from the image using ORB."""
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def findDes(images):
    """Find descriptors for all images in the dataset."""
    desList = []
    for img in images:
        _, des = extract_features(img)
        desList.append(des if des is not None else np.array([]))  # Handle missing descriptors
    return desList

def match_image(input_image_descriptors, dataset_descriptors):
    """Match the input image descriptors with the dataset and find the best match."""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match_index = -1
    max_good_matches = 0
    good_match_threshold = 30  # You can adjust this threshold for stricter matching

    for i, descriptors in enumerate(dataset_descriptors):
        if len(descriptors) == 0 or len(input_image_descriptors) == 0:
            continue
        
        matches = bf.match(descriptors, input_image_descriptors)
        good_matches = [m for m in matches if m.distance < 50]  # Only count matches with a low distance
        
        if len(good_matches) > good_match_threshold:  # Only consider as a match if it meets the threshold
            return i  # Return immediately if a strong match is found

    return None

def detect_currency(input_image_path):
    """Detect the specific Pakistani note based on the uploaded image."""
    input_image = cv2.imread(input_image_path, 0)
    if input_image is None:
        print("Error: Unable to load the input image.")
        return None

    _, input_descriptors = extract_features(input_image)
    if input_descriptors is None or len(input_descriptors) == 0:
        print("Error: No features detected in the input image.")
        return None

    desList = findDes(images)
    best_match_index = match_image(input_descriptors, desList)
    if best_match_index is not None:
        return ClassNames[best_match_index]  # Return the matched note name
    else:
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    """Render the home page with the flag of Pakistan."""
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Process the image and detect currency
            note_name = detect_currency(file_path)
            
            if note_name:
                message = "Successfully Detected Currency of Pakistan"
            else:
                message = "It's not a Pakistani note currency."
                
            return render_template('result.html', message=message, image_path=filename)
        else:
            return redirect(request.url)  # Handle case where file extension is not allowed
    
    # Handle GET request
    return render_template('index.html')  # Make sure you have this template file


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True)
