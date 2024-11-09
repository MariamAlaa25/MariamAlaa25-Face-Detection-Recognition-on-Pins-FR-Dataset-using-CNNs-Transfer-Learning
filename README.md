
Project Overview
------------------------------------------------------------
This project demonstrates how to implement real-time face recognition using embeddings extracted from a pre-trained InceptionV3 model. The goal is to capture video from a webcam, detect faces, extract their embeddings, and classify them as known or unknown based on cosine similarity with pre-saved class embeddings. The system processes live video frames, recognizes faces, and displays results such as the predicted class and similarity score.

Dataset Description
-------------------------------------------------------------
The dataset used for this task contains multiple facial images of various individuals (celebrities) for about 105 celebrities, with each class corresponding to a different person.

Dataset Preprocessing
---------------------------------------------------------------
For this face recognition project, the face embeddings (features) are extracted from a set of known faces and stored in a dictionary. These embeddings are then used to compare and classify faces in real-time.

Preprocessing Steps:

Face Detection: Faces are detected in images using OpenCV’s Haar Cascade classifier.
Face Cropping and Resizing: Detected faces are cropped and resized to the input dimensions expected by the InceptionV3 model (224x224 pixels).
Embedding Extraction: Embeddings (feature vectors) are extracted from the cropped faces using a pre-trained InceptionV3 model.
After preprocessing, embeddings are stored in a pickle file (class_embeddings.pkl) for later use

Model Training and Embedding Extraction
---------------------------------------------------------------
This project uses the InceptionV3 model, which is pre-trained on the ImageNet dataset, to extract embeddings from faces. The InceptionV3 model's output is used to generate feature vectors for each face, and these embeddings serve as the unique representation of faces in the dataset.

Steps:
1-Loading the Pre-Trained InceptionV3 Model:

    -We use the InceptionV3 model from TensorFlow, excluding the top classification layers, to extract features.

    -A custom model is created by using the output from a selected intermediate layer ('mixed6').
2-Embedding Extraction:

    -For each detected face, we preprocess the image (resize, normalize) and pass it through the InceptionV3 model to extract its embedding.

Face Recognition Process
---------------------------------------------------------------
1-Webcam Capture:

-The system continuously captures frames from the webcam using OpenCV's VideoCapture method.

-Each frame is processed to detect faces using OpenCV's Haar Cascade face detector.

2. Face Cropping and Embedding Extraction:

-Once a face is detected, the region of interest (ROI) is cropped from the frame and resized to fit the InceptionV3 model's input size.

-The embedding for the cropped face is extracted using the model.

3. Face Classification:

-The extracted embedding is compared against the saved class embeddings using cosine similarity.

-If the cosine similarity score exceeds a threshold, the face is classified as the corresponding class (known face). Otherwise, it is classified as "not identified."

4. Result Display:

-The predicted class and the similarity score are displayed on the video frame.

-A rectangle is drawn around the detected face with the predicted label and similarity score.

Steps to Run the Code
---------------------------------------------------------------
1-Download the Notebook

2-Open juptyer Notebook

3-Create a New Notebook.

4-Install Required Packages.

5-Run the code by pressing (shift+enter)

Dependencies
--------------------------------------------------------------
-Python 3.x

-TensorFlow: For running the pre-trained InceptionV3 model

-OpenCV: For face detection and webcam capture

-NumPy: For array manipulation

-Scikit-learn: For cosine similarity calculation

-Matplotlib (optional, for visualizations)

pip install tensorflow opencv-python numpy scikit-learn matplotlib

