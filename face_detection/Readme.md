# OpenCV based Face Detection and Swap

This is implementation is based on the excellent tutorial from [PyImageSearch](https://www.pyimagesearch.com/) for detecting faces in images using OpenCV.

It takes an input source image and detects the face in it.

Next it turns on the webcam and swaps the face detected in the images captured live from webcam with the previously detected face.

### Usage:
> python face_detection.py --face-cascade cascades/haarcascade_frontalface_default.xml --output output/faces/face_data.txt --input-img avatars/potter.jpg
