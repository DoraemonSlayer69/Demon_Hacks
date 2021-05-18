from flask import Flask, request, render_template
import os
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import cv2
from keras.preprocessing import image
from PIL import Image
from matplotlib import pyplot
from numpy import asarray

app = Flask(__name__)
app.config["IMAGE_UPLOADS"] = "D:/Personal/DemonHacks"

@app.route('/')
def my_form():
    return render_template('Image_form.html')

@app.route('/upload-image', methods=['POST'])
def my_form_post():
    img = request.files['image']
    img.save(os.path.join(app.config["IMAGE_UPLOADS"], img.filename))
    #Get Face data
    face = extract_face(img.filename)
    face = image.img_to_array(face)
    face = face.reshape((1,) + face.shape)
    network = load_model("GenderDetection_Weights")
    gender_predict = network.predict(face)
    if gender_predict[0][0] < 0.5:
        print(gender_predict[0][0])
        return "Female Gender Detected"
    else:
        print(gender_predict[0][0])
        return "Male Gender Detected"
    #processed_text = text.upper()
    #return processed_text


   
   
def extract_face(filename, required_size=(150, 150)):
	# load image from file
	pixels = cv2.imread(filename)
	# create the detector, using default weights
	detector = MTCNN()
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array


if __name__ == '__main__':
   app.run()