from flask import Flask,request,redirect,url_for,render_template
import camera
import sys,os,glob,re
import numpy as np
import requests
import json
import feat
from keras.preprocessing import image
import cv2
from keras.models import load_model
from keras.applications.imagenet_utils import preprocess_input,decode_predictions
from werkzeug.utils import secure_filename
from keras.models import model_from_json
import keras
from keras.layers import *
from keras.models import Model , load_model
from keras.preprocessing import image
from keras.utils import np_utils
from keras.models import model_from_json
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
import matplotlib.pyplot as plt
app = Flask(__name__,static_url_path="",static_folder="templates")

app = Flask(__name__)

model = load_model('mode.h5')
# model._make_predict_function()

def model_predict(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x,axis=0)
    x = preprocess_input(x,mode='caffe')
    frame = cv2.imread(img_path)
    test_data = cv2.resize(frame, (224, 224))

    test_data = np.array(test_data)
    test_data.shape = (1, 224, 224, 3)

    zz = model.predict(test_data)
    print(zz[0][0])

    if zz[0][0] < 0.24:
        pred = "COVID 19 POSITIVE"
    else:
        pred = "COVID 19 NEGATIVE"
        continous_viol = 0

    print("Prediction ---------->", pred)

    return pred

# model_path = 'model.h5'
# model = load_model(model_path)
# model._make_predict_function()
#
# def model_predict(img_path,model):
#     img = image.load_img(img_path,target_size=(224,224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x,axis=0)
#     x = preprocess_input(x,mode='caffe')
#     preds = model.predict(x)
#     return preds

@app.route('/')
def hello_world():
    data = requests.get("https://disease.sh/v2/countries/India?yesterday=true&strict=true")
    data_dict = data.json()
    var = "hello"
    return render_template('index.html',data = data_dict)

@app.route('/detect')
def detect():
    os.system('python get_pulse.py')
    return render_template('index.html')

@app.route('/xray')
def xray():
    os.system('python X_Ray_Detection.py')
    return render_template('index.html')



@app.route('/analyze', methods=["POST"])
def analyze():

    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname('uploads')
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        print("In the analyze funciton")
        f.save(file_path)
        print("going into model predict")
        preds = model_predict(file_path,model)
        print(preds)

        # pred_class = decode_predictions(preds, top=1)
        # result = str(pred_class[0][0][1])
        # # return result
        #
        # return json.dumps({"image": result})
    return None


if __name__ == '__main__':
    app.run()
