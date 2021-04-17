
import sys
import glob
import re
import tensorflow as tf
# from gevent.pywsgi import WSGIServer

# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config=ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction=0.2
# config.gpu_options.allow_growth=True
# session= InteractiveSession(config=config)

import numpy as np
import os
from flask import send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from flask import Flask,redirect, url_for, request, render_template
from flask_cors import CORS, cross_origin

app= Flask(__name__)


dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER ='Dataset/Train'
STATIC_FOLDER ='static'

#load model
model=load_model('malaria_vgg19.h5')

#call model to predict an image
def model_predict(full_path):
    img = image.load_img(full_path, target_size=(224, 224))

    # Preprocessing the image
    img = np.expand_dims(img, axis=0)

    ## Scaling
    img = img * 1.0 / 255
    predicted = model.predict(img)
    return predicted


@app.route("/")
@cross_origin()
def Home():
    return render_template("home.html")

#procesing uploaded file and predict it
@app.route('/upload', methods=['GET', 'POST'])
@cross_origin()
def upload_file():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            file=request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {0: 'PARASITIC (INFECTED CELL)', 1: 'Uninfected'}
            result = model_predict(full_name)
            print(result)
            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]
            return render_template('result.html', image_file_name=file.filename, label=label, accuracy=accuracy)
        except:
            # flash("Please select the image first !!")
            return render_template('home.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)







if __name__=='__main__':
    app.run(debug=True)