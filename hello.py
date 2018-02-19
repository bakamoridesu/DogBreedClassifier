import os
import numpy as np
from io import BytesIO
from keras.preprocessing import image
import cv2
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
from flask import Flask, render_template
import main
from main import graph
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        print(request.files['file'])
        file = request.files['file']
        filename = secure_filename(file.filename)
        #img = PIL.Image.open(file)
        breed = ''
        with graph[0].as_default():
            breed = main.classify_dog_or_human(file)
        return breed
    return "Error"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
if __name__ == '__main__':
    app.run(debug = True)