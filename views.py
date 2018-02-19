from werkzeug.utils import secure_filename
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
from flask import render_template, request
import classifier
from classifier import graph
from app import app

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/DogBreedClassifier/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        breed = ''
        with graph[0].as_default():
            breed = classifier.classify_dog_or_human(file)
        return breed
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS