graph = []
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.preprocessing import image
import tensorflow as tf
from PIL import Image

ResNet50_model = ResNet50(weights='imagenet')
graph.append(tf.get_default_graph())
ResNet50_model_base = ResNet50(weights='imagenet', include_top=False)
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')
dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

model_classifier = Sequential()
model_classifier.add(GlobalAveragePooling2D(input_shape = ResNet50_model_base.output_shape[1:]))
model_classifier.add(Dense(133, activation = 'softmax'))
model_classifier.load_weights('saved_models/weights.best.Resnet50.hdf5')

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    #img = cv2.imread(img_path)
    in_memory_file = io.BytesIO()
    file.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.float64)
    color_image_flag = 1
    img = cv2.imdecode(data, color_image_flag)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0
def path_to_tensor(img_path):
    img = Image.open(img_path)
    img = np.array(img, dtype = np.float32)
    x = cv2.resize(img,(224,224))
    print(x.shape)
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    print(img)
    prediction = ResNet50_model.predict(img)
    print(prediction)
    return np.argmax(prediction)

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def extract_Resnet50(tensor):
    return ResNet50_model_base.predict(preprocess_input(tensor))

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = model_classifier.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

def classify_dog_or_human(img_path):
    result = ""
    if(dog_detector(img_path)):
        result += "There is a dog on your image. "
        breed = Resnet50_predict_breed(img_path)
        result += "Possible breed is " + breed
    elif(face_detector2(img_path)):
        result += "There is a human on your image. If this human was a dog, he would look like "
        breed = Resnet50_predict_breed(img_path)
        result += breed
    else:
        result += "This image does not contain any human or dog!"
    return result