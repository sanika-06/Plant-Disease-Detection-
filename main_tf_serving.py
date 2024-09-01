import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

from flask import Flask, render_template, request
import os
import pandas as pd
from keras.preprocessing.image import img_to_array , load_img
from keras.applications.vgg19 import  preprocess_input

supplement_info = pd.read_csv('supplement_info_1.csv', encoding='cp1252')
disease_info = pd.read_csv("disease_info_final.csv" , encoding='cp1252')

def prediction(path):
    img = load_img(path, target_size=(256, 256))
    i = img_to_array(img)
    im = preprocess_input(i)
    img = np.expand_dims(im, axis=0)
    index = np.argmax(model.predict(img))
    return index

app = Flask(__name__, template_folder='template', static_folder='static')

endpoint = "http://localhost:8509/v1/models/potatoes_model:predict"
model = tf.keras.models.load_model("../final_models/8")

CLASS_NAMES = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Mango__Diseased',
 'Mango__Healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.route('/')
def home_page():
    return render_template("index.html")

@app.route('/shop')
def shop():
    return render_template("Shop.html")

@app.route('/AIEngine')
def ai_engine_page():
    return render_template("AIEngine.html")

@app.route('/ContactUS')
def contact():
    return render_template("Contact.html")

@app.route('/info')
def more_info():
    return render_template("More_Info.html")

@app.route('/About')
def about():
    return render_template("About.html")


@app.route('/result', methods=['GET', 'POST'])
def result():
    if request.method == 'POST':
        image = request.files['image']

        filename = image.filename
        file_path = os.path.join('static/uploads', filename)
        image.save(file_path)
        print(file_path)
        pred = prediction(file_path)
        title = disease_info['disease_name'][pred]
        description = disease_info['description'][pred]
        prevent = disease_info['Possible Steps'][pred]
        image_url = disease_info['image_url'][pred]
        supplement_name = supplement_info['supplement name'][pred]
        supplement_image_url = supplement_info['supplement image'][pred]
        supplement_buy_link = supplement_info['buy link'][pred]
        return render_template('Result.html', title=title, desc=description, prevent=prevent,
                                image_url = image_url, pred=pred, sname=supplement_name, simage=supplement_image_url,
                               buy_link=supplement_buy_link)



if __name__ == "__main__":
    app.run(debug=True)
    app.run(host='127.0.0.1', port=5500)

