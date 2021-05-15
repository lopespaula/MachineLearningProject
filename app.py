import streamlit as st
import plotly_express as px
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from skimage.transform import resize
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import joblib

def import_and_predict(image_data, model):
    
        size = (28,28)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32)/ 255.0)
        image = image.flatten().reshape(-1, 28*28)

 
        prediction = model.predict(image)
        
        return prediction

model = tf.keras.models.load_model('trained_model.h5')

st.write("""
         # Piece of Clothing Prediction
         """
         )

st.image("Images/default_pic.png")


st.write("Simple image classification app to predict the type of clothing in an image of one piece of clothing.")

file = st.file_uploader("Please upload an image file of a single piece of clothing", type=["jpg", "png"])



if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)

    prediction = import_and_predict(image, model)


    if np.argmax(prediction[0]) == 0:
        st.write("It is a T-shirt/top!")
    elif np.argmax(prediction[0]) == 1:
        st.write("It is a Trouser!")
    elif np.argmax(prediction[0]) == 2:
        st.write("It is a Pullover!")
    elif np.argmax(prediction[0]) == 3:
        st.write("It is a Coat!")
    elif np.argmax(prediction[0]) == 4:
        st.write("It is a Sandal!")
    elif np.argmax(prediction[0]) == 5:
        st.write("It is a Shirt!")
    elif np.argmax(prediction[0]) == 6:
        st.write("It is a Sneaker!")
    elif np.argmax(prediction[0]) == 7:
        st.write("It is a Bag!")
    elif np.argmax(prediction[0]) == 8:
        st.write("It is a Dress!")
    else:
        st.write("It is a Ankle boot!")
    
   
