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

def load_image(image_file):
    uploaded_image = Image.open(image_file)
    return uploaded_image



def import_and_predict(image_data, model):
    
        size = (28,28)    
        img = ImageOps.fit(image_data, size, centering=(0.5,0.5))
        img = img.convert('L')
        img = img_to_array(img)
        img = (img.astype(np.float32)/ 255.0)
        img = img.flatten().reshape(1, 28*28)
        #img_reshape = image[np.newaxis,...]
 
        prediction = model.predict(img)
        
        return prediction

<<<<<<< Updated upstream
model = tf.keras.models.load_model('128nodes.h5')
=======
model = tf.keras.models.load_model('models/128nodes.h5')
display = st.image("Images/default_pic.png", width=600)
>>>>>>> Stashed changes

st.write("""

         # Piece of Clothing Prediction

         """
         )

<<<<<<< Updated upstream
=======


>>>>>>> Stashed changes
st.write("""

         ## Simple image classification app to predict the type of clothing in an image of one piece of clothing.

         """
         )

display = st.image("Images/default_pic.png")


file = st.file_uploader("Please upload an image file of a single piece of clothing", type=["png"])



if file is None:
    st.text("You haven't uploaded an image file")
else:
    userimage = Image.open(file)
    st.image(userimage)
    prediction = import_and_predict(userimage, model)
    class1= prediction[0]
    if np.argmax(prediction) == 0:
            st.write("It is a T-shirt/top!")
    elif np.argmax(prediction) == 1:
                st.write("It is a Trouser!")
    elif np.argmax(prediction) == 2:
                st.write("It is a Pullover!")
    elif np.argmax(prediction) == 3:
                st.write("It is a Coat!")
    elif np.argmax(prediction) == 4:
                st.write("It is a Sandal!")
    elif np.argmax(prediction) == 5:
                st.write("It is a Shirt!")
    elif np.argmax(prediction) == 6:
                st.write("It is a Sneaker!")
    elif np.argmax(prediction) == 7:
                st.write("It is a Bag!")
    elif np.argmax(prediction) == 8:
                st.write("It is a Dress!")
    elif np.argmax(prediction) == 9:
                st.write("It is a Boot!")
        

    st.text("Class Probabilities:")
    st.text("0: T-shirt/top:") 
    st.write(np.round((prediction[0,0]),decimals=5))
    st.text("1: Trouser:")
    st.write(np.round((prediction[0,1]),decimals=5))
    st.text("2: Pullover:")
    st.write(np.round((prediction[0,2]),decimals=5))
    st.text("3: Coat:" )
    st.write(np.round((prediction[0,3]),decimals=5))
    st.text("4: Sandal:") 
    st.write(np.round((prediction[0,4]),decimals=5))
    st.text("5: Shirt:" )
    st.write(np.round((prediction[0,5]),decimals=5))
    st.text("6: Sneaker:")
    st.write(np.round((prediction[0,6]),decimals=5))
    st.text("7: Bag:")
    st.write(np.round((prediction[0,7]),decimals=5))
    st.text("8: Dress:") 
    st.write(np.round((prediction[0,8]),decimals=5))
    st.text("9: Boot:")
    st.write(np.round((prediction[0,9]),decimals=5))
    st.text("Raw prediction array:")
    st.write(prediction)
