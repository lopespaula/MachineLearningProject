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



#st.set_option('deprecation.showfileUploaderEncoding', False)

#streamlit labels
st.title("Clothing classification App")
# st.sidebar.subheader("File uploader")

#streamlit functionsS
image = Image.open('Images/default_pic.png')
show = st.image(image, use_column_width=True)

#declaring uploaded/user file variable
# uploaded_file = st.sidebar.file_uploader(label="Upload your .PNG file.",
#                     type = ['png'])
                    
# if uploaded_file is not None:
    
#     user_image = Image.open(uploaded_file)
#     show.image(user_image, 'Uploaded Image', use_column_width=True)
#     # We preprocess the image to fit in algorithm.
#     image = np.asarray(user_image)/255
    
#     my_image= resize(image, (64,64)).reshape((1, 64*64*3)).T

# st.sidebar.button("Click Here to Classify")

def import_and_predict(image_data, model):
    
        size = (28,28)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        my_image = image[np.newaxis,...]

        prediction = model.predict(my_image)
        
        return prediction

# Load the model
model = tf.keras.models.load_model('mnist_trained.h5')

#file variable
file = st.file_uploader("Please upload an image file", type=["png"])


if file is None:
    st.text("You haven't uploaded an image file")
else:
    

    # image_size = (28, 28)
    # im = image.load_img(file, target_size=image_size, color_mode="grayscale")
    
    # image = img_to_array(im)
    # image /= 255

    # # Flatten into a 1x28*28 array 
    # img = image.flatten().reshape(-1, 28*28)
    # img = 1 - img

    # array = model.predict_classes(img)
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
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
    else:
        st.write("It is a Ankle boot!")
    
    st.text("Probability")
    st.write(prediction)



#['T-shirt/top',
#  'Trouser',
#  'Pullover',
#  'Dress',
#  'Coat',
#  'Sandal',
#  'Shirt',
#  'Sneaker',
#  'Bag',
#  'Ankle boot']