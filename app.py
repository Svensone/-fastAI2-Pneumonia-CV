
# from fastai.learner import load_learner
from fastai.basics import *
from fastai.vision.all import *
import torchvision.transforms as T
import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.image as mpimg
import requests
from io import BytesIO

import pathlib
from pathlib import Path

import base64
from PIL import Image
import PIL.Image

## get local css
##################
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
local_css("style.css")

# Set Background Image *local file"
###################################################
@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
def set_png_as_page_bg(png_file):
        bin_str = get_base64_of_bin_file(png_file)
        page_bg_img = '''
        <style>
        body {
        background-image: url("data:image/png;base64,%s");
        background-size: cover;
        }
        </style>
        ''' % bin_str
        st.markdown(page_bg_img, unsafe_allow_html=True)
        return
set_png_as_page_bg('assets/bg2.jpg')
# adjustment for different systems (share.io PosixPath)
################################
# Option 1: when working on localhost:8501
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# option 2: for when deploying on share.streamlit.io
plt = platform.system()
if plt == 'Linux': 
    pathlib.WindowsPath = pathlib.PosixPath
else:
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
# pathlib.PosixPath = temp

## Layout App
##################
    ### Header
    ##############
st.title('Pneumonia lung x-ray - Classification')
st.markdown("""
## AI- Computer Vision with **fastai/pytorch**
#### Classifing lung scans if pneumonia or healthy 
**Dataset**: 5.000+ Images of lung x-rays, **Accuracy** on ResNet34 Architectur: 67%""")
link1 = 'Model & Data Preprocessing [Github](https://github.com/Svensone/-fastAI2-Pneumonia-CV/blob/master/2021_03_29_%5BfastaiV2%5D_CV_mulitDatasets_Pneumonia_Kaggle.ipynb)'
link2 = 'Deployment [Github](https://github.com/Svensone/-fastAI2-Pneumonia-CV/tree/master)'
st.markdown(link1, unsafe_allow_html=True)
st.markdown(link2, unsafe_allow_html=True)

    ### Test
    ##############
st.markdown("""
## Test the Model yourself
""")
#######################################
### Image Classification
#######################################
def prediction(img, display_img):
    # display the image
    st.image(display_img, use_column_width=True)
    # loading spinner
    with st.spinner('Wait a second .....'):
        time.sleep(3)

    #  load Learner
    learn = load_learner("models/export.pkl")
    
    # Prediction on Image
    predict_class = learn.predict(img)[0]
    predict_proba = learn.predict(img)[2]

    print(f'Prediction Class : {predict_class} and probability {predict_proba}')

    proba = float(predict_proba[1]) if str(predict_class) == 'PNEUMONIA' else float(predict_proba[0])
    proba = round((proba * 100), 2)
    # print(f'predict class is {predict_class} with probab {proba}')

    # Display results
    if str(predict_class) == 'PNEUMONIA':
        st.success(f'This is a scan of a healthy lung. Probabiltiy of Prediction: {proba} %')
        # st.write()
    else:
        st.success(f'Pneumonia infection - please find a doctor! Probabiltiy of Prediction: {proba}%')

#######################################
### Image Selection
#######################################
option1= 'Choose a test image from list'
option2= 'Predict your own Image'

option = st.radio('', [option1, option2 ])

if option == option1:
    # Select an image
    list_test_img = os.listdir('test_images')
    test_img = st.selectbox(
        'Please select an image:', list_test_img)
    # Read the image
    test_img = test_img
    file_path = 'test_images/'+ test_img
    img = PILImage.create(file_path)
    # print(img)
    ##### TEST
    ################
    im_test3 = PIL.Image.open(file_path)
    display_img = np.asarray(im_test3) # Image to display
    # print(img)
    # call predict func with this img as parameters
    prediction(img, display_img)

## Predition from URL Image not yet working - converting to fastAI Image object error
##################################################
else:
    url = st.text_input('URL of the image')
    if url !='':
        # print(url)
        try:
        # Read image from the url
            response = requests.get(url)
            pil_img = PIL.Image.open(BytesIO(response.content))
            display_img = np.asarray(pil_img) # Image to display
            # Transform the image
            timg = TensorImage(image2tensor(pil_img))
            tpil = PILImage.create(timg)
            print(tpil)
            # call predict func
            prediction(tpil, display_img)
        except:
            st.text("Invalid URL")

link3 = '[Return to Portfolio](https://portfolio-sven.netlify.app/)'
st.markdown(link3, unsafe_allow_html=True)
