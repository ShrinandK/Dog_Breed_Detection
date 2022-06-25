# import all required dependencies
import tensorflow as tf
import cv2
import streamlit as st
from PIL import Image
import numpy as np

# function to run model on user provided image
def analyseImage(efficientnetModel,imgPath,classNames):
    # converting the image uploaded by user to numpy array for
    # opencv to read properly
    readImage = np.array(imgPath.convert('RGB'))
    # resizing and expanding dimension for model to predict image
    resizedImg = cv2.resize(readImage,(224,224))
    expandImage = tf.expand_dims(resizedImg,axis=0)
    index = efficientnetModel.predict(expandImage)
    # finding the probability having more than 50%
    findProb = index > 0.5
    # flattening and conversion to list for finding the dog breed
    tempData = findProb.flatten()
    probList = tempData.tolist()
    try:
        return 'The image is a ' + classNames[probList.index(True)]
    except ValueError:
        return 'Sorry Could not recognise image. Please give only the images as displayed above'

# drawing image in columns of streamlit
def drawCols(col,pathString,headerString,capString):
    with col:
        st.header(headerString)
        st.image(pathString,caption = capString)

# loading model and classes
efficientnetModel = tf.keras.models.load_model('effModel')
classNames = ['French Bulldog', 'Husky', 'Malamute', 'Boston Terrier']

# setting some simple streamlit messages
st.set_page_config(layout="wide")
st.title('Dog Breed Detection')
st.subheader('This Web App detects one of the below dog breeds:')

# setting some column
colLength = 4
col0,col1,col2,col3 = st.columns(colLength)

# setting image paths for app
allDogPath = {'Boston Terrier':'Images/For_app/BT129.jpg',
              'French Bulldog':'Images/For_app/FB1.jpg',
              'Husky':'Images/For_app/HD246.jpg',
              'Malamute':'Images/For_app/MD6.jpg'}

# setting column for each breed
allDogColumns = {'Boston Terrier':col0,
              'French Bulldog':col1,
              'Husky':col2,
              'Malamute':col3}

# copyright to their owner
allDogCaptions = {'Boston Terrier': 'Copyright: https://www.thegoodypet.com/how-much-does-a-boston-terrier-cost',
              'French Bulldog':'Copyright: https://thichthucung.com/',
              'Husky':'Copyright: https://siberiianblog.tumblr.com/post/188834198272/officialhuskylovers',
              'Malamute':'Copyright: ©liliya kulianionak - stock.adobe.com'}

# assigning each column data
for key in allDogPath.keys():
    drawCols(allDogColumns[key],allDogPath[key],key,allDogCaptions[key])

# user upload image
with st.form("my-form", clear_on_submit=True):
    uploadedFile = st.file_uploader("Choose a file", type=['jpg','png','jpeg'])
    submitted = st.form_submit_button("UPLOAD!")

    # when submitted run the image for predicting the dog breed
    if submitted and uploadedFile is not None:
        imgPath = Image.open(uploadedFile)
        returnString = analyseImage(efficientnetModel, imgPath, classNames)
        st.image(uploadedFile)
        st.title(returnString)