import streamlit as st
from PIL import Image
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('model1_cifar_10epoch.h5')

# Dictionary to label all the CIFAR-10 dataset classes
classes = {
    0: 'aeroplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

# Streamlit app
st.title("CIFAR-10 Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type="jpeg")

if uploaded_file is not None:
    # Load and display the image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Preprocess the image
    image = image.resize((32, 32))
    image = np.expand_dims(image, axis=0)
    image = np.array(image) / 255.0  # Normalize if required

    # Predict the class
    predictions = model.predict(image)
    pred = np.argmax(predictions, axis=-1)[0]
    label = classes[pred]

    # Display the result
    st.write(f"Prediction: {label}")

