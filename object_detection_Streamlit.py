import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

model=tf.keras.models.load_model(r"C:\Users\haider\Desktop\New folder (3)\Object_Detection_NNN.keras")

st.title("Object Detection")
input_image=st.file_uploader("Enter the image")

if st.button("Detect Object"):
    input_image=Image.open(input_image)
    input_image=input_image.resize((32,32))
    input_image=np.array(input_image)
    input_image=input_image/255.0
    input_image=input_image.reshape(1,32,32,3)
    pred=model.predict(input_image)
    pre=pred.argmax()
    list_of_object={0: 'Airplane', 1: 'Automobile', 2: 'Bird', 3: 'Cat', 4: 'Deer', 5: 'Dog', 6: 'Frog', 7: 'Horse', 8: 'Ship', 9: 'Truck'}
    st.success(f"This image contain: {list_of_object[pre]}")