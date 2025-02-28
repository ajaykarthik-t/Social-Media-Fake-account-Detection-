import streamlit as st
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image
import os

# Load the trained model
model = load_model('my_model.keras')

# Define class names
class_names = ['Fake', 'Real']

# Streamlit UI
st.title("📸 Fake Social Media Account Detector")
st.write("Upload an image of a social media account to check if it's **Fake** or **Real**.")

# Username input
username = st.text_input("Enter the username associated with the account:")

# File uploader
uploaded_file = st.file_uploader("Upload an account screenshot", type=["png", "jpg", "jpeg"])

if uploaded_file and username:
    # Display uploaded image
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image_pil = image_pil.resize((224, 224))
    img_array = np.array(image_pil) / 255.0  # Normalize
    img_array = img_array.reshape(1, 224, 224, 3)
    
    # Predict the label
    label = model.predict(img_array)
    predicted_class_index = np.argmax(label)
    predicted_class = class_names[predicted_class_index]
    
    # Show result
    st.subheader("🔍 Prediction Result")
    if predicted_class == "Fake":
        st.markdown("<h2 style='color:red;'>🚨 Fake Account Detected!</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h2 style='color:green;'>✅ This Account Seems Legit!</h2>", unsafe_allow_html=True)
    
    # Display image with prediction
    fig, ax = plt.subplots()
    ax.imshow(image_pil)
    ax.set_title(predicted_class)
    ax.axis("off")
    st.pyplot(fig)
    
    # Save results to CSV
    csv_file = "prediction_results.csv"
    result_data = pd.DataFrame([[username, predicted_class]], columns=["Username", "Prediction"])
    
    if os.path.exists(csv_file):
        existing_data = pd.read_csv(csv_file)
        result_data = pd.concat([existing_data, result_data], ignore_index=True)
    
    result_data.to_csv(csv_file, index=False)
    st.success("✅ Prediction saved successfully!")
