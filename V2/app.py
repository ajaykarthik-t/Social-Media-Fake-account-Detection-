import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
import os

# Load both models
account_model = joblib.load('fake_account_detector.joblib')
image_model = load_model('my_model.keras')

# Load account dataset
df = pd.read_csv('fake_accounts.csv')
image_class_names = ['Deepfake', 'Real']

# Configure page
st.set_page_config(page_title="Advanced Fake Account Detector", layout="centered")

# Custom styling
st.markdown("""
    <style>
        .stTextInput input, .stFileUploader > div {
            border-radius: 20px;
            padding: 10px;
        }
        .stButton>button {
            background: #4CAF50;
            color: white;
            border-radius: 20px;
            padding: 10px 24px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background: #45a049;
            transform: scale(1.05);
        }
        .result-box {
            padding: 20px;
            border-radius: 15px;
            margin: 20px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header-font { font-size: 1.8rem !important; }
    </style>
""", unsafe_allow_html=True)

# App interface
st.title("🕵️ Advanced Fake Account Detector")
st.markdown("<div class='header-font'>Analyze accounts using both profile data and image analysis</div>", 
            unsafe_allow_html=True)

# Input section
col1, col2 = st.columns(2)
with col1:
    username = st.text_input("Enter username:")
with col2:
    uploaded_file = st.file_uploader("Upload profile image", type=["png", "jpg", "jpeg"])

if st.button("🔍 Analyze Account"):
    if not username or not uploaded_file:
        st.error("Please provide both username and image!")
    else:
        # Initialize results
        account_result = None
        image_result = None
        
        # Account data analysis
        account_data = df[df['Username'] == username]
        if not account_data.empty:
            features = account_data.drop(['Username', 'Label'], axis=1)
            features['Profile Picture Present (Yes/No)'] = features['Profile Picture Present (Yes/No)'].map({'Yes': 1, 'No': 0})
            account_pred = account_model.predict(features)[0]
            account_result = "Real" if account_pred == 0 else "Fake"
        
        # Image analysis
        try:
            image_pil = Image.open(uploaded_file).convert('RGB')
            st.image(image_pil, caption="Uploaded Profile Image", use_column_width=True)
            
            # Preprocess image
            img_array = np.array(image_pil.resize((224, 224))) / 255.0
            img_array = img_array.reshape(1, 224, 224, 3)
            
            # Predict image
            image_pred = image_model.predict(img_array)
            image_result = image_class_names[np.argmax(image_pred)]
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
        
        # Display results
        st.markdown("---")
        st.subheader("Analysis Results")
        
        results_col = st.columns(2)
        
        # Account result
        with results_col[0]:
            st.markdown("<h3 style='text-align: center;'>Account Data Analysis</h3>", 
                        unsafe_allow_html=True)
            if account_result:
                if account_result == "Real":
                    st.markdown("""
                        <div class='result-box' style='background: #e8f5e9;'>
                            <h2 style='color: #4CAF50; text-align: center;'>✅ Genuine Account</h2>
                            <p style='text-align: center;'>Profile data appears legitimate</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class='result-box' style='background: #ffebee;'>
                            <h2 style='color: #f44336; text-align: center;'>❌ Fake Account Detected</h2>
                            <p style='text-align: center;'>Suspicious profile patterns found</p>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='result-box' style='background: #fff3e0;'>
                        <h2 style='color: #ff9800; text-align: center;'>⚠️ Unknown Account</h2>
                        <p style='text-align: center;'>Username not in our database</p>
                    </div>
                """, unsafe_allow_html=True)
        
        # Image result
        with results_col[1]:
            st.markdown("<h3 style='text-align: center;'>Image Analysis</h3>", 
                        unsafe_allow_html=True)
            if image_result:
                if image_result == "Real":
                    st.markdown("""
                        <div class='result-box' style='background: #e8f5e9;'>
                            <h2 style='color: #4CAF50; text-align: center;'>✅ Authentic Image</h2>
                            <p style='text-align: center;'>No deepfake detected</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div class='result-box' style='background: #ffebee;'>
                            <h2 style='color: #f44336; text-align: center;'>❌ Deepfake Detected</h2>
                            <p style='text-align: center;'>Potential AI-generated image</p>
                        </div>
                    """, unsafe_allow_html=True)
        
        # Save results
        if account_result and image_result:
            result_df = pd.DataFrame([{
                'Username': username,
                'Account_Status': account_result,
                'Image_Status': image_result,
                'Timestamp': pd.Timestamp.now()
            }])
            
            if os.path.exists('analysis_history.csv'):
                history_df = pd.read_csv('analysis_history.csv')
                result_df = pd.concat([history_df, result_df])
            
            result_df.to_csv('analysis_history.csv', index=False)
            st.success("Analysis results saved to history")