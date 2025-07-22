import streamlit as st
import tensorflow as tf
import numpy as np
import streamlit.components.v1 as components

# Page Config
st.set_page_config(page_title="LeafCure - Plant Disease Detection", layout="wide")

# Sidebar Title and Navigation
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #0B3D0B, #145214);
        width: 250px !important;
        border-right: 3px solid #003300;
    }

    .sidebar-title {
        font-size: 26px;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 20px 10px 10px;
        border-bottom: 2px solid white;
    }

    .stApp {
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div class='sidebar-title'>🌿LeafCure Menu</div>", unsafe_allow_html=True)
nav_options = ["HOME", "DISEASE DETECTION", "ABOUT"]
selected_page = st.sidebar.selectbox("", nav_options, label_visibility="collapsed")
st.session_state["selected_page"] = selected_page

# Title with dynamic color
title_color = "#ffffff" if selected_page == "ABOUT" else "#006400"
st.markdown(f"<h1 style='text-align: center; color: {title_color};'>🌿 LeafCure: Plant Disease Detection Using Leaf Images</h1>", unsafe_allow_html=True)

# Background image setter
def set_bg_img(image_url):
    st.markdown(
        f"""
        <style>
        html, body, .stApp {{
            height: 100%;
            margin: 0;
            padding: 0;
            background-image: url("{image_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

# Class Labels
class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
              'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
              'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
              'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
              'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
              'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
              'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
              'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
              'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
              'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
              'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
              'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
              'Tomato___healthy']

# HOME PAGE
if selected_page == "HOME":
    st.image("oip2.jpg", use_column_width=True)
    st.markdown("""
  <div style="padding: 25px; font-family: 'Segoe UI', sans-serif; color: #ffffff; line-height: 1.7; background: radial-gradient(circle, #0f3d0f, #042a05); border-radius: 15px;">

  <h2 style="color: #ffcc00; text-align: center;">🌿 Welcome to LeafCure</h2>

  <p style="text-align: justify; font-size: 17px;">
    <strong>LeafCure</strong> is a web-based application designed to assist in the early detection of plant leaf diseases using image classification techniques.
    The tool allows users to upload an image of a plant leaf, and then utilizes a trained Convolutional Neural Network (CNN) model to analyze the image
    and predict whether the plant is healthy or infected with a disease.
  </p>

  <p style="text-align: justify; font-size: 17px;">
    With agriculture being the backbone of many economies, especially in rural regions, plant health plays a vital role in ensuring food security and productivity.
    However, timely identification of diseases can often be challenging due to a lack of resources, access to experts, or awareness. LeafCure aims to bridge this gap
    by offering a simple, fast, and accessible tool that anyone can use—from students and researchers to farmers and hobbyists.
  </p>

  <p style="text-align: justify; font-size: 17px;">
    The model behind LeafCure has been trained on a dataset containing over <strong>87,000 images</strong> of healthy and diseased plant leaves, covering 
    <strong>38 different classes</strong>. These include various plant species such as apple, corn, grape, tomato, and more. The dataset has been carefully curated and 
    structured to support effective training and testing of deep learning models. Additionally, an external set of images has been used for real-time prediction testing 
    to ensure practical usability.
  </p>

  <p style="text-align: justify; font-size: 17px;">
    LeafCure is not just a tech demo—it’s a practical solution that encourages learning, experimentation, and early intervention. It does not require 
    any advanced technical knowledge to operate. Just upload a clear photo of the affected leaf, and within seconds, you will receive a prediction showing
    the likely disease or confirming if the leaf is healthy.
  </p>

  <p style="font-size: 17px;">
    🔍 <strong>Features of LeafCure:</strong>
    <ul style="padding-left: 20px;">
      <li>🚀 Fast and intuitive interface built with Streamlit</li>
      <li>🖼️ Upload image and receive predictions instantly</li>
      <li>🌿 Supports 38 plant disease and healthy classes</li>
      <li>📊 Trained on a large and diverse image dataset</li>
      <li>🌍 Useful for students, farmers, and agri-researchers</li>
    </ul>
  </p>

  <p style="text-align: center; font-size: 16px; font-style: italic;">
    Empower your agricultural decisions with technology — one leaf at a time.
  </p>
<div style="padding: 25px; font-family: 'Segoe UI', sans-serif; color: #ffffff; line-height: 1.7; background-color: rgba(255, 255, 255, 0.05); border-radius: 15px; margin-top: 30px;">

  <h3 style="color: #00ffcc; text-align: center;">🧭 How to Use This App</h3>

  <p style="font-size: 17px;">
    Using <strong>LeafCure</strong> is easy and doesn't require any technical background. Follow these steps:
  </p>

  <p style="font-size: 17px;">
    ✅ <strong>Step 1:</strong> Click on the <em>Upload Image</em> button and select a clear photo of a plant leaf from your device.<br><br>
    🔍 <strong>Step 2:</strong> The app will process the image using a trained CNN model and analyze it for signs of disease.<br><br>
    📋 <strong>Step 3:</strong> You’ll get an instant result showing the disease name or confirmation that the leaf is healthy.
  </p>

  <p style="font-size: 17px;">
    📌 Make sure the leaf is clearly visible in the image and not blurry or dark for best results.
  </p>


</div>


    """, unsafe_allow_html=True)

# DISEASE DETECTION PAGE
elif selected_page == "DISEASE DETECTION":
    set_bg_img("https://tse3.mm.bing.net/th/id/OIP.FNemij4OmrDunwV7_OcdXAHaDe?pid=Api&P=0&h=180")
    st.markdown("<h2 style='color: black;'>🩺 Disease Detection Center</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload a leaf image (JPG/PNG)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption="🖼️ Uploaded Image", use_column_width=False, width=300)

    if st.button("🔍 Predict Disease"):
        if uploaded_file is not None:
            result_index = model_prediction(uploaded_file)
            result = class_name[result_index]
            st.success(f"✅ Model Prediction: **{result}**")
            st.balloons()
        else:
            st.warning("⚠️ Please upload an image first.")

# ABOUT PAGE
elif selected_page == "ABOUT":
    set_bg_img("https://tse1.mm.bing.net/th/id/OIP.d6J1g7pRNIEaWqdisZXDbQHaEK?pid=Api&P=0&h=180")
    about_html = """
    <div style='
        background-color: rgba(255, 255, 255, 0.88);
        padding: 2rem 3rem;
        border-radius: 15px;
        max-width: 850px;
        margin: 3rem auto;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2);
        font-family: "Segoe UI", sans-serif;
        color: black;
    '>

    <h2 style="color: black;">🌿 About LeafCure</h2>

    <h3 style="color: #ffcc00;">🔍 What Does This App Do?</h3>
   <p>
    <strong>LeafCure</strong> is an AI-powered web application that helps users identify plant diseases simply by uploading a photo of a leaf.
    Using deep learning techniques, the app analyzes the image and predicts the specific disease (or healthy status) from among
    <strong>38 different classes</strong>.
  </p>
  <p>
    The process is simple:
    <ol>
      <li>📤 Upload a leaf image (JPEG/PNG format)</li>
      <li>🧠 The trained model processes the image</li>
      <li>📈 Get instant prediction with the most likely disease name</li>
    </ol>
    Whether you're a farmer, gardener, researcher, or student, LeafCure serves as a <strong>digital plant doctor</strong> — enabling early and accurate detection.
  </p>

  <h3 style="color: #ffcc00;">🌱 Why Is This Helpful?</h3>
  <p>
    Plant diseases cause massive crop losses globally. LeafCure helps in:
    <ul>
      <li>✅ Early detection of plant diseases</li>
      <li>💰 Minimizing crop damage through timely action</li>
      <li>📷 Instant diagnosis using only an image</li>
      <li>🌍 Helping remote farmers with limited expert access</li>
      <li>🧪 Supporting agricultural research and education</li>
    </ul>
  </p>

  <h3 style="color: #ffcc00;">🧬 Dataset Description</h3>
  <p>
    LeafCure uses a rich dataset sourced from Kaggle:<br>
    🔗 <a href="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset" target="_blank" style="color: dark green;">
    New Plant Diseases Dataset by Vipoooool</a>
  </p>
  <p>
    The dataset consists of <strong>87,000+ RGB images</strong> of healthy and diseased leaves from <strong>38 plant classes</strong>.
    These images are captured in diverse real-world conditions—making them perfect for building a deep learning-based classification model.
  </p>
  <p>
    - The dataset is split in an <strong>80:20 ratio</strong> into training and validation sets, maintaining the original folder structure.<br>
    - An additional test set of <strong>33 images</strong> was created to evaluate real-time predictions within the app.
  </p>

  <h3 style="color: #ffcc00;">🧰 Technologies Used</h3>

  <h4 style="color: black;">💻 Frontend</h4>
  <ul>
    <li>Streamlit – for building the web interface</li>
    <li>Custom CSS – for styling, layout, fonts, and background</li>
  </ul>

  <h4 style="color: black;">🧠 Machine Learning</h4>
  <ul>
    <li>TensorFlow 2.10.0 – for training and inference</li>
    <li>OpenCV – for image preprocessing</li>
    <li>NumPy, Pandas – for data handling</li>
    <li>Matplotlib, Seaborn – for visualization</li>
  </ul>

  <h4 style="color: black;">📦 Deployment</h4>
  <ul>
    <li>Python Virtual Environment (.venv)</li>
    <li>requirements.txt – for managing dependencies</li>
    <li>Cloud-ready – suitable for Render, Hugging Face Spaces, and more</li>
  </ul>

</div>


    """
    components.html(about_html, height=600, scrolling=True)
