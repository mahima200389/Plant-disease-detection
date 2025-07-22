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

st.sidebar.markdown("<div class='sidebar-title'>🌿 LeafCure Menu</div>", unsafe_allow_html=True)
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
        <div style='text-align: center; font-size: 20px; color: #333; margin-top: 30px;'>
            Welcome to <b>LeafCure</b> - your AI-based assistant for early plant disease detection. Upload a leaf image and let our model do the rest!
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
        <h2 style='color: #2E8B57; text-align: center; text-transform: uppercase; margin-bottom: 2rem;'>📘 About This App</h2>
        <h3 style='color: #2E8B57;'>🔍 Purpose</h3>
        <ul>
            <li>Assist farmers in early disease detection</li>
            <li>Promote sustainable agriculture using AI</li>
        </ul>
        <h3 style='color: #2E8B57;'>🧠 Model Details</h3>
        <ul>
            <li>CNN trained on 87,000+ images from the PlantVillage dataset</li>
            <li>Accuracy: ~95%</li>
            <li>Input image size: 128x128 pixels</li>
        </ul>
        <h3 style='color: #2E8B57;'>👩‍💻 Developer</h3>
        <p><b>Mahima Sharma</b><br>Final Year, B.Tech (CSE)<br>ABES Engineering College</p>
    </div>
    """
    components.html(about_html, height=600, scrolling=True)
