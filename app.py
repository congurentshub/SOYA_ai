import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
import keras

# This tells Keras how to handle the new Dense layer format on an old system
if hasattr(keras.layers, 'Dense'):
    original_dense_init = keras.layers.Dense.__init__
    def patched_dense_init(self, *args, **kwargs):
        kwargs.pop('quantization_config', None) # Remove the troublemaker
        return original_dense_init(self, *args, **kwargs)
    keras.layers.Dense.__init__ = patched_dense_init
# --- 1. DATABASE SETUP ---
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS records 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  date TEXT, disease TEXT, confidence REAL)''')
    conn.commit()
    return conn

# --- 2. MODEL UTILITIES ---
def se_block(input_tensor, ratio=8):
    filters = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu')(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid')(se)
    return tf.keras.layers.Multiply()([input_tensor, se])

@st.cache_resource
def load_resources():
    with open("model_config.json", "r") as f:
        config = json.load(f)
    model = load_model("best_model.keras", custom_objects={'se_block': se_block})
    return model, config

model, config = load_resources()

# --- 3. PREPROCESSING ---
def prepare_image(image):
    img = np.array(image.convert('RGB'))
    img_uint8 = img.astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img_clahe = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    img_resized = cv2.resize(img_clahe, tuple(config['input_specs']['image_size']))
    img_array = np.expand_dims(img_resized, axis=0)
    return preprocess_input(img_array.astype(np.float32))

# --- 4. APP INTERFACE ---
st.set_page_config(page_title="Soybean Health Monitor", layout="wide")
st.title("🌱 Soybean Disease Classifier & Records")

tab1, tab2 = st.tabs(["🔍 New Prediction", "📊 Records & Analytics"])

with tab1:
    st.header("Upload Leaf Image")
    uploaded_file = st.file_uploader("Upload JPG/PNG", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, width=300)
        
        if st.button("Analyze Image"):
            with st.spinner('Running AI Model...'):
                processed = prepare_image(img)
                preds = model.predict(processed)
                idx = str(np.argmax(preds))
                conf = np.max(preds) * 100
                label = config['classes'][idx]
                
                st.success(f"Result: **{label.replace('_',' ').title()}** ({conf:.1f}%)")
                
                # Save to Database
                conn = init_db()
                curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                conn.execute("INSERT INTO records (date, disease, confidence) VALUES (?, ?, ?)", 
                             (curr_time, label, float(conf)))
                conn.commit()
                conn.close()
                st.info("Record saved to database.")

with tab2:
    st.header("Historical Data")
    conn = init_db()
    df = pd.read_sql_query("SELECT * FROM records", conn)
    conn.close()

    if not df.empty:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Recent Records")
            st.dataframe(df.tail(10), use_container_width=True)
            
        with col2:
            st.subheader("Disease Distribution")
            counts = df['disease'].value_counts()
            fig, ax = plt.subplots()
            counts.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_ylabel("Number of Cases")
            plt.xticks(rotation=45)
            st.pyplot(fig)
    else:
        st.write("No records found yet. Run a prediction first!")