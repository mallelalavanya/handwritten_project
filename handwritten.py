import os
import streamlit as st
import numpy as np
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import pandas as pd

# Path to the trained model
MODEL_PATH = "my_model.keras"

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    st.error("🚨 Model file not found! Please upload 'my_model.keras' to the repository.")
    st.stop()

# Try loading the model with error handling
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Function to preprocess the drawn image
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to (28x28)
    image = ImageOps.invert(image)  # Invert colors (black on white)
    image = np.array(image) / 255.0  # Normalize pixel values to [0,1]
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (28,28,1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1,28,28,1)
    return image

# Streamlit UI
st.title("🖊 Handwritten Digit Recognition")
st.write("Draw a digit (0-9) below and get predictions!")

# Sidebar for instructions
st.sidebar.title("🔍 About")
st.sidebar.write("This app uses a deep learning model to recognize handwritten digits (0-9).")
st.sidebar.write("Simply draw a number below and get a prediction in real time!")

st.sidebar.title("⚙️ How to Use")
st.sidebar.write("1️. Draw a digit (0-9) using the white pen.")  
st.sidebar.write("2️. Draw clearly in the middle of the box.")  
st.sidebar.write("3️. Click the delete symbol to erase and try again.")  

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 **Developed by:** Lavanya Mallela")  # Change this if needed

# Create a drawing canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=8,  # Adjusted stroke width for better clarity
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Process the drawn image
if canvas_result.image_data is not None:
    img = Image.fromarray(canvas_result.image_data[:, :, 0:3].astype("uint8"))

    # If user has drawn something, process it
    if img.getbbox():  # Check if the image is not empty
        img = ImageOps.invert(img)  # Invert colors for better model recognition
        st.image(img, caption="Processed Image", use_container_width=True)

        # Preprocess and predict
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        predicted_digit = np.argmax(prediction)

        st.write(f"🎯 **Predicted Digit:** {predicted_digit}")

        # Display confidence scores
        confidence_df = pd.DataFrame({"Digit": list(range(10)), "Confidence": prediction[0]})
        st.bar_chart(confidence_df.set_index("Digit"))

    else:
        st.warning("✏️ Please draw a digit before predicting.")
