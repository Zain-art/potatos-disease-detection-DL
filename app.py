import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os 
# Load the trained model
model = tf.keras.models.load_model('potatos_model.h5')
# Recompile the model (only if needed)
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']  # Add other metrics if needed
)
CLASS_NAMES = ["Healthy", "Early Blight", "Late Blight"]  # Update with your class names

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess the image for model prediction."""
    image = image.resize((256, 256))  # Resize to match model's input shape
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    print('images how ',image)
    return image


def list_images(directory: str):
    """List all image files in a directory."""
    return [f for f in os.listdir(directory) if f.endswith(('jpg', 'jpeg', 'png'))]
# Streamlit app
st.title("Potato Disease Prediction ❤️😊😁")
#  Define the directory containing images
image_dir = 'dataset/'  # Replace with your image folder path

# List all images in the directory
image_files = list_images(image_dir)
st.write("Upload a potato leaf image to predict the disease:")

uploaded_file = st.file_uploader("Choose an image...")

if uploaded_file is not None:
    try:
        # Read and process the image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Results Potatos Disease Detection..😊😊")

        # Preprocess and predict
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])

        st.write(f"Prediction: {predicted_class}")
        st.write(f"Confidence: {confidence:.2f}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    
 



# ===========================================
# import streamlit as st
# from PIL import Image
# import os
# import numpy as np
# import tensorflow as tf

# # Load the trained model
# model = tf.keras.models.load_model('potatos_model1.h5')
# CLASS_NAMES = ["Healthy", "Early Blight", "Late Blight"]  # Update with your class names

# def preprocess_image(image: Image.Image) -> np.ndarray:
#     """Preprocess the image for model prediction."""
#     image = image.resize((256, 256))  # Resize to match model's input shape
#     image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# def list_images(directory: str):
#     """List all image files in a directory."""
#     try:
#             files = os.listdir(directory)
#             image_files = [f for f in files if f.lower().endswith(('jpg', 'jpeg', 'png'))]
#             if not image_files:
#                 st.write("No image files found in the directory.")
#             return image_files
#     except Exception as e:
#             st.error(f"Error accessing directory: {e}")
#             return []

# # Streamlit app
# st.title("Potato Disease Prediction")

# # Define the directory containing images
# image_dir = r'D:\\computer vision practice code\\all projects ml\\potatos disease detector DLp\dataset'  # Replace with your image folder path

# # List all images in the directory
# image_files = list_images(image_dir)

# if image_files:
#     st.write("Select an image from the folder:")
#     selected_image = st.selectbox("Choose an image", image_files)

#     if selected_image:
#         # Load and display the selected image
#         image_path = os.path.join(image_dir, selected_image)
#         image = Image.open(image_path)
#         st.image(image, caption='Selected Image', use_column_width=True)

#         # Predict and display results
#         st.write("Classifying...")
#         processed_image = preprocess_image(image)
#         predictions = model.predict(processed_image)
#         predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#         confidence = np.max(predictions[0])

#         st.write(f"Prediction: {predicted_class}")
#         st.write(f"Confidence: {confidence}")
# else:
#     st.write("No images found in the specified folder.")
