import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Constants and Paths
FOLDERS = ['train', 'test', 'val']
DIR_INPUT = r'C:\Users\sivak\Downloads\pneumonia\chest_xray'
DIR_WORKING = './'
CLASS_LIST = ['normal', 'bacteria', 'virus']
MODEL_FILENAME = 'my_cnn_model.h5'
DIR_MODELS = './models'
CLASS_LIST = ['normal', 'bacteria', 'virus']
# Load the CNN model
@st.cache(allow_output_mutation=True)


@st.cache(allow_output_mutation=True)
def load_cnn_model():
    model_path = os.path.join(DIR_MODELS, MODEL_FILENAME)
    return load_model(model_path)


# Function to preprocess the image
def preprocess_image(img):
    img = image.load_img(img, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.
    return img

# Function to predict class
def predict_class(model, img):
    img = preprocess_image(img)
    prediction = model.predict(img)
    predicted_class = CLASS_LIST[np.argmax(prediction)]
    confidence = np.max(prediction)
    return predicted_class, confidence

def main():
    st.title('Pneumonia Classification App')
    st.text('Upload an X-ray image for classification as normal, bacteria, or virus.')

    # File upload
    uploaded_file = st.file_uploader("Choose an X-ray image ...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
        
        # Make a prediction
        model = load_cnn_model()
        predicted_class, confidence = predict_class(model, uploaded_file)
        
        st.write("")
        st.write("### Prediction:")
        st.write(f"Class: {predicted_class}, Confidence: {confidence:.2f}")

if __name__ == '__main__':
    main()
