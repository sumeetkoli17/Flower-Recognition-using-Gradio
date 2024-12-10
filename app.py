import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('model/flower_recognition_model.h5')  

# Class names for prediction
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

def predict_image(img):
    img = img.resize((180, 180))
    img_array = np.array(img).reshape(-1, 180, 180, 3)
    
    # Make prediction
    prediction = model.predict(img_array)[0]
    return {class_names[i]: float(prediction[i]) for i in range(len(class_names))}

# Create the Gradio interface
image = gr.Image(type="pil", image_mode="RGB")
label = gr.Label(num_top_classes=5)  # Top 5 class predictions

# Launch the Gradio interface
gr.Interface(fn=predict_image, inputs=image, outputs=label).launch(debug=True)



