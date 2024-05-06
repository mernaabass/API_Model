import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('last_edit.h5')

# Initialize FastAPI
app = FastAPI()

# Define image processing and prediction function
def process_image(img):
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255  # Normalize the image data
    return img_array

def predict_class(img_array):
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    class_name = 'hem' if predicted_class == 0 else 'all'
    return class_name, predictions

# Define FastAPI routes
@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    img = image.load_img(BytesIO(contents), target_size=(128, 128))
    
    # Process image and make predictions
    img_array = process_image(img)
    class_name, predictions = predict_class(img_array)
    
    return {"class_name": class_name, "predictions": predictions.tolist()}
