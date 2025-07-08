import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image as PILImage
from io import BytesIO

app = FastAPI()

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:8000",
    "*"  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
IMG_SIZE = (299, 299)
# IMG_SIZE = (224, 224)
CLASS_NAMES = ["ARMD", "Cataract", "Diabetic", "Glaucoma", "Normal"]  
TEMP_IMAGE_DIR = "temp_images"
os.makedirs(TEMP_IMAGE_DIR, exist_ok=True)

# Load model 
try:
    model = tf.keras.models.load_model('MobileNetBaru-86.16.h5')
    # model = tf.keras.models.load_model('EfficientNetB3-93.08.h5')
    # model = tf.keras.models.load_model('InceptionV3-93.08.h5')
    
    print("Custom model loaded successfully")
    
    # base model for Grad-CAM 
    base_model = tf.keras.applications.MobileNetV3Small(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # base_model = tf.keras.applications.efficientnet.EfficientNetB3(
    #     include_top= False, 
    #     weights= "imagenet", 
    #     input_shape= (224, 224, 3), 
    # )
    
    # base_model = tf.keras.applications.InceptionV3(
    #     include_top= False, 
    #     weights= "imagenet", 
    #     input_shape= (299, 299, 3), 
    # )
    
    # Find last convolutional layer
    def find_last_conv_layer(model):
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer.name
        raise ValueError("Could not find convolutional layer in the model")
    
    LAST_CONV_LAYER_NAME = find_last_conv_layer(base_model)
    print(f"Using layer '{LAST_CONV_LAYER_NAME}' for Grad-CAM")
    
    # Grad-CAM model
    grad_cam_model = Model(
        inputs=base_model.input,
        outputs=[base_model.get_layer(LAST_CONV_LAYER_NAME).output, base_model.output]
    )
    
except Exception as e:
    print(f"Error loading model: {e}")
    raise e
    

# Preprocessing functions 
def to_grayscale(image):
    weights = tf.constant([0.2989, 0.5870, 0.1140], dtype=tf.float32)
    grayscale_image = tf.reduce_sum(image * weights, axis=-1, keepdims=True)
    return grayscale_image

def apply_hybrid_filter(image):
    image_np = image.numpy()
    gaussian_blurred = cv2.GaussianBlur(image_np, (5, 5), 0)
    median_filtered = cv2.medianBlur(gaussian_blurred, 5)
    return tf.convert_to_tensor(median_filtered[..., np.newaxis], dtype=tf.float32)

def image_transformation(image):
    return 255.0 * (image / 255.0) ** 2

def histogram_equalization(image):
    image_np = image.numpy().astype(np.uint8)
    equalized = cv2.equalizeHist(image_np)
    return tf.convert_to_tensor(equalized[..., np.newaxis], dtype=tf.float32)

def enhanced_clahe(image, clip_limit=2.0, tile_grid_size=(8,8)):
    image_np = image.numpy().astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_applied = clahe.apply(image_np)
    return tf.convert_to_tensor(clahe_applied[..., np.newaxis], dtype=tf.float32)

def advanced_preprocessing(image):
    image = to_grayscale(image)
    image = apply_hybrid_filter(image)
    image = image_transformation(image)
    image = histogram_equalization(image)
    image = enhanced_clahe(image)
    image = tf.keras.applications.mobilenet_v3.preprocess_input(image)
    return image

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_gradcam(original_img, heatmap, cam_path, alpha=0.5):
    if isinstance(original_img, tf.Tensor):
        original_img = original_img.numpy()
    
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    superimposed_img = heatmap * alpha + original_img * (1 - alpha)
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    
    cv2.imwrite(cam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Generate filename
        file_ext = os.path.splitext(file.filename)[1]
        temp_filename = f"{uuid.uuid4()}{file_ext}"
        temp_filepath = os.path.join(TEMP_IMAGE_DIR, temp_filename)
        cam_filename = f"cam_{uuid.uuid4()}.jpg"
        cam_filepath = os.path.join(TEMP_IMAGE_DIR, cam_filename)
        
        # Save uploaded file temporarily
        contents = await file.read()
        with open(temp_filepath, "wb") as f:
            f.write(contents)
        
        # Load image for prediction with preprocess
        img = cv2.imread(temp_filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        img_processed = advanced_preprocessing(img_tensor)
        
        if img_processed.shape[-1] == 1:
            img_processed = tf.repeat(img_processed, repeats=3, axis=-1)
        
        img_processed = tf.expand_dims(img_processed, axis=0)
        
        # prediction
        preds = model.predict(img_processed)[0]
        pred_index = np.argmax(preds)
        predicted_class = CLASS_NAMES[pred_index]
        confidence = float(preds[pred_index])
        
        # Generate Grad-CAM
        heatmap = make_gradcam_heatmap(
            img_processed, 
            grad_cam_model,
            LAST_CONV_LAYER_NAME,
            pred_index
        )
        
        # Load original image for visualization
        original_img = cv2.imread(temp_filepath)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        original_img = cv2.resize(original_img, IMG_SIZE)
        
        # Save Grad-CAM 
        save_gradcam(original_img, heatmap, cam_filepath, alpha=0.5) 
        
        # confidence scores
        confidence_scores = {
            CLASS_NAMES[i]: float(score) for i, score in enumerate(preds)
        }
        
        # Clean up temp files
        os.remove(temp_filepath)
        
        return JSONResponse({
            "predicted_class": predicted_class,
            "confidence": confidence,
            "confidence_scores": confidence_scores,
            "gradcam_url": f"/gradcam/{cam_filename}"
        })
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gradcam/{filename}")
async def get_gradcam_image(filename: str):
    filepath = os.path.join(TEMP_IMAGE_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(filepath)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)