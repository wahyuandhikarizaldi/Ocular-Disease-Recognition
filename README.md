
# 📦 Project Environment

## 📝 Description
This project involves fundus image classification for eye disease detection using Deep Learning and Explainable AI, implemented in Python (Jupyter Notebook) and served via FastAPI.

---

## 🗂️ Dataset

https://drive.google.com/drive/folders/100sALNdT72NXoUyve6_Bl3BABlKOZ3LE?usp=sharing

## 📚 Required Libraries

The following libraries are used in this project:

| Library                          | Recommended Version        |
|----------------------------------|-----------------------------|
| numpy                            | >= 1.22                     |
| pandas                           | >= 1.4                      |
| matplotlib                       | >= 3.5                      |
| seaborn                          | >= 0.11                     |
| scikit-learn (`sklearn`)         | >= 1.0                      |
| tensorflow                       | >= 2.9                      |
| pillow (`PIL`)                   | >= 9.0                      |
| opencv-python (`cv2`)           | >= 4.5                      |
| lime                             | >= 0.2                      |
| tqdm                             | >= 4.62                     |
| fastapi                          | >= 0.78                     |
| uvicorn                          | >= 0.17                     |

Additional built-in modules:
- os, shutil, pathlib, io, uuid, time, random, itertools

---

## ⚙️ TensorFlow Submodules Used

- tensorflow.keras.models
- tensorflow.keras.layers
- tensorflow.keras.preprocessing.image
- tensorflow.keras.callbacks
- tensorflow.keras.optimizers
- tensorflow.keras.metrics
- tensorflow.keras.regularizers
- tensorflow.keras.utils
- tensorflow.keras.initializers

---



## 🧪 Explainability

Model explainability is performed using:
- lime
- lime.wrappers.scikit_image
- skimage.segmentation

---

## 🌐 API Deployment (Optional)

For deployment via API:
- fastapi
- uvicorn
- fastapi.responses
- fastapi.middleware.cors

---

## 📦 Recommended Environment Setup

Use `virtualenv` or `conda`. Example using `pip`:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow pillow opencv-python lime tqdm fastapi uvicorn
```
