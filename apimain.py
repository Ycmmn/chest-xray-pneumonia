
# -------------------- 1. ایمپورت کتابخانه‌ها --------------------
import uvicorn
from  fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------- 2. بارگذاری مدل --------------------



import tensorflow.keras.backend as K

def focal_loss(gamma=2., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        return K.mean(weight * cross_entropy, axis=-1)
    return focal_loss_fixed

model = tf.keras.models.load_model('my_model.h5' , custom_objects={'focal_loss_fixed': focal_loss()})  

# -------------------- 3. ساخت اپلیکیشن FastAPI --------------------
app = FastAPI(title="Pneumonia Detection API")


# -------------------- 4. پیش‌پردازش تصویر --------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------- 5. ایجاد endpoint اصلی --------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # خواندن و پیش‌پردازش تصویر
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)

        # پیش‌بینی با مدل
        pred_prob = model.predict(img)[0][0]

        # تفسیر نتیجه
        result = "Pneumonia" if pred_prob > 0.5 else "Normal"
        confidence = round(float(pred_prob if pred_prob > 0.5 else 1 - pred_prob), 3)

        # بازگرداندن پاسخ
        return JSONResponse(content={
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(content={
            "error": str(e)
        }, status_code=500)

# -------------------- 6. اجرای سرور (در ترمینال) --------------------
# دستور اجرا:
=======


# -------------------- 1. ایمپورت کتابخانه‌ها --------------------
import uvicorn
from  fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

# -------------------- 2. بارگذاری مدل --------------------



import tensorflow.keras.backend as K

def focal_loss(gamma=2., alpha=0.75):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        cross_entropy = -y_true * K.log(y_pred) - (1 - y_true) * K.log(1 - y_pred)
        weight = alpha * y_true * K.pow(1 - y_pred, gamma) + (1 - alpha) * (1 - y_true) * K.pow(y_pred, gamma)
        return K.mean(weight * cross_entropy, axis=-1)
    return focal_loss_fixed

model = tf.keras.models.load_model('my_model.h5' , custom_objects={'focal_loss_fixed': focal_loss()})  

# -------------------- 3. ساخت اپلیکیشن FastAPI --------------------
app = FastAPI(title="Pneumonia Detection API")


# -------------------- 4. پیش‌پردازش تصویر --------------------
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -------------------- 5. ایجاد endpoint اصلی --------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # خواندن و پیش‌پردازش تصویر
        image_bytes = await file.read()
        img = preprocess_image(image_bytes)

        # پیش‌بینی با مدل
        pred_prob = model.predict(img)[0][0]

        # تفسیر نتیجه
        result = "Pneumonia" if pred_prob > 0.5 else "Normal"
        confidence = round(float(pred_prob if pred_prob > 0.5 else 1 - pred_prob), 3)

        # بازگرداندن پاسخ
        return JSONResponse(content={
            "prediction": result,
            "confidence": confidence
        })

    except Exception as e:
        return JSONResponse(content={
            "error": str(e)
        }, status_code=500)

# -------------------- 6. اجرای سرور (در ترمینال) --------------------
# دستور اجرا:
>>>>>>> dbc88e5 (Initial commit)
# uvicorn main:app --reload