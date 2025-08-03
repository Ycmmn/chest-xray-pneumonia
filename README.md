
# Pneumonia Detection with ResNet50 and FastAPI

This project implements a deep learning model for detecting pneumonia from chest X-ray images using a fine-tuned ResNet50 architecture with focal loss.  
It includes both the training pipeline and a REST API built with FastAPI for real-time predictions.

---

## Project Overview

- **Data preprocessing**: Image augmentation and normalization with `ImageDataGenerator`  
- **Model architecture**: Pretrained ResNet50 (imagenet weights) with custom classification head  
- **Loss function**: Custom focal loss to handle class imbalance  
- **Training**: Transfer learning with partial freezing of base model layers  
- **Evaluation**: Precision, recall, F1-score, confusion matrix, and threshold tuning  
- **Deployment**: FastAPI server exposing a `/predict` endpoint for image input

---

## Requirements

- Python 3.8+  
- TensorFlow  
- FastAPI  
- Uvicorn  
- NumPy, Pillow  
- scikit-learn, matplotlib, seaborn  

Install dependencies with:

```bash
pip install tensorflow fastapi uvicorn numpy pillow scikit-learn matplotlib seaborn
```

---

## Dataset

Place your dataset folder with the following structure inside the specified path (e.g., `C:\Users\Yasaman\Desktop`):

```
chest-xray-pneumonia-project/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
```

---

## Training the Model

Run the training script (e.g., `train.py`) that includes:

* Data loading and augmentation
* Model definition with ResNet50 and focal loss
* Callbacks for early stopping and learning rate reduction
* Saving the best trained model as `myy_model.keras`

```bash
python train.py
```

---

## Running the API Server

Start the FastAPI server with:

```bash
uvicorn main:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

---

## Using the API

Send a POST request with a chest X-ray image file to `/predict` endpoint.
Example using `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -F "file=@path_to_image.jpg"
```

Response example:

```json
{
  "prediction": "Pneumonia",
  "confidence": 0.912
}
```

---

## Notes

* The model uses a threshold tuned based on F1-score for better classification accuracy.
* The API expects RGB images resized to 224x224 pixels.

---

## License

This project is released under the MIT License.

```

If you want, I can help you generate a minimal training script file and the API main file separately too!
```
