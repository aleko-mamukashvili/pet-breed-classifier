import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "resnet18.onnx"
CLASSES_PATH = ROOT / "models" / "classes.json"

# იგივე ნორმალიზაცია, რაც ვარჯიშზე გამოვიყენეთ
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

classes = json.load(open(CLASSES_PATH, encoding="utf-8"))
session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])


def preprocess(img):
    img = img.convert("RGB").resize((224, 224), Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return x.transpose(2, 0, 1)[None]  # [1, 3, 224, 224]


def predict(img, k=3):
    logits = session.run(None, {"input": preprocess(img)})[0][0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    top = probs.argsort()[::-1][:k]
    return {classes[i]: float(probs[i]) for i in top}


import io

import gradio as gr
from fastapi import FastAPI, File, UploadFile

# --- API ---
api = FastAPI(title="Pet Breed Classifier")


@api.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read()))
    return {"top3": predict(img)}


# --- ვებ-ინტერფეისი ---
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="ატვირთე კატის ან ძაღლის სურათი"),
    outputs=gr.Label(num_top_classes=3, label="Top-3 პროგნოზი"),
    title="Oxford Pet-ის ჯიშების ამომცნობი (37 ჯიში)",
    description="ResNet18 (fine-tuned), ONNX Runtime-ზე. Test accuracy 89.1%.",
)

# ორივე ერთ აპლიკაციაში: "/" არის ინტერფეისი, "/predict" არის API
app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
