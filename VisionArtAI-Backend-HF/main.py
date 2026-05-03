import os
import base64
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ultralytics import YOLO
from torchvision import transforms
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

from ctransformers import AutoModelForCausalLM

from collections import Counter

import re

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

YOLO_PATH = Path("yolo_deart.onnx")
STYLE_MODEL_PATH = Path("style_classifier.onnx")

LLM_PATH = hf_hub_download(
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF", 
    filename="qwen2.5-1.5b-instruct-fp16.gguf"
)

FRONTEND_ORIGIN = os.environ.get(
    "FRONTEND_ORIGIN",
    "https://heigon77.github.io"
)

# ─────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN, "http://localhost:4200"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# YOLO
# ─────────────────────────────────────────────

yolo_model = YOLO(str(YOLO_PATH))

# ─────────────────────────────────────────────
# STYLE MODEL (ONNX)
# ─────────────────────────────────────────────

ort_session = ort.InferenceSession(
    str(STYLE_MODEL_PATH),
    providers=["CPUExecutionProvider"]
)

STYLES = [
    "Abstract Expressionism","Action painting","Analytical Cubism","Art Nouveau",
    "Baroque","Color Field Painting","Contemporary Realism","Cubism",
    "Early Renaissance","Expressionism","Fauvism","High Renaissance",
    "Impressionism","Mannerism (Late Renaissance)","Minimalism",
    "Naive Art (Primitivism)","New Realism","Northern Renaissance",
    "Pointillism","Pop Art","Post Impressionism","Realism",
    "Rococo","Romanticism","Symbolism","Synthetic Cubism","Ukiyo-e"
]

# ─────────────────────────────────────────────
# LLM (phi3 GGUF)
# ─────────────────────────────────────────────

# Inicializando o modelo
llm = Llama(
    model_path=LLM_PATH,
    n_ctx=2048, # Tamanho da janela de contexto
    n_threads=2, # O Space free tem 2 vCPUs
    n_gpu_layers=0 # Garante que ele não busque CUDA
)

def generate_poem(detections, style):
    counts = Counter([d["class_name"] for d in detections])

    objects = ", ".join(
        f"{v} {k}{'s' if v > 1 and not k.endswith('s') else ''}"
        for k, v in counts.items()
    ) or "silence"

    prompt = f"""<|im_start|>system
            You are a master of classical poetry. You write only the poem requested, following strict metrical and rhyme rules. No titles, no explanations, no markdown.<|im_end|>
            <|im_start|>user
            Write a Shakespearean sonnet (14 lines, ABAB CDCD EFEF GG) about: {objects}.
            Style of the artwork: {style["style_name"]}.

            Directives:
            - Strictly 14 lines.
            - No markdown (no bold, no italics).
            - No title.
            - Output only the poem text.<|im_end|>
            <|im_start|>assistant
            """

    output = llm(
        prompt,
        max_tokens=300,
        temperature=0.7, # Equilíbrio entre criatividade e ordem
        top_p=0.9,
        repeat_penalty=1.1, # Reduzido, pois o FP16 é naturalmente mais coerente
        stop=["<|im_end|>", "User:", "\n\n\n\n"] 
    )

    poem_text = output["choices"][0]["text"].strip()
    return poem_text

def format_poem(text: str, max_lines: int = 17):
    # ── 1. Remove prefixos tipo [response]: ou "response:"
    text = re.sub(r'^\[.*?\]\s*:\s*', '', text.strip(), flags=re.IGNORECASE)
    text = re.sub(r'^[^:]{0,30}:\s*', '', text.strip())

    # ── 2. Quebra em linhas e limpa espaços
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # ── 3. Remove lixo extra (mantém só primeiras 14 + espaços)
    core = lines[:14]

    # ── 4. Monta estrutura com 3 linhas em branco entre blocos
    # (4 blocos: 4 + 4 + 4 + 2 versos, ajustável se quiser)
    formatted = []

    formatted += core[:4]
    formatted.append("")
    formatted += core[4:8]
    formatted.append("")
    formatted += core[8:12]
    formatted.append("")
    formatted += core[12:14]

    # ── 5. Garante exatamente 17 linhas
    if len(formatted) > max_lines:
        formatted = formatted[:max_lines]
    else:
        while len(formatted) < max_lines:
            formatted.append("")

    return "\n".join(formatted)

# ─────────────────────────────────────────────
# TRANSFORM
# ─────────────────────────────────────────────

style_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

def encode_image(img_bgr: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")

# ─────────────────────────────────────────────
# STYLE INFERENCE
# ─────────────────────────────────────────────

def predict_style(img_bgr: np.ndarray):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = style_tf(img_rgb).unsqueeze(0).numpy()

    outputs = ort_session.run(["logits"], {"image": tensor})[0]

    exp = np.exp(outputs - np.max(outputs))
    probs = exp / exp.sum(axis=1, keepdims=True)

    idx = int(np.argmax(probs))

    return {
        "style_id": idx,
        "style_name": STYLES[idx],
        "confidence": float(np.max(probs))
    }

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "yolo_classes": len(yolo_model.names),
        "style_classes": len(STYLES),
        "llm": "qwen2.5-1.5b-instruct-fp16.gguf"
    }

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    contents = await file.read()

    img_np = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    # ── YOLO ───────────────────────────────
    results = yolo_model(img_np)[0]

    detections = [
        {
            "class_id": int(box.cls[0]),
            "class_name": yolo_model.names[int(box.cls[0])],
            "confidence": round(float(box.conf[0]), 3),
        }
        for box in results.boxes
    ]

    # ── STYLE ──────────────────────────────
    style_pred = predict_style(img_np)

    # ── LLM ────────────────────────────────
    try:
        poem = generate_poem(detections, style_pred)
        poem = format_poem(poem)
    except Exception as e:
        poem = f"LLM error: {str(e)}"

    return JSONResponse({
        "detections": sorted(detections, key=lambda d: -d["confidence"]),
        "style": style_pred,
        "poem": poem,
        "annotated_image": encode_image(results.plot()),
        "image_width": int(img_np.shape[1]),
        "image_height": int(img_np.shape[0]),
    })

@app.get("/")
def root():
    return {
        "status": "ok",
        "description": "Art analysis API: detects objects, classifies artistic style and generates a Shakespearean sonnet from paintings.",
        "models": {
            "object_detection": {
                "file": "yolo_deart.onnx",
                "description": "Custom YOLOv8 fine-tuned for art object detection"
            },
            "style_classification": {
                "file": "style_classifier.onnx",
                "classes": len(STYLES),
                "description": f"ONNX classifier for {len(STYLES)} artistic styles (Baroque, Impressionism, Cubism…)"
            },
            "poem_generation": {
                "model": "Qwen2.5-1.5B-Instruct (FP16 GGUF)",
                "backend": "llama-cpp",
                "description": "Generates a Shakespearean sonnet based on detected objects and art style"
            }
        },
        "endpoints": {
            "GET  /":        "This overview",
            "GET  /health":  "Runtime status and loaded model info",
            "POST /detect":  "Upload a painting image → returns detections, style, poem and annotated image"
        }
    }