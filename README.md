# Vision Art AI — Full System

**Vision Art AI** is an end-to-end platform that combines **computer vision** and **generative AI** to analyze, interpret, and creatively reimagine artworks.

The system is composed of three main parts:

- 🎨 **Frontend** — Interactive web interface for user experience  
- 🧠 **Backend (Hugging Face Space)** — AI inference and orchestration  
- 🏋️ **Training Pipeline** — Model training and export workflows  

---

## 🌐 Live System

- **Frontend (GitHub Pages):**  
  https://heigon77.github.io/VisionArtAI-Frontend/

- **Backend (Hugging Face Space):**  
  https://huggingface.co/spaces/heigon77/VisionArtAI-Backend

---

## 🧩 System Architecture

```

User → Frontend (Angular)
↓
Backend API (HF Space)
↓
┌───────────────────────────────┐
│ Object Detection (YOLOv8)     │
│ Style Classification          │
│ Poetry Generation (Qwen 2.5)  │
└───────────────────────────────┘
↓
Response → Frontend UI

````

---

## ✨ Features

### 🎨 Artwork Understanding
- Detects **134 object categories** in paintings (COCO + DEArt)
- Identifies **artistic styles (27 classes)** from WikiArt

### ✍️ Creative AI
- Generates **context-aware poetry** using **Qwen 2.5 (1.5B)**

### 🖥️ Interactive Experience
- Toggle between original and annotated image
- Real-time feedback and smooth transitions
- Editorial-style UI with artistic focus

---

## 🧠 AI Stack

### Computer Vision
- **YOLOv8** — Object detection (fine-tuned on DEArt + COCO)
- **MobileNetV3** — Style classification (WikiArt, 27 classes)

### Generative AI
- **Qwen 2.5 (1.5B)** — Poem generation based on scene + style

### Optimization
- ONNX export
- INT8 quantization for efficient CPU inference

---

## 🖥️ Frontend

Built with modern web technologies:

- Angular 19+
- TypeScript
- SCSS
- RxJS

### Key Responsibilities
- Image upload & preview
- API communication with backend
- Visualization of:
  - Detected objects
  - Style predictions
  - Generated poem

### Run locally

```bash
npm install
ng serve
````

---

## 🤖 Backend (Hugging Face Space)

The backend is deployed as a **Hugging Face Space**, responsible for:

* Running model inference
* Orchestrating:

  * Object detection
  * Style classification
  * Text generation
* Returning structured results to the frontend

🔗 [https://huggingface.co/spaces/heigon77/VisionArtAI-Backend](https://huggingface.co/spaces/heigon77/VisionArtAI-Backend)

### Notes

* Backend code in this repository is included as a **static snapshot**
* The production environment runs on Hugging Face Spaces
* Designed for lightweight CPU inference using ONNX models

---

## 🏋️ Training Pipeline

This project includes a full pipeline for training and exporting models.

### Models

#### 1. Object Detection — YOLOv8

* 134 classes (80 COCO + 54 DEArt)
* Two-stage fine-tuning:

  * Head warm-up
  * Full fine-tuning

#### 2. Style Classification — MobileNetV3

* 27 WikiArt styles
* Data augmentation (RandAugment)
* Cosine annealing + AdamW

---

## 📊 Datasets

* **DEArt Dataset** — European art object detection
* **COCO** — General object detection
* **WikiArt** — Style classification

Downloaded automatically via Hugging Face `datasets`.

---

## ⚙️ Training Workflow

### 1. Dataset Preparation

```bash
python data_process.py
```

### 2. Train YOLO

```bash
python train_yolo.py
```

### 3. Train Classifier

```bash
python train_class.py
```

### 4. Export Models

```bash
python export_yolo_onnx.py
python export_class_onnx.py
```

---

## 📈 Results

### Object Detection (YOLOv8)

* mAP@0.5: **0.298**
* mAP@0.5:0.95: **0.195**

### Style Classification (MobileNetV3)

* Validation Accuracy: **~60.8%**

---

## 📁 Repository Structure

```
.
├── frontend/        # Angular app (submodule)
├── training/        # Training pipeline (submodule)
├── backend/         # Static backend snapshot
├── README.md
```

---

## 🧠 Design Principles

* **Modular Architecture** — decoupled frontend, backend, and training
* **Efficiency First** — optimized for CPU inference
* **Art-Centric UX** — design aligned with artistic exploration
* **Extensibility** — easy to plug new models or datasets

---

## 📌 Notes

* Frontend and training modules are maintained as **Git submodules**
* Backend is deployed externally (Hugging Face Space)
* This repo acts as a **central integration point**

---

## 📜 License

GNU GENERAL PUBLIC LICENSE

---

## 🌐 Vision

Vision Art AI explores the intersection of:

* Computer Vision
* Generative AI
* Artistic Interpretation

Transforming how we interact with and understand art through AI.
