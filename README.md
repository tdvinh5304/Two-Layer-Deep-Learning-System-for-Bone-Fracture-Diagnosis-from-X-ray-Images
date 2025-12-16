# Bone Fracture Diagnosis & RAG Chatbot System

This repository contains an **AI-powered medical imaging system** for **bone fracture diagnosis from X-ray images**, combined with a **Retrieval-Augmented Generation (RAG) chatbot** that provides bone anatomy and fracture-related medical knowledge.

The project is designed as a **two-layer system**:

1. **Deep Learning models** for automated fracture detection by body part.
2. **RAG chatbot** backed by a medical knowledge base for explainability and user support.

---

## Project Structure

```
├── data/
│   ├── train/                # Training X-ray images
│   └── valid/                # Validation X-ray images
│
├── medical_db/               # Vector database for RAG chatbot
│   ├── chroma.sqlite3        # ChromaDB storage
│   └── <uuid>/               # Embedded medical documents
│
├── models/                   # Trained deep learning models
│   ├── XR_ELBOW_inception3_best.pth
│   ├── XR_FINGER_inception3_best.pth
│   ├── XR_FOREARM_resnet_best.pth
│   ├── XR_HAND_inception3_best.pth
│   ├── XR_HUMERUS_resnet_best.pth
│   ├── XR_SHOULDER_resnet_best.pth
│   ├── XR_WRIST_resnet_best.pth
│   └── c_model_resnet50.pth  # Classification backbone
│
├── src/
│   ├── app.py                # Main application entry point
│   ├── page.py               # UI / inference logic
│   ├── import_db.py          # Load medical documents into ChromaDB
│   ├── chroma.ipynb          # RAG & vector DB experiments
│   ├── DLFinal.ipynb         # Deep learning experiments
│   ├── Training_model_for_specific_class.ipynb
│   └── BOA-and-BAPRAS-Standards-for-...pdf
│
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## Deep Learning Models

* Trained on **X-ray images** for multiple body parts:

  * Elbow, Finger, Forearm, Hand, Humerus, Shoulder, Wrist
* Architectures used:

  * **ResNet**
  * **InceptionV3**
* Each model is optimized for **body-part–specific fracture detection**.

---

## RAG Chatbot (Bone Knowledge Assistant)

The chatbot uses a **Retrieval-Augmented Generation (RAG)** pipeline to:

* Answer questions about **bone anatomy and fractures**
* Provide **medical explanations** to support AI predictions
* Retrieve information from trusted medical documents (e.g. BOA/BAPRAS standards)

**Tech stack:**

* ChromaDB (vector database)
* Embedding-based document retrieval
* LLM-powered response generation

---

## How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ (Optional) Build medical knowledge base

```bash
python src/import_db.py
```

### 3️⃣ Run the application

```bash
python src/app.py
```

---

## Notebooks

* `DLFinal.ipynb` – Model training & evaluation
* `Training_model_for_specific_class.ipynb` – Class-specific training
* `chroma.ipynb` – RAG and vector database experiments

---

## Key Features

* ✅ Automated fracture detection from X-ray images
* ✅ Body-part–specific deep learning models
* ✅ RAG chatbot with medical bone knowledge
* ✅ Explainable AI support for medical understanding

---

## Disclaimer

This project is for **educational and research purposes only**. It is **not intended for clinical diagnosis** or medical decision-making.

---

## Future Improvements

* Add multi-class fracture severity detection
* Integrate Grad-CAM for visual explainability
* Deploy as a web-based medical assistant
* Expand medical knowledge base

---

## 👤 Author

Developed as an academic deep learning & medical AI project by Khanh Vu Quoc, Vinh Tieu Dang, Tai Le Nguyen Minh, Ngoc Bao Doan Gia.
