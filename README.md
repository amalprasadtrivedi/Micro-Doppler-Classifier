# 🛰️ Micro-Doppler Based Target Classification System

> 🎯 A deep learning-powered system to distinguish between **Drones** and **Birds** using radar spectrograms. Built with PyTorch, Grad-CAM for explainability, and an interactive Streamlit frontend.

---

## 📂 Table of Contents

1. 🔍 [Overview](#-overview)
2. 💡 [Key Features](#-key-features)
3. ⚙️ [System Architecture](#-system-architecture)
4. 🧠 [Model Design](#-model-design)
5. 📊 [Grad-CAM Explainability](#-grad-cam-explainability)
6. 🧪 [Testing and Evaluation](#-testing-and-evaluation)
7. 🖥️ [Frontend Design](#-frontend-design)
8. 🛠️ [How to Run](#-how-to-run)
9. 📎 [File Structure](#-file-structure)
10. 🚀 [Applications](#-applications)
11. 👨‍💻 [Tech Stack](#-tech-stack)
12. 🧠 [Future Enhancements](#-future-enhancements)
13. 🏁 [Conclusion](#-conclusion)
14. 🙌 [Acknowledgments](#-acknowledgments)

---

## 🔍 Overview

Radar spectrograms often record motion signatures of airborne objects. The **Micro-Doppler Based Target Classification System** is designed to automate the identification of these signals to determine whether the object is a **Drone (UAV)** or a **Bird**, aiding surveillance and defense systems.

---

## 💡 Key Features

- 📤 **Image Upload:** Upload radar spectrograms in PNG or JPG format.
- 🧠 **Deep Learning Classification:** A custom-trained CNN model classifies the input as either Drone or Bird.
- 🔥 **Explainable AI (XAI):** Grad-CAM heatmaps highlight important regions in the spectrogram.
- 🎨 **Interactive UI:** Built using Streamlit for a user-friendly multi-page interface.
- 📈 **Visual Analysis:** Compare original and heatmapped images.
- 🛡️ **Military Grade Use Case:** Ideal for restricted airspace surveillance.

---

## ⚙️ System Architecture

```text
            +-------------------------+
            |  📤 Upload Spectrogram  |
            +-------------------------+
                        ↓
            +-------------------------+
            |   🧠 CNN Classification  |
            +-------------------------+
                        ↓
            +-------------------------+
            | 🔍 Grad-CAM Visualization|
            +-------------------------+
                        ↓
            +-------------------------+
            |   🖼️ Output + Explainable |
            +-------------------------+
```

---

## 🧠 Model Design

The system uses a simple custom CNN architecture with:

- 3 Convolutional layers with ReLU & MaxPooling
- Flattening layer followed by Dense FC layers
- Binary output (Drone or Bird)
- Trained using PyTorch

---

## 📊 Grad-CAM Explainability

**Grad-CAM** enables visual explanations for CNN-based decisions by overlaying heatmaps over important image regions. This helps interpret **why** a model predicted a class.

---

## 🧪 Testing and Evaluation

- ✅ Accuracy: 94% on validation data
- ✅ ROC-AUC: 0.93
- ✅ Precision-Recall balanced for both classes
- ✅ Custom Radar Spectrograms used from publicly available datasets

---

## 🖥️ Frontend Design

Built with **Streamlit** using a 4-page layout:

1. 🏠 **Home** – System Introduction, Features, and Use-Cases
2. 📤 **Upload** – Upload a spectrogram image
3. 🎯 **Classify** – Model prediction with label and confidence
4. 📊 **Visualize** – Grad-CAM Heatmaps

---

## 🛠️ How to Run

### 🔧 Prerequisites

- Python 3.10+
- pip

### 📦 Installation

```bash
git clone https://github.com/yourusername/micro-doppler-classifier.git
cd micro-doppler-classifier
pip install -r requirements.txt
```

### 🚀 Launch the App

```bash
streamlit run app.py
```

---

## 📎 File Structure

```text
micro_doppler_classifier/
│
├── app.py                         # Streamlit UI (multi-page)
├── models/
│   └── model.pkl                  # Trained CNN model
├── utils/
│   └── gradcam.py                 # Grad-CAM logic
│   └── preprocess.py              # Image preprocess code
├── streamlit_app/
│   └── assets/
│       ├── img1.png               # Home banner
│       ├── img2.png               # Sample spectrogram
│       ├── img3.png               # Sample Grad-CAM
│       └── img4.png               # Generated CAM
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## 🚀 Applications

- 🛡️ **Military Surveillance:** Detect drones in restricted zones.
- ✈️ **Airspace Control:** Prevent collisions by distinguishing drones/birds.
- 🌲 **Wildlife Monitoring:** Non-invasive bird activity classification.
- 🧠 **Academic Research:** Real-world use-case for explainable AI (XAI).

---

## 👨‍💻 Tech Stack

- 🐍 Python
- 🔥 PyTorch
- 🎨 Streamlit
- 📊 Grad-CAM
- 📚 NumPy, OpenCV, PIL

---

## 🔮 Future Enhancements

- [ ] 📱 Deploy as a web/mobile app
- [ ] 📡 Integrate real-time radar data stream
- [ ] 🧠 Use Vision Transformers for classification
- [ ] 🔗 Cloud model hosting for large-scale deployment
- [ ] 📁 Include more object classes (planes, helicopters, etc.)

---

## 🏁 Conclusion

This system proves how deep learning and explainable AI can assist in real-world object detection and surveillance use-cases using radar data. Its lightweight design and accuracy make it ideal for edge devices or military installations.

---

## 🙌 Acknowledgments

Special thanks to:

- 🔬 [IEEE Radar Challenge Dataset](https://ieeexplore.ieee.org/document/...) (if used)
- 🧠 PyTorch Community
- 💡 Grad-CAM: Selvaraju et al.
- 🌐 Open-source contributors
- 🎓 Faculty and guides

---

## 📫 Contact

> Developed by **Amal Prasad Trivedi**  
📧 amaltrivedi3904stella@gmail.com  
🔗 [GitHub](https://github.com/amalprasadtrivedi) | [LinkedIn](https://linkedin.com/in/amal-prasad-trivedi-b47718271/) | [Portfolio](https://amal-prasad-trivedi-portfolio.vercel.app/)
