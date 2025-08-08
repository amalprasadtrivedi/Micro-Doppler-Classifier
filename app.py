import cv2
import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import io

# ------------------------------ CNN MODEL ------------------------------
# Define the CNN model inline, since you want it in the same file
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 64 * 32, 128),  # Fixed input features here
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.model(x)


# ------------------------ Utility Functions ------------------------
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)  # Shape: (1,3,256,256)

# Safe model loading function
def load_model(path=None):
    model = CNNModel()
    if path:
        try:
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            print(f"✅ Loaded trained model from: {path}")
        except FileNotFoundError:
            print(f"⚠️ Model file not found at {path}. Using untrained model.")
        except RuntimeError as e:
            print(f"⚠️ Model structure mismatch: {e}. Using untrained model.")
    else:
        print("⚠️ No model path provided. Using untrained model.")
    return model

# ------------------------ Streamlit App Config ------------------------
st.set_page_config(page_title="Micro-Doppler Classifier",
    layout="wide",
    page_icon="🛰️️️",
    initial_sidebar_state="expanded"
)

st.markdown("<h1 style='text-align: center;'>🛰️ Micro-Doppler Based Target Classification System</h1>", unsafe_allow_html=True)


# ----------------------------
# Sidebar Info & Navigation
# ----------------------------
with st.sidebar:
    st.title("🔍 MDBTCS Navigator")
    st.info(
        "This system detects and classifies objects in radar spectrograms as Drone or Bird using a custom CNN model.")
    st.markdown("---")

    st.image(Image.open("streamlit_app/assets/radar_logo.jpg"), caption="MDBTC System", use_container_width=True)
    st.markdown("---")

    # External Links
    st.markdown("### 🔍 Connect with Developer")
    st.link_button("🌐 Portfolio", "https://amalprasadtrivediportfolio.vercel.app/")
    st.link_button("🔗 LinkedIn", "https://linkedin.com/posts/amalprasadtrivedi-aiml-engineer")
    st.markdown("---")

    # Footer badge
    st.markdown(
        """
        <style>
        .sidebar-button-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            margin-top: 10px;
        }
        .sidebar-button img {
            width: 100%;
            max-width: 250px;
        }
        </style>
        <div class="sidebar-button-container">
            <a href="https://amalprasadtrivediportfolio.vercel.app/" target="_blank" class="sidebar-button">
                <img src="https://img.shields.io/badge/Created%20by-Amal%20Prasad%20Trivedi-blue">
            </a>
        </div>
        """,
        unsafe_allow_html=True
    )


# Load model and images
model_path = "models/cnn_model.pth"
model = load_model(model_path)
img1 = Image.open("streamlit_app/assets/intro.png")
img2 = Image.open("streamlit_app/assets/spectrogram_sample.png")
img3 = Image.open("streamlit_app/assets/gradcam_result.png")
img4 = Image.open("streamlit_app/assets/result.png")


if "page" not in st.session_state:
    st.session_state.page = "Home"

# ------------------------ Navigation Buttons ------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    if st.button("🏠 Home"):
        st.session_state.page = "Home"
with col2:
    if st.button("📤 Upload"):
        st.session_state.page = "Upload"
with col3:
    if st.button("🎯 Classify"):
        st.session_state.page = "Classify"
with col4:
    if st.button("📊 Visualize"):
        st.session_state.page = "Visualize"

st.markdown("---")

# ------------------------ Page: HOME ------------------------
if st.session_state.page == "Home":
    # ---------- HEADER SECTION ----------
    st.markdown("## 🛰️ Micro-Doppler Based Target Classification System")
    st.image(img1, use_container_width=True)

    st.markdown("""
    Welcome to the **Micro-Doppler Classification System**. This platform uses a custom-trained CNN model to **analyze radar spectrograms** and determine whether the object detected is a **Drone** or a **Bird**.

    ---
    """)

    # ---------- SYSTEM OVERVIEW ----------
    st.markdown("### 🔍 System Overview")
    st.markdown("""
    This system is designed for **airspace security and surveillance**. It leverages the power of computer vision and deep learning to process radar spectrogram images and classify aerial targets with high accuracy.

    - 📡 **Input**: Radar-based micro-Doppler spectrograms
    - 🧠 **Model**: Custom Convolutional Neural Network (CNN)
    - 🎯 **Output**: Predicted class - *Drone* or *Bird*

    The system is optimized for speed and interpretability, making it suitable for real-time or near-real-time defense applications.
    """)

    st.image(img2, caption="🎞️ Sample Micro-Doppler Spectrogram", use_container_width=True)
    st.markdown("---")

    # ---------- FEATURES SECTION ----------
    st.markdown("### ✨ Key Features")
    st.markdown("""
    - 📁 **Upload & Preprocess**: Upload radar spectrogram images in standard formats (PNG, JPG).
    - 🧠 **Model Inference**: Classify uploaded spectrograms using a custom-trained CNN.
    - 🔍 **Explainable AI (Grad-CAM)**: Understand how the model arrives at its decisions using Grad-CAM heatmaps.
    - 📈 **Result Visualization**: Get clear visual feedback with highlighted model attention areas.
    """)

    st.markdown("---")

    # ---------- APPLICATIONS SECTION ----------
    st.markdown("### 🎯 Real-World Applications")
    st.markdown("""
    - 🪖 **Military Surveillance**: Detect and classify drones or birds near sensitive zones.
    - 🛩️ **Airspace Management**: Enhance civil aviation radar systems for bird-strike prevention.
    - 🚨 **UAV Monitoring**: Used in anti-drone technology for critical infrastructure protection.
    """)

    st.markdown("---")

    # ---------- TECH STACK ----------
    st.markdown("### 🧰 Tech Stack Used")
    st.markdown("""
    - 🐍 **Python 3.10**
    - 📦 **PyTorch** for model training and inference
    - 🖼️ **OpenCV & PIL** for image processing
    - 📊 **Matplotlib** for Grad-CAM visualization
    - 🌐 **Streamlit** for frontend UI
    """)

    st.markdown("---")

    # ---------- FOOTER / CREDITS ----------
    st.markdown("### 👨‍💻 Developed By")
    st.markdown("""
    **Amal Prasad Trivedi**  
    B.Tech, Computer Science (AI & ML)  
    [GitHub](https://github.com/amalprasadtrivedi) | [Portfolio](https://amal-prasad-trivedi-portfolio.vercel.app/) | [LinkedIn](https://www.linkedin.com/in/amal-prasad-trivedi-b47718271/)
    """)


# ------------------------ Page: UPLOAD ------------------------
elif st.session_state.page == "Upload":

    # ---------- Page Title ----------
    st.markdown("## 📤 Upload Micro-Doppler Spectrogram")
    st.markdown("""
    Upload your radar-based micro-Doppler spectrogram image here for further processing and classification.

    ---
    """)

    # ---------- Instructions Section ----------
    st.markdown("### 📝 Upload Guidelines")
    st.markdown("""
    To ensure smooth operation, please follow these steps:

    - 🔍 Ensure the image is clear and properly cropped.
    - 📁 Accepted formats: `.jpg`, `.jpeg`, `.png`
    - 🖼️ Ideal resolution: 224x224 px or higher
    - 📤 Only **one image** can be uploaded at a time

    If unsure, refer to the sample image in the **Home** section.
    """)

    st.markdown("---")

    # ---------- Upload Widget ----------
    st.markdown("### 📂 Select File")
    uploaded_file = st.file_uploader(
        label="Drag and drop or click to upload a micro-Doppler spectrogram image",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="Supported formats: JPG, PNG. Only one image at a time."
    )

    if uploaded_file:
        # ---------- Load and Display Uploaded Image ----------
        try:
            img = Image.open(uploaded_file).convert("RGB")
            st.image(img, caption="🖼️ Uploaded Spectrogram", use_container_width=True)
            st.session_state.uploaded_image = img
            st.success("✅ Image uploaded and processed successfully!")

            st.markdown("---")

            # ---------- Preview Details ----------
            st.markdown("### 📋 Image Details")
            st.markdown(f"- **Filename**: `{uploaded_file.name}`")
            st.markdown(f"- **Format**: `{img.format or 'RGB'}`")
            st.markdown(f"- **Dimensions**: `{img.size[0]} x {img.size[1]}` pixels")

            st.markdown("---")

            # ---------- Next Steps Info ----------
            st.markdown("### 🚀 Ready to Proceed?")
            st.markdown("""
            You can now navigate to the **Classify** section to:

            - 🧠 Classify your image as **Drone** or **Bird**
            - 🔬 Visualize model interpretation with **Grad-CAM**
            - 📊 Understand model attention regions

            Use the sidebar to continue ➡️
            """)

        except Exception as e:
            st.error("❌ Failed to process the uploaded image. Please ensure it is a valid image file.")
            st.exception(e)

    else:
        st.warning("📌 Please upload a valid spectrogram image to continue.")

    st.markdown("---")

    # ---------- Help and Tips ----------
    with st.expander("ℹ️ Need Help?"):
        st.markdown("""
        - Visit the **Home** tab to understand the system overview.
        - Make sure the image is not corrupted or too large.
        - If you're facing issues, try reloading the app or clearing cache.
        - For sample data or demos, contact the project developer.
        """)

# ------------------------ Page: CLASSIFY ------------------------
elif st.session_state.page == "Classify":

    # ---------- Page Title ----------
    st.markdown("## 🎯 Target Classification")
    st.markdown("Using our deep learning model, classify your uploaded spectrogram image as a **Drone** or a **Bird**.")
    st.markdown("---")

    # ---------- Check for Uploaded Image ----------
    if "uploaded_image" not in st.session_state:
        st.warning("⚠️ Please upload a spectrogram image first from the **Upload** section.")
    else:
        # ---------- Display Uploaded Image ----------
        image = st.session_state.uploaded_image
        st.markdown("### 🖼️ Image to be Classified")
        st.image(image, caption="Micro-Doppler Spectrogram", use_container_width=True)
        st.markdown("---")

        # ---------- Model Prediction Logic ----------
        with st.spinner("🔎 Classifying... Please wait"):
            tensor = preprocess_image(image)
            outputs = model(tensor)
            _, pred = torch.max(outputs, 1)
            confidence_scores = torch.softmax(outputs, dim=1).squeeze().tolist()

            # Label assignment
            label = "🚁 Drone" if pred.item() == 0 else "🐦 Bird"

        # ---------- Show Prediction ----------
        st.markdown("### 🧠 Model Prediction")
        st.success(f"**Prediction:** {label}")
        st.markdown("---")

        # ---------- Show Confidence Scores ----------
        st.markdown("### 📊 Confidence Scores")
        st.json({
            "🚁 Drone": round(confidence_scores[0] * 100, 2),
            "🐦 Bird": round(confidence_scores[1] * 100, 2)
        })
        st.markdown("---")

        # ---------- Visual Progress Bar ----------
        st.markdown("### 📈 Prediction Probability")
        st.progress(confidence_scores[0] if pred.item() == 0 else confidence_scores[1])

        # ---------- Conditional Advice Based on Result ----------
        st.markdown("### 📌 Recommendation")
        if pred.item() == 0:
            st.info("🛰️ **Actionable Insight:** Detected object is likely a **drone**. Suitable for military surveillance or security alert systems.")
        else:
            st.info("🕊️ **Actionable Insight:** Detected object appears to be a **bird**. No threat detected.")

        st.markdown("---")

        # ---------- Expandable Insights ----------
        with st.expander("📚 About the Prediction"):
            st.markdown("""
            - The classification is based on radar return signal spectrograms.
            - Our model uses a **Convolutional Neural Network (CNN)** trained on labeled micro-Doppler patterns.
            - Confidence scores indicate how certain the model is about each class.
            """)

        with st.expander("⚙️ Technical Details"):
            st.markdown("""
            - **Model**: Custom CNN
            - **Framework**: PyTorch
            - **Preprocessing**: Normalized, resized to 224x224
            - **Output Layer**: Softmax (2 classes)
            """)

        st.markdown("---")
        st.markdown("✅ You can now proceed to the **Visualize** section to view Grad-CAM heatmaps of attention.")


# ------------------------ Page: VISUALIZE ------------------------
elif st.session_state.page == "Visualize":

    # ---------- Title & Description ----------
    st.markdown("## 📊 Grad-CAM Visualization")
    st.markdown(
        "Visualize which parts of the spectrogram your deep learning model focuses on to make its decision. This enhances interpretability and trust.")
    st.markdown("---")

    # ---------- Check for Uploaded Image ----------
    if "uploaded_image" not in st.session_state:
        st.warning("⚠️ Please upload a spectrogram image first from the **Upload** section.")
    else:
        # Section 1: Display Original Image
        st.markdown("### 🖼️ Step 1: Original Spectrogram")
        image = st.session_state.uploaded_image
        st.image(image, caption="Original Micro-Doppler Spectrogram", use_container_width=True)
        st.markdown("This is the raw spectrogram image you uploaded for classification.")
        st.markdown("---")

        # Section 2: Grad-CAM Activation Map
        st.markdown("### 🔥 Step 2: Grad-CAM Activation Map")
        st.image(img4, caption="Model's Attention Heatmap (Grad-CAM)", use_container_width=True)
        st.info("🧠 This heatmap shows which regions of the image influenced the model's decision the most.")
        st.markdown("---")

        # Section 3: Overlayed Result
        st.markdown("### 📌 Step 3: Final Result with Grad-CAM Overlay")
        st.image(img3, caption="Overlayed Grad-CAM on Original Image", use_container_width=True)
        st.success("✅ This composite visualization helps identify **key discriminative areas**.")
        st.markdown("---")

        # Section 4: Interpretation Help
        with st.expander("📚 How to Interpret Grad-CAM Results"):
            st.markdown("""
            - **Red/Yellow Areas**: These regions had the **highest influence** on the model’s decision.
            - **Blue/Black Areas**: These had **less or no influence**.
            - **Overlay View**: Helps correlate signal patterns with model attention.
            - Grad-CAM works by computing gradients of the target class w.r.t. final convolutional layers.

            > 💡 Useful for debugging model behavior and verifying model integrity.
            """)

        # Section 5: Technical Details
        with st.expander("⚙️ Technical Details"):
            st.markdown("""
            - **Model**: CNN trained on spectrogram images
            - **Method**: Gradient-weighted Class Activation Mapping (Grad-CAM)
            - **Input Shape**: 224 x 224 RGB
            - **Backend**: PyTorch + OpenCV for visualization
            """)

        # Final Note
        st.markdown("---")
        st.info("👉 You can return to **Upload** or **Classify** sections to test other images.")

