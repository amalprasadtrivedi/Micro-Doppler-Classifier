import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from models.cnn_model import CNNModel

# --------------------------------------------
# 🔄 Transform applied to input images
# Should match the training pipeline
# --------------------------------------------
image_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])


# ------------------------------------------------------
# 📥 Function to load and preprocess an image from path
# ------------------------------------------------------
def load_image(image_path: str) -> torch.Tensor:
    """
    Loads and preprocesses an image for prediction or Grad-CAM.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Transformed image tensor (with batch dimension).
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = image_transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor


# ------------------------------------------------------
# 🔁 Decode predicted class index into a label
# ------------------------------------------------------
def decode_prediction(pred_idx: int) -> str:
    """
    Converts prediction index to label.

    Args:
        pred_idx (int): Index predicted by model (0 or 1).

    Returns:
        str: Class label ('Bird' or 'Drone').
    """
    label_map = {0: "Bird", 1: "Drone"}
    return label_map.get(pred_idx, "Unknown")


# ------------------------------------------------------
# 🧠 Load the trained CNN model from file
# ------------------------------------------------------
def load_model(model_path: str = "models/cnn_model.pth") -> CNNModel:
    """
    Loads the trained CNN model.

    Args:
        model_path (str): Path to the saved model weights.

    Returns:
        CNNModel: Loaded and ready-to-use model.
    """
    model = CNNModel()
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model


# ------------------------------------------------------
# 📊 Predict the class for a given image using the model
# ------------------------------------------------------
def predict_image(model: CNNModel, image_tensor: torch.Tensor) -> tuple:
    """
    Predicts the class of a given image using the model.

    Args:
        model (CNNModel): Loaded model.
        image_tensor (torch.Tensor): Input image tensor.

    Returns:
        tuple: (predicted_label, predicted_index, confidence)
    """
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
        label = decode_prediction(pred_idx.item())
        return label, pred_idx.item(), confidence.item()
