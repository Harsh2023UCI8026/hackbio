import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os

# --- Page Configuration ---
st.set_page_config(page_title="Fracture.AI", page_icon="🦴")

# --- Model Architecture (Same as your training) ---
class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None) 
        self.features = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        return torch.flatten(self.features(x), 1)

class VisionMamba(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, 768))
    def forward(self, x):
        return self.layer(x)

class ViTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = models.vit_b_16(weights=None)
        self.vit.heads = nn.Identity()
    def forward(self, x):
        return self.vit(x)

class HybridModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = CNNBackbone()
        self.vit = ViTEncoder()
        self.mamba = VisionMamba()
        self.classifier = nn.Sequential(
            nn.Linear(2048+768, 512), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, num_classes)
        )
    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        vit_feat = self.mamba(vit_feat)
        return self.classifier(torch.cat([cnn_feat, vit_feat], dim=1))

# --- Load Model Function ---
@st.cache_resource
def load_model():
    model = HybridModel(num_classes=2)
    MODEL_PATH = "fracture_model.pth"
    if os.path.exists(MODEL_PATH):
        # Streamlit cloud uses CPU by default
        checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    return None

model = load_model()

# --- UI ---
st.title("🦴 Fracture.AI")
st.write("Upload an X-ray image to detect fractures.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Analyze'):
        # Transform
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            prob = torch.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)
        
        classes = ['Fractured', 'Normal']
        result = classes[pred.item()]
        confidence = conf.item() * 100
        
        if result == 'Fractured':
            st.error(f"Prediction: {result} ({confidence:.2f}%)")
        else:
            st.success(f"Prediction: {result} ({confidence:.2f}%)")