import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import gdown
import gc
import time

# --- Page Config ---
st.set_page_config(page_title="Fracture.AI", page_icon="🦴", layout="centered")

# --- Model Architecture ---
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

# --- Google Drive Downloader ---
@st.cache_resource
def download_model():
    file_id = '1n2lrck3Upzk-eoJ7h00zlRJZ-kHKnUgH'
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'fracture_model.pth'
    
    if not os.path.exists(output):
        with st.spinner('Downloading Model from Google Drive... Please wait.'):
            try:
                gdown.download(url, output, quiet=False)
            except Exception as e:
                st.error(f"Download failed: {e}")
                return None
    return output

# --- Load Model with Memory Optimization ---
@st.cache_resource
def load_my_model():
    model_path = download_model()
    if model_path is None or not os.path.exists(model_path):
        return None
        
    model = HybridModel(num_classes=2)
    try:
        # Loading directly to CPU to save RAM
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        model.load_state_dict(checkpoint)
        model.eval()
        
        # Free up RAM immediately
        del checkpoint
        gc.collect()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# --- X-Ray Validation Logic ---
def is_valid_xray(img):
    img_gray = img.convert('L')
    img_np = np.array(img_gray)
    hist, _ = np.histogram(img_np.flatten(), 256, [0, 256])
    peak_count = np.sum(hist > (img_np.size / 50))
    if img.mode != 'L':
        color_diff = np.mean(np.array(img)) - np.mean(img_np)
        if abs(color_diff) > 20: return False
    return peak_count <= 35

# --- UI Layout ---
st.title("🦴 Fracture.AI")
st.markdown("### Hybrid Deep Learning Diagnostic Tool")

if model is None:
    st.error("Model could not be initialized. Please check Google Drive link.")
    st.stop()

uploaded_file = st.file_uploader("Upload a Bone X-ray...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if st.button('Analyze Scan'):
        if not is_valid_xray(image):
            st.error("❌ Invalid Input: Please upload a clear grayscale Bone X-ray.")
        else:
            with st.spinner('Neural Network is analyzing...'):
                # Image Preprocessing
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                img_tensor = transform(image.convert('RGB')).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    prob = torch.softmax(output, dim=1)
                    conf, pred = torch.max(prob, 1)
                
                confidence = conf.item() * 100
                result = "Fractured" if pred.item() == 0 else "Normal"
                
                if result == "Fractured":
                    st.error(f"## Result: {result} ({confidence:.2f}%)")
                else:
                    st.success(f"## Result: {result} ({confidence:.2f}%)")
                
                st.info("Disclaimer: AI generated result. Consult a doctor.")