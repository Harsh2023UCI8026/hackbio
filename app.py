import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import time

# --- Page Config ---
st.set_page_config(
    page_title="Fracture.AI | Bone Diagnostic Tool",
    page_icon="🦴",
    layout="centered"
)

# --- CSS for Better UI ---
st.markdown("""
    <style>
    .main { background-color: #0f172a; }
    .stButton>button { width: 100%; border-radius: 10px; height: 3em; background-image: linear-gradient(to right, #ec4899, #f43f5e); color: white; border: none; }
    .stAlert { border-radius: 15px; }
    </style>
    """, unsafe_allow_html=True)

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

# --- Load Model with Cache ---
@st.cache_resource
def load_my_model():
    model = HybridModel(num_classes=2)
    MODEL_PATH = "fracture_model.pth"
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint)
            model.eval()
            return model
        except Exception as e:
            st.error(f"Error loading model weights: {e}")
    return None

model = load_my_model()

# --- Advanced Pre-validation Logic ---
def is_valid_xray(img):
    img_gray = img.convert('L')
    img_np = np.array(img_gray)
    # Histogram analysis
    hist, _ = np.histogram(img_np.flatten(), 256, [0, 256])
    peak_count = np.sum(hist > (img_np.size / 50))
    
    # Color check
    if img.mode != 'L' and img.mode != '1':
        color_diff = np.mean(np.array(img)) - np.mean(img_np)
        if abs(color_diff) > 15: return False, "Image contains colors. X-rays should be grayscale."
        
    if peak_count > 38: 
        return False, "Complex texture detected. Please upload a clear bone X-ray scan."
        
    return True, "Valid"

# --- Sidebar ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2869/2869376.png", width=100)
    st.title("About Project")
    st.info("Fracture.AI uses a Hybrid CNN-ViT-Mamba architecture to analyze bone scans with high precision.")
    st.divider()
    st.write("📊 **Model Info:**")
    st.write("- Backbone: ResNet50")
    st.write("- Encoder: ViT-B16")
    st.write("- Logic: Mamba Layer")

# --- Main UI ---
st.title("🦴 Fracture.AI")
st.subheader("Hybrid Deep Learning Diagnostic Tool")

if model is None:
    st.warning("⚠️ Model file 'fracture_model.pth' not found. Please upload it to your GitHub repo.")
    st.stop()

uploaded_file = st.file_uploader("Upload Bone X-ray (PNG/JPG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption='Uploaded Scan', use_container_width=True)
    
    with col2:
        st.write("### Analysis Dashboard")
        if st.button('🚀 Start AI Scan'):
            valid, msg = is_valid_xray(image)
            
            if not valid:
                st.error(f"**Invalid Input:** {msg}")
            else:
                with st.spinner('Neural Networks Processing...'):
                    time.sleep(1.5) # Aesthetic delay
                    
                    # Transform
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
                    
                    # Display Result
                    if result == "Fractured":
                        st.markdown(f"<h2 style='color: #f43f5e;'>Result: {result}</h2>", unsafe_allow_html=True)
                        st.progress(confidence/100)
                    else:
                        st.markdown(f"<h2 style='color: #4ade80;'>Result: {result}</h2>", unsafe_allow_html=True)
                        st.progress(confidence/100)
                        
                    st.write(f"**Confidence Score:** {confidence:.2f}%")
                    
                    if confidence > 95:
                        st.success("High confidence prediction.")
                    elif confidence < 75:
                        st.warning("Low confidence. Result may be unreliable.")

st.divider()
st.caption("⚖️ **Disclaimer:** This tool is for research purposes only. Always consult a certified medical professional for diagnosis.")