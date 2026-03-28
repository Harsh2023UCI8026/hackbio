import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# --- Force Device to CPU for Stability on Local PC ---
device = torch.device("cpu")

# --- Model Architecture (CNN + ViT + Mamba Hybrid) ---

class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights=None) 
        self.features = nn.Sequential(*list(resnet.children())[:-1])
    def forward(self, x):
        x = self.features(x)
        return torch.flatten(x, 1)

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
            nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        cnn_feat = self.cnn(x)
        vit_feat = self.vit(x)
        vit_feat = self.mamba(vit_feat)
        x = torch.cat([cnn_feat, vit_feat], dim=1)
        return self.classifier(x)

# --- Initialize and Load Model with Error Handling ---
num_classes = 2 
model = HybridModel(num_classes)

MODEL_PATH = "fracture_model.pth"

if os.path.exists(MODEL_PATH):
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        print("✅ Success: Model loaded to CPU perfectly!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ Error: {MODEL_PATH} not found in current directory!")

# Transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def is_valid_xray_image(pil_img):
    """
    Checks if a PIL image is grayscale and looks like an X-ray histogram-wise.
    Real-world X-rays have a specific grayscale distribution. Desktop screenshots don't.
    """
    # 1. Grayscale check
    grayscale_img = pil_img.convert('L')
    img_np = np.array(grayscale_img)
    
    # Simple check for color variety that X-rays don't have. Desktop screenshots do.
    # We compare the grayscale version vs. original colors (if original was color)
    if pil_img.mode != 'L':
        original_np = np.array(pil_img)
        # If mean diff between RGB and L is high, it's colored
        color_diff = np.mean(original_np) - np.mean(grayscale_img)
        if abs(color_diff) > 20: # Colored images are definitely not X-rays
            return False, "Image is colored. X-rays are grayscale."
            
    # 2. Basic Gray-Histogram check: Real X-rays have a narrow-ish peak and long tail.
    # Desktop screenshots have wild distributions with multiple peaks.
    hist, _ = np.histogram(img_np.flatten(), 256, [0, 256])
    peak_count = np.sum(hist > (img_np.size / 50)) # How many gray levels have substantial pixels
    if peak_count > 30: # If too many gray levels are heavily represented, it's not a standard X-ray.
        return False, "Grayscale distribution is too wide for a standard X-ray."

    return True, "Valid grayscale X-ray candidate."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file uploaded'})
    
    file = request.files['file']
    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # --- Pre-Validation Step ---
        is_valid, validation_message = is_valid_xray_image(img)
        if not is_valid:
             return jsonify({
                'status': 'invalid',
                'prediction': 'Invalid Input',
                'message': f"Prediction aborted. {validation_message} Please upload a clear Bone X-ray image.",
                'confidence': 0.0
            })

        # Preprocess
        input_tensor = transform(img.convert('RGB')).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(input_tensor)
            prob = torch.softmax(output, dim=1)
            confidence, pred = torch.max(prob, 1)
        
        conf_score = round(confidence.item() * 100, 2)
        classes = ['Fractured', 'Normal']
        
        result = classes[pred.item()]

        # UI level Suspect Detection: Aclinical model being 100% confident on a desktop screenshot is suspect.
        # This part should be handled in the frontend UI based on the response structure.
        return jsonify({
            'status': 'success',
            'prediction': result,
            'confidence': conf_score,
            'prediction_status': 'processed' # Added status field to help frontend
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    print("🚀 Fracture AI Server Starting...")
    app.run(debug=True, port=5000)