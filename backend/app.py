from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import cv2
import base64
import os
import time
from typing import Dict, List, Optional

# Define the model architecture
class RetinalModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Load the torchvision EfficientNet-B3
        base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
        
        # Freeze feature extractor
        for p in base.features.parameters():
            p.requires_grad = False
        
        # Replace classifier with custom head
        in_feats = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.3, inplace=True),
            nn.Linear(in_feats, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Expose pieces for Grad-CAM
        self.features = base.features
        self.avgpool = base.avgpool
        self.classifier = base.classifier

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

# Class names mapping
CLASS_NAMES = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# Initialize FastAPI app
app = FastAPI(title="Retinal OCT Image Analyzer API",
              description="API for analyzing OCT retinal images and predicting disease categories",
              version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Global variables
IMG_SIZE = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model: Optional[RetinalModel] = None
model_ready = False

# Transform pipeline
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def load_model_task():
    """Background task to load the model"""
    global model, model_ready
    print("Starting model loading...")
    start_time = time.time()
    
    try:
        # Initialize the model
        model = RetinalModel(num_classes=len(CLASS_NAMES)).to(device)
        
        # Look for model in different possible locations
        checkpoint_paths = [
            "best_model.pth",  # In the current directory
            os.path.join(os.getcwd(), "best_model.pth"),  # Absolute path
            "/etc/secrets/best_model.pth"  # Render secret files location
        ]
        
        model_loaded = False
        for path in checkpoint_paths:
            if os.path.exists(path):
                print(f"Found model at {path}")
                # Setting weights_only=True for security and better loading performance
                model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                model_loaded = True
                break
        
        if not model_loaded:
            print("WARNING: No model file found. Using random weights.")
        
        # Set model to evaluation mode
        model.eval()
        model_ready = True
        print(f"Model loaded successfully in {time.time() - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        # Initialize with random weights if loading fails
        model = RetinalModel(num_classes=len(CLASS_NAMES)).to(device)
        model.eval()
        model_ready = True
        print("Initialized model with random weights after error")

@app.on_event("startup")
async def startup_event(background_tasks: BackgroundTasks):
    """Start model loading in background"""
    background_tasks.add_task(load_model_task)
    print("Application startup - model loading initiated in background")

def generate_grad_cam(model, img_tensor, target_class=None):
    """Generate Grad-CAM for the given image tensor"""
    # Make sure tensor requires grad
    img_tensor = img_tensor.clone().detach().requires_grad_(True)
    
    # Get the feature module
    feature_module = model.features
    layers = list(feature_module.children())
    target_layer = layers[-1]  # last conv block
    
    # Hook definition
    activations, gradients = [], []
    def forward_hook(_, __, out):
        out.requires_grad_(True)
        activations.append(out)
        return out
    
    def backward_hook(_, __, grad_out):
        gradients.append(grad_out[0])
        return None
    
    # Register hooks
    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(img_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Zero gradients and backward pass
    model.zero_grad()
    score = output[0, target_class]
    score.backward(retain_graph=True)
    
    # Remove hooks
    forward_handle.remove()
    backward_handle.remove()
    
    # Get activations and gradients
    act = activations[0].detach()
    grad = gradients[0].detach()
    
    # Calculate weights and CAM
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * act).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    return cam, target_class

def image_to_base64(img_array):
    """Convert image array to base64 string"""
    success, encoded_img = cv2.imencode('.png', img_array)
    if success:
        return base64.b64encode(encoded_img).decode('utf-8')
    return None

@app.get("/")
async def root():
    """Root endpoint"""
    global model_ready
    return {
        "message": "OCT Image Analysis API. POST an image to /predict/ to get predictions.",
        "model_status": "ready" if model_ready else "loading"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global model_ready
    return {"status": "healthy", "model_loaded": model_ready}

@app.post("/predict/", response_model=Dict)
async def predict(file: UploadFile = File(...)):
    """Endpoint to predict and generate Grad-CAM for uploaded OCT image"""
    global model, model_ready
    
    # Check if model is ready
    if not model_ready:
        raise HTTPException(status_code=503, detail="Model is still loading, please try again in a moment")
    
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    
    try:
        # Read and process the image
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Original image for display
        orig_img = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        
        # Process for model
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = CLASS_NAMES[pred_idx]
            confidence = probs[0, pred_idx].item()
        
        # Generate Grad-CAM
        cam, _ = generate_grad_cam(model, img_tensor, target_class=pred_idx)
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
        
        # Get prediction probabilities for all classes
        class_probs = {CLASS_NAMES[i]: float(probs[0, i].item()) for i in range(len(CLASS_NAMES))}
        
        # Convert images to base64 for response
        orig_b64 = image_to_base64(cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
        heatmap_b64 = image_to_base64(heatmap)
        overlay_b64 = image_to_base64(overlay)
        
        # Prepare response
        response = {
            "prediction": pred_label,
            "confidence": confidence,
            "class_probabilities": class_probs,
            "images": {
                "original": orig_b64,
                "heatmap": heatmap_b64,
                "overlay": overlay_b64
            }
        }
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Run app with uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)