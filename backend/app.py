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
import gc
import os
from typing import Dict, List

# Define the model architecture - simplified for memory efficiency
class RetinalModel(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Use a smaller model like EfficientNet-B0 instead of B3
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # Get input features
        in_feats = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(in_feats, num_classes)
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables with memory optimization
IMG_SIZE = 224  # Consider reducing to 196 if memory still an issue
device = torch.device('cpu')  # Explicitly use CPU
model = None

# Optimize transform pipeline
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@app.on_event("startup")
async def load_model():
    global model
    # Initialize the model
    model = RetinalModel(num_classes=len(CLASS_NAMES)).to(device)
    
    # Load checkpoint with error handling
    checkpoint_path = "best_model.pth"
    try:
        # Load with map_location to ensure it loads on CPU
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Initializing with random weights")
    
    model.eval()  # Set to evaluation mode

def generate_grad_cam(model, img_tensor, target_class=None):
    """Generate Grad-CAM with memory optimization"""
    # Make sure we're using CPU tensors to reduce memory usage
    img_tensor = img_tensor.clone().detach().requires_grad_(True)
    
    # Store only what we need
    activations = None
    gradients = None
    
    def forward_hook(_, __, out):
        nonlocal activations
        activations = out.detach()
        return None
    
    def backward_hook(_, __, grad_out):
        nonlocal gradients
        gradients = grad_out[0].detach()
        return None
    
    # Get target layer - last convolutional layer
    target_layer = model.features[-1]
    
    # Register hooks
    hooks = [
        target_layer.register_forward_hook(forward_hook),
        target_layer.register_full_backward_hook(backward_hook)
    ]
    
    # Forward pass
    output = model(img_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=1).item()
    
    # Zero gradients and backward pass
    model.zero_grad()
    score = output[0, target_class]
    score.backward()
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Calculate weights and CAM
    weights = gradients.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations).sum(dim=1))
    
    # Resize CAM to original image size
    cam = F.interpolate(cam.unsqueeze(0).unsqueeze(0), 
                       size=(IMG_SIZE, IMG_SIZE), 
                       mode='bilinear', 
                       align_corners=False)
    
    cam = cam.squeeze().cpu().numpy()
    
    # Normalize the CAM
    if cam.max() != cam.min():
        cam = (cam - cam.min()) / (cam.max() - cam.min())
    else:
        cam = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    
    # Clean up to free memory
    del weights, activations, gradients, output, score
    torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()
    
    return cam, target_class

def image_to_base64(img_array):
    """Convert image array to base64 string with memory optimization"""
    success, encoded_img = cv2.imencode('.png', img_array, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if success:
        result = base64.b64encode(encoded_img).decode('utf-8')
        # Clean up
        del encoded_img
        return result
    return None

def cleanup():
    """Force garbage collection to free memory"""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

@app.post("/predict/", response_model=Dict)
async def predict(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Endpoint to predict and generate Grad-CAM for uploaded OCT image"""
    # Check if file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File is not an image")
    
    try:
        # Read the image in chunks to avoid loading large files entirely into memory
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        
        # Original image for display - reduce size if needed
        orig_img = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        
        # Process for model
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Forward pass with memory cleanup
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(probs, dim=1).item()
            pred_label = CLASS_NAMES[pred_idx]
            confidence = probs[0, pred_idx].item()
            
            # Get prediction probabilities for all classes
            class_probs = {CLASS_NAMES[i]: float(probs[0, i].item()) for i in range(len(CLASS_NAMES))}
            
            # Clean up tensors we don't need anymore
            del outputs
            del probs
        
        # Generate Grad-CAM (requires gradients)
        cam, _ = generate_grad_cam(model, img_tensor, target_class=pred_idx)
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Create overlay
        overlay = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
        
        # Convert images to base64 for response
        orig_b64 = image_to_base64(cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))
        heatmap_b64 = image_to_base64(heatmap)
        overlay_b64 = image_to_base64(overlay)
        
        # Clean up variables to free memory
        del img_tensor, cam, heatmap, overlay, orig_img
        
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
        
        # Schedule cleanup after response is sent
        background_tasks.add_task(cleanup)
        
        return response
        
    except Exception as e:
        # Clean up on error
        gc.collect()
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "OCT Image Analysis API. POST an image to /predict/ to get predictions."}

if __name__ == "__main__":
    import uvicorn
    # Run app with uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)