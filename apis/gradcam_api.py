# gradcam_api.py
import io, torch, cv2, numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn

app = FastAPI()

# --- load model ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
num_classes = 7  
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_features, num_classes))
model.load_state_dict(torch.load("mobilenetv2_ham10000_balanced.pth", map_location=device))
model.eval().to(device)

# --- Grad-CAM helper ---
class GradCAM:
    def __init__(self, model, target_layer_name="18"):
        self.model = model
        self.target_layer = dict([*model.features.named_children()])[target_layer_name]
        self.gradients = None
        self.activations = None
        self.hook_layers()
    def hook_layers(self):
        def forward_hook(_, __, output): self.activations = output.detach()
        def backward_hook(_, __, grad_out): self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    def generate_cam(self, input_tensor):
        output = self.model(input_tensor)
        target_class = output.argmax(dim=1).item()
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        weights = self.gradients.mean(dim=[0, 2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam - cam.min()
        cam = cam / cam.max()
        return cam

gradcam = GradCAM(model)

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[.485,.456,.406], std=[.229,.224,.225])
])

@app.post("/gradcam")
async def gradcam_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = val_transform(img_pil).unsqueeze(0).to(device)
    cam = gradcam.generate_cam(input_tensor)

    # Overlay CAM on image
    img = np.array(img_pil.resize((224,224))) / 255.0
    cam_np = cam.squeeze().cpu().numpy()
    cam_np = cv2.resize(cam_np, (224,224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_np), cv2.COLORMAP_JET)
    overlay = np.float32(heatmap)/255 * 0.4 + img
    overlay = overlay / overlay.max()

    _, buffer = cv2.imencode('.jpg', cv2.cvtColor((overlay*255).astype(np.uint8), cv2.COLOR_RGB2BGR))
    return Response(buffer.tobytes(), media_type="image/jpeg")
