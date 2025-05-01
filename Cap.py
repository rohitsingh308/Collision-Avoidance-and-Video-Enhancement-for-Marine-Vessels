import os
import cv2
import time
import torch
import numpy as np
from PIL import Image
from runpy import run_path
from ultralytics import YOLO
from torchvision.transforms import ToTensor, ToPILImage
import matplotlib.pyplot as plt  # Import matplotlib

def get_weights_and_parameters(task, parameters):
    if task == 'lowlight_enhancement':
        weights = r"E:\\Machine Learning\\Capstone\\MIRNetv2\\Enhancement\\pretrained_models\\enhancement_lol.pth"

    return weights, parameters

# Set task and parameters      
task = 'lowlight_enhancement'
parameters = {
    'inp_channels': 3,
    'out_channels': 3,
    'n_feat': 80,
    'chan_factor': 1.5,
    'n_RRG': 4,
    'n_MRB': 2,
    'height': 3,
    'width': 2,
    'bias': False,
    'scale': 1,
    'task': task
}

weights, parameters = get_weights_and_parameters(task, parameters)

# Load MIRNetv2 architecture
load_arch = run_path(os.path.join('MIRNetv2','basicsr', 'models', 'archs', 'mirnet_v2_arch.py'))
mir_model = load_arch['MIRNet_v2'](**parameters)
checkpoint = torch.load(weights)
mir_model.load_state_dict(checkpoint['params'])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = mir_model.to(device).eval()

yolo_model = YOLO("E:\Machine Learning\Capstone\yolov8n.pt")  # Provide correct path if not in working dir

def preprocess_frame(frame):
    h, w, _ = frame.shape
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8
    padded = cv2.copyMakeBorder(frame, 0, new_h - h, 0, new_w - w, cv2.BORDER_REFLECT)
    return padded

cap = cv2.VideoCapture(0)  # Use 0 for default camera
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('enhanced_output.mp4', fourcc, 20.0, (frame_width, frame_height))

to_tensor = ToTensor()
to_pil_image = ToPILImage()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # YOLO Detection
    results = yolo_model(frame, verbose=False)[0]
    boxes = results.boxes

    # Frame Enhancement
    padded = preprocess_frame(frame)
    img_rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    img_tensor = to_tensor(Image.fromarray(img_rgb)).unsqueeze(0).cpu()

    with torch.no_grad():
        enhanced = mir_model(img_tensor)
    enhanced_img = enhanced.squeeze().detach().cpu()
    enhanced_img = to_pil_image(enhanced_img)
    enhanced_img = cv2.cvtColor(np.array(enhanced_img), cv2.COLOR_RGB2BGR)

    # Crop back to original size
    enhanced_img = enhanced_img[:frame.shape[0], :frame.shape[1]]

    # Draw YOLO boxes
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = box.conf[0]
        label = f"{yolo_model.names[cls]} {conf:.2f}"
        cv2.rectangle(enhanced_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(enhanced_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # FPS Calculation
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(enhanced_img, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display enhanced frame using matplotlib
    enhanced_img_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
    
    # Display image using matplotlib
    plt.imshow(enhanced_img_rgb)
    plt.axis('off')  # Remove axis labels
    plt.show(block=False)  # Show the image without blocking further execution
    plt.pause(0.001)  # Pause to update the frame and allow for continuous display

    out.write(enhanced_img)

cap.release()
out.release()
