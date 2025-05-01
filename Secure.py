import os
import cv2
import time
import torch
import numpy as np
import io
import hashlib
import hmac
from PIL import Image
from runpy import run_path
from cryptography.fernet import Fernet
from ultralytics import YOLO
from torchvision.transforms import ToTensor

# ====================== SECURITY SETUP ======================
MODEL_ENCRYPTION_KEY = b'h6CSQ3wAocZJqSA4NgbajRHQBAoBtBbx1KH8abqAQEM='
COMMAND_HMAC_KEY = "Capstone"
cipher_suite = Fernet(MODEL_ENCRYPTION_KEY)

# ====================== SECURE FUNCTIONS ======================
def generate_secure_command(direction, confidence):
    """Generate a command with integrity verification"""
    command = f"{direction}|{confidence:.2f}"
    signature = hmac.new(COMMAND_HMAC_KEY.encode(), command.encode(), hashlib.sha256).hexdigest()
    return f"{command}|{signature}"

def verify_command(secure_command):
    """Verify command integrity"""
    try:
        command_part, signature = secure_command.rsplit('|', 1)
        expected_signature = hmac.new(COMMAND_HMAC_KEY.encode(), command_part.encode(), hashlib.sha256).hexdigest()
        return signature == expected_signature, command_part.split('|')
    except:
        return False, None

def load_secure_yolo():
    """Special loader for encrypted YOLO model"""
    try:
        temp_path = "temp_yolo.pt"
        with open(r"E:\Machine Learning\Capstone\yolov8n.enc", 'rb') as f:
            encrypted_data = f.read()
        decrypted_data = cipher_suite.decrypt(encrypted_data)
        
        with open(temp_path, 'wb') as f:
            f.write(decrypted_data)
        
        model = YOLO(temp_path)
        os.remove(temp_path)
        return model
    except Exception as e:
        print(f"YOLO load failed: {str(e)}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        exit(1)

def load_encrypted_model(model_path):
    """Load encrypted PyTorch model"""
    with open(model_path, 'rb') as f:
        encrypted_data = f.read()
    decrypted_data = cipher_suite.decrypt(encrypted_data)
    buffer = io.BytesIO(decrypted_data)
    return torch.load(buffer, map_location='cpu')

def preprocess_frame(frame):
    h, w, _ = frame.shape
    new_h = ((h + 7) // 8) * 8
    new_w = ((w + 7) // 8) * 8
    padded = cv2.copyMakeBorder(frame, 0, new_h - h, 0, new_w - w, cv2.BORDER_REFLECT)
    return padded

def get_maneuver_direction(boxes, frame_width):
    """Determine maneuver direction based on object positions"""
    if not boxes:
        return "Hold Course", 1.0
    
    right_objects = sum(1 for box in boxes if (box.xyxy[0][0] + box.xyxy[0][2]) / 2 > frame_width / 2)
    left_objects = len(boxes) - right_objects
    
    if right_objects > left_objects:
        return "Turn Left", min(0.99, right_objects / (right_objects + left_objects))
    elif left_objects > right_objects:
        return "Turn Right", min(0.99, left_objects / (right_objects + left_objects))
    return "Hold Course", 1.0

# ====================== MAIN SYSTEM INIT ======================
try:
    # Load MIRNetv2
    load_arch = run_path(os.path.join('MIRNetv2', 'basicsr', 'models', 'archs', 'mirnet_v2_arch.py'))
    mir_model = load_arch['MIRNet_v2'](
        inp_channels=3,
        out_channels=3,
        n_feat=80,
        chan_factor=1.5,
        n_RRG=4,
        n_MRB=2,
        height=3,
        width=2,
        bias=False,
        scale=1,
        task='lowlight_enhancement'
    )
    checkpoint = load_encrypted_model(r"E:\Machine Learning\Capstone\MIRNetv2\Enhancement\pretrained_models\enhancement_lol.enc")
    mir_model.load_state_dict(checkpoint['params'])
    
    # Load YOLO
    yolo_model = load_secure_yolo()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = mir_model.to(device).eval()

except Exception as e:
    print(f"Secure model loading failed: {str(e)}")
    exit(1)

# ====================== MAIN LOOP ======================
cap = cv2.VideoCapture(0)
to_tensor = ToTensor()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()
    
    # Detection
    results = yolo_model(frame, verbose=False)[0]
    boxes = results.boxes
    
    # Enhancement
    padded = preprocess_frame(frame)
    img_tensor = to_tensor(Image.fromarray(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
    
    with torch.no_grad():
        enhanced = model(img_tensor)
    
    enhanced_img = enhanced.squeeze().clamp(0, 1).cpu().numpy()
    enhanced_img = (enhanced_img * 255).astype(np.uint8)
    enhanced_img = np.transpose(enhanced_img, (1, 2, 0))
    enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)[:frame.shape[0], :frame.shape[1]]
    
    # Determine maneuver command
    direction, confidence = get_maneuver_direction(boxes, frame.shape[1])
    secure_command = generate_secure_command(direction, confidence)
    
    # Verify command
    is_valid, (cmd, conf) = verify_command(secure_command)
    if not is_valid:
        cmd = "SECURITY ALERT: Invalid command"
    
    # Display
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(enhanced_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(enhanced_img, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(enhanced_img, f"Cmd: {cmd}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imshow('Secure Collision Avoidance', enhanced_img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()