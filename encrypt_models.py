from cryptography.fernet import Fernet
import os

# Generate and save your encryption key (keep this secret!)
KEY = Fernet.generate_key()
with open("model_key.key", "wb") as f:
    f.write(KEY)
print(f"Generated Key: {KEY.decode()}")  # Save this for later

# Initialize cipher
cipher = Fernet(KEY)

# Define model paths (UPDATE THESE TO YOUR ACTUAL PATHS)
mirnet_path = r"E:\Machine Learning\Capstone\MIRNetv2\Enhancement\pretrained_models\enhancement_lol.pth"
yolo_path = r"E:\Machine Learning\Capstone\yolov8n.pt"  # Update if different

# Encrypt MIRNet model
try:
    with open(mirnet_path, "rb") as f:
        mirnet_encrypted = cipher.encrypt(f.read())
    with open("enhancement_lol.enc", "wb") as f:
        f.write(mirnet_encrypted)
    print("MIRNet model encrypted successfully!")
except FileNotFoundError:
    print(f"Error: MIRNet model not found at {mirnet_path}")
    print("Please verify the path and try again")

# Encrypt YOLO model
try:
    with open(yolo_path, "rb") as f:
        yolo_encrypted = cipher.encrypt(f.read())
    with open("yolov8n.enc", "wb") as f:
        f.write(yolo_encrypted)
    print("YOLO model encrypted successfully!")
except FileNotFoundError:
    print(f"Error: YOLO model not found at {yolo_path}")
    print("Please verify the path and try again")