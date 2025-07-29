import sys
import os
import cv2
import numpy as np
import torch
import segmentation_models_pytorch as smp
import albumentations as A
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QLabel, QFileDialog, QMessageBox, QCheckBox)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class VideoSegmentationApp(QMainWindow):
    def __init__(self, model_path="UNet_model.pth", model_folder="./models"):
        super().__init__()
        self.setWindowTitle("Video Segmentation and Diameter Measurement")
        self.setGeometry(100, 100, 1000, 600)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load segmentation model
        self.model = self.load_model(model_path, self.device, model_folder)
        
        # Define normalization pipeline
        self.transform = A.Compose([
            A.Resize(
                height=256,
                width=256,
                interpolation=cv2.INTER_LINEAR,
                p=1.0
            ),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)
            ),
            A.pytorch.ToTensorV2()
        ])
        
        # Initialize video variables
        self.video_path = None
        self.cap = None
        self.frames = []
        self.segmented_frames = []
        self.diameters = []
        self.current_frame_idx = 0
        
        # Setup GUI
        self.setup_gui()
        
    def load_model(self, path, device, parent_folder="./models"):
        name = os.path.basename(path)
        model_type = name.split("_")[0]
        try:
            model = getattr(smp, model_type)
        except AttributeError:
            raise ValueError(f"Invalid model type in filename: {name}")
        
        model = model(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1
        )
        model.name = os.path.splitext(name)[0]
        try:
            model.load_state_dict(torch.load(os.path.join(parent_folder, path), weights_only=True))
        except FileNotFoundError:
            raise FileNotFoundError(f"Model file {path} not found in {parent_folder}")
        model.to(device)
        model.eval()
        return model
    
    def setup_gui(self):
        # Main widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        
        # Left panel: Video display and controls
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        
        # Video display label
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setFixedSize(640, 480)
        self.left_layout.addWidget(self.video_label)
        
        # Checkboxes for display options
        self.show_segmentation = QCheckBox("Show Segmentation Mask", self)
        self.show_segmentation.setChecked(True)
        self.left_layout.addWidget(self.show_segmentation)
        
        self.show_diameter = QCheckBox("Show Diameter Value", self)
        self.show_diameter.setChecked(True)
        self.left_layout.addWidget(self.show_diameter)
        
        # Diameter display label
        self.diameter_label = QLabel("Diameter: N/A", self)
        self.left_layout.addWidget(self.diameter_label)
        
        # Buttons
        self.upload_btn = QPushButton("Upload Video", self)
        self.upload_btn.clicked.connect(self.upload_video)
        self.left_layout.addWidget(self.upload_btn)
        
        self.play_btn = QPushButton("Play Segmented Video", self)
        self.play_btn.clicked.connect(self.start_playback)
        self.play_btn.setEnabled(False)
        self.left_layout.addWidget(self.play_btn)
        
        self.main_layout.addWidget(self.left_panel)
        
        # Right panel: Diameter graph
        self.figure = Figure(figsize=(4, 3))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Frame Number")
        self.ax.set_ylabel("Diameter (pixels)")
        self.ax.set_title("Diameter Over Time")
        self.main_layout.addWidget(self.canvas)
        
        # Timer for video playback
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
    def upload_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Could not open video file.")
                return
            self.process_video()
            self.play_btn.setEnabled(True)
            
    def process_video(self):
        self.frames = []
        self.segmented_frames = []
        self.diameters = []
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.frames.append(frame)
            
            # Apply segmentation
            segmented = self.apply_segmentation(frame)
            self.segmented_frames.append(segmented)
            
            # Calculate diameter
            diameter = self.calculate_diameter(segmented)
            self.diameters.append(diameter)
            
        self.cap.release()
        self.update_graph()
        
    def apply_segmentation(self, frame):
        # Convert frame to RGB (OpenCV uses BGR)
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Apply normalization pipeline
        transformed = self.transform(image=img)
        img_tensor = transformed['image']  # Shape: (C, H, W)
        
        # Add batch dimension and move to device
        img_tensor = img_tensor.unsqueeze(0).to(self.device)  # Shape: (1, C, H, W)
        
        # Perform model inference
        with torch.no_grad():
            output = self.model(img_tensor)  # Assume output is (1, 1, H, W)
        
        # Process output to binary mask
        mask = torch.sigmoid(output).squeeze().cpu().numpy()  # Shape: (H, W)
        mask = (mask > 0.5).astype(np.uint8) * 255  # Binary mask (0 or 255)
        
        # Resize mask back to original frame size
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask
    
    def calculate_diameter(self, mask):
        # Find contours in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0.0
        
        # Assume largest contour is the object of interest
        contour = max(contours, key=cv2.contourArea)
        # Fit a circle to the contour and get diameter
        (x, y), radius = cv2.minEnclosingCircle(contour)
        diameter = 2 * radius
        return diameter
    
    def overlay_segmentation(self, frame, mask):
        # Create colored overlay for segmentation
        overlay = frame.copy()
        if self.show_segmentation.isChecked():
            mask_colored = np.zeros_like(frame)
            mask_colored[mask == 255] = [0, 255, 0]  # Green for segmented area
            cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0, overlay)
        return overlay
    
    def update_graph(self):
        # Update diameter graph up to current frame
        self.ax.clear()
        self.ax.plot(range(self.current_frame_idx + 1), self.diameters[:self.current_frame_idx + 1], 'b-')
        self.ax.set_xlabel("Frame Number")
        self.ax.set_ylabel("Diameter (pixels)")
        self.ax.set_title("Diameter Over Time")
        self.ax.grid(True)
        self.canvas.draw()
        
    def start_playback(self):
        self.current_frame_idx = 0
        self.timer.start(1000 // 30)  # 30 FPS
        
    def update_frame(self):
        if self.current_frame_idx >= len(self.frames):
            self.timer.stop()
            return
            
        frame = self.frames[self.current_frame_idx]
        mask = self.segmented_frames[self.current_frame_idx]
        diameter = self.diameters[self.current_frame_idx]
        
        # Overlay segmentation based on checkbox
        display_frame = self.overlay_segmentation(frame, mask)
        
        # Overlay diameter text if checked
        if self.show_diameter.isChecked():
            cv2.putText(display_frame, f"Diameter: {diameter:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Update diameter label
        self.diameter_label.setText(f"Diameter: {diameter:.2f} pixels")
        
        # Convert frame to QImage for display
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
        
        # Update graph
        self.update_graph()
        
        self.current_frame_idx += 1

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoSegmentationApp(model_path="UNet_model.pth", model_folder="./models")
    window.show()
    sys.exit(app.exec())