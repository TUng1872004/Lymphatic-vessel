import sys
import os
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import albumentations as A
import logging
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QPushButton, QLabel, QSlider, QCheckBox, 
                            QLineEdit, QFileDialog, QMessageBox, QSpinBox,
                            QProgressBar, QFrame, QGroupBox, QGridLayout)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QFont
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VideoSegmentationApp(QMainWindow):
    def __init__(self, model_path="UnetPlusPlus_human_aug.pth", model_folder="./human"):
        super().__init__()
        self.setWindowTitle("Video Segmentation and Diameter Measurement - Enhanced")
        self.setGeometry(100, 100, 1400, 900)
        
        # Video properties
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.current_mask = None
        self.current_diameter = 0.0
        self.current_time = 0.0
        self.current_volume = 0.0
        self.times = []
        self.diameters = []
        self.volumes = []
        self.playing = False
        self.total_frames = 0
        self.current_frame_number = 0
        self.fps = 30
        self.calibration_factor = 1.0
        self.video_width = 0
        self.video_height = 0
        
        # Timer for playback
        self.timer = QTimer()
        self.timer.timeout.connect(self.process_next_frame)
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Load segmentation model
        try:
            self.model = self.load_model(model_path, self.device, model_folder)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load model: {str(e)}")
            self.model = None
        
        # Define normalization pipeline
        self.transform = A.Compose([
            A.Resize(height=256, width=256, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2()
        ])
        
        # Setup GUI
        self.setup_gui()
        logging.info("GUI initialized")
        
    def load_model(self, path, device, parent_folder="./human"):
        model_path = os.path.join(parent_folder, path)
        logging.info(f"Loading model from {model_path}")
        
        from huggingface_hub import hf_hub_download
        huggingface_model_repo = "tungDKT/Unet"
        huggingface_model_filename = "UnetPlusPlus_human_aug.pth"

        try:
            # Download from Hugging Face
            downloaded_model_path = hf_hub_download(
                repo_id=huggingface_model_repo,
                filename=huggingface_model_filename,
                repo_type="model"
            )

            # Infer model type from file name
            model_type = huggingface_model_filename.split("_")[0]
            model = getattr(smp, model_type)(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1
            )

            model.load_state_dict(torch.load(downloaded_model_path, map_location=device))
            model.name = huggingface_model_filename.replace(".pth", "")
        except Exception as e:
            logging.error(f"Failed to load model from Hugging Face: {e}")
            raise
            
        model.to(device)
        model.eval()
        logging.info(f"Model {model.name} loaded successfully")
        return model
        
    def setup_gui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel for controls
        left_panel = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setFixedWidth(350)
        
        # Video controls group
        video_group = QGroupBox("Video Controls")
        video_layout = QVBoxLayout(video_group)
        
        self.upload_btn = QPushButton("Upload Video")
        self.upload_btn.clicked.connect(self.upload_video)
        video_layout.addWidget(self.upload_btn)
        
        self.play_btn = QPushButton("Play Video")
        self.play_btn.clicked.connect(self.toggle_playback)
        self.play_btn.setEnabled(False)
        video_layout.addWidget(self.play_btn)
        
        # Time slider for seeking
        self.time_slider = QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(100)
        self.time_slider.valueChanged.connect(self.seek_video)
        video_layout.addWidget(QLabel("Seek:"))
        video_layout.addWidget(self.time_slider)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        video_layout.addWidget(self.progress_bar)
        
        # Time display
        self.time_label = QLabel("Time: 00:00 / 00:00")
        video_layout.addWidget(self.time_label)
        
        left_panel.addWidget(video_group)
        
        # Calibration group
        calib_group = QGroupBox("Calibration")
        calib_layout = QGridLayout(calib_group)
        
        calib_layout.addWidget(QLabel("Known size (mm):"), 0, 0)
        self.known_mm = QLineEdit()
        calib_layout.addWidget(self.known_mm, 0, 1)
        
        calib_layout.addWidget(QLabel("Known size (pixels):"), 1, 0)
        self.known_pixels = QLineEdit()
        calib_layout.addWidget(self.known_pixels, 1, 1)
        
        self.calibrate_btn = QPushButton("Calibrate")
        self.calibrate_btn.clicked.connect(self.calibrate)
        calib_layout.addWidget(self.calibrate_btn, 2, 0, 1, 2)
        
        left_panel.addWidget(calib_group)
        
        # Measurement controls group
        measure_group = QGroupBox("Measurement Controls")
        measure_layout = QVBoxLayout(measure_group)
        
        # X position slider
        measure_layout.addWidget(QLabel("X Position for Diameter:"))
        self.x_slider = QSlider(Qt.Horizontal)
        self.x_slider.setMinimum(0)
        self.x_slider.setMaximum(100)
        self.x_slider.setValue(50)
        self.x_slider.valueChanged.connect(self.update_display_if_paused)
        measure_layout.addWidget(self.x_slider)
        
        self.x_value_label = QLabel("X: 50")
        measure_layout.addWidget(self.x_value_label)
        
        # Checkboxes
        self.overlay_mask_cb = QCheckBox("Overlay Mask")
        self.overlay_mask_cb.setChecked(True)
        self.overlay_mask_cb.stateChanged.connect(self.update_display_if_paused)
        measure_layout.addWidget(self.overlay_mask_cb)
        
        self.manual_diameter_cb = QCheckBox("Manual Diameter")
        self.manual_diameter_cb.stateChanged.connect(self.update_display_if_paused)
        measure_layout.addWidget(self.manual_diameter_cb)
        
        left_panel.addWidget(measure_group)
        
        # Results group
        results_group = QGroupBox("Current Measurements")
        results_layout = QVBoxLayout(results_group)
        
        self.diameter_label = QLabel("Diameter: N/A")
        self.diameter_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.diameter_label.setStyleSheet("color: red;")
        results_layout.addWidget(self.diameter_label)
        
        self.volume_label = QLabel("Volume: N/A")
        self.volume_label.setFont(QFont("Arial", 12, QFont.Bold))
        self.volume_label.setStyleSheet("color: blue;")
        results_layout.addWidget(self.volume_label)
        
        self.time_display_label = QLabel("Current Time: N/A")
        results_layout.addWidget(self.time_display_label)
        
        left_panel.addWidget(results_group)
        
        left_panel.addStretch()
        
        # Right panel for video and graphs
        right_panel = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right_panel)
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("border: 1px solid gray; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setText("No video loaded")
        right_panel.addWidget(self.video_label)
        
        # Graphs
        self.setup_graphs(right_panel)
        
        # Add panels to main layout
        main_layout.addWidget(left_widget)
        main_layout.addWidget(right_widget, 1)
        
    def setup_graphs(self, parent_layout):
        # Create matplotlib figures
        self.diameter_fig = Figure(figsize=(8, 3))
        self.diameter_canvas = FigureCanvas(self.diameter_fig)
        self.diameter_ax = self.diameter_fig.add_subplot(111)
        self.diameter_line, = self.diameter_ax.plot([], [], 'b-', linewidth=2)
        self.diameter_ax.set_xlabel("Time (seconds)")
        self.diameter_ax.set_ylabel("Diameter (mm)")
        self.diameter_ax.set_title("Diameter Over Time")
        self.diameter_ax.grid(True, alpha=0.3)
        
        self.volume_fig = Figure(figsize=(8, 3))
        self.volume_canvas = FigureCanvas(self.volume_fig)
        self.volume_ax = self.volume_fig.add_subplot(111)
        self.volume_line, = self.volume_ax.plot([], [], 'g-', linewidth=2)
        self.volume_ax.set_xlabel("Time (seconds)")
        self.volume_ax.set_ylabel("Volume (mm³)")
        self.volume_ax.set_title("Volume Over Time")
        self.volume_ax.grid(True, alpha=0.3)
        
        parent_layout.addWidget(self.diameter_canvas)
        parent_layout.addWidget(self.volume_canvas)
        
    def calibrate(self):
        try:
            known_mm = float(self.known_mm.text())
            known_pixels = float(self.known_pixels.text())
            self.calibration_factor = known_mm / known_pixels
            QMessageBox.information(self, "Calibration", 
                                  f"Calibration factor set to {self.calibration_factor:.4f} mm/pixel")
        except ValueError:
            QMessageBox.critical(self, "Error", 
                               "Invalid input for calibration. Please enter numeric values.")
    
    def upload_video(self):
        if not self.model:
            QMessageBox.critical(self, "Error", "No valid model loaded. Cannot process video.")
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video files (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        
        if file_path:
            self.video_path = file_path
            self.cap = cv2.VideoCapture(self.video_path)
            
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Could not open video file.")
                self.cap = None
                return
                
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if self.fps <= 0:
                self.fps = 30
                
            # Update UI components
            self.x_slider.setMaximum(self.video_width - 1)
            self.x_slider.setValue(self.video_width // 2)
            self.time_slider.setMaximum(self.total_frames - 1)
            self.time_slider.setValue(0)
            
            self.play_btn.setEnabled(True)
            self.current_frame_number = 0
            
            # Load first frame
            self.seek_to_frame(0)
            
            logging.info(f"Video loaded: {self.video_width}x{self.video_height}, "
                        f"{self.fps} FPS, {self.total_frames} frames")
    
    def toggle_playback(self):
        if not self.cap or not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "No video loaded.")
            return
            
        if self.playing:
            self.stop_playback()
        else:
            self.start_playback()
    
    def start_playback(self):
        self.playing = True
        self.play_btn.setText("Pause")
        
        # Reset data if starting from beginning
        if self.current_frame_number == 0:
            self.times = []
            self.diameters = []
            self.volumes = []
            self.update_graphs()
            
        # Start timer for playback
        frame_interval = int(1000 / self.fps)
        self.timer.start(frame_interval)
        
    def stop_playback(self):
        self.playing = False
        self.timer.stop()
        self.play_btn.setText("Play")
        
    def seek_video(self, frame_number):
        if not self.playing and self.cap and self.cap.isOpened():
            self.seek_to_frame(frame_number)
            
    def seek_to_frame(self, frame_number):
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame_number = frame_number
            self.current_time = frame_number / self.fps
            
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.current_mask = self.apply_segmentation(frame)
                self.calculate_measurements()
                self.update_display()
                self.update_time_info()
                
    def process_next_frame(self):
        if not self.playing or not self.cap or not self.cap.isOpened():
            self.stop_playback()
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop_playback()
            return
            
        self.current_frame = frame
        self.current_frame_number += 1
        self.current_time = self.current_frame_number / self.fps
        
        # Apply segmentation and calculate measurements
        self.current_mask = self.apply_segmentation(frame)
        self.calculate_measurements()
        
        # Store measurements
        self.times.append(self.current_time)
        self.diameters.append(self.current_diameter)
        self.volumes.append(self.current_volume)
        
        # Update displays
        self.update_display()
        self.update_graphs()
        self.update_time_info()
        self.update_progress()
        
    def apply_segmentation(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=img)
        img_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(img_tensor)
            
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), 
                         interpolation=cv2.INTER_NEAREST)
        return mask
    
    def calculate_measurements(self):
        if self.current_mask is None:
            return
            
        # Calculate diameter
        if self.manual_diameter_cb.isChecked():
            x = self.x_slider.value()
            diameter_pixels = self.calculate_manual_diameter(self.current_mask, x)
        else:
            diameter_pixels, x = self.calculate_diameter(self.current_mask)
            self.x_slider.setValue(x)
            
        self.current_diameter = diameter_pixels * self.calibration_factor
        
        # Calculate volume (assuming circular cross-section)
        if self.current_diameter > 0:
            radius_mm = self.current_diameter / 2
            # Volume of a sphere approximation: (4/3) * π * r³
            # Or for a cylinder slice: π * r² * height (assuming height = 1mm slice)
            self.current_volume = np.pi * (radius_mm ** 2)  # Area in mm²
        else:
            self.current_volume = 0.0
            
    def calculate_diameter(self, mask):
        max_diameter = 0.0
        best_x = 0
        
        for x in range(mask.shape[1]):
            ys = np.where(mask[:, x] == 255)[0]
            if len(ys) > 0:
                diameter = ys.max() - ys.min() + 1
                if diameter > max_diameter:
                    max_diameter = diameter
                    best_x = x
                    
        return max_diameter, best_x
    
    def calculate_manual_diameter(self, mask, x):
        ys = np.where(mask[:, x] == 255)[0]
        if len(ys) == 0:
            return 0.0
        return ys.max() - ys.min() + 1
    
    def update_display(self):
        if self.current_frame is None or self.current_mask is None:
            return
            
        display_frame = self.current_frame.copy()
        
        # Overlay mask if checked
        if self.overlay_mask_cb.isChecked():
            display_frame = self.overlay_segmentation(display_frame, self.current_mask)
        
        # Draw measurement lines
        x = self.x_slider.value()
        cv2.line(display_frame, (x, 0), (x, self.video_height), (255, 0, 0), 2)
        
        ys = np.where(self.current_mask[:, x] == 255)[0]
        if len(ys) > 0:
            min_y = ys.min()
            max_y = ys.max()
            cv2.line(display_frame, (0, min_y), (self.video_width, min_y), (0, 0, 255), 1)
            cv2.line(display_frame, (0, max_y), (self.video_width, max_y), (0, 0, 255), 1)
        
        # Convert to QImage and display
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        # Scale image to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(scaled_pixmap)
        
        # Update measurement labels
        self.diameter_label.setText(f"Diameter: {self.current_diameter:.2f} mm")
        self.volume_label.setText(f"Cross-sectional Area: {self.current_volume:.2f} mm²")
        self.x_value_label.setText(f"X: {x}")
        
    def overlay_segmentation(self, frame, mask):
        overlay = frame.copy()
        mask_colored = np.zeros_like(frame)
        mask_colored[mask == 255] = [0, 255, 0]  # Green overlay
        cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0, overlay)
        return overlay
    
    def update_graphs(self):
        if len(self.times) > 0:
            # Update diameter graph
            self.diameter_line.set_data(self.times, self.diameters)
            self.diameter_ax.set_xlim(0, max(self.times))
            self.diameter_ax.set_ylim(0, max(self.diameters) * 1.1 if self.diameters else 1)
            self.diameter_canvas.draw()
            
            # Update volume graph
            self.volume_line.set_data(self.times, self.volumes)
            self.volume_ax.set_xlim(0, max(self.times))
            self.volume_ax.set_ylim(0, max(self.volumes) * 1.1 if self.volumes else 1)
            self.volume_canvas.draw()
        else:
            # Clear graphs when no data
            self.diameter_line.set_data([], [])
            self.volume_line.set_data([], [])
            self.diameter_ax.set_xlim(0, 1)
            self.diameter_ax.set_ylim(0, 1)
            self.volume_ax.set_xlim(0, 1)
            self.volume_ax.set_ylim(0, 1)
            self.diameter_canvas.draw()
            self.volume_canvas.draw()
    
    def update_time_info(self):
        if self.total_frames > 0:
            current_time_str = self.format_time(self.current_time)
            total_time_str = self.format_time(self.total_frames / self.fps)
            self.time_label.setText(f"Time: {current_time_str} / {total_time_str}")
            self.time_display_label.setText(f"Current Time: {current_time_str}")
            
            # Update time slider position
            if not self.playing:  # Only update if not playing to avoid conflicts
                self.time_slider.setValue(self.current_frame_number)
    
    def update_progress(self):
        if self.total_frames > 0:
            progress = int((self.current_frame_number / self.total_frames) * 100)
            self.progress_bar.setValue(progress)
            self.time_slider.setValue(self.current_frame_number)
    
    def update_display_if_paused(self):
        if not self.playing and self.current_frame is not None:
            self.calculate_measurements()
            self.update_display()
    
    def format_time(self, seconds):
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def closeEvent(self, event):
        if self.cap:
            self.cap.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = VideoSegmentationApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()