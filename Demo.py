import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import segmentation_models_pytorch as smp
from PIL import Image, ImageTk
import torch
import albumentations as A
import os
import logging
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class VideoSegmentationApp:
    def __init__(self, root, model_path="UnetPlusPlus_human_aug.pth", model_folder="./human", frame_delay=33):
        self.root = root
        self.root.title("Video Segmentation and Diameter Measurement")
        self.video_path = None
        self.cap = None
        self.current_frame = None
        self.current_mask = None
        self.current_diameter = 0.0
        self.times = []  # Store all frame times
        self.diameters = []  # Store all diameters
        self.playing = False
        self.frame_count = 0
        self.fps = 30  # Default FPS
        self.frame_delay = frame_delay  # Delay in ms (~30 FPS)
        self.calibration_factor = 1.0  # Default to 1.0 (pixels per mm)
        self.video_width = 0
        self.video_height = 0
        self.selected_x = tk.IntVar()
        self.overlay_mask_var = tk.IntVar(value=1)  # Default checked
        self.manual_diameter_var = tk.IntVar(value=0)  # Default unchecked
        
        # Initialize device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")
        
        # Load segmentation model
        try:
            self.model = self.load_model(model_path, self.device, model_folder)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.model = None
        
        # Define normalization pipeline
        self.transform = A.Compose([
            A.Resize(height=256, width=256, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            A.pytorch.ToTensorV2()
        ])
        
        # GUI components
        self.setup_gui()
        logging.info("GUI initialized")
        
    def load_model(self, path, device, parent_folder="./human"):
        model_path = os.path.join(parent_folder, path)
        logging.info(f"Loading model from {model_path}")
        #if not os.path.exists(model_path):
        if True:
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
        else:
            name = os.path.basename(path)
            model_type = name.split("_")[0]
            model = getattr(smp, model_type)(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1
            )
            model.name = os.path.splitext(name)[0]
            state_dict = torch.load(os.path.join(parent_folder, path), weights_only=True)
            model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        logging.info(f"Model {model.name} loaded successfully")
        return model
        
    def setup_gui(self):
        # Upload button
        self.upload_btn = tk.Button(self.root, text="Upload Video", command=self.upload_video)
        self.upload_btn.pack(pady=10)
        
        # Play button
        self.play_btn = tk.Button(self.root, text="Play Segmented Video", command=self.start_playback, state=tk.DISABLED)
        self.play_btn.pack(pady=10)
        
        # Canvas for video display
        self.canvas = tk.Canvas(self.root, width=640, height=480, bg="black")
        self.canvas.pack()
        
        # Label for diameter display
        self.diameter_label = tk.Label(self.root, text="Diameter: N/A", font=("Arial", 16, "bold"), fg="red")
        self.diameter_label.pack(pady=10)
        
        # Calibration input
        self.calibration_frame = tk.Frame(self.root)
        self.calibration_frame.pack(pady=10)
        tk.Label(self.calibration_frame, text="Known size (mm):").pack(side=tk.LEFT)
        self.known_mm = tk.Entry(self.calibration_frame, width=10)
        self.known_mm.pack(side=tk.LEFT)
        tk.Label(self.calibration_frame, text="Known size (pixels):").pack(side=tk.LEFT)
        self.known_pixels = tk.Entry(self.calibration_frame, width=10)
        self.known_pixels.pack(side=tk.LEFT)
        self.calibrate_btn = tk.Button(self.calibration_frame, text="Calibrate", command=self.calibrate)
        self.calibrate_btn.pack(side=tk.LEFT)
        
        # Slider for selecting x-coordinate
        self.x_slider = tk.Scale(self.root, from_=0, to=100, orient=tk.HORIZONTAL, label="Select X for Diameter", variable=self.selected_x)
        self.x_slider.pack(pady=10)
        
        # Checkboxes
        self.overlay_cb = tk.Checkbutton(self.root, text="Overlay Mask", variable=self.overlay_mask_var)
        self.overlay_cb.pack(pady=5)
        self.manual_cb = tk.Checkbutton(self.root, text="Manual Diameter", variable=self.manual_diameter_var)
        self.manual_cb.pack(pady=5)
        
        # Matplotlib graph for diameter history
        self.fig, self.ax = plt.subplots(figsize=(5, 3))
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_xlabel("Time (seconds)")
        self.ax.set_ylabel("Diameter (mm)")
        self.ax.set_title("Diameter Over Time")
        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_graph.get_tk_widget().pack(pady=10)
        
    def calibrate(self):
        try:
            known_mm = float(self.known_mm.get())
            known_pixels = float(self.known_pixels.get())
            self.calibration_factor = known_mm / known_pixels
            messagebox.showinfo("Calibration", f"Calibration factor set to {self.calibration_factor:.4f} mm/pixel")
        except ValueError:
            messagebox.showerror("Error", "Invalid input for calibration. Please enter numeric values.")
        
    def upload_video(self):
        if not self.model:
            messagebox.showerror("Error", "No valid model loaded. Cannot process video.")
            return
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video file.")
                self.cap = None
                return
            self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30  # Default to 30 FPS if FPS retrieval fails
            self.frame_delay = int(1000 / self.fps)
            self.x_slider.config(to=self.video_width - 1)
            self.selected_x.set(self.video_width // 2)
            self.play_btn.config(state=tk.NORMAL)
            logging.info("Video loaded, play button enabled")
            
    def start_playback(self):
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Error", "No video loaded.")
            return
        self.playing = True
        self.frame_count = 0
        self.times = []
        self.diameters = []
        self.line.set_data([], [])
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.canvas_graph.draw()
        self.play_btn.config(text="Stop Playback", command=self.stop_playback)
        self.process_next_frame()
        
    def stop_playback(self):
        self.playing = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.current_frame = None
        self.current_mask = None
        self.diameter_label.config(text="Diameter: N/A")
        self.canvas.delete("all")
        self.play_btn.config(text="Play Segmented Video", command=self.start_playback)
        self.play_btn.config(state=tk.DISABLED)
        
    def process_next_frame(self):
        if not self.playing or not self.cap or not self.cap.isOpened():
            self.stop_playback()
            return
            
        ret, frame = self.cap.read()
        if not ret:
            self.stop_playback()
            return
            
        self.current_frame = frame
        self.frame_count += 1
        
        # Apply segmentation
        self.current_mask = self.apply_segmentation(frame)
        
        # Calculate diameter based on mode
        if self.manual_diameter_var.get():
            x = self.selected_x.get()
            diameter_pixels = self.calculate_manual_diameter(self.current_mask, x)
        else:
            diameter_pixels, x = self.calculate_diameter(self.current_mask)
            self.selected_x.set(x) 

        self.current_diameter = diameter_pixels * self.calibration_factor
        
        # Store time and diameter
        self.times.append(self.frame_count / self.fps)
        self.diameters.append(self.current_diameter)
        self.update_graph()
        
        # Update display
        self.update_display()
        
        # Schedule next frame
        self.root.after(self.frame_delay, self.process_next_frame)
        
    def apply_segmentation(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        transformed = self.transform(image=img)
        img_tensor = transformed['image'].unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img_tensor)
        mask = torch.sigmoid(output).squeeze().cpu().numpy()
        mask = (mask > 0.5).astype(np.uint8) * 255
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        return mask
    
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
        min_y = ys.min()
        max_y = ys.max()
        return max_y - min_y + 1
    
    def overlay_segmentation(self, frame, mask):
        overlay = frame.copy()
        mask_colored = np.zeros_like(frame)
        mask_colored[mask == 255] = [0, 255, 0]  # Green overlay
        cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0, overlay)
        return overlay
    
    def update_display(self):
        if self.current_frame is None or self.current_mask is None:
            return
            
        display_frame = self.current_frame.copy()
        
        if self.overlay_mask_var.get():
            display_frame = self.overlay_segmentation(display_frame, self.current_mask)
        
        # Draw vertical line at selected x
        x = self.selected_x.get()
        cv2.line(display_frame, (x, 0), (x, self.video_height), (255, 0, 0), 2)  # Blue line

        ys = np.where(self.current_mask[:, x] == 255)[0]
        if len(ys) > 0:
                min_y = ys.min()
                max_y = ys.max()
                # Draw horizontal lines to represent diameter
                cv2.line(display_frame, (0, min_y), (self.video_width, min_y), (0, 0, 255), 1)  # Red line at top
                cv2.line(display_frame, (0, max_y), (self.video_width, max_y), (0, 0, 255), 1)  # Red line at bottom
        
        # Update diameter display
        self.diameter_label.config(text=f"Diameter: {self.current_diameter:.2f} mm")
        
        # Convert frame to PhotoImage
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display_frame)
        img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.image = img  # Keep reference
        
    def update_graph(self):
        if len(self.times) > 0:
            start_idx = max(0, len(self.times) - 1000)
            times_to_plot = self.times[start_idx:]
            diameters_to_plot = self.diameters[start_idx:]
            self.line.set_data(times_to_plot, diameters_to_plot)
            if len(times_to_plot) > 1:
                self.ax.set_xlim(times_to_plot[0], times_to_plot[-1])
                self.ax.set_ylim(0, max(diameters_to_plot)*1.3)
            else:
                self.ax.set_xlim(0, 1)
                self.ax.set_ylim(0, 1)
        else:
            self.line.set_data([], [])
            self.ax.set_xlim(0, 1)
            self.ax.set_ylim(0, 1)
        self.canvas_graph.draw()
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoSegmentationApp(root, model_path="UnetPlusPlus_human_aug.pth", model_folder="./human")
    app.run()