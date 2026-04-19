import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

class CarDamageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Car Damage Segmentation")
        self.root.geometry("1000x700")
        self.root.configure(bg="#1E1E2E")  # Deep modern dark background
        
        # Try to load the model immediately
        try:
            self.model = YOLO("cardmgv3.pt")
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load 'cardmgv3.pt'. Ensure the file is in the same directory.\n\nError: {e}")
            self.model = None

        self.original_image = None
        self.display_image = None
        
        self.setup_ui()

    def setup_ui(self):
        # Header Frame
        header = tk.Frame(self.root, bg="#181825", height=80)
        header.pack(fill=tk.X)
        
        title = tk.Label(
            header, 
            text="Car Damage Segmentation", 
            font=("Segoe UI", 24, "bold"), 
            bg="#181825", 
            fg="#CDD6F4"
        )
        title.pack(pady=20)
        
        # Main content frame (Canvas container)
        self.main_frame = tk.Frame(self.root, bg="#1E1E2E")
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=30, pady=20)
        
        # Image Display Area (Canvas)
        self.image_canvas = tk.Canvas(
            self.main_frame, 
            bg="#313244", 
            bd=0, 
            highlightthickness=2, 
            highlightbackground="#45475A"
        )
        self.image_canvas.pack(expand=True, fill=tk.BOTH, side=tk.TOP, pady=(0, 20))
        
        # Initial placeholder text
        self.placeholder_text = self.image_canvas.create_text(
            500, 250, 
            text="No Image Loaded", 
            fill="#A6ADC8", 
            font=("Segoe UI", 16, "italic")
        )
        
        # Bottom Control Panel
        control_panel = tk.Frame(self.main_frame, bg="#1E1E2E")
        control_panel.pack(fill=tk.X, side=tk.BOTTOM)
        
        # Button styles (Flat, colored, bold text)
        btn_style = {
            "font": ("Segoe UI", 12, "bold"),
            "fg": "#FFFFFF",
            "bd": 0,
            "padx": 25,
            "pady": 12,
            "cursor": "hand2"
        }
        
        # Load Button (Blue)
        self.btn_load = tk.Button(
            control_panel, 
            text="📁 Load Image", 
            bg="#89B4FA", 
            activebackground="#74C7EC", 
            activeforeground="#11111B",
            command=self.load_image, 
            **btn_style
        )
        self.btn_load.pack(side=tk.LEFT, padx=(0, 15))
        
        # Predict Button (Green)
        self.btn_predict = tk.Button(
            control_panel, 
            text="✨ Predict", 
            bg="#A6E3A1", 
            activebackground="#94E2D5", 
            activeforeground="#11111B",
            command=self.predict, 
            **btn_style
        )
        self.btn_predict.pack(side=tk.LEFT, padx=15)
        
        # Reset Button (Red)
        self.btn_reset = tk.Button(
            control_panel, 
            text="🗑️ Reset", 
            bg="#F38BA8", 
            activebackground="#F9E2AF", 
            activeforeground="#11111B",
            command=self.reset, 
            **btn_style
        )
        self.btn_reset.pack(side=tk.RIGHT, padx=(15, 0))

        # Bind resize event to keep the canvas text/images centered
        self.image_canvas.bind("<Configure>", self.on_resize)

    def load_image(self):
        file_path = filedialog.askopenfilename(
            title="Select a Car Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.original_image = Image.open(file_path)
            self.update_canvas_image(self.original_image)
            
    def predict(self):
        if not self.original_image:
            messagebox.showwarning("Notice", "Please load an image first before predicting.")
            return
            
        if not self.model:
            messagebox.showerror("Error", "Model is not loaded. Cannot perform prediction.")
            return
            
        # Run prediction on the loaded PIL image
        results = self.model.predict(source=self.original_image, save=False, show=False)
        
        # Extract the resulting plotted image with masks/boxes (returns a BGR numpy array)
        res_plotted_bgr = results[0].plot() 
        
        # Convert BGR (OpenCV) back to RGB (Pillow)
        res_rgb = cv2.cvtColor(res_plotted_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert array back to a PIL Image
        predicted_img = Image.fromarray(res_rgb)
        
        # Update canvas with the segmented image
        self.update_canvas_image(predicted_img)
        
    def reset(self):
        self.original_image = None
        self.image_canvas.delete("all")
        
        # Restore the placeholder text
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        self.placeholder_text = self.image_canvas.create_text(
            canvas_width // 2, canvas_height // 2, 
            text="No Image Loaded", 
            fill="#A6ADC8", 
            font=("Segoe UI", 16, "italic")
        )

    def update_canvas_image(self, img):
        # Grab current canvas dimensions
        canvas_width = self.image_canvas.winfo_width()
        canvas_height = self.image_canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            return 
            
        # Create a copy and resize it to fit the canvas while preserving aspect ratio
        img_copy = img.copy()
        img_copy.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
        
        self.display_image = ImageTk.PhotoImage(img_copy)
        
        self.image_canvas.delete("all")
        self.image_canvas.create_image(
            canvas_width // 2, canvas_height // 2, 
            anchor=tk.CENTER, 
            image=self.display_image
        )

    def on_resize(self, event):
        # Handle centering the UI placeholder dynamically if the window size changes
        if self.original_image is None:
            self.image_canvas.delete("all")
            self.placeholder_text = self.image_canvas.create_text(
                event.width // 2, event.height // 2, 
                text="No Image Loaded", 
                fill="#A6ADC8", 
                font=("Segoe UI", 16, "italic")
            )

if __name__ == "__main__":
    root = tk.Tk()
    app = CarDamageApp(root)
    root.mainloop()