import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, colorchooser, messagebox, Menu, simpledialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import os
import time
import threading
from collections import defaultdict, deque
import torch
from ultralytics import YOLO
import json
import csv
from datetime import datetime
import math


class TrafficAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intermediate Traffic Analysis System")
        self.root.geometry("1200x800")

        # Enable dark mode by default for better visibility
        self.dark_mode = True

        # Set theme colors
        self.bg_color = "#1e1e1e"  # Dark background
        self.fg_color = "#ffffff"  # White text
        self.accent_color = "#0078d7"  # Blue accent
        self.canvas_bg = "black"  # Black video background
        self.plot_bg = "#2d2d2d"  # Dark gray plot background
        self.plot_fg = "white"  # White plot text

        # YOLO model and settings
        self.model = None
        self.model_loaded = False
        self.class_names = {}
        self.detection_threshold = 0.4

        # Define vehicle classes
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.person_class = 0

        # Video processing variables
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.processing_thread = None
        self.update_id = None
        self.video_fps = 30  # Default assumption

        # Analysis results
        self.detected_objects = defaultdict(int)
        self.time_series_data = defaultdict(list)
        self.timestamps = []
        self.frame_count = 0
        self.start_time = 0

        # Object tracking
        self.object_history = {}
        self.history_window = 30
        self.next_object_id = 0
        self.object_speeds = {}

        # Counting lines
        self.counting_lines = []
        self.line_counts = defaultdict(int)
        self.total_vehicle_count = 0

        # Drawing mode
        self.drawing_mode = None
        self.drawing_start = None
        self.temp_line = None

        # Performance settings
        self.frame_skip = 2
        self.current_skip_count = 0
        self.downscale_factor = 0.7

        # Visualization settings
        self.show_tracking = True
        self.show_boxes = True
        self.show_labels = True
        self.show_counts = True

        # GUI update control
        self.last_gui_update = 0
        self.gui_update_interval = 0.03

        # Buffer for processed frames
        self.processed_frames = deque(maxlen=5)

        # Data export
        self.data_log = []
        self.export_directory = "traffic_data_exports"
        if not os.path.exists(self.export_directory):
            os.makedirs(self.export_directory)

        # Store last processed frame
        self.last_frame = None

        # Setup UI
        self.setup_ui()

        # Initialize plotting
        self.setup_plots()

        # Debug flag
        self.debug = True

    def setup_ui(self):
        """Set up the user interface"""
        # Set application background
        self.root.configure(bg=self.bg_color)

        # Configure ttk styles
        style = ttk.Style()
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        style.configure("TButton", background=self.accent_color)

        # Set up menu bar
        self.menu_bar = Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # File menu
        file_menu = Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_command(label="Select Video", command=self.select_video)
        file_menu.add_separator()
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        # View menu
        view_menu = Menu(self.menu_bar, tearoff=0)

        # Create boolean variables for checkbuttons
        self.show_tracking_var = tk.BooleanVar(value=self.show_tracking)
        self.show_boxes_var = tk.BooleanVar(value=self.show_boxes)
        self.show_labels_var = tk.BooleanVar(value=self.show_labels)
        self.show_counts_var = tk.BooleanVar(value=self.show_counts)

        view_menu.add_checkbutton(
            label="Show Tracking",
            variable=self.show_tracking_var,
            command=self.update_display_settings,
        )
        view_menu.add_checkbutton(
            label="Show Bounding Boxes",
            variable=self.show_boxes_var,
            command=self.update_display_settings,
        )
        view_menu.add_checkbutton(
            label="Show Labels",
            variable=self.show_labels_var,
            command=self.update_display_settings,
        )
        view_menu.add_checkbutton(
            label="Show Count Info",
            variable=self.show_counts_var,
            command=self.update_display_settings,
        )
        self.menu_bar.add_cascade(label="View", menu=view_menu)

        # Settings menu
        settings_menu = Menu(self.menu_bar, tearoff=0)

        settings_menu.add_command(label="Set Frame Skip", command=self.set_frame_skip)
        settings_menu.add_command(
            label="Set Detection Threshold", command=self.set_threshold
        )
        settings_menu.add_separator()
        settings_menu.add_command(label="Reset Settings", command=self.reset_settings)

        self.menu_bar.add_cascade(label="Settings", menu=settings_menu)

        # Help menu
        help_menu = Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)

        # Main layout frames
        self.control_frame = ttk.Frame(self.root, padding=10)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        self.content_frame = ttk.Frame(self.root)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left panel for video
        self.video_frame = ttk.LabelFrame(
            self.content_frame, text="Video Feed", padding=10
        )
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.video_canvas = tk.Canvas(self.video_frame, bg=self.canvas_bg)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Bind click events for setting count line
        self.video_canvas.bind("<Button-1>", self.canvas_click)
        self.video_canvas.bind("<B1-Motion>", self.canvas_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.canvas_release)

        # Right panel for visualizations
        self.viz_frame = ttk.LabelFrame(
            self.content_frame, text="Analytics", padding=10
        )
        self.viz_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Control buttons
        self.control_buttons_frame = ttk.Frame(self.control_frame)
        self.control_buttons_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.load_model_btn = ttk.Button(
            self.control_buttons_frame, text="Load Model", command=self.load_model
        )
        self.load_model_btn.pack(side=tk.LEFT, padx=5)

        self.select_video_btn = ttk.Button(
            self.control_buttons_frame, text="Select Video", command=self.select_video
        )
        self.select_video_btn.pack(side=tk.LEFT, padx=5)

        self.play_button = ttk.Button(
            self.control_buttons_frame,
            text="Play",
            command=self.toggle_play,
            state=tk.DISABLED,
        )
        self.play_button.pack(side=tk.LEFT, padx=5)

        # Add export button
        self.export_btn = ttk.Button(
            self.control_buttons_frame,
            text="Export Data",
            command=self.export_data,
            state=tk.DISABLED,
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)

        # Line controls frame
        self.line_frame = ttk.LabelFrame(self.control_frame, text="Counting Lines")
        self.line_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        self.add_line_btn = ttk.Button(
            self.line_frame, text="Add Count Line", command=self.start_add_line
        )
        self.add_line_btn.pack(side=tk.TOP, padx=5, pady=2)

        self.clear_lines_btn = ttk.Button(
            self.line_frame, text="Clear All Lines", command=self.clear_lines
        )
        self.clear_lines_btn.pack(side=tk.TOP, padx=5, pady=2)

        # Detection settings frame
        self.detection_frame = ttk.LabelFrame(
            self.control_frame, text="Detection Settings"
        )
        self.detection_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        ttk.Label(self.detection_frame, text="Threshold:").pack(
            side=tk.TOP, anchor=tk.W
        )
        self.threshold_var = tk.DoubleVar(value=self.detection_threshold)
        threshold_slider = ttk.Scale(
            self.detection_frame,
            from_=0.1,
            to=0.9,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            length=100,
            command=lambda v: self.update_threshold(float(v)),
        )
        threshold_slider.pack(side=tk.TOP, fill=tk.X, padx=5)

        self.threshold_label = ttk.Label(
            self.detection_frame, text=f"{self.detection_threshold:.2f}"
        )
        self.threshold_label.pack(side=tk.TOP)

        # Stats display frame
        self.stats_frame = ttk.LabelFrame(self.control_frame, text="Statistics")
        self.stats_frame.pack(side=tk.RIGHT, padx=10, fill=tk.Y)

        # Vehicle count display
        self.total_count_var = tk.StringVar(value="0")
        ttk.Label(self.stats_frame, text="Total Vehicles:").pack(
            side=tk.TOP, anchor=tk.W
        )
        ttk.Label(
            self.stats_frame,
            textvariable=self.total_count_var,
            font=("Arial", 16, "bold"),
        ).pack(side=tk.TOP)

        # Current FPS display
        self.fps_var = tk.StringVar(value="0 FPS")
        ttk.Label(self.stats_frame, text="Processing Rate:").pack(
            side=tk.TOP, anchor=tk.W, pady=(5, 0)
        )
        ttk.Label(self.stats_frame, textvariable=self.fps_var).pack(side=tk.TOP)

        # Current time display
        self.time_var = tk.StringVar(value="00:00")
        ttk.Label(self.stats_frame, text="Video Time:").pack(
            side=tk.TOP, anchor=tk.W, pady=(5, 0)
        )
        ttk.Label(self.stats_frame, textvariable=self.time_var).pack(side=tk.TOP)

        # Status bar
        self.status_var = tk.StringVar(value="Ready. Please load a YOLO model.")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Progress bar for lengthy operations
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            self.root, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X, before=status_bar)
        self.progress_bar.pack_forget()  # Hide initially

    def setup_plots(self):
        """Initialize the visualization plots"""
        # Create matplotlib figure for plots
        self.fig = Figure(figsize=(5, 8), dpi=100)
        self.fig.patch.set_facecolor(self.plot_bg)

        # Create multiple subplots
        self.ax1 = self.fig.add_subplot(3, 1, 1)  # Vehicle counts
        self.ax2 = self.fig.add_subplot(3, 1, 2)  # Time series
        self.ax3 = self.fig.add_subplot(3, 1, 3)  # Speed data

        # Configure plot appearance
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.set_facecolor(self.plot_bg)
            ax.tick_params(colors=self.plot_fg)
            ax.xaxis.label.set_color(self.plot_fg)
            ax.yaxis.label.set_color(self.plot_fg)
            ax.title.set_color(self.plot_fg)

        # Set up titles
        self.ax1.set_title("Vehicle Types Detected", color=self.plot_fg)
        self.ax2.set_title("Traffic Flow Over Time", color=self.plot_fg)
        self.ax3.set_title("Average Speed by Vehicle Type", color=self.plot_fg)

        # Add axis labels
        self.ax1.set_ylabel("Count", color=self.plot_fg)
        self.ax2.set_xlabel("Time (s)", color=self.plot_fg)
        self.ax2.set_ylabel("Vehicle Count", color=self.plot_fg)
        self.ax3.set_xlabel("Speed (km/h)", color=self.plot_fg)

        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Add tight layout
        self.fig.tight_layout(pad=3.0)

    def load_model(self):
        """Load the YOLOv8 model"""
        try:
            self.status_var.set("Loading YOLOv8 model (may take a moment)...")
            self.root.update()

            # Show progress bar
            self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X)
            self.progress_var.set(10)
            self.root.update()

            # Use the nano model for better performance on CPU
            model_path = "yolov8n.pt"

            # Special handling for systems with limited resources
            torch.set_num_threads(4)  # Limit CPU threads to prevent overload

            self.progress_var.set(30)
            self.root.update()

            # Load the model
            self.model = YOLO(model_path)

            # Get class names
            self.class_names = self.model.names

            self.progress_var.set(80)
            self.root.update()

            self.model_loaded = True

            # Check if using CPU or GPU
            device = "GPU" if torch.cuda.is_available() else "CPU"
            self.status_var.set(
                f"Model loaded successfully. Using {device}. Ready to analyze videos."
            )

            # Hide progress bar
            self.progress_var.set(100)
            self.root.update()
            self.progress_bar.pack_forget()

        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            self.progress_bar.pack_forget()
            if self.debug:
                import traceback

                traceback.print_exc()

    def select_video(self):
        """Open dialog to select a video file"""
        video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if video_path:
            # Close any previously opened video
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.video_path = video_path
            self.status_var.set(f"Video selected: {os.path.basename(video_path)}")

            # Try to open the video to get properties and show the first frame
            try:
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    self.status_var.set(
                        f"Error: Could not open video file: {self.video_path}"
                    )
                    return

                # Get video properties
                self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.video_fps <= 0:
                    self.video_fps = 30  # Default if not available

                self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                # Now reset analysis data after we have the video dimensions
                self.reset_analysis_data()

                # Enable play button if model is loaded
                if self.model_loaded:
                    self.play_button.config(state=tk.NORMAL)
                    self.export_btn.config(state=tk.NORMAL)

                ret, frame = self.cap.read()
                if ret:
                    # Resize frame to fit canvas while maintaining aspect ratio
                    frame = self.resize_frame(frame)
                    self.show_frame(frame)

                    # Set default counting line at middle of the frame
                    middle_y = frame.shape[0] // 2
                    new_line = {
                        "start": (0, middle_y),
                        "end": (frame.shape[1], middle_y),
                        "name": "Main Line",
                        "color": (255, 0, 0),
                        "count": 0,
                    }
                    self.counting_lines.append(new_line)

                    self.status_var.set(
                        f"Video loaded: {os.path.basename(video_path)} - "
                        f"{self.video_width}x{self.video_height} at {self.video_fps:.1f} FPS"
                    )
                else:
                    self.status_var.set(
                        "Error: Could not read the first frame of the video."
                    )
            except Exception as e:
                self.status_var.set(f"Error opening video: {str(e)}")
                if self.debug:
                    import traceback

                    traceback.print_exc()

    def reset_analysis_data(self):
        """Reset all analysis data"""
        # Reset all analysis-related data
        self.detected_objects = defaultdict(int)
        self.time_series_data = defaultdict(list)
        self.timestamps = []
        self.frame_count = 0
        self.total_vehicle_count = 0
        self.object_history = {}
        self.next_object_id = 0
        self.total_count_var.set("0")
        self.time_var.set("00:00")
        self.fps_var.set("0 FPS")

        # Reset line counts but keep the lines
        for line in self.counting_lines:
            line["count"] = 0

        # Reset speed data
        self.object_speeds = {}

        # Update plots
        self.update_plots()

    def resize_frame(self, frame):
        """Resize a frame to fit in the canvas while maintaining aspect ratio"""
        if frame is None:
            return None

        # Get canvas dimensions
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()

        # If canvas hasn't been drawn yet, use default dimensions
        if canvas_width <= 1:
            canvas_width = 640
        if canvas_height <= 1:
            canvas_height = 480

        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]

        # Calculate scaling factor
        width_ratio = canvas_width / frame_width
        height_ratio = canvas_height / frame_height
        scale = min(width_ratio, height_ratio)

        # Calculate new dimensions
        new_width = int(frame_width * scale)
        new_height = int(frame_height * scale)

        # Resize the frame
        try:
            resized_frame = cv2.resize(frame, (new_width, new_height))
            return resized_frame
        except Exception as e:
            self.status_var.set(f"Error resizing frame: {str(e)}")
            if self.debug:
                import traceback

                traceback.print_exc()
            return frame  # Return original frame if resize fails

    def show_frame(self, frame):
        """Display a frame on the canvas"""
        if frame is None:
            self.status_var.set("Error: Invalid frame (None)")
            return

        try:
            # Convert frame to RGB (from BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PhotoImage
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)

            # Clear previous content
            self.video_canvas.delete("all")

            # Update canvas
            self.video_canvas.config(width=img.width, height=img.height)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.video_canvas.image = img_tk  # Keep a reference

            # Draw counting lines
            for line in self.counting_lines:
                start_point = line["start"]
                end_point = line["end"]
                color = line["color"]
                name = line["name"]
                count = line["count"]

                # Convert color from BGR to hex for tkinter
                hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"

                # Draw the line
                self.video_canvas.create_line(
                    start_point[0],
                    start_point[1],
                    end_point[0],
                    end_point[1],
                    fill=hex_color,
                    width=2,
                    dash=(5, 5),
                )

                # Add label for the line with count
                if self.show_counts:
                    self.video_canvas.create_text(
                        start_point[0] + 10,
                        start_point[1] - 10,
                        text=f"{name}: {count}",
                        anchor=tk.W,
                        fill="white",
                        font=("Arial", 10, "bold"),
                    )

            # Draw temporary elements for interactive line creation
            if self.drawing_mode == "line" and self.drawing_start and self.temp_line:
                start_x, start_y = self.drawing_start
                end_x, end_y = self.temp_line
                self.video_canvas.create_line(
                    start_x, start_y, end_x, end_y, fill="yellow", width=2, dash=(5, 5)
                )

            # Draw a message if in drawing mode
            if self.drawing_mode == "line":
                self.video_canvas.create_text(
                    10,
                    20,
                    text="Click and drag to create a counting line. Press ESC to cancel.",
                    anchor=tk.W,
                    fill="white",
                    font=("Arial", 10),
                )

        except Exception as e:
            self.status_var.set(f"Error displaying frame: {str(e)}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def start_add_line(self):
        """Start the process of adding a new counting line"""
        self.drawing_mode = "line"
        self.drawing_start = None
        self.temp_line = None
        self.status_var.set("Click and drag to create a new counting line")

        # Bind Escape key to cancel
        self.root.bind("<Escape>", self.cancel_drawing)

    def cancel_drawing(self, event=None):
        """Cancel the current drawing operation"""
        self.drawing_mode = None
        self.drawing_start = None
        self.temp_line = None
        self.status_var.set("Drawing cancelled")

        # Redraw the current frame
        if self.last_frame is not None:
            self.show_frame(self.last_frame)

        # Unbind escape key
        self.root.unbind("<Escape>")

    def clear_lines(self):
        """Clear all counting lines"""
        self.counting_lines = []
        self.line_counts = defaultdict(int)

        # Redraw the current frame
        if self.last_frame is not None:
            self.show_frame(self.last_frame)

        self.status_var.set("All counting lines cleared")

    def canvas_click(self, event):
        """Handle mouse click on the canvas"""
        if self.drawing_mode == "line":
            # Start drawing a line
            self.drawing_start = (event.x, event.y)
            self.temp_line = (event.x, event.y)

    def canvas_drag(self, event):
        """Handle mouse drag on the canvas"""
        if self.drawing_mode == "line" and self.drawing_start:
            # Update temporary line endpoint
            self.temp_line = (event.x, event.y)

            # Redraw the frame to show the updated line
            if self.last_frame is not None:
                self.show_frame(self.last_frame)

    def canvas_release(self, event):
        """Handle mouse release on the canvas"""
        if self.drawing_mode == "line" and self.drawing_start:
            # Complete the line
            start_point = self.drawing_start
            end_point = (event.x, event.y)

            # Only create if it's a valid line (not too short)
            line_length = np.sqrt(
                (end_point[0] - start_point[0]) ** 2
                + (end_point[1] - start_point[1]) ** 2
            )

            if line_length > 20:  # Minimum length threshold
                # Ask for a name
                line_name = simpledialog.askstring(
                    "Line Name",
                    "Enter a name for this counting line:",
                    initialvalue=f"Line {len(self.counting_lines) + 1}",
                )

                if line_name:
                    # Ask for a color
                    color_rgb = colorchooser.askcolor(
                        title="Select Line Color", initialcolor="#FF0000"
                    )[0]

                    if color_rgb:
                        # Convert RGB to BGR for OpenCV
                        color_bgr = (
                            int(color_rgb[2]),
                            int(color_rgb[1]),
                            int(color_rgb[0]),
                        )

                        # Add the new line
                        new_line = {
                            "start": start_point,
                            "end": end_point,
                            "name": line_name,
                            "color": color_bgr,
                            "count": 0,
                        }

                        self.counting_lines.append(new_line)
                        self.status_var.set(f"Added counting line: {line_name}")

            # Reset drawing state
            self.drawing_mode = None
            self.drawing_start = None
            self.temp_line = None

            # Redraw the frame
            if self.last_frame is not None:
                self.show_frame(self.last_frame)

            # Unbind Escape
            self.root.unbind("<Escape>")

    def toggle_play(self):
        """Toggle video playback"""
        if not self.is_playing:
            # Start playing
            if self.cap is None or not self.cap.isOpened():
                self.status_var.set(
                    "Cannot play: No video loaded or video cannot be opened."
                )
                return

            self.is_playing = True
            self.play_button.config(text="Pause")

            # Reset video if it's at the end
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if current_frame >= total_frames - 1:
                self.cap.release()
                self.cap = cv2.VideoCapture(self.video_path)
                self.frame_count = 0

            # Start processing in a separate thread
            self.start_time = time.time() - (self.frame_count / self.video_fps)
            self.processing_thread = threading.Thread(target=self.process_video)
            self.processing_thread.daemon = True
            self.processing_thread.start()

            # Start GUI update loop
            self.update_gui()
        else:
            # Pause
            self.is_playing = False
            self.play_button.config(text="Play")

            # Cancel scheduled GUI updates
            if self.update_id is not None:
                self.root.after_cancel(self.update_id)
                self.update_id = None

    def update_threshold(self, value):
        """Update the detection threshold"""
        self.detection_threshold = value
        self.threshold_label.config(text=f"{self.detection_threshold:.2f}")

    def update_display_settings(self):
        """Update display settings based on checkboxes"""
        self.show_tracking = self.show_tracking_var.get()
        self.show_boxes = self.show_boxes_var.get()
        self.show_labels = self.show_labels_var.get()
        self.show_counts = self.show_counts_var.get()

        # Redraw the current frame if available
        if self.last_frame is not None:
            self.show_frame(self.last_frame)

    def set_frame_skip(self):
        """Open dialog to set frame skip"""
        value = simpledialog.askinteger(
            "Frame Skip",
            "Enter frame skip value (1-10):\n\nHigher values improve performance but reduce smoothness.",
            minvalue=1,
            maxvalue=10,
            initialvalue=self.frame_skip,
        )
        if value is not None:
            self.frame_skip = value
            self.status_var.set(f"Frame skip set to {value}")

    def set_threshold(self):
        """Open dialog to set detection threshold"""
        value = simpledialog.askfloat(
            "Detection Threshold",
            "Enter detection threshold (0.1-0.9):\n\nLower values detect more objects but may include false positives.",
            minvalue=0.1,
            maxvalue=0.9,
            initialvalue=self.detection_threshold,
        )
        if value is not None:
            self.detection_threshold = value
            self.threshold_var.set(value)
            self.threshold_label.config(text=f"{self.detection_threshold:.2f}")
            self.status_var.set(f"Detection threshold set to {value:.2f}")

    def reset_settings(self):
        """Reset all settings to defaults"""
        confirm = messagebox.askyesno(
            "Reset Settings", "Are you sure you want to reset all settings to defaults?"
        )
        if confirm:
            # Reset performance settings
            self.frame_skip = 2
            self.downscale_factor = 0.7
            self.detection_threshold = 0.4
            self.threshold_var.set(0.4)
            self.threshold_label.config(text=f"{self.detection_threshold:.2f}")

            # Reset display settings
            self.show_tracking = True
            self.show_tracking_var.set(True)
            self.show_boxes = True
            self.show_boxes_var.set(True)
            self.show_labels = True
            self.show_labels_var.set(True)
            self.show_counts = True
            self.show_counts_var.set(True)

            self.status_var.set("All settings reset to defaults")

    def process_video(self):
        """Thread function for processing video frames"""
        try:
            processing_times = deque(
                maxlen=30
            )  # Track processing times for FPS calculation

            while self.is_playing and self.cap is not None and self.cap.isOpened():
                # Measure frame processing time
                start_time = time.time()

                ret, frame = self.cap.read()
                if not ret:
                    self.is_playing = False
                    self.root.after(
                        0, lambda: self.status_var.set("End of video reached.")
                    )
                    self.root.after(0, lambda: self.play_button.config(text="Play"))
                    break

                # Frame skipping for better performance on CPU
                self.current_skip_count += 1
                if self.current_skip_count % self.frame_skip != 0:
                    # Skip this frame for processing but still increment frame counter
                    self.frame_count += 1

                    # For smoother video, add the unprocessed frame to the queue
                    # but only if we're not overloaded
                    if len(self.processed_frames) < self.processed_frames.maxlen:
                        resized_frame = self.resize_frame(frame)
                        self.processed_frames.append(resized_frame)
                        self.last_frame = resized_frame

                    continue

                # Downscale for processing if requested
                if self.downscale_factor != 1.0:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * self.downscale_factor), int(
                        w * self.downscale_factor
                    )
                    processing_frame = cv2.resize(frame, (new_w, new_h))
                else:
                    processing_frame = frame.copy()

                # Process frame with detection and tracking
                processed_frame = self.process_frame(processing_frame, frame)

                # Store for possible redrawing
                self.last_frame = processed_frame

                # Calculate processing time and FPS
                end_time = time.time()
                processing_time = end_time - start_time
                processing_times.append(processing_time)

                # Update FPS display every second
                if len(processing_times) >= 5:
                    avg_time = sum(processing_times) / len(processing_times)
                    fps = 1 / avg_time if avg_time > 0 else 0
                    self.root.after(0, lambda f=fps: self.fps_var.set(f"{f:.1f} FPS"))

                # Add to our frame queue for the GUI thread to pick up
                if processed_frame is not None:
                    self.processed_frames.append(processed_frame)

                # Update video time
                current_time = (
                    self.frame_count / self.video_fps if self.video_fps > 0 else 0
                )
                mins = int(current_time // 60)
                secs = int(current_time % 60)
                self.root.after(
                    0, lambda m=mins, s=secs: self.time_var.set(f"{m:02d}:{s:02d}")
                )

                # Adaptive sleep based on processing time to maintain real-time feel
                # but avoid consuming too much CPU
                if processing_time < 0.03:  # If processing is fast
                    time.sleep(
                        max(0, 0.03 - processing_time)
                    )  # Sleep to target ~30 FPS UI updates

        except Exception as e:
            self.is_playing = False
            self.root.after(
                0,
                lambda msg=f"Error during video processing: {str(e)}": self.status_var.set(
                    msg
                ),
            )
            if self.debug:
                import traceback

                traceback.print_exc()

    def update_gui(self):
        """Update GUI from the main thread at a controlled rate"""
        if not self.is_playing:
            return

        try:
            current_time = time.time()

            # Update GUI at controlled rate
            if current_time - self.last_gui_update >= self.gui_update_interval:
                self.last_gui_update = current_time

                # If we have processed frames, display the latest one
                if len(self.processed_frames) > 0:
                    frame = self.processed_frames.pop()  # Get the latest frame
                    self.show_frame(frame)

                # Update plots occasionally
                if self.frame_count % 30 == 0:
                    self.update_plots()

            # Schedule the next update
            self.update_id = self.root.after(33, self.update_gui)  # ~30fps max

        except Exception as e:
            self.status_var.set(f"Error updating GUI: {str(e)}")
            if self.debug:
                import traceback

                traceback.print_exc()
            self.is_playing = False
            self.play_button.config(text="Play")

    def process_frame(self, frame, original_frame=None):
        """Process a video frame with object detection and tracking"""
        if frame is None or not self.model_loaded:
            return frame

        try:
            # Use original frame if provided, otherwise use the input frame
            if original_frame is not None:
                display_frame = original_frame.copy()
            else:
                display_frame = frame.copy()

            # Increment frame counter
            self.frame_count += 1

            # Get current timestamp
            current_time = time.time() - self.start_time

            # Run detection with YOLOv8
            results = self.model(frame, conf=self.detection_threshold, verbose=False)

            # Process detections
            current_detections = []

            # Process YOLOv8 detection results
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    try:
                        # Get class and confidence
                        class_id = int(box.cls[0].item())

                        # We're only interested in vehicles in this application
                        if class_id not in self.vehicle_classes:
                            continue

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Skip if object is too small (likely noise)
                        if (x2 - x1) < 20 or (y2 - y1) < 20:
                            continue

                        class_name = self.class_names[class_id]
                        confidence = float(box.conf[0].item())

                        # Calculate center point (for tracking)
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        # Store detection
                        current_detections.append(
                            {
                                "class_id": class_id,
                                "class_name": class_name,
                                "confidence": confidence,
                                "box": (x1, y1, x2, y2),
                                "center": (center_x, center_y),
                            }
                        )

                        # Update global detection counts
                        self.detected_objects[class_name] += 1

                        # Draw bounding box and label if enabled
                        if self.show_boxes:
                            # Choose color based on class
                            color = (0, 255, 0)  # Default green for vehicles

                            # Draw rectangle
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)

                            # Draw label if enabled
                            if self.show_labels:
                                # Create label
                                label = f"{class_name}: {confidence:.2f}"

                                # Get text size for background
                                text_size, _ = cv2.getTextSize(
                                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                                )

                                # Draw label background
                                cv2.rectangle(
                                    display_frame,
                                    (x1, y1 - text_size[1] - 5),
                                    (x1 + text_size[0], y1),
                                    color,
                                    -1,
                                )

                                # Draw label text
                                cv2.putText(
                                    display_frame,
                                    label,
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    (0, 0, 0),
                                    2,
                                )

                    except Exception as e:
                        # Skip this detection if there's an error
                        if self.debug:
                            print(f"Error processing detection: {str(e)}")

            # Track objects across frames
            self.track_objects(current_detections, current_time, display_frame)

            # Update status less frequently to reduce overhead
            if self.frame_count % 30 == 0:
                self.status_var.set(
                    f"Processing frame {self.frame_count} | "
                    f"Time: {current_time:.1f}s | "
                    f"Total vehicles: {self.total_vehicle_count}"
                )

                # Update the count display
                self.total_count_var.set(str(self.total_vehicle_count))

            return display_frame

        except Exception as e:
            self.status_var.set(f"Error processing frame: {str(e)}")
            if self.debug:
                import traceback

                traceback.print_exc()
            return frame  # Return original frame if processing fails

    def track_objects(self, current_detections, current_time, display_frame=None):
        """Track objects across frames and check for line crossings"""
        try:
            # If no history yet, initialize with current detections
            if not self.object_history:
                for det in current_detections:
                    # Only track vehicles
                    if det["class_id"] in self.vehicle_classes:
                        obj_id = self.next_object_id
                        self.next_object_id += 1

                        self.object_history[obj_id] = {
                            "positions": deque(maxlen=self.history_window),
                            "timestamps": deque(maxlen=self.history_window),
                            "class_id": det["class_id"],
                            "class_name": det["class_name"],
                            "counted": False,
                            "last_seen": current_time,
                            "boxes": deque(maxlen=self.history_window),
                            "direction": None,
                            "speed": None,
                        }

                        self.object_history[obj_id]["positions"].append(det["center"])
                        self.object_history[obj_id]["timestamps"].append(current_time)
                        self.object_history[obj_id]["boxes"].append(det["box"])
                return

            # Match current detections with existing tracked objects
            matched_indices = set()
            unmatched_detections = []

            # First, try to match by IoU for better tracking
            for det in current_detections:
                if det["class_id"] not in self.vehicle_classes:
                    continue

                max_iou = 0.0
                best_match = None
                det_box = det["box"]

                # Calculate IoU with existing tracked objects
                for obj_id, obj_data in self.object_history.items():
                    if (
                        obj_id in matched_indices
                        or obj_data["class_id"] != det["class_id"]
                    ):
                        continue

                    if not obj_data["boxes"]:
                        continue

                    # Get the last box
                    last_box = obj_data["boxes"][-1]

                    # Calculate IoU
                    iou = self.calculate_iou(det_box, last_box)

                    if iou > max_iou and iou > 0.3:  # IoU threshold
                        max_iou = iou
                        best_match = obj_id

                if best_match is not None:
                    # Update the matched object
                    self.object_history[best_match]["positions"].append(det["center"])
                    self.object_history[best_match]["timestamps"].append(current_time)
                    self.object_history[best_match]["last_seen"] = current_time
                    self.object_history[best_match]["boxes"].append(det["box"])
                    matched_indices.add(best_match)

                    # Check for line crossings
                    self.check_line_crossings(best_match)

                    # Calculate speed
                    self.calculate_speed(best_match)
                else:
                    unmatched_detections.append(det)

            # For unmatched detections, try to match by distance
            for det in unmatched_detections:
                center = det["center"]
                min_dist = float("inf")
                best_match = None

                # Find the closest tracked object of the same class
                for obj_id, obj_data in self.object_history.items():
                    if obj_id in matched_indices:
                        continue

                    if obj_data["class_id"] != det["class_id"]:
                        continue

                    if not obj_data["positions"]:
                        continue

                    # Skip if the object hasn't been seen for too long
                    if current_time - obj_data["last_seen"] > 1.0:  # 1 second threshold
                        continue

                    last_pos = obj_data["positions"][-1]
                    # Calculate Euclidean distance
                    dist = np.sqrt(
                        (center[0] - last_pos[0]) ** 2 + (center[1] - last_pos[1]) ** 2
                    )

                    # If within reasonable distance, consider a match
                    if dist < 50 and dist < min_dist:  # 50 pixels threshold
                        min_dist = dist
                        best_match = obj_id

                if best_match is not None:
                    # Update the matched object
                    self.object_history[best_match]["positions"].append(center)
                    self.object_history[best_match]["timestamps"].append(current_time)
                    self.object_history[best_match]["last_seen"] = current_time
                    self.object_history[best_match]["boxes"].append(det["box"])
                    matched_indices.add(best_match)

                    # Check for line crossings
                    self.check_line_crossings(best_match)

                    # Calculate speed
                    self.calculate_speed(best_match)
                else:
                    # Create a new tracked object
                    obj_id = self.next_object_id
                    self.next_object_id += 1

                    self.object_history[obj_id] = {
                        "positions": deque(maxlen=self.history_window),
                        "timestamps": deque(maxlen=self.history_window),
                        "class_id": det["class_id"],
                        "class_name": det["class_name"],
                        "counted": False,
                        "last_seen": current_time,
                        "boxes": deque(maxlen=self.history_window),
                        "direction": None,
                        "speed": None,
                    }

                    self.object_history[obj_id]["positions"].append(center)
                    self.object_history[obj_id]["timestamps"].append(current_time)
                    self.object_history[obj_id]["boxes"].append(det["box"])

            # Record data for time series plot
            if (
                len(self.timestamps) == 0 or current_time - self.timestamps[-1] >= 0.5
            ):  # Add data point every 0.5 seconds
                self.timestamps.append(current_time)

                # Count visible objects of each class for the time series
                class_counts = defaultdict(int)
                for obj_id, obj_data in self.object_history.items():
                    if (
                        current_time - obj_data["last_seen"] < 0.5
                    ):  # Object is currently visible
                        class_counts[obj_data["class_name"]] += 1

                # Update time series data for vehicle classes
                for class_id in self.vehicle_classes:
                    if class_id in self.class_names:
                        class_name = self.class_names[class_id]
                        self.time_series_data[class_name].append(
                            class_counts[class_name]
                        )

            # Prune history to remove objects not seen recently
            current_objects = list(self.object_history.keys())
            for obj_id in current_objects:
                if (
                    current_time - self.object_history[obj_id]["last_seen"] > 2.0
                ):  # 2 seconds threshold
                    del self.object_history[obj_id]

            # Draw tracking trails if enabled
            if display_frame is not None and self.show_tracking:
                for obj_id, obj_data in self.object_history.items():
                    if (
                        current_time - obj_data["last_seen"] < 0.5
                        and len(obj_data["positions"]) > 1
                    ):
                        # Choose color (green for vehicles)
                        color = (0, 255, 0)

                        # Draw tracking line
                        positions = list(obj_data["positions"])
                        for i in range(1, len(positions)):
                            cv2.line(
                                display_frame, positions[i - 1], positions[i], color, 2
                            )

        except Exception as e:
            if self.debug:
                print(f"Error in object tracking: {str(e)}")
                import traceback

                traceback.print_exc()

    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two bounding boxes"""
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        # Calculate area of each box
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        # Check if boxes intersect
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        # Calculate intersection area
        intersection = (x2_i - x1_i) * (y2_i - y1_i)

        # Calculate union area
        union = area1 + area2 - intersection

        # Calculate IoU
        iou = intersection / union if union > 0 else 0.0

        return iou

    def check_line_crossings(self, obj_id):
        """Check if an object has crossed any counting lines"""
        obj_data = self.object_history[obj_id]

        # Need at least two positions to detect crossing
        if len(obj_data["positions"]) < 2:
            return

        # Get current and previous positions
        curr_pos = obj_data["positions"][-1]
        prev_pos = obj_data["positions"][-2]

        # Check each counting line
        for line_index, line in enumerate(self.counting_lines):
            if self.line_segment_intersection(
                prev_pos, curr_pos, line["start"], line["end"]
            ):
                # Object has crossed this line
                if not obj_data["counted"]:
                    # Update vehicle count
                    self.total_vehicle_count += 1
                    obj_data["counted"] = True

                    # Update line-specific count
                    line["count"] += 1

                    # Log the event for data export
                    crossing_data = {
                        "time": obj_data["timestamps"][-1],
                        "object_id": obj_id,
                        "class": obj_data["class_name"],
                        "line": line["name"],
                        "speed": obj_data["speed"] if obj_data["speed"] else 0,
                    }
                    self.data_log.append(crossing_data)

    def calculate_speed(self, obj_id):
        """Calculate speed of a tracked object"""
        obj_data = self.object_history[obj_id]

        # Need at least two positions with timestamps to calculate
        if len(obj_data["positions"]) < 2 or len(obj_data["timestamps"]) < 2:
            return

        # Get positions and timestamps
        positions = list(obj_data["positions"])
        timestamps = list(obj_data["timestamps"])

        # Use last few points for better speed estimation
        if len(positions) >= 5 and len(timestamps) >= 5:
            # Calculate distance traveled in pixels
            start_pos = positions[-5]
            end_pos = positions[-1]
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            distance_px = np.sqrt(dx**2 + dy**2)

            # Calculate time elapsed
            dt = timestamps[-1] - timestamps[-5]

            if dt > 0:
                # Convert to approximate real-world measures
                # Assuming 100 pixels is roughly 1 meter (this is a simplification)
                meters = distance_px / 100

                # Calculate speed in km/h
                speed_kmh = (meters / dt) * 3.6

                # Apply smoothing
                if obj_data["speed"] is None:
                    obj_data["speed"] = speed_kmh
                else:
                    # Simple exponential smoothing
                    alpha = 0.3  # Smoothing factor
                    obj_data["speed"] = (
                        alpha * speed_kmh + (1 - alpha) * obj_data["speed"]
                    )

                # Store speed in object history
                self.object_speeds[obj_id] = obj_data["speed"]

    def line_segment_intersection(self, p1, p2, p3, p4):
        """Check if two line segments intersect"""

        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def update_plots(self):
        """Update the visualization plots with current data"""
        try:
            # Clear the axes
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()

            # Configure plot appearance
            for ax in [self.ax1, self.ax2, self.ax3]:
                ax.set_facecolor(self.plot_bg)
                ax.tick_params(colors=self.plot_fg)
                ax.xaxis.label.set_color(self.plot_fg)
                ax.yaxis.label.set_color(self.plot_fg)

            # 1. Vehicle type counts (bar chart)
            self.ax1.set_title("Vehicle Types Detected", color=self.plot_fg)
            self.ax1.set_ylabel("Count", color=self.plot_fg)

            # Filter for vehicle classes and sort by count
            vehicle_counts = [
                (
                    self.class_names[class_id],
                    self.detected_objects[self.class_names[class_id]],
                )
                for class_id in self.vehicle_classes
                if self.class_names[class_id] in self.detected_objects
            ]

            vehicle_counts.sort(key=lambda x: x[1], reverse=True)

            if vehicle_counts:
                labels = [item[0] for item in vehicle_counts]
                values = [item[1] for item in vehicle_counts]

                bars = self.ax1.bar(labels, values, color=self.accent_color)

                # Add count labels above bars
                for bar in bars:
                    height = bar.get_height()
                    self.ax1.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.1,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        color=self.plot_fg,
                    )
            else:
                self.ax1.text(
                    0.5,
                    0.5,
                    "No vehicle data yet",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )

            # 2. Time series plot
            self.ax2.set_title("Traffic Flow Over Time", color=self.plot_fg)
            self.ax2.set_xlabel("Time (s)", color=self.plot_fg)
            self.ax2.set_ylabel("Vehicle Count", color=self.plot_fg)

            if self.timestamps and any(self.time_series_data.values()):
                # Plot data for each vehicle class
                for class_id in self.vehicle_classes:
                    if class_id in self.class_names:
                        class_name = self.class_names[class_id]
                        if (
                            class_name in self.time_series_data
                            and self.time_series_data[class_name]
                        ):
                            data_len = min(
                                len(self.timestamps),
                                len(self.time_series_data[class_name]),
                            )
                            if data_len > 1:  # Need at least 2 points to plot a line
                                self.ax2.plot(
                                    self.timestamps[:data_len],
                                    self.time_series_data[class_name][:data_len],
                                    label=class_name,
                                )

                self.ax2.legend(loc="upper left")
            else:
                self.ax2.text(
                    0.5,
                    0.5,
                    "Waiting for time series data...",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )

            # 3. Speed data
            self.ax3.set_title("Average Speed by Vehicle Type", color=self.plot_fg)

            # Collect speed data by class
            class_speeds = defaultdict(list)
            for obj_id, speed in self.object_speeds.items():
                if obj_id in self.object_history:
                    class_name = self.object_history[obj_id]["class_name"]
                    class_speeds[class_name].append(speed)

            # Calculate average speeds
            avg_speeds = {}
            for class_name, speeds in class_speeds.items():
                if speeds:
                    avg_speeds[class_name] = sum(speeds) / len(speeds)

            if avg_speeds:
                # Sort by speed
                sorted_speeds = sorted(
                    avg_speeds.items(), key=lambda x: x[1], reverse=True
                )

                # Create horizontal bar chart
                labels = [item[0] for item in sorted_speeds]
                values = [item[1] for item in sorted_speeds]

                bars = self.ax3.barh(labels, values, color=self.accent_color)

                # Add labels
                for bar in bars:
                    width = bar.get_width()
                    self.ax3.text(
                        width + 0.5,
                        bar.get_y() + bar.get_height() / 2,
                        f"{width:.1f} km/h",
                        va="center",
                        color=self.plot_fg,
                    )

                self.ax3.set_xlabel("Speed (km/h)", color=self.plot_fg)
            else:
                self.ax3.text(
                    0.5,
                    0.5,
                    "No speed data available yet",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )

            # Update the canvas
            self.fig.tight_layout()
            self.canvas.draw()

        except Exception as e:
            if self.debug:
                print(f"Error updating plots: {str(e)}")
                import traceback

                traceback.print_exc()

    def export_data(self):
        """Export collected data to CSV and JSON files"""
        if not self.data_log and self.total_vehicle_count == 0:
            messagebox.showinfo("No Data", "No traffic data has been collected yet.")
            return

        try:
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create export directory if it doesn't exist
            if not os.path.exists(self.export_directory):
                os.makedirs(self.export_directory)

            # Export crossing events to CSV
            if self.data_log:
                csv_path = os.path.join(
                    self.export_directory, f"traffic_crossings_{timestamp}.csv"
                )
                with open(csv_path, "w", newline="") as csvfile:
                    fieldnames = ["time", "object_id", "class", "line", "speed"]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                    writer.writeheader()
                    for event in self.data_log:
                        writer.writerow(event)

            # Export summary data to JSON
            json_path = os.path.join(
                self.export_directory, f"traffic_summary_{timestamp}.json"
            )

            summary_data = {
                "total_vehicle_count": self.total_vehicle_count,
                "video_file": (
                    os.path.basename(self.video_path) if self.video_path else "None"
                ),
                "analysis_duration": time.time() - self.start_time,
                "counted_by_class": dict(self.detected_objects),
                "counting_lines": [
                    {"name": line["name"], "count": line["count"]}
                    for line in self.counting_lines
                ],
            }

            with open(json_path, "w") as jsonfile:
                json.dump(summary_data, jsonfile, indent=4)

            messagebox.showinfo(
                "Export Successful", f"Data exported to {self.export_directory}"
            )

        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting data: {str(e)}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def show_instructions(self):
        """Show application instructions"""
        instructions = """
        Traffic Analysis Application Instructions
        
        Getting Started:
        1. Load a YOLO model using the "Load Model" button
        2. Select a video file for analysis
        3. Press "Play" to start the analysis
        
        Counting Lines:
        • Add counting lines to track vehicles crossing specific boundaries
        • Click the "Add Count Line" button, then click and drag on the video
        • Each line will count vehicles that cross it
        
        Analytics:
        • View real-time analytics in the visualization panel
        • Export data for detailed analysis
        
        Display Options:
        • Use the View menu to toggle different visualization elements
        
        Performance Settings:
        • Adjust frame skip to balance between performance and smoothness
        • Change detection threshold to control sensitivity
        
        Tips for CPU Optimization:
        • Increase frame skip on slower systems
        • Lower the detection threshold if objects are being missed
        """

        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("Application Instructions")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        dialog.grab_set()

        # Add text widget with instructions
        text = tk.Text(dialog, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, instructions)
        text.config(state=tk.DISABLED)  # Make read-only

        # Add scrollbar
        scrollbar = ttk.Scrollbar(text, command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.config(yscrollcommand=scrollbar.set)

        # Add close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

    def show_about(self):
        """Show about dialog"""
        about_text = """
        Intermediate Traffic Analysis System
        Version 1.0
        
        A traffic monitoring and analysis application 
        using YOLOv8 for real-time vehicle detection and tracking.
        
        Features:
        • Real-time vehicle detection and classification
        • Traffic flow analysis with counting lines
        • Speed estimation
        • Data visualization and analytics
        • Data export
        
        Optimized for CPU systems with limited resources.
        """

        # Create dialog window
        dialog = tk.Toplevel(self.root)
        dialog.title("About Traffic Analysis")
        dialog.geometry("400x300")
        dialog.transient(self.root)
        dialog.grab_set()

        # Add text widget with about info
        text = tk.Text(dialog, wrap=tk.WORD, padx=10, pady=10)
        text.pack(fill=tk.BOTH, expand=True)
        text.insert(tk.END, about_text)
        text.config(state=tk.DISABLED)  # Make read-only

        # Add close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=10)

    def on_closing(self):
        """Handle application closing"""
        # Stop playback if running
        self.is_playing = False

        # Cancel scheduled GUI updates
        if self.update_id is not None:
            self.root.after_cancel(self.update_id)

        # Release video
        if self.cap is not None:
            self.cap.release()

        # Close window
        self.root.destroy()


if __name__ == "__main__":
    # Create the Tkinter root window
    root = tk.Tk()

    # Set application theme - try to use a modern theme if available
    try:
        style = ttk.Style()
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")
        elif "alt" in available_themes:
            style.theme_use("alt")
    except:
        pass  # Use default theme if custom themes fail

    # Create the app
    app = TrafficAnalysisApp(root)

    # Set up closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start the main loop
    root.mainloop()
