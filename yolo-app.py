import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, colorchooser, messagebox, Menu, simpledialog
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import time
import threading
from collections import defaultdict, deque
import torch
from ultralytics import YOLO
import pandas as pd
from datetime import datetime
import json
import csv
import math


class TrafficAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Analysis System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)

        # Setup theme and colors
        self.setup_theme()

        # Initialize variables
        self.initialize_variables()

        # Setup UI elements
        self.create_ui()

        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Display welcome message
        self.status_var.set("Welcome! Start by loading a YOLO model.")

    def setup_theme(self):
        """Configure application theme and colors"""
        # Set dark theme by default
        self.dark_mode = True

        # Configure colors
        if self.dark_mode:
            self.bg_color = "#212121"
            self.fg_color = "#f0f0f0"
            self.accent_color = "#2196F3"
            self.accent_light = "#42A5F5"
            self.accent_dark = "#1976D2"
            self.canvas_bg = "#333333"
            self.success_color = "#4CAF50"
            self.warning_color = "#FFC107"
            self.error_color = "#F44336"
        else:
            self.bg_color = "#f5f5f5"
            self.fg_color = "#212121"
            self.accent_color = "#2196F3"
            self.accent_light = "#BBDEFB"
            self.accent_dark = "#1976D2"
            self.canvas_bg = "#ffffff"
            self.success_color = "#4CAF50"
            self.warning_color = "#FFC107"
            self.error_color = "#F44336"

        # Configure ttk styles
        self.style = ttk.Style()
        self.style.theme_use("clam")  # Use a theme that supports customization

        # Configure style colors
        self.style.configure("TFrame", background=self.bg_color)
        self.style.configure(
            "TButton", background=self.accent_color, foreground=self.fg_color
        )
        self.style.map(
            "TButton",
            background=[("active", self.accent_light), ("pressed", self.accent_dark)],
            foreground=[("active", self.fg_color)],
        )
        self.style.configure(
            "TLabel", background=self.bg_color, foreground=self.fg_color
        )
        self.style.configure(
            "TCheckbutton", background=self.bg_color, foreground=self.fg_color
        )
        self.style.configure(
            "TNotebook", background=self.bg_color, tabmargins=[2, 5, 2, 0]
        )
        self.style.configure(
            "TNotebook.Tab",
            background=self.bg_color,
            foreground=self.fg_color,
            padding=[10, 5],
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", self.accent_color)],
            foreground=[("selected", self.fg_color)],
        )

        # Configure the root window
        self.root.configure(bg=self.bg_color)

    def initialize_variables(self):
        """Initialize all application variables"""
        # YOLO model settings
        self.model = None
        self.model_loaded = False
        self.class_names = {}
        self.detection_threshold = 0.4
        self.model_size = "nano"  # nano, small, medium

        # Optimization settings
        self.frame_skip = 2
        self.current_skip_count = 0
        self.downscale_factor = 0.7

        # Video processing variables
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.processing_thread = None
        self.update_id = None
        self.video_fps = 30
        self.video_resolution = (640, 480)

        # Object detection settings
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.person_class = 0
        self.detection_mode = "vehicles"  # vehicles, people, all, custom
        self.custom_classes = []

        # Class colors for visualization
        self.class_colors = {
            2: (0, 255, 0),  # Car: Green
            3: (0, 0, 255),  # Motorcycle: Blue
            5: (255, 0, 0),  # Bus: Red
            7: (255, 255, 0),  # Truck: Yellow
            0: (255, 0, 255),  # Person: Magenta
        }

        # Counting and analysis data
        self.detected_objects = defaultdict(int)
        self.total_vehicle_count = 0

        # For object tracking
        self.object_history = {}
        self.next_object_id = 0
        self.max_tracked_objects = 50

        # For line counting
        self.counting_lines = []
        self.line_counts = defaultdict(int)

        # For region analysis
        self.regions_of_interest = []
        self.roi_counts = defaultdict(int)

        # For heatmap generation
        self.heatmap_data = np.zeros((480, 640), dtype=np.float32)

        # For speed estimation
        self.object_speeds = {}
        self.avg_speeds = defaultdict(list)

        # For direction analysis
        self.direction_counts = {"north": 0, "south": 0, "east": 0, "west": 0}

        # For time series analysis
        self.time_series_data = defaultdict(list)
        self.timestamps = []

        # Visualization settings
        self.show_bounding_boxes = True
        self.show_labels = True
        self.show_tracking = True
        self.show_heatmap = False
        self.show_count_info = True
        self.show_direction_arrows = True

        # For zone drawing
        self.drawing_mode = None
        self.drawing_start = None
        self.temp_line = None
        self.temp_roi = []

        # Analysis data
        self.data_log = []
        self.export_directory = "traffic_data_exports"
        if not os.path.exists(self.export_directory):
            os.makedirs(self.export_directory)

        # Frame processing
        self.processed_frames = deque(maxlen=5)
        self.last_frame = None
        self.frame_count = 0
        self.start_time = 0

        # GUI update control
        self.last_gui_update = 0
        self.gui_update_interval = 0.03

    def create_ui(self):
        """Create the user interface"""
        # Main layout - use a grid layout for more control
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=0)  # Header
        self.root.rowconfigure(1, weight=1)  # Content
        self.root.rowconfigure(2, weight=0)  # Status bar

        # Create UI components
        self.create_menu()
        self.create_toolbar()
        self.create_main_content()
        self.create_status_bar()

    def create_menu(self):
        """Create the application menu"""
        self.menu_bar = Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # File menu
        file_menu = Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_command(label="Open Video", command=self.select_video)
        file_menu.add_separator()
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_command(label="Generate Report", command=self.generate_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        # View menu
        view_menu = Menu(self.menu_bar, tearoff=0)

        self.show_boxes_var = tk.BooleanVar(value=self.show_bounding_boxes)
        self.show_labels_var = tk.BooleanVar(value=self.show_labels)
        self.show_tracking_var = tk.BooleanVar(value=self.show_tracking)
        self.show_heatmap_var = tk.BooleanVar(value=self.show_heatmap)
        self.show_counts_var = tk.BooleanVar(value=self.show_count_info)
        self.show_arrows_var = tk.BooleanVar(value=self.show_direction_arrows)

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
            label="Show Tracking",
            variable=self.show_tracking_var,
            command=self.update_display_settings,
        )
        view_menu.add_checkbutton(
            label="Show Heatmap",
            variable=self.show_heatmap_var,
            command=self.update_display_settings,
        )
        view_menu.add_checkbutton(
            label="Show Count Info",
            variable=self.show_counts_var,
            command=self.update_display_settings,
        )
        view_menu.add_checkbutton(
            label="Show Direction Arrows",
            variable=self.show_arrows_var,
            command=self.update_display_settings,
        )
        view_menu.add_separator()
        view_menu.add_checkbutton(
            label="Dark Mode",
            command=self.toggle_theme,
            variable=tk.BooleanVar(value=self.dark_mode),
        )
        self.menu_bar.add_cascade(label="View", menu=view_menu)

        # Analysis menu
        analysis_menu = Menu(self.menu_bar, tearoff=0)
        analysis_menu.add_command(
            label="Speed Analysis", command=self.show_speed_analysis
        )
        analysis_menu.add_command(
            label="Density Analysis", command=self.show_density_analysis
        )
        analysis_menu.add_command(
            label="Direction Analysis", command=self.show_direction_analysis
        )
        analysis_menu.add_command(
            label="Time Series Analysis", command=self.show_time_analysis
        )
        analysis_menu.add_separator()
        analysis_menu.add_command(
            label="Analytics Dashboard", command=self.show_dashboard
        )
        self.menu_bar.add_cascade(label="Analysis", menu=analysis_menu)

        # Settings menu
        settings_menu = Menu(self.menu_bar, tearoff=0)

        # Model settings
        model_menu = Menu(settings_menu, tearoff=0)
        self.model_size_var = tk.StringVar(value=self.model_size)
        model_menu.add_radiobutton(
            label="YOLOv8 Nano (fastest)", variable=self.model_size_var, value="nano"
        )
        model_menu.add_radiobutton(
            label="YOLOv8 Small (balanced)", variable=self.model_size_var, value="small"
        )
        model_menu.add_radiobutton(
            label="YOLOv8 Medium (accurate)",
            variable=self.model_size_var,
            value="medium",
        )
        settings_menu.add_cascade(label="Model Size", menu=model_menu)

        # Detection settings
        detection_menu = Menu(settings_menu, tearoff=0)
        self.detection_mode_var = tk.StringVar(value=self.detection_mode)
        detection_menu.add_radiobutton(
            label="Vehicles Only",
            variable=self.detection_mode_var,
            value="vehicles",
            command=self.update_detection_mode,
        )
        detection_menu.add_radiobutton(
            label="People Only",
            variable=self.detection_mode_var,
            value="people",
            command=self.update_detection_mode,
        )
        detection_menu.add_radiobutton(
            label="All Objects",
            variable=self.detection_mode_var,
            value="all",
            command=self.update_detection_mode,
        )
        detection_menu.add_radiobutton(
            label="Custom Classes",
            variable=self.detection_mode_var,
            value="custom",
            command=self.select_custom_classes,
        )
        settings_menu.add_cascade(label="Detection Mode", menu=detection_menu)

        # Performance settings
        perf_menu = Menu(settings_menu, tearoff=0)
        perf_menu.add_command(
            label=f"Frame Skip: {self.frame_skip}", command=self.set_frame_skip
        )
        perf_menu.add_command(
            label=f"Downscale Factor: {self.downscale_factor}",
            command=self.set_downscale,
        )
        settings_menu.add_cascade(label="Performance", menu=perf_menu)

        settings_menu.add_separator()
        settings_menu.add_command(
            label="Detection Threshold", command=self.set_detection_threshold
        )
        settings_menu.add_separator()
        settings_menu.add_command(
            label="Reset All Settings", command=self.reset_settings
        )

        self.menu_bar.add_cascade(label="Settings", menu=settings_menu)

        # Help menu
        help_menu = Menu(self.menu_bar, tearoff=0)
        help_menu.add_command(label="Instructions", command=self.show_instructions)
        help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=help_menu)

    def create_toolbar(self):
        """Create the application toolbar"""
        toolbar_frame = ttk.Frame(self.root)
        toolbar_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=5)

        # Load model button
        self.load_model_btn = ttk.Button(
            toolbar_frame, text="Load Model", command=self.load_model
        )
        self.load_model_btn.pack(side=tk.LEFT, padx=5)

        # Open video button
        self.open_video_btn = ttk.Button(
            toolbar_frame, text="Open Video", command=self.select_video
        )
        self.open_video_btn.pack(side=tk.LEFT, padx=5)

        # Play/pause button
        self.play_button = ttk.Button(
            toolbar_frame, text="Play", command=self.toggle_play, state=tk.DISABLED
        )
        self.play_button.pack(side=tk.LEFT, padx=5)

        # Separator
        ttk.Separator(toolbar_frame, orient="vertical").pack(
            side=tk.LEFT, fill="y", padx=10, pady=5
        )

        # Zone tools
        self.add_line_btn = ttk.Button(
            toolbar_frame, text="Add Counting Line", command=self.start_add_line
        )
        self.add_line_btn.pack(side=tk.LEFT, padx=5)

        self.add_roi_btn = ttk.Button(
            toolbar_frame, text="Add Region", command=self.start_add_roi
        )
        self.add_roi_btn.pack(side=tk.LEFT, padx=5)

        self.clear_zones_btn = ttk.Button(
            toolbar_frame, text="Clear Zones", command=self.clear_zones
        )
        self.clear_zones_btn.pack(side=tk.LEFT, padx=5)

        # Separator
        ttk.Separator(toolbar_frame, orient="vertical").pack(
            side=tk.LEFT, fill="y", padx=10, pady=5
        )

        # Export button
        self.export_btn = ttk.Button(
            toolbar_frame,
            text="Export Data",
            command=self.export_data,
            state=tk.DISABLED,
        )
        self.export_btn.pack(side=tk.LEFT, padx=5)

        # Stats display on the right
        self.total_count_var = tk.StringVar(value="0")
        stats_frame = ttk.Frame(toolbar_frame)
        stats_frame.pack(side=tk.RIGHT, padx=10)

        ttk.Label(stats_frame, text="Total Vehicles:").pack(side=tk.LEFT)
        ttk.Label(
            stats_frame, textvariable=self.total_count_var, font=("Arial", 12, "bold")
        ).pack(side=tk.LEFT, padx=5)

    def create_main_content(self):
        """Create the main content area with video and analysis panels"""
        content_frame = ttk.Frame(self.root)
        content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        # Configure grid layout
        content_frame.columnconfigure(0, weight=7)  # Video takes more space
        content_frame.columnconfigure(1, weight=3)  # Analysis panel
        content_frame.rowconfigure(0, weight=1)

        # Video panel (left side)
        video_frame = ttk.LabelFrame(content_frame, text="Video Feed")
        video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.video_canvas = tk.Canvas(video_frame, bg=self.canvas_bg)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Bind canvas events for zone drawing
        self.video_canvas.bind("<Button-1>", self.canvas_click)
        self.video_canvas.bind("<B1-Motion>", self.canvas_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.canvas_release)

        # Analysis panel (right side)
        analysis_frame = ttk.Frame(content_frame)
        analysis_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Analysis tabs
        self.analysis_notebook = ttk.Notebook(analysis_frame)
        self.analysis_notebook.pack(fill=tk.BOTH, expand=True)

        # Create each analysis tab
        self.create_summary_tab()
        self.create_vehicle_tab()
        self.create_time_tab()
        self.create_spatial_tab()
        self.create_speed_tab()

    def create_summary_tab(self):
        """Create the summary analysis tab"""
        summary_tab = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(summary_tab, text="Summary")

        # Overall stats
        stats_frame = ttk.LabelFrame(summary_tab, text="Statistics")
        stats_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        # Video info
        info_frame = ttk.Frame(stats_frame)
        info_frame.pack(fill=tk.X, expand=False, padx=5, pady=5)

        ttk.Label(info_frame, text="Video:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.video_name_var = tk.StringVar(value="No video loaded")
        ttk.Label(info_frame, textvariable=self.video_name_var).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2
        )

        ttk.Label(info_frame, text="Duration:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.video_time_var = tk.StringVar(value="00:00")
        ttk.Label(info_frame, textvariable=self.video_time_var).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=2
        )

        ttk.Label(info_frame, text="Processing:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2
        )
        self.fps_var = tk.StringVar(value="0 FPS")
        ttk.Label(info_frame, textvariable=self.fps_var).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=2
        )

        # Create a summary plot
        plot_frame = ttk.Frame(summary_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.summary_figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.summary_canvas = FigureCanvasTkAgg(self.summary_figure, plot_frame)
        self.summary_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.setup_summary_plot()

    def create_vehicle_tab(self):
        """Create the vehicle analysis tab"""
        vehicle_tab = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(vehicle_tab, text="Vehicles")

        # Create a plot for vehicle distribution
        plot_frame = ttk.Frame(vehicle_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.vehicle_figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.vehicle_canvas = FigureCanvasTkAgg(self.vehicle_figure, plot_frame)
        self.vehicle_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.setup_vehicle_plot()

    def create_time_tab(self):
        """Create the time analysis tab"""
        time_tab = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(time_tab, text="Time")

        # Create a plot for time series analysis
        plot_frame = ttk.Frame(time_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.time_figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.time_canvas = FigureCanvasTkAgg(self.time_figure, plot_frame)
        self.time_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.setup_time_plot()

    def create_spatial_tab(self):
        """Create the spatial analysis tab"""
        spatial_tab = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(spatial_tab, text="Spatial")

        # Create a plot for heatmap and direction analysis
        plot_frame = ttk.Frame(spatial_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.spatial_figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.spatial_canvas = FigureCanvasTkAgg(self.spatial_figure, plot_frame)
        self.spatial_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.setup_spatial_plot()

    def create_speed_tab(self):
        """Create the speed analysis tab"""
        speed_tab = ttk.Frame(self.analysis_notebook)
        self.analysis_notebook.add(speed_tab, text="Speed")

        # Create a plot for speed analysis
        plot_frame = ttk.Frame(speed_tab)
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.speed_figure = plt.Figure(figsize=(5, 4), dpi=100)
        self.speed_canvas = FigureCanvasTkAgg(self.speed_figure, plot_frame)
        self.speed_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.setup_speed_plot()

    def create_status_bar(self):
        """Create the status bar"""
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=2, column=0, sticky="ew")

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            status_frame, variable=self.progress_var, maximum=100
        )
        self.progress_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(5, 0))
        self.progress_bar.pack_forget()  # Hide initially

        # Status label
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(
            status_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_label.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 5))

    def setup_summary_plot(self):
        """Set up the summary plot"""
        self.summary_figure.clear()
        ax = self.summary_figure.add_subplot(111)
        ax.set_title("Total Vehicles Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count")
        ax.text(
            0.5,
            0.5,
            "No data available yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_facecolor(self.canvas_bg)
        self.summary_figure.set_facecolor(self.canvas_bg)
        self.summary_canvas.draw()

    def setup_vehicle_plot(self):
        """Set up the vehicle distribution plot"""
        self.vehicle_figure.clear()
        ax = self.vehicle_figure.add_subplot(111)
        ax.set_title("Vehicle Type Distribution")
        ax.text(
            0.5,
            0.5,
            "No data available yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_facecolor(self.canvas_bg)
        self.vehicle_figure.set_facecolor(self.canvas_bg)
        self.vehicle_canvas.draw()

    def setup_time_plot(self):
        """Set up the time series plot"""
        self.time_figure.clear()
        ax = self.time_figure.add_subplot(111)
        ax.set_title("Traffic Over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Count")
        ax.text(
            0.5,
            0.5,
            "No data available yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_facecolor(self.canvas_bg)
        self.time_figure.set_facecolor(self.canvas_bg)
        self.time_canvas.draw()

    def setup_spatial_plot(self):
        """Set up the spatial analysis plot"""
        self.spatial_figure.clear()
        ax = self.spatial_figure.add_subplot(111)
        ax.set_title("Traffic Direction Distribution")
        ax.text(
            0.5,
            0.5,
            "No data available yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_facecolor(self.canvas_bg)
        self.spatial_figure.set_facecolor(self.canvas_bg)
        ax.set_xticks([])
        ax.set_yticks([])
        self.spatial_canvas.draw()

    def setup_speed_plot(self):
        """Set up the speed analysis plot"""
        self.speed_figure.clear()
        ax = self.speed_figure.add_subplot(111)
        ax.set_title("Speed Distribution")
        ax.set_xlabel("Speed (km/h)")
        ax.set_ylabel("Frequency")
        ax.text(
            0.5,
            0.5,
            "No data available yet",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_facecolor(self.canvas_bg)
        self.speed_figure.set_facecolor(self.canvas_bg)
        self.speed_canvas.draw()

    # Core functionality methods
    def load_model(self):
        """Load the YOLO model"""
        try:
            self.status_var.set("Loading YOLOv8 model...")
            self.progress_bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(5, 0))
            self.progress_var.set(10)
            self.root.update()

            # Determine model size based on user selection
            model_path = f"yolov8{self.model_size[0]}.pt"  # n, s, or m

            # Load the model
            self.progress_var.set(30)
            self.root.update()

            self.model = YOLO(model_path)

            # Get class names
            self.class_names = self.model.names

            self.progress_var.set(80)
            self.root.update()

            self.model_loaded = True

            # Check if using CPU or GPU
            device = "GPU" if torch.cuda.is_available() else "CPU"
            self.status_var.set(f"Model loaded successfully. Using {device}.")

            # Hide progress bar
            self.progress_var.set(100)
            self.root.update()
            self.progress_bar.pack_forget()

            # Enable play button if video is loaded
            if self.cap is not None:
                self.play_button.config(state=tk.NORMAL)

        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            self.progress_bar.pack_forget()
            messagebox.showerror("Model Loading Error", str(e))

    def select_video(self):
        """Select and open a video file"""
        video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
        )

        if video_path:
            # Close any previously opened video
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.video_path = video_path
            self.status_var.set(f"Opening video: {os.path.basename(video_path)}")
            self.video_name_var.set(os.path.basename(video_path))

            try:
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    self.status_var.set(f"Error: Could not open video file")
                    return

                # Get video properties
                self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.video_fps <= 0:
                    self.video_fps = 30  # Default if not available

                self.video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_resolution = (self.video_width, self.video_height)

                # Initialize heatmap with correct dimensions
                self.heatmap_data = np.zeros(
                    (self.video_height, self.video_width), dtype=np.float32
                )

                # Reset analysis data
                self.reset_analysis_data()

                # Enable play button if model is loaded
                if self.model_loaded:
                    self.play_button.config(state=tk.NORMAL)
                    self.export_btn.config(state=tk.NORMAL)

                # Show first frame
                ret, frame = self.cap.read()
                if ret:
                    # Set default counting line
                    middle_y = frame.shape[0] // 2
                    new_line = {
                        "start": (0, middle_y),
                        "end": (frame.shape[1], middle_y),
                        "name": "Main Line",
                        "color": (255, 0, 0),
                        "count": 0,
                    }
                    self.counting_lines.append(new_line)

                    # Display first frame
                    self.show_frame(frame)

                    self.status_var.set(
                        f"Video loaded: {os.path.basename(video_path)} - "
                        + f"{self.video_width}x{self.video_height} at {self.video_fps:.1f} FPS"
                    )
                else:
                    self.status_var.set(
                        "Error: Could not read the first frame of the video."
                    )

            except Exception as e:
                self.status_var.set(f"Error opening video: {str(e)}")
                messagebox.showerror("Video Loading Error", str(e))

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

            # Reset video if at the end
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

    def process_video(self):
        """Process video frames in a separate thread with improved stability"""
        processing_times = deque(maxlen=30)  # For FPS calculation

        try:
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

                # Frame skipping for better performance
                self.current_skip_count += 1
                if self.current_skip_count % self.frame_skip != 0:
                    # Skip processing but still increment frame counter
                    self.frame_count += 1
                    continue

                # Process the frame
                processed_frame, detections = self.process_frame(frame, frame.copy())

                # Store for possible redrawing
                self.last_frame = processed_frame

                # Calculate processing time and FPS
                end_time = time.time()
                processing_time = end_time - start_time
                processing_times.append(processing_time)

                # Update FPS display every second
                if len(processing_times) >= 10:
                    avg_time = sum(processing_times) / len(processing_times)
                    fps = 1 / avg_time if avg_time > 0 else 0
                    self.root.after(0, lambda f=fps: self.fps_var.set(f"{f:.1f} FPS"))

                # Add to frame queue for GUI thread
                # Only add if queue is not already full to prevent memory buildup
                if len(self.processed_frames) < self.processed_frames.maxlen:
                    self.processed_frames.append(processed_frame)

                # Update video time
                current_time = (
                    self.frame_count / self.video_fps if self.video_fps > 0 else 0
                )
                mins = int(current_time // 60)
                secs = int(current_time % 60)
                self.root.after(
                    0,
                    lambda m=mins, s=secs: self.video_time_var.set(f"{m:02d}:{s:02d}"),
                )

                # Adaptive sleep to maintain smooth playback
                # Target a consistent frame rate based on video FPS
                target_time = 1.0 / self.video_fps if self.video_fps > 0 else 0.033
                if processing_time < target_time:
                    time.sleep(target_time - processing_time)

        except Exception as e:
            self.is_playing = False
            self.root.after(
                0,
                lambda msg=f"Error during video processing: {str(e)}": self.status_var.set(
                    msg
                ),
            )
            messagebox.showerror("Processing Error", str(e))

    def update_gui(self):
        """Update GUI from the main thread at a controlled rate"""
        if not self.is_playing:
            return

        try:
            # Use a more consistent update approach with double buffering
            # Only update when we have a new frame to show
            if len(self.processed_frames) > 0:
                frame = self.processed_frames.pop()
                self.show_frame(frame)

            # Update plots less frequently to reduce processing load
            if self.frame_count % 60 == 0:  # Update plots every 60 frames instead of 30
                self.update_plots()

            # Schedule next update with a fixed interval for smoother playback
            self.update_id = self.root.after(33, self.update_gui)  # ~30fps max

        except Exception as e:
            self.status_var.set(f"Error updating GUI: {str(e)}")
            self.is_playing = False
            self.play_button.config(text="Play")
            messagebox.showerror("GUI Update Error", str(e))

    def process_frame(self, frame, original_frame=None):
        """Process a frame with object detection and tracking"""
        if frame is None or not self.model_loaded:
            return frame, []

        try:
            # Use original frame if provided, otherwise use input frame
            if original_frame is not None:
                display_frame = original_frame.copy()
            else:
                display_frame = frame.copy()

            # Increment frame counter
            self.frame_count += 1

            # Get current timestamp
            current_time = time.time() - self.start_time

            # Run YOLO detection
            results = self.model(frame, conf=self.detection_threshold, verbose=False)

            # Process detections
            current_detections = []

            # Filter detections based on detection mode
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class and confidence
                    class_id = int(box.cls[0].item())

                    # Skip based on detection mode
                    if (
                        (
                            self.detection_mode == "vehicles"
                            and class_id not in self.vehicle_classes
                        )
                        or (
                            self.detection_mode == "people"
                            and class_id != self.person_class
                        )
                        or (
                            self.detection_mode == "custom"
                            and class_id not in self.custom_classes
                        )
                    ):
                        continue

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    # Skip if object is too small
                    if (x2 - x1) < 20 or (y2 - y1) < 20:
                        continue

                    class_name = self.class_names.get(class_id, f"Class {class_id}")
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

                    # Update heatmap
                    # Scale coordinates back if needed
                    if self.downscale_factor != 1.0:
                        scale_x = display_frame.shape[1] / frame.shape[1]
                        scale_y = display_frame.shape[0] / frame.shape[0]
                        orig_center_x = int(center_x * scale_x)
                        orig_center_y = int(center_y * scale_y)
                    else:
                        orig_center_x, orig_center_y = center_x, center_y

                    # Update heatmap only if within bounds
                    if (
                        0 <= orig_center_y < self.heatmap_data.shape[0]
                        and 0 <= orig_center_x < self.heatmap_data.shape[1]
                    ):
                        self.heatmap_data[orig_center_y, orig_center_x] += 1

                        # Add a gaussian blob around center for better visualization
                        y, x = np.ogrid[
                            -orig_center_y : self.heatmap_data.shape[0] - orig_center_y,
                            -orig_center_x : self.heatmap_data.shape[1] - orig_center_x,
                        ]
                        mask = x * x + y * y <= 25  # Circle with radius 5
                        self.heatmap_data[
                            mask
                        ] += 0.2  # Lower intensity for surrounding pixels

            # Track objects across frames
            self.track_objects(current_detections, current_time, display_frame)

            # Draw visualization overlays
            self.draw_visualizations(display_frame, current_detections)

            # Update status occasionally
            if self.frame_count % 30 == 0:
                self.status_var.set(
                    f"Processing frame {self.frame_count} | "
                    + f"Time: {current_time:.1f}s | "
                    + f"Total vehicles: {self.total_vehicle_count}"
                )
                self.total_count_var.set(str(self.total_vehicle_count))

            return display_frame, current_detections

        except Exception as e:
            self.status_var.set(f"Error processing frame: {str(e)}")
            return frame, []  # Return original frame if processing fails

    def track_objects(self, current_detections, current_time, display_frame=None):
        """Track objects across frames and compute metrics"""
        try:
            # If no history yet, initialize with current detections
            if not self.object_history:
                for det in current_detections:
                    # Only track objects we're interested in based on detection mode
                    if (
                        (
                            self.detection_mode == "vehicles"
                            and det["class_id"] in self.vehicle_classes
                        )
                        or (
                            self.detection_mode == "people"
                            and det["class_id"] == self.person_class
                        )
                        or (self.detection_mode == "all")
                        or (
                            self.detection_mode == "custom"
                            and det["class_id"] in self.custom_classes
                        )
                    ):

                        obj_id = self.next_object_id
                        self.next_object_id += 1

                        self.object_history[obj_id] = {
                            "positions": deque(maxlen=30),  # Last 30 positions
                            "timestamps": deque(maxlen=30),  # Corresponding timestamps
                            "class_id": det["class_id"],
                            "class_name": det["class_name"],
                            "counted": False,
                            "line_counted": set(),  # Which lines this object has crossed
                            "roi_counted": set(),  # Which ROIs this object has been counted in
                            "last_seen": current_time,
                            "boxes": deque(maxlen=30),  # Last 30 bounding boxes
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
                if (
                    (
                        self.detection_mode == "vehicles"
                        and det["class_id"] not in self.vehicle_classes
                    )
                    or (
                        self.detection_mode == "people"
                        and det["class_id"] != self.person_class
                    )
                    or (
                        self.detection_mode == "custom"
                        and det["class_id"] not in self.custom_classes
                        and self.detection_mode != "all"
                    )
                ):
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

                    # Check for line crossings and ROI presence
                    self.check_line_crossings(best_match)
                    self.check_roi_presence(best_match)

                    # Calculate direction and speed
                    self.calculate_direction_and_speed(best_match)
                else:
                    unmatched_detections.append(det)

            # For unmatched detections, try to match by distance
            for det in unmatched_detections:
                center = det["center"]
                min_dist = float("inf")
                best_match = None

                # Find the closest tracked object of the same class
                for obj_id, obj_data in self.object_history.items():
                    if (
                        obj_id in matched_indices
                        or obj_data["class_id"] != det["class_id"]
                    ):
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

                    # Check for line crossings and ROI presence
                    self.check_line_crossings(best_match)
                    self.check_roi_presence(best_match)

                    # Calculate direction and speed
                    self.calculate_direction_and_speed(best_match)
                else:
                    # Create a new tracked object
                    # Limit the number of tracked objects for performance
                    if len(self.object_history) < self.max_tracked_objects:
                        obj_id = self.next_object_id
                        self.next_object_id += 1

                        self.object_history[obj_id] = {
                            "positions": deque(maxlen=30),
                            "timestamps": deque(maxlen=30),
                            "class_id": det["class_id"],
                            "class_name": det["class_name"],
                            "counted": False,
                            "line_counted": set(),
                            "roi_counted": set(),
                            "last_seen": current_time,
                            "boxes": deque(maxlen=30),
                            "direction": None,
                            "speed": None,
                        }

                        self.object_history[obj_id]["positions"].append(center)
                        self.object_history[obj_id]["timestamps"].append(current_time)
                        self.object_history[obj_id]["boxes"].append(det["box"])

            # Record data for time series plot
            if (
                len(self.timestamps) == 0 or current_time - self.timestamps[-1] >= 0.5
            ):  # Add data point every 0.5s
                self.timestamps.append(current_time)

                # Count visible objects of each class for the time series
                class_counts = defaultdict(int)
                for obj_id, obj_data in self.object_history.items():
                    if (
                        current_time - obj_data["last_seen"] < 0.5
                    ):  # Object is currently visible
                        class_counts[obj_data["class_name"]] += 1

                # Update time series data for relevant classes
                for class_name in class_counts:
                    self.time_series_data[class_name].append(class_counts[class_name])

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
                        # Choose color based on class
                        class_id = obj_data["class_id"]
                        if class_id in self.class_colors:
                            color = self.class_colors[class_id]
                        else:
                            color = (0, 255, 255)  # Default to cyan

                        # Draw tracking line
                        positions = list(obj_data["positions"])
                        for i in range(1, len(positions)):
                            cv2.line(
                                display_frame, positions[i - 1], positions[i], color, 2
                            )

        except Exception as e:
            self.status_var.set(f"Error in object tracking: {str(e)}")

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
            line_name = line["name"]

            # Skip if already counted for this line
            if line_name in obj_data["line_counted"]:
                continue

            if self.line_segment_intersection(
                prev_pos, curr_pos, line["start"], line["end"]
            ):
                # Object has crossed this line
                obj_data["line_counted"].add(line_name)

                # Update total count if not already counted
                if not obj_data["counted"]:
                    self.total_vehicle_count += 1
                    obj_data["counted"] = True

                # Update line-specific count
                line["count"] += 1

                # Log the event for data export
                crossing_data = {
                    "time": obj_data["timestamps"][-1],
                    "object_id": obj_id,
                    "class": obj_data["class_name"],
                    "line": line_name,
                    "direction": (
                        obj_data["direction"] if obj_data["direction"] else "unknown"
                    ),
                    "speed": obj_data["speed"] if obj_data["speed"] else 0,
                }
                self.data_log.append(crossing_data)

    def check_roi_presence(self, obj_id):
        """Check if an object is present in any region of interest"""
        obj_data = self.object_history[obj_id]

        # Get current position
        curr_pos = obj_data["positions"][-1]

        # Check each ROI
        for roi in self.regions_of_interest:
            roi_name = roi["name"]

            # Skip if already counted for this ROI
            if roi_name in obj_data["roi_counted"]:
                continue

            # Check if point is inside polygon
            if self.point_in_polygon(curr_pos, roi["points"]):
                # Update ROI count
                self.roi_counts[roi_name] += 1
                obj_data["roi_counted"].add(roi_name)

    def calculate_direction_and_speed(self, obj_id):
        """Calculate direction and speed of a tracked object"""
        obj_data = self.object_history[obj_id]

        # Need at least two positions with timestamps to calculate
        if len(obj_data["positions"]) < 2 or len(obj_data["timestamps"]) < 2:
            return

        # Get positions and timestamps
        positions = list(obj_data["positions"])
        timestamps = list(obj_data["timestamps"])

        # Calculate direction vector from the first position to the last
        if len(positions) >= 5:  # Use more points for better direction estimation
            start_pos = positions[0]
            end_pos = positions[-1]

            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]

            # Determine cardinal direction
            direction = None
            if abs(dx) > abs(dy):
                direction = "east" if dx > 0 else "west"
            else:
                direction = "south" if dy > 0 else "north"

            # Store direction
            obj_data["direction"] = direction

            # Update direction counts if changed
            if direction and obj_data["direction"] != direction:
                self.direction_counts[direction] += 1

            # Calculate speed
            if len(timestamps) >= 5:
                dt = timestamps[-1] - timestamps[0]
                if dt > 0:
                    # Calculate distance in pixels
                    distance = np.sqrt(dx**2 + dy**2)

                    # Convert to real-world distance (approximate)
                    # Assume 100 pixels is roughly 1 meter
                    meters = distance / 100

                    # Calculate speed in km/h
                    speed_kmh = (meters / dt) * 3.6

                    # Store speed
                    obj_data["speed"] = speed_kmh

                    # Add to average speeds
                    self.avg_speeds[obj_data["class_name"]].append(speed_kmh)

                    # Keep only recent speeds for average calculation
                    if len(self.avg_speeds[obj_data["class_name"]]) > 100:
                        self.avg_speeds[obj_data["class_name"]] = self.avg_speeds[
                            obj_data["class_name"]
                        ][-100:]

    def line_segment_intersection(self, p1, p2, p3, p4):
        """Check if two line segments intersect using cross-product method"""

        def ccw(a, b, c):
            # Determine if three points make a counter-clockwise turn
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        # Two line segments intersect if and only if:
        # 1. p1 and p2 are on opposite sides of line p3-p4, AND
        # 2. p3 and p4 are on opposite sides of line p1-p2
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using ray casting algorithm"""
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]  # Wrap around to first point

            # Check if point is vertically between the edge endpoints
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        # Calculate intersection with the edge
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        # Handle special case of horizontal edge
                        if p1x == p2x or x <= xinters:
                            inside = not inside  # Toggle inside status

            # Move to next edge
            p1x, p1y = p2x, p2y

        return inside

    def draw_visualizations(self, frame, detections):
        """Draw visual elements on the frame"""
        if frame is None:
            return frame

        # Draw bounding boxes and labels
        if self.show_bounding_boxes:
            for det in detections:
                # Get bounding box
                x1, y1, x2, y2 = det["box"]

                # Get class color
                class_id = det["class_id"]
                if class_id in self.class_colors:
                    color = self.class_colors[class_id]
                else:
                    color = (0, 255, 255)  # Default to cyan

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label if enabled
                if self.show_labels:
                    # Create label
                    label = f"{det['class_name']}: {det['confidence']:.2f}"

                    # Get text size for background
                    text_size, _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )

                    # Draw label background
                    cv2.rectangle(
                        frame,
                        (x1, y1 - text_size[1] - 5),
                        (x1 + text_size[0], y1),
                        color,
                        -1,
                    )

                    # Draw label text
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),  # Black text on colored background
                        2,
                    )

        # Draw direction arrows if enabled
        if self.show_direction_arrows:
            for obj_id, obj_data in self.object_history.items():
                # Only show for recent objects with direction
                if (
                    time.time() - self.start_time - obj_data["last_seen"] < 0.5
                    and obj_data["direction"]
                ):
                    # Need at least one position
                    if not obj_data["positions"]:
                        continue

                    # Get current position
                    curr_pos = obj_data["positions"][-1]

                    # Set arrow parameters based on direction
                    if obj_data["direction"] == "north":
                        arrow_end = (curr_pos[0], curr_pos[1] - 30)
                    elif obj_data["direction"] == "south":
                        arrow_end = (curr_pos[0], curr_pos[1] + 30)
                    elif obj_data["direction"] == "east":
                        arrow_end = (curr_pos[0] + 30, curr_pos[1])
                    elif obj_data["direction"] == "west":
                        arrow_end = (curr_pos[0] - 30, curr_pos[1])
                    else:
                        continue

                    # Get class color
                    class_id = obj_data["class_id"]
                    if class_id in self.class_colors:
                        color = self.class_colors[class_id]
                    else:
                        color = (0, 255, 255)  # Default to cyan

                    # Draw arrow
                    cv2.arrowedLine(frame, curr_pos, arrow_end, color, 2, tipLength=0.3)

                    # Draw speed if available
                    if obj_data["speed"] is not None:
                        speed_label = f"{obj_data['speed']:.1f} km/h"
                        cv2.putText(
                            frame,
                            speed_label,
                            (curr_pos[0] + 5, curr_pos[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                        )

        # Overlay heatmap if enabled
        if self.show_heatmap:
            try:
                # Create a normalized copy of the heatmap
                norm_heatmap = self.heatmap_data.copy()
                if np.max(norm_heatmap) > 0:
                    norm_heatmap = norm_heatmap / np.max(norm_heatmap)

                # Resize to frame size if needed
                if norm_heatmap.shape[:2] != frame.shape[:2]:
                    norm_heatmap = cv2.resize(
                        norm_heatmap, (frame.shape[1], frame.shape[0])
                    )

                # Convert to colormap
                heatmap_colored = cv2.applyColorMap(
                    (norm_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
                )

                # Create overlay with transparency
                overlay = frame.copy()
                cv2.addWeighted(heatmap_colored, 0.5, overlay, 0.5, 0, frame)
            except Exception as e:
                self.status_var.set(f"Error drawing heatmap: {str(e)}")

        # Draw counting lines
        for line in self.counting_lines:
            start_point = line["start"]
            end_point = line["end"]
            color = line["color"]

            # Draw the line
            cv2.line(frame, start_point, end_point, color, 2, cv2.LINE_AA)

            # Add label with count if enabled
            if self.show_count_info:
                # Draw line name and count
                label = f"{line['name']}: {line['count']}"
                # Position text near line start
                text_x = start_point[0] + 10
                text_y = start_point[1] - 10

                # Draw background for better visibility
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    frame,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 0),
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # White text
                    2,
                )

        # Draw regions of interest
        for roi in self.regions_of_interest:
            points = roi["points"]

            # Draw the ROI polygon
            points_array = np.array(points, np.int32)
            cv2.polylines(frame, [points_array], True, (0, 255, 255), 2, cv2.LINE_AA)

            # Add ROI label with count if enabled
            if self.show_count_info:
                # Calculate center point of ROI for label placement
                center_x = sum(p[0] for p in points) // len(points)
                center_y = sum(p[1] for p in points) // len(points)

                label = f"{roi['name']}: {self.roi_counts[roi['name']]}"

                # Draw background for better visibility
                text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(
                    frame,
                    (
                        center_x - text_size[0] // 2 - 5,
                        center_y - text_size[1] // 2 - 5,
                    ),
                    (
                        center_x + text_size[0] // 2 + 5,
                        center_y + text_size[1] // 2 + 5,
                    ),
                    (0, 0, 0),
                    -1,
                )

                cv2.putText(
                    frame,
                    label,
                    (center_x - text_size[0] // 2, center_y + text_size[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),  # White text
                    2,
                )

        # Draw temporary elements for interactive zone creation
        if self.drawing_mode == "line" and self.drawing_start and self.temp_line:
            start_x, start_y = self.drawing_start
            end_x, end_y = self.temp_line
            cv2.line(
                frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 2, cv2.LINE_AA
            )

        elif self.drawing_mode == "roi" and len(self.temp_roi) > 0:
            # Draw lines connecting the points
            for i in range(len(self.temp_roi) - 1):
                x1, y1 = self.temp_roi[i]
                x2, y2 = self.temp_roi[i + 1]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

            # Draw points
            for x, y in self.temp_roi:
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            # Connect last point to first if we have at least 3 points
            if len(self.temp_roi) >= 3:
                x1, y1 = self.temp_roi[-1]
                x2, y2 = self.temp_roi[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

        # Draw video time and processing rate
        current_time = time.time() - self.start_time
        mins = int(current_time // 60)
        secs = int(current_time % 60)
        time_text = f"Time: {mins:02d}:{secs:02d}"

        cv2.putText(
            frame,
            time_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        count_text = f"Total: {self.total_vehicle_count}"
        cv2.putText(
            frame,
            count_text,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        return frame

    # UI Utility Methods
    def resize_frame(self, frame):
        """Resize a frame to fit the canvas while maintaining aspect ratio"""
        if frame is None:
            return None

        try:
            # Get current canvas dimensions
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()

            # Use reasonable defaults if canvas dimensions aren't available
            if canvas_width <= 1:
                canvas_width = 640
            if canvas_height <= 1:
                canvas_height = 480

            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]

            # Calculate scaling factor
            width_ratio = canvas_width / frame_width
            height_ratio = canvas_height / frame_height

            # Use the smaller ratio to ensure the image fits
            scale = min(width_ratio, height_ratio)

            # Calculate new dimensions
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)

            # Resize the frame
            resized_frame = cv2.resize(frame, (new_width, new_height))
            return resized_frame
        except Exception as e:
            self.status_var.set(f"Error resizing frame: {str(e)}")
            return frame  # Return original frame if resize fails

    def show_frame(self, frame):
        """Display a frame on the canvas with improved stability"""
        if frame is None:
            return

        try:
            # Convert frame to RGB (from BGR)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Convert to PhotoImage - using a more efficient approach
            img = Image.fromarray(frame_rgb)

            # Store the current image to prevent garbage collection issues
            self.current_image = ImageTk.PhotoImage(image=img)

            # Get canvas dimensions
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()

            # Use reasonable defaults if canvas dimensions aren't available
            if canvas_width <= 1:
                canvas_width = 640
            if canvas_height <= 1:
                canvas_height = 480

            # Clear canvas only once per update to prevent flickering
            self.video_canvas.delete("all")

            # Calculate position to center image in canvas
            x_offset = max(0, (canvas_width - img.width) // 2)
            y_offset = max(0, (canvas_height - img.height) // 2)

            # Create image centered in the canvas - using our persistent reference
            self.video_canvas.create_image(
                x_offset, y_offset, anchor=tk.NW, image=self.current_image
            )

            # Draw any overlays (lines, regions, etc.) after the image is already displayed
            self.draw_canvas_overlays()

        except Exception as e:
            self.status_var.set(f"Error displaying frame: {str(e)}")

    def update_plots(self):
        """Update all analysis plots"""
        try:
            # Update summary plot
            self.update_summary_plot()

            # Update vehicle distribution plot
            self.update_vehicle_plot()

            # Update time series plot
            self.update_time_plot()

            # Update spatial analysis plot
            self.update_spatial_plot()

            # Update speed analysis plot
            self.update_speed_plot()
        except Exception as e:
            self.status_var.set(f"Error updating plots: {str(e)}")

    def update_summary_plot(self):
        """Update the summary plot with latest data"""
        self.summary_figure.clear()

        # Create time series of total vehicles
        ax = self.summary_figure.add_subplot(111)
        ax.set_facecolor(self.canvas_bg)

        if self.timestamps and len(self.timestamps) > 1:
            # Combine all vehicle counts by timestamp
            combined_counts = []
            valid_series = [data for data in self.time_series_data.values() if data]

            if valid_series:
                min_length = min(
                    len(self.timestamps), min(len(data) for data in valid_series)
                )

                for i in range(min_length):
                    total = sum(
                        data[i]
                        for data in self.time_series_data.values()
                        if i < len(data)
                    )
                    combined_counts.append(total)

                # Plot the combined data
                ax.plot(
                    self.timestamps[:min_length],
                    combined_counts,
                    color=self.accent_color,
                    linewidth=2,
                )
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Vehicle Count")
                ax.set_title("Total Vehicles Over Time")

                # Set background and text colors
                ax.tick_params(colors=self.fg_color)
                ax.xaxis.label.set_color(self.fg_color)
                ax.yaxis.label.set_color(self.fg_color)
                ax.title.set_color(self.fg_color)

            else:
                ax.text(
                    0.5,
                    0.5,
                    "No time series data available yet",
                    ha="center",
                    va="center",
                    color=self.fg_color,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.text(
                0.5,
                0.5,
                "No time series data available yet",
                ha="center",
                va="center",
                color=self.fg_color,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        self.summary_figure.tight_layout()
        self.summary_canvas.draw()

    def update_vehicle_plot(self):
        """Update the vehicle distribution plot"""
        self.vehicle_figure.clear()

        # Create pie chart of vehicle types
        ax = self.vehicle_figure.add_subplot(111)
        ax.set_facecolor(self.canvas_bg)

        # Filter for vehicle classes
        vehicle_counts = {}
        for class_id in self.vehicle_classes:
            if class_id in self.class_names:
                class_name = self.class_names[class_id]
                count = self.detected_objects.get(class_name, 0)
                if count > 0:
                    vehicle_counts[class_name] = count

        if vehicle_counts:
            labels = list(vehicle_counts.keys())
            sizes = list(vehicle_counts.values())

            # Use custom colors
            colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

            # Create pie chart
            ax.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"color": self.fg_color},
            )
            ax.set_title("Vehicle Type Distribution", color=self.fg_color)
        else:
            ax.text(
                0.5,
                0.5,
                "No vehicle data available yet",
                ha="center",
                va="center",
                color=self.fg_color,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        self.vehicle_figure.tight_layout()
        self.vehicle_canvas.draw()

    def update_time_plot(self):
        """Update the time series plot"""
        self.time_figure.clear()

        # Create time series plot
        ax = self.time_figure.add_subplot(111)
        ax.set_facecolor(self.canvas_bg)

        if self.timestamps and len(self.timestamps) > 1:
            # Plot individual vehicle classes
            for class_id in self.vehicle_classes:
                if class_id in self.class_names:
                    class_name = self.class_names[class_id]
                    if (
                        class_name in self.time_series_data
                        and len(self.time_series_data[class_name]) > 1
                    ):
                        # Make sure data lengths match
                        data_len = min(
                            len(self.timestamps), len(self.time_series_data[class_name])
                        )
                        ax.plot(
                            self.timestamps[:data_len],
                            self.time_series_data[class_name][:data_len],
                            label=class_name,
                        )

            ax.set_xlabel("Time (s)", color=self.fg_color)
            ax.set_ylabel("Count", color=self.fg_color)
            ax.set_title("Traffic Volume Over Time", color=self.fg_color)
            ax.tick_params(colors=self.fg_color)

            if any(
                len(self.time_series_data.get(self.class_names.get(class_id, ""), []))
                > 0
                for class_id in self.vehicle_classes
                if class_id in self.class_names
            ):
                ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No time series data available yet",
                ha="center",
                va="center",
                color=self.fg_color,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        self.time_figure.tight_layout()
        self.time_canvas.draw()

    def update_spatial_plot(self):
        """Update the spatial analysis plot"""
        self.spatial_figure.clear()

        # Create direction distribution pie chart
        ax = self.spatial_figure.add_subplot(111)
        ax.set_facecolor(self.canvas_bg)

        if sum(self.direction_counts.values()) > 0:
            # Filter out zero counts
            labels = []
            sizes = []
            for direction, count in self.direction_counts.items():
                if count > 0:
                    labels.append(direction.capitalize())
                    sizes.append(count)

            if sizes:
                # Use custom colors
                colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

                # Create pie chart
                ax.pie(
                    sizes,
                    labels=labels,
                    colors=colors,
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"color": self.fg_color},
                )
                ax.set_title("Traffic Direction Distribution", color=self.fg_color)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No direction data available yet",
                    ha="center",
                    va="center",
                    color=self.fg_color,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.text(
                0.5,
                0.5,
                "No direction data available yet",
                ha="center",
                va="center",
                color=self.fg_color,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        self.spatial_figure.tight_layout()
        self.spatial_canvas.draw()

    def update_speed_plot(self):
        """Update the speed analysis plot"""
        self.speed_figure.clear()

        # Create speed distribution histogram
        ax = self.speed_figure.add_subplot(111)
        ax.set_facecolor(self.canvas_bg)

        # Combine all speed data
        all_speeds = []
        for speeds in self.avg_speeds.values():
            all_speeds.extend(speeds)

        if all_speeds:
            # Create histogram
            bins = min(20, max(5, len(all_speeds) // 5 + 1))
            ax.hist(all_speeds, bins=bins, color=self.accent_color, alpha=0.7)
            ax.set_xlabel("Speed (km/h)", color=self.fg_color)
            ax.set_ylabel("Frequency", color=self.fg_color)
            ax.set_title("Speed Distribution", color=self.fg_color)
            ax.tick_params(colors=self.fg_color)

            # Add vertical line for average speed
            avg_speed = sum(all_speeds) / len(all_speeds)
            ax.axvline(x=avg_speed, color="red", linestyle="--", linewidth=2)
            ax.text(
                avg_speed + 0.5,
                ax.get_ylim()[1] * 0.9,
                f"Avg: {avg_speed:.1f} km/h",
                color="red",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No speed data available yet",
                ha="center",
                va="center",
                color=self.fg_color,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        self.speed_figure.tight_layout()
        self.speed_canvas.draw()

    # Zone drawing methods
    def start_add_line(self):
        """Start the process of adding a new counting line"""
        self.drawing_mode = "line"
        self.drawing_start = None
        self.temp_line = None
        self.status_var.set("Click and drag to create a new counting line")

        # Bind Escape key to cancel
        self.root.bind("<Escape>", self.cancel_drawing)

    def start_add_roi(self):
        """Start the process of adding a new region of interest"""
        self.drawing_mode = "roi"
        self.temp_roi = []
        self.status_var.set(
            "Click to add points for the region of interest. Double-click to complete."
        )

        # Bind Escape key to cancel
        self.root.bind("<Escape>", self.cancel_drawing)

        # Bind double-click to finish ROI
        self.video_canvas.bind("<Double-Button-1>", self.finish_roi)

    def canvas_click(self, event):
        """Handle mouse click on the canvas"""
        if self.drawing_mode == "line":
            # Start drawing a line
            self.drawing_start = (event.x, event.y)
            self.temp_line = (event.x, event.y)

        elif self.drawing_mode == "roi":
            # Add a point to the ROI
            self.temp_roi.append((event.x, event.y))

            # Redraw the frame to show the updated ROI
            if hasattr(self, "last_frame") and self.last_frame is not None:
                self.show_frame(self.last_frame)

    def canvas_drag(self, event):
        """Handle mouse drag on the canvas"""
        if self.drawing_mode == "line" and self.drawing_start:
            # Update temporary line endpoint
            self.temp_line = (event.x, event.y)

            # Redraw the frame to show the updated line
            if hasattr(self, "last_frame") and self.last_frame is not None:
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
            if hasattr(self, "last_frame") and self.last_frame is not None:
                self.show_frame(self.last_frame)

            # Unbind Escape
            self.root.unbind("<Escape>")

    def finish_roi(self, event=None):
        """Complete the ROI when double-clicked"""
        if self.drawing_mode == "roi" and len(self.temp_roi) >= 3:
            # Ask for a name
            roi_name = simpledialog.askstring(
                "ROI Name",
                "Enter a name for this region of interest:",
                initialvalue=f"Region {len(self.regions_of_interest) + 1}",
            )

            if roi_name:
                # Add the new ROI
                new_roi = {"points": self.temp_roi.copy(), "name": roi_name}

                self.regions_of_interest.append(new_roi)
                self.roi_counts[roi_name] = 0

                self.status_var.set(f"Added region of interest: {roi_name}")

        # Reset drawing state
        self.drawing_mode = None
        self.temp_roi = []

        # Redraw the frame
        if hasattr(self, "last_frame") and self.last_frame is not None:
            self.show_frame(self.last_frame)

        # Unbind events
        self.root.unbind("<Escape>")
        self.video_canvas.unbind("<Double-Button-1>")

    def cancel_drawing(self, event=None):
        """Cancel the current drawing operation"""
        self.drawing_mode = None
        self.drawing_start = None
        self.temp_line = None
        self.temp_roi = []
        self.status_var.set("Drawing cancelled")

        # Redraw the current frame
        if hasattr(self, "last_frame") and self.last_frame is not None:
            self.show_frame(self.last_frame)

        # Unbind double-click
        self.video_canvas.unbind("<Double-Button-1>")

    def clear_zones(self):
        """Clear all counting lines and regions of interest"""
        if not messagebox.askyesno(
            "Clear Zones",
            "Are you sure you want to clear all counting lines and regions?",
        ):
            return

        self.counting_lines = []
        self.regions_of_interest = []
        self.roi_counts = defaultdict(int)

        # Redraw the current frame
        if hasattr(self, "last_frame") and self.last_frame is not None:
            self.show_frame(self.last_frame)

        self.status_var.set("All zones cleared")

    # Settings methods
    def toggle_theme(self):
        """Toggle between dark and light themes"""
        self.dark_mode = not self.dark_mode
        self.setup_theme()

        # Update all plot backgrounds
        self.update_plots()

        # Redraw the current frame
        if hasattr(self, "last_frame") and self.last_frame is not None:
            self.show_frame(self.last_frame)

        self.status_var.set(
            f"Theme changed to {'Dark' if self.dark_mode else 'Light'} mode"
        )

    def set_frame_skip(self):
        """Opens a dialog to set the frame skip value"""
        new_value = simpledialog.askinteger(
            "Frame Skip",
            "Enter frame skip value (higher values improve performance but reduce smoothness):",
            initialvalue=self.frame_skip,
            minvalue=1,
            maxvalue=10,
        )

        if new_value is not None:
            self.frame_skip = new_value
            self.status_var.set(f"Frame skip set to {self.frame_skip}")

    def set_downscale(self):
        """Opens a dialog to set the downscale factor"""
        new_value = simpledialog.askfloat(
            "Downscale Factor",
            "Enter downscale factor (lower values improve performance but reduce quality):",
            initialvalue=self.downscale_factor,
            minvalue=0.1,
            maxvalue=1.0,
        )

        if new_value is not None:
            self.downscale_factor = new_value
            self.status_var.set(f"Downscale factor set to {self.downscale_factor:.1f}")

    def set_detection_threshold(self):
        """Opens a dialog to set detection confidence threshold"""
        new_value = simpledialog.askfloat(
            "Detection Threshold",
            "Enter detection confidence threshold (0.1-0.9):",
            initialvalue=self.detection_threshold,
            minvalue=0.1,
            maxvalue=0.9,
        )

        if new_value is not None:
            self.detection_threshold = new_value
            self.status_var.set(
                f"Detection threshold set to {self.detection_threshold:.2f}"
            )

    def update_display_settings(self):
        """Update display settings based on checkboxes"""
        self.show_tracking = self.show_tracking_var.get()
        self.show_bounding_boxes = self.show_boxes_var.get()
        self.show_labels = self.show_labels_var.get()
        self.show_count_info = self.show_counts_var.get()
        self.show_heatmap = self.show_heatmap_var.get()
        self.show_direction_arrows = self.show_arrows_var.get()

        # Redraw the current frame if available
        if hasattr(self, "last_frame") and self.last_frame is not None:
            self.show_frame(self.last_frame)

    def update_detection_mode(self):
        """Update the detection mode"""
        self.detection_mode = self.detection_mode_var.get()
        self.status_var.set(f"Detection mode set to: {self.detection_mode}")

    def reset_settings(self):
        """Reset all settings to default values"""
        # Confirm with user
        if not messagebox.askyesno(
            "Reset Settings", "Are you sure you want to reset all settings to defaults?"
        ):
            return

        # Reset detection settings
        self.detection_threshold = 0.4

        # Reset performance settings
        self.frame_skip = 2
        self.downscale_factor = 0.7

        # Reset display settings
        self.show_tracking = True
        self.show_tracking_var.set(True)
        self.show_bounding_boxes = True
        self.show_boxes_var.set(True)
        self.show_labels = True
        self.show_labels_var.set(True)
        self.show_count_info = True
        self.show_counts_var.set(True)
        self.show_heatmap = False
        self.show_heatmap_var.set(False)
        self.show_direction_arrows = True
        self.show_arrows_var.set(True)

        # Reset detection mode
        self.detection_mode = "vehicles"
        self.detection_mode_var.set("vehicles")

        # Reset model size
        self.model_size = "nano"
        self.model_size_var.set("nano")

        # Update UI
        self.status_var.set("All settings reset to defaults")

        # Redraw current frame if available
        if hasattr(self, "last_frame") and self.last_frame is not None:
            self.show_frame(self.last_frame)

    def select_custom_classes(self):
        """Open a dialog to select custom classes to detect"""
        if not self.model_loaded or not self.class_names:
            messagebox.showwarning(
                "No Model Loaded",
                "Please load a YOLO model first to see available classes.",
            )
            return

        # Create a dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Classes to Detect")
        dialog.geometry("300x400")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=self.bg_color)

        # Create a frame with scrollbar
        frame = ttk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create a canvas for scrolling
        canvas = tk.Canvas(frame, yscrollcommand=scrollbar.set, bg=self.bg_color)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar.config(command=canvas.yview)

        # Create a frame inside canvas for checkbuttons
        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor=tk.NW)

        # Create checkbuttons for each class
        class_vars = {}
        for class_id, class_name in self.class_names.items():
            var = tk.BooleanVar(value=class_id in self.custom_classes)
            class_vars[class_id] = var

            cb = ttk.Checkbutton(
                inner_frame, text=f"{class_id}: {class_name}", variable=var
            )
            cb.pack(anchor=tk.W, pady=2)

        # Update canvas scroll region when inner frame changes size
        inner_frame.update_idletasks()
        canvas.config(scrollregion=canvas.bbox(tk.ALL))

        # Buttons
        button_frame = ttk.Frame(dialog)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(
            button_frame,
            text="Select All",
            command=lambda: [var.set(True) for var in class_vars.values()],
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Clear All",
            command=lambda: [var.set(False) for var in class_vars.values()],
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame,
            text="Apply",
            command=lambda: self.apply_custom_classes(class_vars, dialog),
        ).pack(side=tk.RIGHT, padx=5)

        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(
            side=tk.RIGHT, padx=5
        )

    def apply_custom_classes(self, class_vars, dialog):
        """Apply the selected custom classes"""
        self.custom_classes = [
            class_id for class_id, var in class_vars.items() if var.get()
        ]

        if not self.custom_classes:
            messagebox.showwarning(
                "No Classes Selected",
                "Please select at least one class to detect.",
                parent=dialog,
            )
            return

        # Update detection mode
        self.detection_mode = "custom"
        self.detection_mode_var.set("custom")

        # Update status
        class_names = [self.class_names[class_id] for class_id in self.custom_classes]
        self.status_var.set(f"Custom detection enabled for: {', '.join(class_names)}")

        # Close dialog
        dialog.destroy()

    def reset_analysis_data(self):
        """Reset all analysis-related data"""
        self.detected_objects = defaultdict(int)
        self.time_series_data = defaultdict(list)
        self.timestamps = []
        self.frame_count = 0
        self.total_vehicle_count = 0
        self.object_history = {}
        self.next_object_id = 0
        self.total_count_var.set("0")
        self.video_time_var.set("00:00")
        self.fps_var.set("0 FPS")

        # Reset line counts but keep the lines
        for line in self.counting_lines:
            line["count"] = 0

        # Reset ROI counts but keep the ROIs
        self.roi_counts = defaultdict(int)

        # Reset heatmap - use default dimensions if video dimensions aren't available yet
        if hasattr(self, "video_height") and hasattr(self, "video_width"):
            self.heatmap_data = np.zeros(
                (self.video_height, self.video_width), dtype=np.float32
            )
        else:
            self.heatmap_data = np.zeros((480, 640), dtype=np.float32)

        # Reset speed data
        self.object_speeds = {}
        self.avg_speeds = defaultdict(list)

        # Reset direction data
        self.direction_counts = {"north": 0, "south": 0, "east": 0, "west": 0}

        # Update plots
        self.update_plots()

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
                    fieldnames = [
                        "time",
                        "object_id",
                        "class",
                        "line",
                        "direction",
                        "speed",
                    ]
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
                "analysis_duration": (
                    time.time() - self.start_time if hasattr(self, "start_time") else 0
                ),
                "counted_by_class": dict(self.detected_objects),
                "counting_lines": [
                    {"name": line["name"], "count": line["count"]}
                    for line in self.counting_lines
                ],
                "regions_of_interest": [
                    {"name": roi["name"], "count": self.roi_counts[roi["name"]]}
                    for roi in self.regions_of_interest
                ],
                "direction_counts": dict(self.direction_counts),
                "average_speeds": {
                    class_name: sum(speeds) / len(speeds) if speeds else 0
                    for class_name, speeds in self.avg_speeds.items()
                },
            }

            with open(json_path, "w") as jsonfile:
                json.dump(summary_data, jsonfile, indent=4)

            # Export heatmap as an image if we have data
            if np.sum(self.heatmap_data) > 0:
                heatmap_path = os.path.join(
                    self.export_directory, f"traffic_heatmap_{timestamp}.png"
                )

                # Normalize heatmap
                norm_heatmap = self.heatmap_data.copy()
                if np.max(norm_heatmap) > 0:
                    norm_heatmap = norm_heatmap / np.max(norm_heatmap) * 255

                # Apply colormap and save
                heatmap_colored = cv2.applyColorMap(
                    norm_heatmap.astype(np.uint8), cv2.COLORMAP_JET
                )
                cv2.imwrite(heatmap_path, heatmap_colored)

            messagebox.showinfo(
                "Export Successful",
                f"Data exported to {self.export_directory}\n\n"
                + f"Files:\n- {os.path.basename(csv_path)}\n"
                + f"- {os.path.basename(json_path)}",
            )

        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting data: {str(e)}")

    def generate_report(self):
        """Generate a comprehensive analysis report in HTML format"""
        if not self.data_log and self.total_vehicle_count == 0:
            messagebox.showinfo(
                "No Data", "No traffic data available for report generation."
            )
            return

        try:
            # Create timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create export directory if it doesn't exist
            if not os.path.exists(self.export_directory):
                os.makedirs(self.export_directory)

            # Export data first to ensure we have the latest
            self.export_data()

            # Create HTML report
            report_path = os.path.join(
                self.export_directory, f"traffic_report_{timestamp}.html"
            )

            # Get report data
            total_vehicles = self.total_vehicle_count
            analysis_duration = (
                time.time() - self.start_time if hasattr(self, "start_time") else 0
            )
            mins = int(analysis_duration // 60)
            secs = int(analysis_duration % 60)

            # Generate class distribution data
            class_labels = []
            class_counts = []

            for class_name, count in sorted(
                self.detected_objects.items(), key=lambda x: x[1], reverse=True
            ):
                if count > 0:
                    class_labels.append(class_name)
                    class_counts.append(count)

            # Generate line crossing data
            line_labels = []
            line_counts = []

            for line in self.counting_lines:
                line_labels.append(line["name"])
                line_counts.append(line["count"])

            # Generate direction data
            direction_labels = []
            direction_counts = []

            for direction, count in self.direction_counts.items():
                if count > 0:
                    direction_labels.append(direction)
                    direction_counts.append(count)

            # Generate speed data
            speed_labels = []
            speed_values = []

            for class_name, speeds in self.avg_speeds.items():
                if speeds:
                    speed_labels.append(class_name)
                    speed_values.append(sum(speeds) / len(speeds))

            # Create HTML content with modern styling
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Traffic Analysis Report</title>
                <style>
                    body {{ 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                        margin: 0; 
                        padding: 0; 
                        background-color: #f7f9fc; 
                        color: #333; 
                    }}
                    .container {{ 
                        max-width: 1000px; 
                        margin: 0 auto; 
                        padding: 20px; 
                    }}
                    header {{ 
                        background-color: #2196F3; 
                        color: white; 
                        padding: 20px; 
                        text-align: center; 
                        margin-bottom: 30px; 
                        border-radius: 8px; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
                    }}
                    h1, h2, h3 {{ 
                        color: #1976D2; 
                        margin-top: 30px; 
                    }}
                    header h1 {{ 
                        color: white; 
                        margin: 0; 
                    }}
                    .summary-box {{ 
                        background-color: white; 
                        padding: 20px; 
                        border-radius: 8px; 
                        margin-bottom: 30px; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05); 
                        display: flex;
                        flex-wrap: wrap;
                        justify-content: space-between;
                    }}
                    .summary-item {{
                        width: 48%;
                        margin-bottom: 15px;
                    }}
                    .summary-value {{
                        font-size: 24px;
                        font-weight: bold;
                        color: #2196F3;
                        margin: 5px 0;
                    }}
                    .summary-label {{
                        font-size: 14px;
                        color: #757575;
                    }}
                    .data-section {{
                        background-color: white;
                        padding: 20px;
                        border-radius: 8px;
                        margin-bottom: 30px;
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                    }}
                    table {{ 
                        width: 100%; 
                        border-collapse: collapse; 
                        margin: 20px 0; 
                        background-color: white;
                    }}
                    th, td {{ 
                        padding: 12px 15px; 
                        text-align: left; 
                        border-bottom: 1px solid #e0e0e0; 
                    }}
                    th {{ 
                        background-color: #f5f5f5; 
                        font-weight: 600; 
                        color: #333;
                    }}
                    tr:hover {{ 
                        background-color: #f5f9ff; 
                    }}
                    .footer {{
                        text-align: center;
                        margin-top: 40px;
                        color: #757575;
                        font-size: 12px;
                    }}
                </style>
            </head>
            <body>
                <header>
                    <h1>Traffic Analysis Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </header>
                
                <div class="container">
                    <div class="summary-box">
                        <div class="summary-item">
                            <div class="summary-label">Total Vehicles</div>
                            <div class="summary-value">{total_vehicles}</div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-label">Analysis Duration</div>
                            <div class="summary-value">{mins:02d}:{secs:02d}</div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-label">Video File</div>
                            <div class="summary-value" style="font-size: 16px;">{os.path.basename(self.video_path) if self.video_path else "N/A"}</div>
                        </div>
                        <div class="summary-item">
                            <div class="summary-label">Average Vehicles Per Minute</div>
                            <div class="summary-value">{(total_vehicles / (analysis_duration / 60)):.1f}</div>
                        </div>
                    </div>
                    
                    <div class="data-section">
                        <h2>Vehicle Distribution</h2>
                        <table>
                            <tr>
                                <th>Vehicle Type</th>
                                <th>Count</th>
                                <th>Percentage</th>
                            </tr>
            """

            # Add vehicle distribution table rows
            for label, count in zip(class_labels, class_counts):
                percentage = (count / total_vehicles * 100) if total_vehicles > 0 else 0
                html_content += f"""
                            <tr>
                                <td>{label}</td>
                                <td>{count}</td>
                                <td>{percentage:.1f}%</td>
                            </tr>
                """

            html_content += """
                        </table>
                    </div>
                    
                    <div class="data-section">
                        <h2>Crossing Point Analysis</h2>
                        <table>
                            <tr>
                                <th>Counting Line</th>
                                <th>Count</th>
                                <th>Percentage</th>
                            </tr>
            """

            # Add counting line table rows
            total_line_count = sum(line_counts)
            for label, count in zip(line_labels, line_counts):
                percentage = (
                    (count / total_line_count * 100) if total_line_count > 0 else 0
                )
                html_content += f"""
                            <tr>
                                <td>{label}</td>
                                <td>{count}</td>
                                <td>{percentage:.1f}%</td>
                            </tr>
                """

            html_content += """
                        </table>
                    </div>
                    
                    <div class="data-section">
                        <h2>Direction Analysis</h2>
                        <table>
                            <tr>
                                <th>Direction</th>
                                <th>Count</th>
                                <th>Percentage</th>
                            </tr>
            """

            # Add direction table rows
            total_direction_count = sum(direction_counts)
            for label, count in zip(direction_labels, direction_counts):
                percentage = (
                    (count / total_direction_count * 100)
                    if total_direction_count > 0
                    else 0
                )
                html_content += f"""
                            <tr>
                                <td>{label.capitalize()}</td>
                                <td>{count}</td>
                                <td>{percentage:.1f}%</td>
                            </tr>
                """

            html_content += """
                        </table>
                    </div>
                    
                    <div class="data-section">
                        <h2>Speed Analysis</h2>
                        <table>
                            <tr>
                                <th>Vehicle Type</th>
                                <th>Average Speed (km/h)</th>
                            </tr>
            """

            # Add speed table rows
            for label, speed in zip(speed_labels, speed_values):
                html_content += f"""
                            <tr>
                                <td>{label}</td>
                                <td>{speed:.1f}</td>
                            </tr>
                """

            html_content += """
                        </table>
                    </div>
                    
                    <div class="data-section">
                        <h2>Conclusions</h2>
                        <p>This report provides a comprehensive analysis of traffic patterns captured during the video analysis.</p>
                        <p>The data shows traffic flow characteristics, vehicle type distribution, directional trends, and speed profiles that can be used for traffic management and planning.</p>
                    </div>
                    
                    <div class="footer">
                        <p>Generated by Traffic Analysis System</p>
                    </div>
                </div>
            </body>
            </html>
            """

            # Write HTML file
            with open(report_path, "w") as f:
                f.write(html_content)

            # Open the report in the default browser
            import webbrowser

            webbrowser.open(report_path)

            messagebox.showinfo(
                "Report Generated", f"Report generated and saved to:\n{report_path}"
            )

        except Exception as e:
            messagebox.showerror("Report Error", f"Error generating report: {str(e)}")

    # Analysis visualization methods
    def show_speed_analysis(self):
        """Show dedicated speed analysis window"""
        # Create a new toplevel window
        window = tk.Toplevel(self.root)
        window.title("Speed Analysis")
        window.geometry("800x600")
        window.configure(bg=self.bg_color)
        window.transient(self.root)

        # Create main frame
        main_frame = ttk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Speed histogram
        hist_frame = ttk.LabelFrame(main_frame, text="Speed Distribution")
        hist_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create speed histogram
        fig = plt.Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.canvas_bg)
        fig.patch.set_facecolor(self.canvas_bg)

        # Combine all speed data
        all_speeds = []
        for speeds in self.avg_speeds.values():
            all_speeds.extend(speeds)

        if all_speeds:
            # Create histogram
            bins = min(20, max(5, len(all_speeds) // 5 + 1))
            ax.hist(all_speeds, bins=bins, color=self.accent_color, alpha=0.7)
            ax.set_xlabel("Speed (km/h)", color=self.fg_color)
            ax.set_ylabel("Frequency", color=self.fg_color)
            ax.set_title("Speed Distribution", color=self.fg_color)
            ax.tick_params(colors=self.fg_color)

            # Add vertical line for average speed
            avg_speed = sum(all_speeds) / len(all_speeds)
            ax.axvline(x=avg_speed, color="red", linestyle="--", linewidth=2)
            ax.text(
                avg_speed + 0.5,
                ax.get_ylim()[1] * 0.9,
                f"Avg: {avg_speed:.1f} km/h",
                color="red",
            )
        else:
            ax.text(
                0.5,
                0.5,
                "No speed data available yet",
                ha="center",
                va="center",
                color=self.fg_color,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        canvas = FigureCanvasTkAgg(fig, master=hist_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Speed statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Speed Statistics")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        # Stats grid
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X, padx=10, pady=10)

        if all_speeds:
            # Calculate statistics
            avg_speed = sum(all_speeds) / len(all_speeds)
            median_speed = sorted(all_speeds)[len(all_speeds) // 2]
            max_speed = max(all_speeds)
            min_speed = min(all_speeds)

            # Calculate percentiles
            p85 = sorted(all_speeds)[int(len(all_speeds) * 0.85)]
            p95 = sorted(all_speeds)[int(len(all_speeds) * 0.95)]

            # Display statistics
            ttk.Label(stats_grid, text="Average Speed:").grid(
                row=0, column=0, sticky=tk.W, padx=5, pady=3
            )
            ttk.Label(stats_grid, text=f"{avg_speed:.1f} km/h").grid(
                row=0, column=1, sticky=tk.W, padx=5, pady=3
            )

            ttk.Label(stats_grid, text="Median Speed:").grid(
                row=0, column=2, sticky=tk.W, padx=5, pady=3
            )
            ttk.Label(stats_grid, text=f"{median_speed:.1f} km/h").grid(
                row=0, column=3, sticky=tk.W, padx=5, pady=3
            )

            ttk.Label(stats_grid, text="Maximum Speed:").grid(
                row=1, column=0, sticky=tk.W, padx=5, pady=3
            )
            ttk.Label(stats_grid, text=f"{max_speed:.1f} km/h").grid(
                row=1, column=1, sticky=tk.W, padx=5, pady=3
            )

            ttk.Label(stats_grid, text="Minimum Speed:").grid(
                row=1, column=2, sticky=tk.W, padx=5, pady=3
            )
            ttk.Label(stats_grid, text=f"{min_speed:.1f} km/h").grid(
                row=1, column=3, sticky=tk.W, padx=5, pady=3
            )

            ttk.Label(stats_grid, text="85th Percentile:").grid(
                row=2, column=0, sticky=tk.W, padx=5, pady=3
            )
            ttk.Label(stats_grid, text=f"{p85:.1f} km/h").grid(
                row=2, column=1, sticky=tk.W, padx=5, pady=3
            )

            ttk.Label(stats_grid, text="95th Percentile:").grid(
                row=2, column=2, sticky=tk.W, padx=5, pady=3
            )
            ttk.Label(stats_grid, text=f"{p95:.1f} km/h").grid(
                row=2, column=3, sticky=tk.W, padx=5, pady=3
            )

            ttk.Label(stats_grid, text="Total Measurements:").grid(
                row=3, column=0, sticky=tk.W, padx=5, pady=3
            )
            ttk.Label(stats_grid, text=f"{len(all_speeds)}").grid(
                row=3, column=1, sticky=tk.W, padx=5, pady=3
            )
        else:
            ttk.Label(stats_grid, text="No speed data available yet").grid(
                row=0, column=0, columnspan=4, padx=5, pady=10
            )

        # Button frame
        button_frame = ttk.Frame(window)
        button_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Button(button_frame, text="Close", command=window.destroy).pack(
            side=tk.RIGHT
        )
        ttk.Button(button_frame, text="Export Data", command=self.export_data).pack(
            side=tk.RIGHT, padx=10
        )

    def show_density_analysis(self):
        """Show dedicated density analysis window"""
        # Create a new toplevel window
        window = tk.Toplevel(self.root)
        window.title("Density Analysis")
        window.geometry("800x600")
        window.configure(bg=self.bg_color)
        window.transient(self.root)

        # Create main frame
        main_frame = ttk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Heatmap frame
        heatmap_frame = ttk.LabelFrame(main_frame, text="Traffic Density Heatmap")
        heatmap_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create heatmap visualization
        fig = plt.Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        fig.patch.set_facecolor(self.canvas_bg)

        if np.sum(self.heatmap_data) > 0:
            # Normalize heatmap
            norm_heatmap = self.heatmap_data.copy()
            norm_heatmap = (
                norm_heatmap / np.max(norm_heatmap)
                if np.max(norm_heatmap) > 0
                else norm_heatmap
            )

            # Display heatmap
            im = ax.imshow(norm_heatmap, cmap="jet", interpolation="gaussian")
            fig.colorbar(im, ax=ax)
            ax.set_title("Traffic Density Heatmap", color=self.fg_color)

            # Hide axis ticks
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.text(
                0.5,
                0.5,
                "Insufficient data for heatmap",
                ha="center",
                va="center",
                color=self.fg_color,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        canvas = FigureCanvasTkAgg(fig, master=heatmap_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Hotspot analysis frame
        hotspot_frame = ttk.LabelFrame(main_frame, text="Hotspot Analysis")
        hotspot_frame.pack(fill=tk.X, padx=10, pady=10)

        # Check if we have hotspot data
        if np.sum(self.heatmap_data) > 0:
            # Find hotspots (areas with highest density)
            flat_heatmap = self.heatmap_data.flatten()
            # Get indices of top 3 hotspots
            top_indices = np.argsort(flat_heatmap)[-3:][::-1]

            # Convert flat indices to 2D coordinates
            hotspots = []
            for idx in top_indices:
                y, x = np.unravel_index(idx, self.heatmap_data.shape)
                intensity = flat_heatmap[idx]
                if intensity > 0:  # Only include actual hotspots
                    hotspots.append((x, y, intensity))

            if hotspots:
                # Create a grid to display hotspots
                for i, (x, y, intensity) in enumerate(hotspots):
                    ttk.Label(hotspot_frame, text=f"Hotspot {i+1}:").grid(
                        row=i, column=0, sticky=tk.W, padx=5, pady=5
                    )
                    ttk.Label(hotspot_frame, text=f"X: {x}, Y: {y}").grid(
                        row=i, column=1, sticky=tk.W, padx=5, pady=5
                    )
                    ttk.Label(hotspot_frame, text=f"Intensity: {intensity:.1f}").grid(
                        row=i, column=2, sticky=tk.W, padx=5, pady=5
                    )
            else:
                ttk.Label(hotspot_frame, text="No significant hotspots detected").grid(
                    row=0, column=0, columnspan=3, padx=5, pady=10
                )
        else:
            ttk.Label(
                hotspot_frame, text="Insufficient data for hotspot analysis"
            ).grid(row=0, column=0, padx=5, pady=10)

        # Button frame
        button_frame = ttk.Frame(window)
        button_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Button(button_frame, text="Close", command=window.destroy).pack(
            side=tk.RIGHT
        )
        ttk.Button(button_frame, text="Export Heatmap", command=self.export_data).pack(
            side=tk.RIGHT, padx=10
        )

    def show_direction_analysis(self):
        """Show dedicated direction analysis window"""
        # Create a new toplevel window
        window = tk.Toplevel(self.root)
        window.title("Direction Analysis")
        window.geometry("800x600")
        window.configure(bg=self.bg_color)
        window.transient(self.root)

        # Create main frame
        main_frame = ttk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Direction pie chart frame
        pie_frame = ttk.LabelFrame(main_frame, text="Traffic Direction Distribution")
        pie_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create pie chart
        fig = plt.Figure(figsize=(7, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.canvas_bg)
        fig.patch.set_facecolor(self.canvas_bg)

        if sum(self.direction_counts.values()) > 0:
            # Filter out zero counts
            labels = []
            sizes = []
            for direction, count in self.direction_counts.items():
                if count > 0:
                    labels.append(direction.capitalize())
                    sizes.append(count)

            if sizes:
                # Use custom colors
                colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

                # Create pie chart
                ax.pie(
                    sizes,
                    labels=labels,
                    colors=colors,
                    autopct="%1.1f%%",
                    startangle=90,
                    shadow=False,
                    textprops={"color": self.fg_color},
                )
                ax.set_title("Traffic Direction Distribution", color=self.fg_color)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No direction data available yet",
                    ha="center",
                    va="center",
                    color=self.fg_color,
                )
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.text(
                0.5,
                0.5,
                "No direction data available yet",
                ha="center",
                va="center",
                color=self.fg_color,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        canvas = FigureCanvasTkAgg(fig, master=pie_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Direction statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Direction Statistics")
        stats_frame.pack(fill=tk.X, padx=10, pady=10)

        # Create a grid to display direction stats
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X, padx=10, pady=10)

        # Get direction counts
        total_count = sum(self.direction_counts.values())
        if total_count > 0:
            row = 0
            for direction, count in sorted(
                self.direction_counts.items(), key=lambda x: x[1], reverse=True
            ):
                percentage = (count / total_count * 100) if total_count > 0 else 0

                ttk.Label(stats_grid, text=f"{direction.capitalize()}:").grid(
                    row=row, column=0, sticky=tk.W, padx=5, pady=3
                )
                ttk.Label(stats_grid, text=f"{count} vehicles").grid(
                    row=row, column=1, sticky=tk.W, padx=5, pady=3
                )
                ttk.Label(stats_grid, text=f"{percentage:.1f}%").grid(
                    row=row, column=2, sticky=tk.W, padx=5, pady=3
                )

                # Simple progress bar visualization
                bar_frame = ttk.Frame(stats_grid)
                bar_frame.grid(row=row, column=3, sticky=tk.W, padx=5, pady=3)

                bar_width = int(percentage * 2)  # 2 pixels per percent
                bar = tk.Canvas(
                    bar_frame,
                    width=200,
                    height=15,
                    bg=self.bg_color,
                    highlightthickness=0,
                )
                bar.pack(side=tk.LEFT)
                bar.create_rectangle(
                    0, 0, bar_width, 15, fill=self.accent_color, outline=""
                )

                row += 1
        else:
            ttk.Label(stats_grid, text="No direction data available yet").grid(
                row=0, column=0, columnspan=4, padx=5, pady=10
            )

        # Button frame
        button_frame = ttk.Frame(window)
        button_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Button(button_frame, text="Close", command=window.destroy).pack(
            side=tk.RIGHT
        )
        ttk.Button(button_frame, text="Export Data", command=self.export_data).pack(
            side=tk.RIGHT, padx=10
        )

    def show_time_analysis(self):
        """Show dedicated time series analysis window"""
        # Create a new toplevel window
        window = tk.Toplevel(self.root)
        window.title("Time Series Analysis")
        window.geometry("800x600")
        window.configure(bg=self.bg_color)
        window.transient(self.root)

        # Create main frame
        main_frame = ttk.Frame(window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Time series frame
        time_frame = ttk.LabelFrame(main_frame, text="Traffic Volume Over Time")
        time_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Create time series plot
        fig = plt.Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.canvas_bg)
        fig.patch.set_facecolor(self.canvas_bg)

        if self.timestamps and len(self.timestamps) > 1:
            # Plot individual vehicle classes
            for class_id in self.vehicle_classes:
                if class_id in self.class_names:
                    class_name = self.class_names[class_id]
                    if (
                        class_name in self.time_series_data
                        and len(self.time_series_data[class_name]) > 1
                    ):
                        # Make sure data lengths match
                        data_len = min(
                            len(self.timestamps), len(self.time_series_data[class_name])
                        )
                        ax.plot(
                            self.timestamps[:data_len],
                            self.time_series_data[class_name][:data_len],
                            label=class_name,
                        )

            ax.set_xlabel("Time (s)", color=self.fg_color)
            ax.set_ylabel("Count", color=self.fg_color)
            ax.set_title("Traffic Volume Over Time", color=self.fg_color)
            ax.tick_params(colors=self.fg_color)

            if any(
                len(self.time_series_data.get(self.class_names.get(class_id, ""), []))
                > 0
                for class_id in self.vehicle_classes
                if class_id in self.class_names
            ):
                ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No time series data available yet",
                ha="center",
                va="center",
                color=self.fg_color,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        canvas = FigureCanvasTkAgg(fig, master=time_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Peak times frame
        peak_frame = ttk.LabelFrame(main_frame, text="Peak Traffic Times")
        peak_frame.pack(fill=tk.X, padx=10, pady=10)

        # Get peak times if we have time series data
        if self.timestamps and len(self.timestamps) > 1:
            # Calculate combined vehicle count over time
            combined_data = []
            valid_series = [data for data in self.time_series_data.values() if data]

            if valid_series:
                min_length = min(
                    len(self.timestamps), min(len(data) for data in valid_series)
                )

                if min_length > 0:
                    for i in range(min_length):
                        total = sum(
                            data[i]
                            for data in self.time_series_data.values()
                            if i < len(data)
                        )
                        combined_data.append((self.timestamps[i], total))

                    # Sort by count to find peak times
                    sorted_data = sorted(
                        combined_data, key=lambda x: x[1], reverse=True
                    )

                    # Display top 5 peak times
                    for i, (timestamp, count) in enumerate(sorted_data[:5]):
                        mins = int(timestamp // 60)
                        secs = int(timestamp % 60)

                        frame = ttk.Frame(peak_frame)
                        frame.pack(fill=tk.X, padx=10, pady=3)

                        ttk.Label(frame, text=f"Peak {i+1}:").pack(side=tk.LEFT, padx=5)
                        ttk.Label(frame, text=f"Time {mins:02d}:{secs:02d}").pack(
                            side=tk.LEFT, padx=5
                        )
                        ttk.Label(frame, text=f"Count: {count} vehicles").pack(
                            side=tk.LEFT, padx=5
                        )

                        # Simple visual indicator of peak intensity
                        max_count = sorted_data[0][1]
                        bar_width = int(
                            (count / max_count) * 200
                        )  # Scale to max 200 pixels

                        bar = tk.Canvas(
                            frame,
                            width=200,
                            height=15,
                            bg=self.bg_color,
                            highlightthickness=0,
                        )
                        bar.pack(side=tk.LEFT, padx=5)
                        bar.create_rectangle(
                            0, 0, bar_width, 15, fill=self.accent_color, outline=""
                        )
                else:
                    ttk.Label(
                        peak_frame, text="Insufficient data for peak analysis"
                    ).pack(padx=10, pady=10)
            else:
                ttk.Label(peak_frame, text="Insufficient data for peak analysis").pack(
                    padx=10, pady=10
                )
        else:
            ttk.Label(peak_frame, text="No time series data available yet").pack(
                padx=10, pady=10
            )

        # Button frame
        button_frame = ttk.Frame(window)
        button_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Button(button_frame, text="Close", command=window.destroy).pack(
            side=tk.RIGHT
        )
        ttk.Button(button_frame, text="Export Data", command=self.export_data).pack(
            side=tk.RIGHT, padx=10
        )

    def show_dashboard(self):
        """Show a comprehensive analytics dashboard"""
        # Create a new toplevel window
        dashboard = tk.Toplevel(self.root)
        dashboard.title("Traffic Analytics Dashboard")
        dashboard.geometry("1000x700")
        dashboard.configure(bg=self.bg_color)
        dashboard.transient(self.root)

        # Add title
        title_label = tk.Label(
            dashboard,
            text="Real-time Traffic Analytics Dashboard",
            font=("Arial", 16, "bold"),
            bg=self.bg_color,
            fg=self.fg_color,
        )
        title_label.pack(pady=10)

        # Create a frame for the dashboard content
        content_frame = ttk.Frame(dashboard)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create a notebook for different dashboard tabs
        notebook = ttk.Notebook(content_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create tabs
        overview_tab = ttk.Frame(notebook)
        vehicles_tab = ttk.Frame(notebook)
        temporal_tab = ttk.Frame(notebook)
        spatial_tab = ttk.Frame(notebook)
        speed_tab = ttk.Frame(notebook)

        notebook.add(overview_tab, text="Overview")
        notebook.add(vehicles_tab, text="Vehicles")
        notebook.add(temporal_tab, text="Temporal")
        notebook.add(spatial_tab, text="Spatial")
        notebook.add(speed_tab, text="Speed")

        # Set up each tab with charts and statistics
        self.setup_dashboard_overview(overview_tab)
        self.setup_dashboard_vehicles(vehicles_tab)
        self.setup_dashboard_temporal(temporal_tab)
        self.setup_dashboard_spatial(spatial_tab)
        self.setup_dashboard_speed(speed_tab)

        # Add export buttons at the bottom
        button_frame = ttk.Frame(dashboard)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        ttk.Button(button_frame, text="Export All Data", command=self.export_data).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Button(
            button_frame, text="Generate Report", command=self.generate_report
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            button_frame, text="Close Dashboard", command=dashboard.destroy
        ).pack(side=tk.RIGHT, padx=5)

    def setup_dashboard_overview(self, parent):
        """Set up the overview dashboard tab"""
        # Create grid layout
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Configure grid
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        # Create summary statistics box
        stats_frame = ttk.LabelFrame(frame, text="Summary Statistics")
        stats_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Total vehicles
        ttk.Label(stats_frame, text="Total Vehicles:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=2
        )
        total_var = tk.StringVar(value=str(self.total_vehicle_count))
        ttk.Label(stats_frame, textvariable=total_var, font=("Arial", 14, "bold")).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2
        )

        # Analysis duration
        duration = time.time() - self.start_time if hasattr(self, "start_time") else 0
        mins = int(duration // 60)
        secs = int(duration % 60)
        ttk.Label(stats_frame, text="Analysis Duration:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=2
        )
        duration_var = tk.StringVar(value=f"{mins:02d}:{secs:02d}")
        ttk.Label(stats_frame, textvariable=duration_var).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=2
        )

        # Average density
        ttk.Label(stats_frame, text="Average Density:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=2
        )
        density_var = tk.StringVar(
            value=(
                f"{self.total_vehicle_count / (duration / 60):.1f} vehicles/minute"
                if duration > 0
                else "0"
            )
        )
        ttk.Label(stats_frame, textvariable=density_var).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=2
        )

        # Create count over time plot
        count_frame = ttk.LabelFrame(frame, text="Vehicle Count Over Time")
        count_frame.grid(row=0, column=1, rowspan=1, padx=5, pady=5, sticky="nsew")

        # Set up the plot
        fig = plt.Figure(figsize=(5, 3), dpi=100)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.canvas_bg)
        fig.patch.set_facecolor(self.canvas_bg)

        # Plot time series data if available
        if self.timestamps and len(self.timestamps) > 1:
            # Combine all vehicle counts
            combined_data = []
            valid_series = [data for data in self.time_series_data.values() if data]

            if valid_series:
                min_length = min(
                    len(self.timestamps), min(len(data) for data in valid_series)
                )

                for i in range(min_length):
                    total = sum(
                        data[i]
                        for data in self.time_series_data.values()
                        if i < len(data)
                    )
                    combined_data.append(total)

                ax.plot(
                    self.timestamps[:min_length], combined_data, color=self.accent_color
                )
                ax.set_xlabel("Time (s)", color=self.fg_color)
                ax.set_ylabel("Count", color=self.fg_color)
                ax.set_title("Total Vehicles Over Time", color=self.fg_color)
                ax.tick_params(colors=self.fg_color)
        else:
            ax.text(
                0.5,
                0.5,
                "No time series data available yet",
                ha="center",
                va="center",
                color=self.fg_color,
            )
            ax.set_xticks([])
            ax.set_yticks([])

        count_canvas = FigureCanvasTkAgg(fig, master=count_frame)
        count_canvas.draw()
        count_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create vehicle distribution chart
        dist_frame = ttk.LabelFrame(frame, text="Vehicle Distribution")
        dist_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Set up the plot
        fig2 = plt.Figure(figsize=(5, 3), dpi=100)
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor(self.canvas_bg)
        fig2.patch.set_facecolor(self.canvas_bg)

        # Plot vehicle distribution if available
        vehicle_counts = {}
        for class_id in self.vehicle_classes:
            if class_id in self.class_names:
                class_name = self.class_names[class_id]
                count = self.detected_objects.get(class_name, 0)
                if count > 0:
                    vehicle_counts[class_name] = count

        if vehicle_counts:
            labels = list(vehicle_counts.keys())
            sizes = list(vehicle_counts.values())

            # Use custom colors
            colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

            ax2.pie(
                sizes,
                labels=labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"color": self.fg_color},
            )
            ax2.set_title("Vehicle Type Distribution", color=self.fg_color)
        else:
            ax2.text(
                0.5,
                0.5,
                "No vehicle data available yet",
                ha="center",
                va="center",
                color=self.fg_color,
            )
            ax2.set_xticks([])
            ax2.set_yticks([])

        dist_canvas = FigureCanvasTkAgg(fig2, master=dist_frame)
        dist_canvas.draw()
        dist_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Create direction distribution chart
        dir_frame = ttk.LabelFrame(frame, text="Direction Distribution")
        dir_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # Set up the plot
        fig3 = plt.Figure(figsize=(5, 3), dpi=100)
        ax3 = fig3.add_subplot(111)
        ax3.set_facecolor(self.canvas_bg)
        fig3.patch.set_facecolor(self.canvas_bg)

        # Plot direction data if available
        if sum(self.direction_counts.values()) > 0:
            labels = []
            sizes = []
            for direction, count in self.direction_counts.items():
                if count > 0:
                    labels.append(direction.capitalize())
                    sizes.append(count)

            if sizes:
                # Use custom colors
                colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

                ax3.pie(
                    sizes,
                    labels=labels,
                    colors=colors,
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"color": self.fg_color},
                )
                ax3.set_title("Traffic Direction", color=self.fg_color)
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "No direction data available yet",
                    ha="center",
                    va="center",
                    color=self.fg_color,
                )
                ax3.set_xticks([])
                ax3.set_yticks([])
        else:
            ax3.text(
                0.5,
                0.5,
                "No direction data available yet",
                ha="center",
                va="center",
                color=self.fg_color,
            )
            ax3.set_xticks([])
            ax3.set_yticks([])

        dir_canvas = FigureCanvasTkAgg(fig3, master=dir_frame)
        dir_canvas.draw()
        dir_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Function to update the dashboard
        def update_overview():
            # Update statistics
            total_var.set(str(self.total_vehicle_count))

            duration = (
                time.time() - self.start_time if hasattr(self, "start_time") else 0
            )
            mins = int(duration // 60)
            secs = int(duration % 60)
            duration_var.set(f"{mins:02d}:{secs:02d}")

            density = self.total_vehicle_count / (duration / 60) if duration > 0 else 0
            density_var.set(f"{density:.1f} vehicles/minute")

            # Update count plot
            ax.clear()
            ax.set_facecolor(self.canvas_bg)

            if self.timestamps and len(self.timestamps) > 1:
                # Combine all vehicle counts
                combined_data = []
                valid_series = [data for data in self.time_series_data.values() if data]

                if valid_series:
                    min_length = min(
                        len(self.timestamps), min(len(data) for data in valid_series)
                    )

                    for i in range(min_length):
                        total = sum(
                            data[i]
                            for data in self.time_series_data.values()
                            if i < len(data)
                        )
                        combined_data.append(total)

                    ax.plot(
                        self.timestamps[:min_length],
                        combined_data,
                        color=self.accent_color,
                    )
                    ax.set_xlabel("Time (s)", color=self.fg_color)
                    ax.set_ylabel("Count", color=self.fg_color)
                    ax.set_title("Total Vehicles Over Time", color=self.fg_color)
                    ax.tick_params(colors=self.fg_color)
            else:
                ax.text(
                    0.5,
                    0.5,
                    "No time series data available yet",
                    ha="center",
                    va="center",
                    color=self.fg_color,
                )
                ax.set_xticks([])
                ax.set_yticks([])

            count_canvas.draw()

            # Update vehicle distribution
            ax2.clear()
            ax2.set_facecolor(self.canvas_bg)

            vehicle_counts = {}
            for class_id in self.vehicle_classes:
                if class_id in self.class_names:
                    class_name = self.class_names[class_id]
                    count = self.detected_objects.get(class_name, 0)
                    if count > 0:
                        vehicle_counts[class_name] = count

            if vehicle_counts:
                labels = list(vehicle_counts.keys())
                sizes = list(vehicle_counts.values())

                colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

                ax2.pie(
                    sizes,
                    labels=labels,
                    colors=colors,
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"color": self.fg_color},
                )
                ax2.set_title("Vehicle Type Distribution", color=self.fg_color)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No vehicle data available yet",
                    ha="center",
                    va="center",
                    color=self.fg_color,
                )
                ax2.set_xticks([])
                ax2.set_yticks([])

            dist_canvas.draw()

            # Update direction distribution
            ax3.clear()
            ax3.set_facecolor(self.canvas_bg)

            if sum(self.direction_counts.values()) > 0:
                labels = []
                sizes = []
                for direction, count in self.direction_counts.items():
                    if count > 0:
                        labels.append(direction.capitalize())
                        sizes.append(count)

                if sizes:
                    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]

                    ax3.pie(
                        sizes,
                        labels=labels,
                        colors=colors,
                        autopct="%1.1f%%",
                        startangle=90,
                        textprops={"color": self.fg_color},
                    )
                    ax3.set_title("Traffic Direction", color=self.fg_color)
                else:
                    ax3.text(
                        0.5,
                        0.5,
                        "No direction data available yet",
                        ha="center",
                        va="center",
                        color=self.fg_color,
                    )
                    ax3.set_xticks([])
                    ax3.set_yticks([])
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "No direction data available yet",
                    ha="center",
                    va="center",
                    color=self.fg_color,
                )
                ax3.set_xticks([])
                ax3.set_yticks([])

            dir_canvas.draw()

            # Schedule next update
            parent.after(1000, update_overview)

        # Start the update loop
        parent.after(1000, update_overview)

    def setup_dashboard_vehicles(self, parent):
        """Set up the vehicles dashboard tab"""
        # Implementation details would go here
        # For brevity, I'll provide a placeholder implementation
        ttk.Label(parent, text="Detailed vehicle analysis will be shown here").pack(
            pady=20
        )

    def setup_dashboard_temporal(self, parent):
        """Set up the temporal analysis dashboard tab"""
        # Implementation details would go here
        # For brevity, I'll provide a placeholder implementation
        ttk.Label(parent, text="Temporal analysis will be shown here").pack(pady=20)

    def setup_dashboard_spatial(self, parent):
        """Set up the spatial analysis dashboard tab"""
        # Implementation details would go here
        # For brevity, I'll provide a placeholder implementation
        ttk.Label(parent, text="Spatial analysis will be shown here").pack(pady=20)

    def setup_dashboard_speed(self, parent):
        """Set up the speed analysis dashboard tab"""
        # Implementation details would go here
        # For brevity, I'll provide a placeholder implementation
        ttk.Label(parent, text="Speed analysis will be shown here").pack(pady=20)

    # Help and info methods
    def show_instructions(self):
        """Show application instructions"""
        instructions = """
Traffic Analysis Application Instructions

Getting Started:
1. Load a YOLO model using the "Load Model" button
2. Open a video file for analysis
3. Press "Play" to start the analysis

Analysis Features:
- Detect and track vehicles, people, or custom object classes
- Count objects as they cross counting lines
- Monitor regions of interest for activity
- Analyze traffic patterns, speeds, and directions
- Generate heatmaps of traffic density

Creating Analysis Zones:
- Add counting lines to track vehicles crossing specific boundaries
   - Click "Add Counting Line" and drag on the video
- Add regions of interest (ROIs) to monitor activity in specific areas
   - Click "Add Region" and click points to define a polygon

Analysis Tools:
- View real-time analytics in the visualization tabs
- Access detailed analysis through specialized windows
- Export data and generate reports

Tips for Better Performance:
- Use the smallest suitable YOLO model (nano is fastest)
- Increase frame skip on slower systems
- Reduce detection threshold for better accuracy
- Downscale video for faster processing

Keyboard Shortcuts:
- ESC - Cancel drawing operation
"""

        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Instructions")
        dialog.geometry("600x500")
        dialog.configure(bg=self.bg_color)
        dialog.transient(self.root)
        dialog.grab_set()

        # Create text widget
        text = tk.Text(
            dialog, wrap=tk.WORD, bg=self.bg_color, fg=self.fg_color, padx=15, pady=15
        )
        text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Insert instructions
        text.insert(tk.END, instructions)
        text.config(state=tk.DISABLED)  # Make read-only

        # Add scrollbar
        scrollbar = ttk.Scrollbar(text, command=text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        text.config(yscrollcommand=scrollbar.set)

        # Add close button
        close_btn = ttk.Button(dialog, text="Close", command=dialog.destroy)
        close_btn.pack(pady=10)

    def show_about(self):
        """Show about dialog"""
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("About Traffic Analysis")
        dialog.geometry("400x400")
        dialog.configure(bg=self.bg_color)
        dialog.transient(self.root)
        dialog.grab_set()

        # Create content frame
        content_frame = ttk.Frame(dialog)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # App title
        ttk.Label(
            content_frame, text="Traffic Analysis System", font=("Arial", 16, "bold")
        ).pack(pady=10)

        # Version
        ttk.Label(content_frame, text="Version 1.0", font=("Arial", 10)).pack()

        # Separator
        ttk.Separator(content_frame, orient="horizontal").pack(fill="x", pady=15)

        # Description
        description = """
A comprehensive traffic monitoring and analysis application
using YOLOv8 for real-time vehicle detection and tracking.

This application provides tools for traffic flow analysis,
speed estimation, direction tracking, and density visualization
to help understand traffic patterns from video sources.

Features:
- Real-time vehicle detection and classification
- Object tracking across video frames
- Traffic counting with virtual lines
- Speed and direction estimation
- Heatmap visualization of traffic density
- Comprehensive data export and reporting
"""
        text = tk.Text(
            content_frame,
            wrap=tk.WORD,
            height=12,
            bg=self.bg_color,
            fg=self.fg_color,
            bd=0,
            padx=0,
            pady=0,
        )
        text.pack(fill=tk.X, pady=10)
        text.insert(tk.END, description)
        text.config(state=tk.DISABLED)  # Make read-only

        # Technologies used
        ttk.Label(content_frame, text="Technologies:", font=("Arial", 10, "bold")).pack(
            anchor=tk.W
        )

        ttk.Label(
            content_frame, text="Python, OpenCV, YOLO, TensorFlow, Tkinter, Matplotlib"
        ).pack(anchor=tk.W)

        # Close button
        ttk.Button(dialog, text="Close", command=dialog.destroy).pack(pady=15)

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


# Main function to run the application
def main():
    root = tk.Tk()

    # Try to use a modern theme if available
    try:
        style = ttk.Style()
        available_themes = style.theme_names()
        if "clam" in available_themes:
            style.theme_use("clam")
        elif "alt" in available_themes:
            style.theme_use("alt")
    except:
        pass  # Use default theme if custom themes fail

    app = TrafficAnalysisApp(root)

    # Set up closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start the main loop
    root.mainloop()


if __name__ == "__main__":
    main()
