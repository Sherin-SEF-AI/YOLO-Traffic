import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk, colorchooser, messagebox, Menu, simpledialog
from PIL import Image, ImageTk, ImageDraw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.patches as patches
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
import seaborn as sns
import math
from tqdm import tqdm


class EnhancedTrafficAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Traffic Analysis System")
        # Setting a larger default window size and minimum size to ensure enough space
        self.root.geometry("1280x900")
        self.root.minsize(1000, 700)

        # Set application icon if available
        try:
            self.root.iconbitmap("traffic_icon.ico")
        except:
            pass  # Icon not found, continue without it

        # Enable dark mode by default for better visibility
        self.dark_mode = True
        self.configure_theme()

        # YOLO model and settings
        self.model = None
        self.model_loaded = False
        self.class_names = {}
        self.detection_threshold = 0.4  # Lower default for better detection

        # CPU optimization options
        self.use_tiny_model = True
        self.optimize_for_cpu = True
        self.use_tensorrt = False

        # Define vehicle classes early to avoid attribute errors
        self.vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.person_class = 0  # Person class ID

        # Video processing variables
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.processing_thread = None
        self.update_id = None
        self.video_fps = 30  # Default assumption
        self.video_resolution = (640, 480)  # Default

        # Analysis results
        self.detected_objects = defaultdict(int)
        self.time_series_data = defaultdict(list)
        self.timestamps = []
        self.frame_count = 0
        self.start_time = 0

        # Traffic data collection
        self.hourly_data = defaultdict(int)
        self.daily_data = defaultdict(int)

        # History for tracking
        self.object_history = {}
        self.history_window = 30
        self.next_object_id = 0
        self.object_speeds = {}  # For speed estimation
        self.object_direction_counts = defaultdict(int)  # For direction analysis

        # Zones and lines
        self.counting_lines = []  # Multiple counting lines
        self.counting_line_names = []  # Names for each line
        self.counting_line_colors = []  # Colors for each line
        self.regions_of_interest = []  # ROIs for specific analysis
        self.roi_names = []  # Names for each ROI
        self.roi_counts = defaultdict(int)  # Counts for each ROI
        self.heatmap_data = np.zeros((480, 640), dtype=np.float32)  # Default size

        # For vehicle counting
        self.line_counts = defaultdict(int)  # Counts for each line
        self.direction_counts = {"north": 0, "south": 0, "east": 0, "west": 0}
        self.total_vehicle_count = 0

        # Drawing mode
        self.drawing_mode = None
        self.drawing_start = None
        self.temp_line = None
        self.temp_roi = []

        # CPU optimization settings
        self.frame_skip = 2
        self.current_skip_count = 0
        self.downscale_factor = 0.7  # More aggressive downscaling
        self.max_processing_time = 50  # Max milliseconds to spend on a frame

        # Visualization settings
        self.show_tracking = True
        self.show_bounding_boxes = True
        self.show_labels = True
        self.show_count_info = True
        self.show_heatmap = False
        self.show_direction_arrows = True

        # Advanced settings
        self.apply_non_max_suppression = True
        self.nms_threshold = 0.4
        self.minimum_object_size = 20  # Minimum object width/height in pixels
        self.max_tracked_objects = 50  # Limit tracked objects for better performance

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

        # Statistics
        self.avg_speeds = defaultdict(list)
        self.peak_times = []
        self.traffic_density = []

        # Detection mode
        self.detection_mode = "vehicles"  # "vehicles", "people", "all", "custom"
        self.custom_classes = []

        # Vehicle class colors
        self.class_colors = {
            2: (0, 255, 0),  # Car: Green
            3: (0, 0, 255),  # Motorcycle: Blue
            5: (255, 0, 0),  # Bus: Red
            7: (255, 255, 0),  # Truck: Yellow
            0: (255, 0, 255),  # Person: Magenta
        }

        # Setup UI
        self.setup_ui()

        # Initialize plotting
        self.setup_plots()

        # Debug flag
        self.debug = True

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
            # Update the menu label
            for index, item in enumerate(self.menu_bar.winfo_children()):
                if isinstance(item, Menu) and "Settings" in item.entrycget(0, "label"):
                    for i in range(item.index("end") + 1):
                        if "Performance" in item.entrycget(i, "label"):
                            performance_menu = item.nametowidget(
                                item.entrycget(i, "menu")
                            )
                            for j in range(performance_menu.index("end") + 1):
                                if "Frame Skip" in performance_menu.entrycget(
                                    j, "label"
                                ):
                                    performance_menu.entryconfig(
                                        j, label=f"Frame Skip: {self.frame_skip}"
                                    )
                                    break
                            break
                    break

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
            # Update the menu label
            for index, item in enumerate(self.menu_bar.winfo_children()):
                if isinstance(item, Menu) and "Settings" in item.entrycget(0, "label"):
                    for i in range(item.index("end") + 1):
                        if "Performance" in item.entrycget(i, "label"):
                            performance_menu = item.nametowidget(
                                item.entrycget(i, "menu")
                            )
                            for j in range(performance_menu.index("end") + 1):
                                if "Downscale Factor" in performance_menu.entrycget(
                                    j, "label"
                                ):
                                    performance_menu.entryconfig(
                                        j,
                                        label=f"Downscale Factor: {self.downscale_factor}",
                                    )
                                    break
                            break
                    break

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

    def reset_settings(self):
        """Reset all settings to default values"""
        # Confirm with user
        if not messagebox.askyesno(
            "Reset Settings", "Are you sure you want to reset all settings to defaults?"
        ):
            return

        # Reset detection settings
        self.detection_threshold = 0.4
        self.threshold_var.set(self.detection_threshold)
        self.threshold_label.config(text=f"{self.detection_threshold:.2f}")

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
        self.model_size_var.set("nano")

        # Reset CPU optimization
        self.optimize_for_cpu = True
        self.optimize_cpu_var.set(True)

        # Update UI
        self.status_var.set("All settings reset to defaults")

        # Redraw current frame if available
        if hasattr(self, "last_frame") and self.last_frame is not None:
            self.show_frame(self.last_frame)

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

    def reset_settings(self):
        """Reset all settings to default values"""
        # Confirm with user
        if not messagebox.askyesno(
            "Reset Settings", "Are you sure you want to reset all settings to defaults?"
        ):
            return

        # Reset detection settings
        self.detection_threshold = 0.4
        self.threshold_var.set(self.detection_threshold)
        self.threshold_label.config(text=f"{self.detection_threshold:.2f}")

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
        self.model_size_var.set("nano")

        # Reset CPU optimization
        self.optimize_for_cpu = True
        self.optimize_cpu_var.set(True)

        # Update UI
        self.status_var.set("All settings reset to defaults")

        # Redraw current frame if available
        if hasattr(self, "last_frame") and self.last_frame is not None:
            self.show_frame(self.last_frame)

    def show_speed_analysis(self):
        """Show dedicated speed analysis window"""
        # Create a simple window with speed analysis
        window = tk.Toplevel(self.root)
        window.title("Speed Analysis")
        window.geometry("800x600")

        # Add visualization
        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)

        # Check if we have speed data
        all_speeds = []
        for speeds in self.avg_speeds.values():
            all_speeds.extend(speeds)

        if all_speeds:
            # Create plots
            ax.hist(all_speeds, bins=10, color=self.accent_color)
            ax.set_xlabel("Speed (km/h)")
            ax.set_ylabel("Frequency")
            ax.set_title("Speed Distribution")
        else:
            ax.text(0.5, 0.5, "No speed data available yet", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add close button
        tk.Button(window, text="Close", command=window.destroy).pack(pady=10)

    def show_density_analysis(self):
        """Show dedicated density analysis window"""
        # Create a simple window with density analysis
        window = tk.Toplevel(self.root)
        window.title("Density Analysis")
        window.geometry("800x600")

        # Add visualization
        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)

        if np.sum(self.heatmap_data) > 0:
            # Normalize heatmap
            norm_heatmap = self.heatmap_data.copy()
            norm_heatmap = (
                norm_heatmap / np.max(norm_heatmap)
                if np.max(norm_heatmap) > 0
                else norm_heatmap
            )

            # Create heatmap
            im = ax.imshow(norm_heatmap, cmap="jet", interpolation="gaussian")
            fig.colorbar(im, ax=ax)
            ax.set_title("Traffic Density Heatmap")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.text(0.5, 0.5, "No density data available yet", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add close button
        tk.Button(window, text="Close", command=window.destroy).pack(pady=10)

    def show_direction_analysis(self):
        """Show dedicated direction analysis window"""
        # Create a simple window with direction analysis
        window = tk.Toplevel(self.root)
        window.title("Direction Analysis")
        window.geometry("800x600")

        # Add visualization
        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)

        if sum(self.direction_counts.values()) > 0:
            labels = list(self.direction_counts.keys())
            sizes = list(self.direction_counts.values())

            # Filter out zero values
            non_zero_labels = []
            non_zero_sizes = []
            for label, size in zip(labels, sizes):
                if size > 0:
                    non_zero_labels.append(label.capitalize())
                    non_zero_sizes.append(size)

            if non_zero_sizes:
                colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
                ax.pie(
                    non_zero_sizes,
                    labels=non_zero_labels,
                    colors=colors,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax.set_title("Traffic Direction Distribution")
            else:
                ax.text(
                    0.5, 0.5, "No direction data available", ha="center", va="center"
                )
                ax.set_xticks([])
                ax.set_yticks([])
        else:
            ax.text(
                0.5, 0.5, "No direction data available yet", ha="center", va="center"
            )
            ax.set_xticks([])
            ax.set_yticks([])

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add close button
        tk.Button(window, text="Close", command=window.destroy).pack(pady=10)

    def show_time_analysis(self):
        """Show dedicated time series analysis window"""
        # Create a simple window with time analysis
        window = tk.Toplevel(self.root)
        window.title("Time Series Analysis")
        window.geometry("800x600")

        # Add visualization
        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)

        if self.timestamps and len(self.timestamps) > 1:
            # Plot data for each vehicle class
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

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Count")
            ax.set_title("Traffic Volume Over Time")
            ax.legend()
        else:
            ax.text(
                0.5, 0.5, "No time series data available yet", ha="center", va="center"
            )
            ax.set_xticks([])
            ax.set_yticks([])

        canvas = FigureCanvasTkAgg(fig, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add close button
        tk.Button(window, text="Close", command=window.destroy).pack(pady=10)

    def show_dashboard(self):
        """Simple stub for dashboard"""
        messagebox.showinfo(
            "Dashboard",
            "Full analytics dashboard functionality will be implemented in the next version.",
        )

    def export_data(self):
        """Stub for export data function"""
        if not hasattr(self, "data_log") or not self.data_log:
            messagebox.showinfo("No Data", "No traffic data has been collected yet.")
            return

        try:
            # Create a placeholder for actual implementation
            messagebox.showinfo(
                "Export", "Data export functionality will be implemented soon."
            )
        except Exception as e:
            messagebox.showerror("Export Error", f"Error: {str(e)}")

    def generate_report(self):
        """Stub for generate report function"""
        if not hasattr(self, "data_log") or not self.data_log:
            messagebox.showinfo(
                "No Data", "No traffic data available for report generation."
            )
            return

        try:
            # Create a placeholder for actual implementation
            messagebox.showinfo(
                "Report", "Report generation functionality will be implemented soon."
            )
        except Exception as e:
            messagebox.showerror("Report Error", f"Error: {str(e)}")

    def select_custom_classes(self):
        """Placeholder for custom class selection"""
        messagebox.showinfo(
            "Custom Classes", "Custom class selection will be implemented soon."
        )

    def configure_theme(self):
        # Configure application theme based on dark/light mode
        if self.dark_mode:
            self.bg_color = "#1e1e1e"
            self.fg_color = "#ffffff"
            self.accent_color = "#0078d7"
            self.canvas_bg = "black"
            self.plot_bg = "#2d2d2d"
            self.plot_fg = "white"
        else:
            self.bg_color = "#f0f0f0"
            self.fg_color = "#000000"
            self.accent_color = "#0078d7"
            self.canvas_bg = "white"
            self.plot_bg = "#f8f8f8"
            self.plot_fg = "black"

        # Apply theme to root
        self.root.configure(bg=self.bg_color)

        # Configure ttk styles
        style = ttk.Style()
        style.configure("TFrame", background=self.bg_color)
        style.configure("TLabel", background=self.bg_color, foreground=self.fg_color)
        style.configure("TButton", background=self.accent_color)
        style.configure(
            "TCheckbutton", background=self.bg_color, foreground=self.fg_color
        )
        style.configure("TScale", background=self.bg_color)

    def toggle_theme(self):
        # Toggle between dark and light mode
        self.dark_mode = not self.dark_mode
        self.configure_theme()

        # Update matplotlib plots
        self.update_plots()

    def setup_ui(self):
        # Set up menu bar
        self.menu_bar = Menu(self.root)
        self.root.config(menu=self.menu_bar)

        # File menu
        file_menu = Menu(self.menu_bar, tearoff=0)
        file_menu.add_command(label="Load Model", command=self.load_model)
        file_menu.add_command(label="Select Video", command=self.select_video)
        file_menu.add_separator()
        file_menu.add_command(label="Export Data", command=self.export_data)
        file_menu.add_command(label="Generate Report", command=self.generate_report)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        self.menu_bar.add_cascade(label="File", menu=file_menu)

        # View menu
        view_menu = Menu(self.menu_bar, tearoff=0)
        view_menu.add_checkbutton(
            label="Dark Mode",
            command=self.toggle_theme,
            variable=tk.BooleanVar(value=self.dark_mode),
        )
        view_menu.add_separator()

        # Create boolean variables for checkbuttons
        self.show_tracking_var = tk.BooleanVar(value=self.show_tracking)
        self.show_boxes_var = tk.BooleanVar(value=self.show_bounding_boxes)
        self.show_labels_var = tk.BooleanVar(value=self.show_labels)
        self.show_counts_var = tk.BooleanVar(value=self.show_count_info)
        self.show_heatmap_var = tk.BooleanVar(value=self.show_heatmap)
        self.show_arrows_var = tk.BooleanVar(value=self.show_direction_arrows)

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
        view_menu.add_checkbutton(
            label="Show Heatmap",
            variable=self.show_heatmap_var,
            command=self.update_display_settings,
        )
        view_menu.add_checkbutton(
            label="Show Direction Arrows",
            variable=self.show_arrows_var,
            command=self.update_display_settings,
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
            label="Full Analytics Dashboard", command=self.show_dashboard
        )
        self.menu_bar.add_cascade(label="Analysis", menu=analysis_menu)

        # Settings menu
        settings_menu = Menu(self.menu_bar, tearoff=0)

        # Model submenu
        model_menu = Menu(settings_menu, tearoff=0)
        self.model_size_var = tk.StringVar(value="nano")
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

        # CPU optimization menu
        cpu_menu = Menu(settings_menu, tearoff=0)
        self.optimize_cpu_var = tk.BooleanVar(value=self.optimize_for_cpu)
        cpu_menu.add_checkbutton(
            label="CPU Optimization", variable=self.optimize_cpu_var
        )

        self.frame_skip_var = tk.IntVar(value=self.frame_skip)
        cpu_menu.add_command(
            label=f"Frame Skip: {self.frame_skip}", command=self.set_frame_skip
        )

        self.downscale_var = tk.DoubleVar(value=self.downscale_factor)
        cpu_menu.add_command(
            label=f"Downscale Factor: {self.downscale_factor}",
            command=self.set_downscale,
        )

        settings_menu.add_cascade(label="Performance", menu=cpu_menu)

        # Detection settings
        self.detection_mode_var = tk.StringVar(value=self.detection_mode)
        detection_menu = Menu(settings_menu, tearoff=0)
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

        # Main layout frames
        self.control_frame = ttk.Frame(self.root, padding=10)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        # Create the content frame - this will hold both the video and viz panels
        self.content_frame = ttk.Frame(self.root)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Configure grid columns with weights for proper space distribution
        self.content_frame.columnconfigure(0, weight=3)  # Video gets 60% of width
        self.content_frame.columnconfigure(1, weight=2)  # Viz gets 40% of width
        self.content_frame.rowconfigure(0, weight=1)  # Both panels get full height

        # Left panel for video - using grid instead of pack for better control
        self.video_frame = ttk.LabelFrame(
            self.content_frame, text="Video Feed", padding=10
        )
        self.video_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        self.video_canvas = tk.Canvas(self.video_frame, bg=self.canvas_bg)
        self.video_canvas.pack(fill=tk.BOTH, expand=True)

        # Bind click events for setting count line and ROIs
        self.video_canvas.bind("<Button-1>", self.canvas_click)
        self.video_canvas.bind("<B1-Motion>", self.canvas_drag)
        self.video_canvas.bind("<ButtonRelease-1>", self.canvas_release)

        # Right panel for visualizations - using grid for proper sizing
        self.viz_notebook = ttk.Notebook(self.content_frame)
        self.viz_notebook.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Create different visualization tabs
        self.counts_tab = ttk.Frame(self.viz_notebook)
        self.time_series_tab = ttk.Frame(self.viz_notebook)
        self.heatmap_tab = ttk.Frame(self.viz_notebook)
        self.direction_tab = ttk.Frame(self.viz_notebook)
        self.speed_tab = ttk.Frame(self.viz_notebook)

        self.viz_notebook.add(self.counts_tab, text="Counts")
        self.viz_notebook.add(self.time_series_tab, text="Time Series")
        self.viz_notebook.add(self.heatmap_tab, text="Heatmap")
        self.viz_notebook.add(self.direction_tab, text="Direction")
        self.viz_notebook.add(self.speed_tab, text="Speed")

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

        # Zone controls frame
        self.zone_frame = ttk.LabelFrame(self.control_frame, text="Analysis Zones")
        self.zone_frame.pack(side=tk.LEFT, padx=10, fill=tk.Y)

        self.add_line_btn = ttk.Button(
            self.zone_frame, text="Add Count Line", command=self.start_add_line
        )
        self.add_line_btn.pack(side=tk.TOP, padx=5, pady=2)

        self.add_roi_btn = ttk.Button(
            self.zone_frame, text="Add ROI", command=self.start_add_roi
        )
        self.add_roi_btn.pack(side=tk.TOP, padx=5, pady=2)

        self.clear_zones_btn = ttk.Button(
            self.zone_frame, text="Clear All Zones", command=self.clear_zones
        )
        self.clear_zones_btn.pack(side=tk.TOP, padx=5, pady=2)

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

        # Mode selection
        ttk.Label(self.detection_frame, text="Detect:").pack(
            side=tk.TOP, anchor=tk.W, pady=(5, 0)
        )
        self.mode_combo = ttk.Combobox(
            self.detection_frame,
            textvariable=self.detection_mode_var,
            values=["vehicles", "people", "all", "custom"],
        )
        self.mode_combo.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        self.mode_combo.bind(
            "<<ComboboxSelected>>", lambda e: self.update_detection_mode()
        )

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
        # Set up plots for each tab with smaller, more conservative sizes

        # 1. Counts tab - smaller figure size to fit in panel
        self.counts_fig = Figure(figsize=(4, 3.5), dpi=100)
        self.counts_canvas = FigureCanvasTkAgg(self.counts_fig, self.counts_tab)
        self.counts_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 2. Time series tab
        self.time_fig = Figure(figsize=(4, 3.5), dpi=100)
        self.time_canvas = FigureCanvasTkAgg(self.time_fig, self.time_series_tab)
        self.time_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 3. Heatmap tab
        self.heatmap_fig = Figure(figsize=(4, 3.5), dpi=100)
        self.heatmap_canvas = FigureCanvasTkAgg(self.heatmap_fig, self.heatmap_tab)
        self.heatmap_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 4. Direction tab
        self.direction_fig = Figure(figsize=(4, 3.5), dpi=100)
        self.direction_canvas = FigureCanvasTkAgg(
            self.direction_fig, self.direction_tab
        )
        self.direction_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 5. Speed tab
        self.speed_fig = Figure(figsize=(4, 3.5), dpi=100)
        self.speed_canvas = FigureCanvasTkAgg(self.speed_fig, self.speed_tab)
        self.speed_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configure plot background colors based on theme
        for fig in [
            self.counts_fig,
            self.time_fig,
            self.heatmap_fig,
            self.direction_fig,
            self.speed_fig,
        ]:
            fig.patch.set_facecolor(self.plot_bg)
            # Use explicit margins instead of tight_layout to prevent errors
            fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)

            for ax in fig.get_axes():
                ax.set_facecolor(self.plot_bg)
                ax.tick_params(colors=self.plot_fg)
                ax.xaxis.label.set_color(self.plot_fg)
                ax.yaxis.label.set_color(self.plot_fg)
                ax.title.set_color(self.plot_fg)

    def load_model(self):
        try:
            self.status_var.set("Loading YOLOv8 model (may take a moment)...")
            self.root.update()

            # Show progress bar
            self.progress_bar.pack(side=tk.BOTTOM, fill=tk.X)
            self.progress_var.set(10)
            self.root.update()

            # Determine model size based on user selection
            model_size = self.model_size_var.get()
            model_path = f"yolov8{model_size[0]}.pt"  # n, s, or m

            # Special handling for systems with limited resources
            if self.optimize_cpu_var.get():
                # Lower precision for better performance on CPU
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
                    self.counting_line_names.append("Main Line")
                    self.counting_line_colors.append((255, 0, 0))

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
        self.object_direction_counts = defaultdict(int)

        # Update plots
        self.update_plots()

    def resize_frame(self, frame):
        """Resize a frame to fit the canvas while respecting layout constraints"""
        if frame is None:
            return None

        try:
            # Get current canvas dimensions
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()

            # If canvas hasn't been drawn yet or is too small, use reasonable defaults
            if canvas_width <= 1:
                # Use at most 60% of total window width to leave room for viz panel
                total_width = self.root.winfo_width()
                if total_width > 1:
                    canvas_width = min(640, int(total_width * 0.6))
                else:
                    canvas_width = 480  # Smaller default
            if canvas_height <= 1:
                canvas_height = 360  # Smaller default

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
            if self.debug:
                import traceback

                traceback.print_exc()
            return frame  # Return original frame if resize fails

    def show_frame(self, frame):
        """Display a frame on the canvas with proper sizing constraints"""
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

            # Important: Set fixed canvas size based on parent container
            # This prevents canvas from expanding beyond its allocated space
            video_frame_width = self.video_frame.winfo_width()
            if video_frame_width > 1:
                # Leave room for padding
                canvas_width = video_frame_width - 20
            else:
                canvas_width = 640

            video_frame_height = self.video_frame.winfo_height()
            if video_frame_height > 1:
                canvas_height = video_frame_height - 20
            else:
                canvas_height = 480

            self.video_canvas.config(width=canvas_width, height=canvas_height)

            # Calculate position to center image in canvas
            x_offset = max(0, (canvas_width - img.width) // 2)
            y_offset = max(0, (canvas_height - img.height) // 2)

            # Create image centered in the canvas
            self.video_canvas.create_image(
                x_offset, y_offset, anchor=tk.NW, image=img_tk
            )
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
                if self.show_count_info:
                    self.video_canvas.create_text(
                        start_point[0] + 10,
                        start_point[1] - 10,
                        text=f"{name}: {count}",
                        anchor=tk.W,
                        fill="white",
                        font=("Arial", 10, "bold"),
                    )

            # Draw regions of interest
            for i, roi in enumerate(self.regions_of_interest):
                points = roi["points"]
                name = roi["name"]
                count = self.roi_counts[name]

                # Convert points for polygon
                flat_points = []
                for point in points:
                    flat_points.extend([point[0], point[1]])

                # Draw the ROI polygon
                if len(flat_points) >= 6:  # Need at least 3 points (6 values)
                    self.video_canvas.create_polygon(
                        flat_points, outline="yellow", fill="", dash=(5, 5), width=2
                    )

                    # Add ROI label with count
                    if self.show_count_info:
                        center_x = sum(p[0] for p in points) // len(points)
                        center_y = sum(p[1] for p in points) // len(points)
                        self.video_canvas.create_text(
                            center_x,
                            center_y,
                            text=f"{name}: {count}",
                            fill="white",
                            font=("Arial", 10, "bold"),
                        )

            # Draw temporary elements for interactive zone creation
            if self.drawing_mode == "line" and self.drawing_start and self.temp_line:
                start_x, start_y = self.drawing_start
                end_x, end_y = self.temp_line
                self.video_canvas.create_line(
                    start_x, start_y, end_x, end_y, fill="yellow", width=2, dash=(5, 5)
                )

            elif self.drawing_mode == "roi" and len(self.temp_roi) > 0:
                # Draw lines connecting the points
                for i in range(len(self.temp_roi) - 1):
                    x1, y1 = self.temp_roi[i]
                    x2, y2 = self.temp_roi[i + 1]
                    self.video_canvas.create_line(
                        x1, y1, x2, y2, fill="yellow", width=2, dash=(5, 5)
                    )

                # Draw points
                for x, y in self.temp_roi:
                    self.video_canvas.create_oval(
                        x - 3, y - 3, x + 3, y + 3, fill="yellow", outline="black"
                    )

                # Connect last point to first if we have at least 3 points
                if len(self.temp_roi) >= 3:
                    x1, y1 = self.temp_roi[-1]
                    x2, y2 = self.temp_roi[0]
                    self.video_canvas.create_line(
                        x1, y1, x2, y2, fill="yellow", width=2, dash=(5, 5)
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
            elif self.drawing_mode == "roi":
                self.video_canvas.create_text(
                    10,
                    20,
                    text="Click to add points. Double-click to complete the ROI. Press ESC to cancel.",
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
        self.counting_lines = []
        self.counting_line_names = []
        self.counting_line_colors = []
        self.regions_of_interest = []
        self.roi_names = []
        self.roi_counts = defaultdict(int)

        # Redraw the current frame
        if hasattr(self, "last_frame") and self.last_frame is not None:
            self.show_frame(self.last_frame)

        self.status_var.set("All zones cleared")

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
                        self.counting_line_names.append(line_name)
                        self.counting_line_colors.append(color_bgr)

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
                initialvalue=f"ROI {len(self.regions_of_interest) + 1}",
            )

            if roi_name:
                # Add the new ROI
                new_roi = {"points": self.temp_roi.copy(), "name": roi_name}

                self.regions_of_interest.append(new_roi)
                self.roi_names.append(roi_name)
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

    def toggle_play(self):
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

        # Create a frame with scrollbar
        frame = ttk.Frame(dialog)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        scrollbar = ttk.Scrollbar(frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Create a canvas for scrolling
        canvas = tk.Canvas(frame, yscrollcommand=scrollbar.set)
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

            ttk.Checkbutton(
                inner_frame, text=f"{class_id}: {class_name}", variable=var
            ).pack(anchor=tk.W, pady=2)

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

                # Store original frame size for reference
                if not hasattr(self, "original_frame_size"):
                    self.original_frame_size = frame.shape[:2]  # (height, width)

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
                processed_frame, detections = self.process_frame(
                    processing_frame, frame
                )

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

    def update_plots(self):
        try:
            # 1. Object Counts Plot
            self.counts_fig.clear()
            ax1 = self.counts_fig.add_subplot(111)

            # Sort objects by count, get top 6
            sorted_objects = sorted(
                self.detected_objects.items(), key=lambda x: x[1], reverse=True
            )[:6]

            if sorted_objects:
                labels = [item[0] for item in sorted_objects]
                values = [item[1] for item in sorted_objects]

                bars = ax1.bar(labels, values, color=[self.accent_color])
                ax1.set_title("Top Detected Objects", color=self.plot_fg)
                ax1.set_ylabel("Count", color=self.plot_fg)

                # Add counts above bars
                for bar in bars:
                    height = bar.get_height()
                    ax1.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.1,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        color=self.plot_fg,
                    )

                # Rotate labels if needed
                if any(len(label) > 6 for label in labels):
                    ax1.set_xticklabels(labels, rotation=45, ha="right")

                # Set background color
                ax1.set_facecolor(self.plot_bg)
                ax1.tick_params(colors=self.plot_fg)

            # Use explicit subplots_adjust instead of tight_layout()
            self.counts_fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
            self.counts_canvas.draw()

            # 2. Time Series Plot
            self.time_fig.clear()
            ax2 = self.time_fig.add_subplot(111)

            # Make sure we have data
            if self.timestamps:
                # Get vehicle classes for legend
                vehicle_classes = []
                if self.detection_mode == "vehicles":
                    for class_id in self.vehicle_classes:
                        if class_id in self.class_names:
                            vehicle_classes.append(self.class_names[class_id])
                elif self.detection_mode == "people":
                    vehicle_classes = [self.class_names[self.person_class]]
                elif self.detection_mode == "custom" and self.custom_classes:
                    vehicle_classes = [
                        self.class_names[class_id] for class_id in self.custom_classes
                    ]
                else:  # "all" or fallback
                    # Just get the top 5 most frequent classes
                    top_classes = sorted(
                        self.detected_objects.items(), key=lambda x: x[1], reverse=True
                    )[:5]
                    vehicle_classes = [item[0] for item in top_classes]

                # Only plot classes that have data
                for class_name in vehicle_classes:
                    if (
                        class_name in self.time_series_data
                        and self.time_series_data[class_name]
                    ):
                        # Only plot last 100 points to keep visualization clean
                        max_points = 100
                        timestamps = (
                            self.timestamps[-max_points:]
                            if len(self.timestamps) > max_points
                            else self.timestamps
                        )
                        values = (
                            self.time_series_data[class_name][-max_points:]
                            if len(self.time_series_data[class_name]) > max_points
                            else self.time_series_data[class_name]
                        )

                        if len(timestamps) == len(values) and len(timestamps) > 0:
                            ax2.plot(timestamps, values, label=class_name)

                ax2.set_title("Objects Over Time", color=self.plot_fg)
                ax2.set_xlabel("Time (s)", color=self.plot_fg)
                ax2.set_ylabel("Count", color=self.plot_fg)

                # Only add legend if we have data
                if any(
                    len(self.time_series_data[class_name]) > 0
                    for class_name in vehicle_classes
                ):
                    ax2.legend(loc="upper left")

                # Set background color
                ax2.set_facecolor(self.plot_bg)
                ax2.tick_params(colors=self.plot_fg)

            # Use explicit subplots_adjust instead of tight_layout()
            self.time_fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
            self.time_canvas.draw()

            # 3. Heatmap Plot
            self.heatmap_fig.clear()
            ax3 = self.heatmap_fig.add_subplot(111)

            # Only show heatmap if we have data
            if np.sum(self.heatmap_data) > 0:
                # Normalize heatmap data
                normalized_heatmap = (
                    self.heatmap_data / np.max(self.heatmap_data)
                    if np.max(self.heatmap_data) > 0
                    else self.heatmap_data
                )

                # Use a colormap for better visualization
                cmap = plt.cm.jet

                # Create heatmap
                im = ax3.imshow(normalized_heatmap, cmap=cmap, interpolation="gaussian")

                # Add colorbar
                cbar = self.heatmap_fig.colorbar(im, ax=ax3)
                cbar.set_label("Normalized Density", color=self.plot_fg)
                cbar.ax.tick_params(colors=self.plot_fg)

                ax3.set_title("Traffic Density Heatmap", color=self.plot_fg)

                # Hide axis ticks for cleaner look
                ax3.set_xticks([])
                ax3.set_yticks([])
            else:
                ax3.text(
                    0.5,
                    0.5,
                    "Insufficient data for heatmap",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax3.set_xticks([])
                ax3.set_yticks([])

            # Use explicit subplots_adjust instead of tight_layout()
            self.heatmap_fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
            self.heatmap_canvas.draw()

            # 4. Direction Plot
            self.direction_fig.clear()
            ax4 = self.direction_fig.add_subplot(111)

            # Plot direction data as a pie chart
            if sum(self.direction_counts.values()) > 0:
                labels = list(self.direction_counts.keys())
                sizes = list(self.direction_counts.values())

                # Filter out zero values
                non_zero_labels = []
                non_zero_sizes = []
                for label, size in zip(labels, sizes):
                    if size > 0:
                        non_zero_labels.append(label)
                        non_zero_sizes.append(size)

                if non_zero_sizes:
                    # Use custom colors for directions
                    colors = ["#FF9999", "#66B2FF", "#99FF99", "#FFCC99"]
                    ax4.pie(
                        non_zero_sizes,
                        labels=non_zero_labels,
                        colors=colors,
                        autopct="%1.1f%%",
                        startangle=90,
                        textprops={"color": self.plot_fg},
                    )
                    ax4.set_title("Traffic Direction Distribution", color=self.plot_fg)
                else:
                    ax4.text(
                        0.5,
                        0.5,
                        "No direction data available",
                        ha="center",
                        va="center",
                        color=self.plot_fg,
                    )
                    ax4.set_xticks([])
                    ax4.set_yticks([])
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "Insufficient data for direction analysis",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax4.set_xticks([])
                ax4.set_yticks([])

            # Use explicit subplots_adjust instead of tight_layout()
            self.direction_fig.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9)
            self.direction_canvas.draw()

            # 5. Speed Plot
            self.speed_fig.clear()
            ax5 = self.speed_fig.add_subplot(111)

            # Check if we have speed data
            if any(len(speeds) > 0 for speeds in self.avg_speeds.values()):
                # Compute average speeds for each vehicle class
                avg_speed_data = {}
                for class_name, speeds in self.avg_speeds.items():
                    if speeds:
                        avg_speed_data[class_name] = sum(speeds) / len(speeds)

                if avg_speed_data:
                    # Sort by speed
                    sorted_speeds = sorted(
                        avg_speed_data.items(), key=lambda x: x[1], reverse=True
                    )

                    classes = [item[0] for item in sorted_speeds]
                    speeds = [item[1] for item in sorted_speeds]

                    # Create horizontal bar chart
                    bars = ax5.barh(classes, speeds, color=self.accent_color)

                    # Add speed values
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax5.text(
                            width + 0.5,
                            bar.get_y() + bar.get_height() / 2,
                            f"{width:.1f} km/h",
                            va="center",
                            color=self.plot_fg,
                        )

                    ax5.set_title("Average Speed by Vehicle Type", color=self.plot_fg)
                    ax5.set_xlabel("Speed (km/h)", color=self.plot_fg)

                    # Set background color
                    ax5.set_facecolor(self.plot_bg)
                    ax5.tick_params(colors=self.plot_fg)
                else:
                    ax5.text(
                        0.5,
                        0.5,
                        "Processing speed data...",
                        ha="center",
                        va="center",
                        color=self.plot_fg,
                    )
                    ax5.set_xticks([])
                    ax5.set_yticks([])
            else:
                ax5.text(
                    0.5,
                    0.5,
                    "Insufficient data for speed analysis",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax5.set_xticks([])
                ax5.set_yticks([])

            # Use explicit subplots_adjust instead of tight_layout()
            self.speed_fig.subplots_adjust(left=0.15, bottom=0.15, right=0.9, top=0.9)
            self.speed_canvas.draw()

        except Exception as e:
            if self.debug:
                print(f"Error updating plots: {str(e)}")
                import traceback

                traceback.print_exc()

    def process_frame(self, frame, original_frame=None):
        """Process a video frame with object detection and tracking"""
        if frame is None or not self.model_loaded:
            return frame, []

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

            # Filter detections based on detection mode
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    try:
                        # Get class and confidence
                        class_id = int(box.cls[0].item())

                        # Skip based on detection mode
                        if (
                            self.detection_mode == "vehicles"
                            and class_id not in self.vehicle_classes
                        ):
                            continue
                        elif (
                            self.detection_mode == "people"
                            and class_id != self.person_class
                        ):
                            continue
                        elif (
                            self.detection_mode == "custom"
                            and class_id not in self.custom_classes
                        ):
                            continue

                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Skip if object is too small (likely noise)
                        if (x2 - x1) < self.minimum_object_size or (
                            y2 - y1
                        ) < self.minimum_object_size:
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

                        # Update heatmap
                        if self.downscale_factor != 1.0:
                            # Scale coordinates back to original frame size
                            scale_x = display_frame.shape[1] / frame.shape[1]
                            scale_y = display_frame.shape[0] / frame.shape[0]

                            orig_x1 = int(x1 * scale_x)
                            orig_y1 = int(y1 * scale_y)
                            orig_x2 = int(x2 * scale_x)
                            orig_y2 = int(y2 * scale_y)
                            orig_center_x = (orig_x1 + orig_x2) // 2
                            orig_center_y = (orig_y1 + orig_y2) // 2

                            # Update heatmap with original coordinates
                            if (
                                0 <= orig_center_y < self.heatmap_data.shape[0]
                                and 0 <= orig_center_x < self.heatmap_data.shape[1]
                            ):
                                self.heatmap_data[orig_center_y, orig_center_x] += 1

                                # Add a gaussian blob around the center for better visualization
                                y, x = np.ogrid[
                                    -orig_center_y : self.heatmap_data.shape[0]
                                    - orig_center_y,
                                    -orig_center_x : self.heatmap_data.shape[1]
                                    - orig_center_x,
                                ]
                                mask = x * x + y * y <= 25  # Circle with radius 5
                                self.heatmap_data[
                                    mask
                                ] += 0.2  # Lower intensity for surrounding pixels
                        else:
                            # Update heatmap directly
                            if (
                                0 <= center_y < self.heatmap_data.shape[0]
                                and 0 <= center_x < self.heatmap_data.shape[1]
                            ):
                                self.heatmap_data[center_y, center_x] += 1

                                # Add a gaussian blob around the center for better visualization
                                y, x = np.ogrid[
                                    -center_y : self.heatmap_data.shape[0] - center_y,
                                    -center_x : self.heatmap_data.shape[1] - center_x,
                                ]
                                mask = x * x + y * y <= 25  # Circle with radius 5
                                self.heatmap_data[
                                    mask
                                ] += 0.2  # Lower intensity for surrounding pixels

                    except Exception as e:
                        # Skip this detection if there's an error
                        if self.debug:
                            print(f"Error processing detection: {str(e)}")

            # Track objects across frames
            self.track_objects(current_detections, current_time, display_frame)

            # Draw visualization overlays
            self.draw_visualizations(display_frame, current_detections)

            # Update status less frequently to reduce overhead
            if self.frame_count % 30 == 0:
                self.status_var.set(
                    f"Processing frame {self.frame_count} | "
                    f"Time: {current_time:.1f}s | "
                    f"Total vehicles: {self.total_vehicle_count}"
                )

                # Update the count display
                self.total_count_var.set(str(self.total_vehicle_count))

            return display_frame, current_detections

        except Exception as e:
            self.status_var.set(f"Error processing frame: {str(e)}")
            if self.debug:
                import traceback

                traceback.print_exc()
            return (
                frame,
                [],
            )  # Return original frame and empty detections if processing fails

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
                            "positions": deque(maxlen=self.history_window),
                            "timestamps": deque(maxlen=self.history_window),
                            "class_id": det["class_id"],
                            "class_name": det["class_name"],
                            "counted": False,
                            "roi_counted": set(),  # Track which ROIs this object has been counted in
                            "last_seen": current_time,
                            "boxes": deque(
                                maxlen=self.history_window
                            ),  # Store bounding boxes for size tracking
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
                            "positions": deque(maxlen=self.history_window),
                            "timestamps": deque(maxlen=self.history_window),
                            "class_id": det["class_id"],
                            "class_name": det["class_name"],
                            "counted": False,
                            "roi_counted": set(),
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

                # Update time series data for relevant classes
                if self.detection_mode == "vehicles":
                    for class_id in self.vehicle_classes:
                        if class_id in self.class_names:
                            class_name = self.class_names[class_id]
                            self.time_series_data[class_name].append(
                                class_counts[class_name]
                            )
                elif self.detection_mode == "people":
                    class_name = self.class_names[self.person_class]
                    self.time_series_data[class_name].append(class_counts[class_name])
                elif self.detection_mode == "custom":
                    for class_id in self.custom_classes:
                        if class_id in self.class_names:
                            class_name = self.class_names[class_id]
                            self.time_series_data[class_name].append(
                                class_counts[class_name]
                            )
                else:  # "all"
                    # Add all classes that have been detected
                    for class_name in class_counts:
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
                        "direction": (
                            obj_data["direction"]
                            if obj_data["direction"]
                            else "unknown"
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
        for roi_index, roi in enumerate(self.regions_of_interest):
            roi_name = roi["name"]

            # Check if point is inside polygon
            if self.point_in_polygon(curr_pos, roi["points"]):
                # Object is in this ROI
                if roi_name not in obj_data["roi_counted"]:
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

            # Update direction counts
            if direction and obj_data["direction"] != direction:
                self.direction_counts[direction] += 1

            # Calculate speed
            if len(timestamps) >= 5:
                dt = timestamps[-1] - timestamps[0]
                if dt > 0:
                    # Calculate distance in pixels
                    distance = np.sqrt(dx**2 + dy**2)

                    # Convert to real-world distance (approximate)
                    # This assumes that 100 pixels is roughly 1 meter (adjust based on your camera setup)
                    meters = distance / 100

                    # Calculate speed in km/h (dt is in seconds)
                    speed_kmh = (meters / dt) * 3.6

                    # Store speed
                    obj_data["speed"] = speed_kmh

                    # Add to average speeds
                    self.avg_speeds[obj_data["class_name"]].append(speed_kmh)

                    # Keep only the recent speeds for average calculation
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
            return frame  # Nothing to do if no frame is provided

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
                        -1,  # Fill rectangle
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
                if self.debug:
                    print(f"Error drawing heatmap: {str(e)}")

        # Draw general statistics
        stats_y_position = 30
        stats_increment = 25

        # Add timestamp
        current_time = time.time() - self.start_time
        mins = int(current_time // 60)
        secs = int(current_time % 60)
        cv2.putText(
            frame,
            f"Time: {mins:02d}:{secs:02d}",
            (10, stats_y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        stats_y_position += stats_increment

        # Add total count
        cv2.putText(
            frame,
            f"Total Vehicles: {self.total_vehicle_count}",
            (10, stats_y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        return frame

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
                "analysis_duration": time.time() - self.start_time,
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

            # Export heatmap as an image
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
                "Export Successful", f"Data exported to {self.export_directory}"
            )

        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting data: {str(e)}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def generate_report(self):
        """Generate a comprehensive analysis report"""
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
            analysis_duration = time.time() - self.start_time
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

            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Traffic Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                    .container {{ max-width: 900px; margin: 0 auto; }}
                    h1, h2, h3 {{ color: #333; }}
                    .summary-box {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .chart-container {{ display: flex; flex-wrap: wrap; justify-content: space-between; }}
                    .chart {{ width: 48%; margin-bottom: 20px; background-color: white; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                    th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Traffic Analysis Report</h1>
                    <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <div class="summary-box">
                        <h2>Summary</h2>
                        <p>Video: {os.path.basename(self.video_path) if self.video_path else "N/A"}</p>
                        <p>Analysis Duration: {mins:02d}:{secs:02d}</p>
                        <p>Total Vehicles Counted: {total_vehicles}</p>
                        <p>Average Vehicles Per Minute: {(total_vehicles / (analysis_duration / 60)):.2f}</p>
                    </div>
                    
                    <h2>Vehicle Distribution</h2>
                    <table>
                        <tr>
                            <th>Vehicle Type</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
            """

            # Add vehicle distribution table rows
            for i, (label, count) in enumerate(zip(class_labels, class_counts)):
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
            for i, (label, count) in enumerate(zip(line_labels, line_counts)):
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
            for i, (label, count) in enumerate(zip(direction_labels, direction_counts)):
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
                    
                    <h2>Speed Analysis</h2>
                    <table>
                        <tr>
                            <th>Vehicle Type</th>
                            <th>Average Speed (km/h)</th>
                        </tr>
            """

            # Add speed table rows
            for i, (label, speed) in enumerate(zip(speed_labels, speed_values)):
                html_content += f"""
                        <tr>
                            <td>{label}</td>
                            <td>{speed:.1f}</td>
                        </tr>
                """

            html_content += """
                    </table>
                    
                    <h2>Conclusions</h2>
                    <p>This report provides a comprehensive analysis of traffic patterns captured during the video analysis.</p>
                    <p>The data shows traffic flow characteristics, vehicle type distribution, directional trends, and speed profiles that can be used for traffic management and planning.</p>
                    
                    <p><small>Generated by Advanced Traffic Analysis System</small></p>
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
            if self.debug:
                import traceback

                traceback.print_exc()

    def show_dashboard(self):
        """Show a comprehensive analytics dashboard"""
        # Create a new toplevel window
        dashboard = tk.Toplevel(self.root)
        dashboard.title("Traffic Analytics Dashboard")
        dashboard.geometry("1200x800")

        # Configure the same theme as the main window
        dashboard.configure(bg=self.bg_color)

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
        notebook.add(vehicles_tab, text="Vehicle Types")
        notebook.add(temporal_tab, text="Temporal")
        notebook.add(spatial_tab, text="Spatial")
        notebook.add(speed_tab, text="Speed")

        # Overview Tab
        self.setup_overview_tab(overview_tab)

        # Vehicles Tab
        self.setup_vehicles_tab(vehicles_tab)

        # Temporal Tab
        self.setup_temporal_tab(temporal_tab)

        # Spatial Tab
        self.setup_spatial_tab(spatial_tab)

        # Speed Tab
        self.setup_speed_tab(speed_tab)

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

        # Make sure dashboard stays on top
        dashboard.transient(self.root)
        dashboard.grab_set()

    def setup_overview_tab(self, parent):
        """Setup the overview dashboard tab"""
        # Create a grid layout
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Add summary statistics
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
        duration = time.time() - self.start_time
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

        # Add real-time plot for vehicle count
        plot_frame = ttk.LabelFrame(frame, text="Vehicle Count Over Time")
        plot_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")

        fig = Figure(figsize=(6, 4), dpi=100)
        fig.patch.set_facecolor(self.plot_bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(self.plot_bg)
        ax.tick_params(colors=self.plot_fg)
        ax.set_title("Vehicle Count Over Time", color=self.plot_fg)
        ax.set_xlabel("Time (s)", color=self.plot_fg)
        ax.set_ylabel("Count", color=self.plot_fg)

        # Plot time series data if available
        if self.timestamps:
            # Plot for vehicles
            vehicle_data = []
            for class_id in self.vehicle_classes:
                if class_id in self.class_names:
                    class_name = self.class_names[class_id]
                    if (
                        class_name in self.time_series_data
                        and self.time_series_data[class_name]
                    ):
                        vehicle_data.append(self.time_series_data[class_name])

            # Combine all vehicle data
            if vehicle_data:
                combined_data = [sum(counts) for counts in zip(*vehicle_data)]
                ax.plot(self.timestamps, combined_data, color=self.accent_color)

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add direction distribution
        direction_frame = ttk.LabelFrame(frame, text="Direction Distribution")
        direction_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Show direction counts in a table
        row = 0
        ttk.Label(direction_frame, text="Direction", font=("Arial", 10, "bold")).grid(
            row=row, column=0, sticky=tk.W, padx=5, pady=2
        )
        ttk.Label(direction_frame, text="Count", font=("Arial", 10, "bold")).grid(
            row=row, column=1, sticky=tk.W, padx=5, pady=2
        )

        # Sort by count
        sorted_directions = sorted(
            self.direction_counts.items(), key=lambda x: x[1], reverse=True
        )

        for direction, count in sorted_directions:
            row += 1
            ttk.Label(direction_frame, text=direction.capitalize()).grid(
                row=row, column=0, sticky=tk.W, padx=5, pady=1
            )
            ttk.Label(direction_frame, text=str(count)).grid(
                row=row, column=1, sticky=tk.W, padx=5, pady=1
            )

        # Configure grid weights
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=3)
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        # Update function for real-time updates
        def update_overview():
            total_var.set(str(self.total_vehicle_count))

            duration = time.time() - self.start_time
            mins = int(duration // 60)
            secs = int(duration % 60)
            duration_var.set(f"{mins:02d}:{secs:02d}")

            density_var.set(
                f"{self.total_vehicle_count / (duration / 60):.1f} vehicles/minute"
                if duration > 0
                else "0"
            )

            # Update plot
            ax.clear()
            ax.set_facecolor(self.plot_bg)
            ax.tick_params(colors=self.plot_fg)
            ax.set_title("Vehicle Count Over Time", color=self.plot_fg)
            ax.set_xlabel("Time (s)", color=self.plot_fg)
            ax.set_ylabel("Count", color=self.plot_fg)

            if self.timestamps:
                # Plot for vehicles
                vehicle_data = []
                for class_id in self.vehicle_classes:
                    if class_id in self.class_names:
                        class_name = self.class_names[class_id]
                        if (
                            class_name in self.time_series_data
                            and self.time_series_data[class_name]
                        ):
                            vehicle_data.append(self.time_series_data[class_name])

                # Combine all vehicle data
                if vehicle_data:
                    # Make sure all lists are the same length
                    min_length = min(len(data) for data in vehicle_data)
                    vehicle_data = [data[:min_length] for data in vehicle_data]
                    timestamps = self.timestamps[:min_length]

                    combined_data = [sum(counts) for counts in zip(*vehicle_data)]
                    ax.plot(timestamps, combined_data, color=self.accent_color)

            canvas.draw()

            # Schedule next update if window is still open
            try:
                parent.after(1000, update_overview)
            except:
                pass  # Window was closed

        # Start update loop
        parent.after(1000, update_overview)

    def setup_vehicles_tab(self, parent):
        """Setup the vehicles dashboard tab"""
        # Create a grid layout
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top frame for pie chart
        pie_frame = ttk.LabelFrame(frame, text="Vehicle Type Distribution")
        pie_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Create pie chart
        fig1 = Figure(figsize=(4, 4), dpi=100)
        fig1.patch.set_facecolor(self.plot_bg)
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor(self.plot_bg)

        # Get vehicle class data
        vehicle_labels = []
        vehicle_counts = []

        for class_id in self.vehicle_classes:
            if class_id in self.class_names:
                class_name = self.class_names[class_id]
                count = self.detected_objects.get(class_name, 0)
                if count > 0:
                    vehicle_labels.append(class_name)
                    vehicle_counts.append(count)

        if vehicle_counts:
            # Custom colors
            colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
            ax1.pie(
                vehicle_counts,
                labels=vehicle_labels,
                colors=colors,
                autopct="%1.1f%%",
                startangle=90,
                textprops={"color": self.plot_fg},
            )
        else:
            ax1.text(
                0.5,
                0.5,
                "No vehicle data",
                ha="center",
                va="center",
                color=self.plot_fg,
            )
            ax1.set_xticks([])
            ax1.set_yticks([])

        canvas1 = FigureCanvasTkAgg(fig1, master=pie_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom frame for bar chart
        bar_frame = ttk.LabelFrame(frame, text="Vehicle Counts by Type")
        bar_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # Create bar chart
        fig2 = Figure(figsize=(5, 4), dpi=100)
        fig2.patch.set_facecolor(self.plot_bg)
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor(self.plot_bg)
        ax2.tick_params(colors=self.plot_fg)

        if vehicle_labels and vehicle_counts:
            # Sort by count
            sorted_data = sorted(
                zip(vehicle_labels, vehicle_counts), key=lambda x: x[1], reverse=True
            )
            sorted_labels = [item[0] for item in sorted_data]
            sorted_counts = [item[1] for item in sorted_data]

            bars = ax2.bar(sorted_labels, sorted_counts, color=self.accent_color)

            # Add count labels
            for bar in bars:
                height = bar.get_height()
                ax2.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.1,
                    f"{int(height)}",
                    ha="center",
                    va="bottom",
                    color=self.plot_fg,
                )

            ax2.set_xlabel("Vehicle Type", color=self.plot_fg)
            ax2.set_ylabel("Count", color=self.plot_fg)
        else:
            ax2.text(
                0.5,
                0.5,
                "No vehicle data",
                ha="center",
                va="center",
                color=self.plot_fg,
            )
            ax2.set_xticks([])
            ax2.set_yticks([])

        canvas2 = FigureCanvasTkAgg(fig2, master=bar_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom frame for detailed stats
        stats_frame = ttk.LabelFrame(frame, text="Vehicle Statistics")
        stats_frame.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # Create a table of statistics
        columns = ("Type", "Count", "Avg Speed", "% of Total")
        tree = ttk.Treeview(stats_frame, columns=columns, show="headings")

        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=100, anchor="center")

        # Calculate total for percentage
        total = sum(self.detected_objects.values())

        # Add rows for each vehicle type
        for class_id in self.vehicle_classes:
            if class_id in self.class_names:
                class_name = self.class_names[class_id]
                count = self.detected_objects.get(class_name, 0)

                # Calculate average speed
                avg_speed = 0
                if class_name in self.avg_speeds and self.avg_speeds[class_name]:
                    avg_speed = sum(self.avg_speeds[class_name]) / len(
                        self.avg_speeds[class_name]
                    )

                # Calculate percentage
                percentage = (count / total * 100) if total > 0 else 0

                tree.insert(
                    "",
                    "end",
                    values=(
                        class_name,
                        count,
                        f"{avg_speed:.1f} km/h",
                        f"{percentage:.1f}%",
                    ),
                )

        # Add scrollbar
        scrollbar = ttk.Scrollbar(stats_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)

        # Configure grid weights
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=2)
        frame.rowconfigure(1, weight=1)

        # Update function for real-time updates
        def update_vehicles():
            # Update pie chart
            ax1.clear()

            # Get updated vehicle class data
            vehicle_labels = []
            vehicle_counts = []

            for class_id in self.vehicle_classes:
                if class_id in self.class_names:
                    class_name = self.class_names[class_id]
                    count = self.detected_objects.get(class_name, 0)
                    if count > 0:
                        vehicle_labels.append(class_name)
                        vehicle_counts.append(count)

            if vehicle_counts:
                colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
                ax1.pie(
                    vehicle_counts,
                    labels=vehicle_labels,
                    colors=colors,
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"color": self.plot_fg},
                )
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "No vehicle data",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax1.set_xticks([])
                ax1.set_yticks([])

            canvas1.draw()

            # Update bar chart
            ax2.clear()
            ax2.set_facecolor(self.plot_bg)
            ax2.tick_params(colors=self.plot_fg)

            if vehicle_labels and vehicle_counts:
                # Sort by count
                sorted_data = sorted(
                    zip(vehicle_labels, vehicle_counts),
                    key=lambda x: x[1],
                    reverse=True,
                )
                sorted_labels = [item[0] for item in sorted_data]
                sorted_counts = [item[1] for item in sorted_data]

                bars = ax2.bar(sorted_labels, sorted_counts, color=self.accent_color)

                # Add count labels
                for bar in bars:
                    height = bar.get_height()
                    ax2.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height + 0.1,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        color=self.plot_fg,
                    )

                ax2.set_xlabel("Vehicle Type", color=self.plot_fg)
                ax2.set_ylabel("Count", color=self.plot_fg)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No vehicle data",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax2.set_xticks([])
                ax2.set_yticks([])

            canvas2.draw()

            # Update table
            # Clear existing rows
            for item in tree.get_children():
                tree.delete(item)

            # Calculate total for percentage
            total = sum(self.detected_objects.values())

            # Add updated rows
            for class_id in self.vehicle_classes:
                if class_id in self.class_names:
                    class_name = self.class_names[class_id]
                    count = self.detected_objects.get(class_name, 0)

                    # Calculate average speed
                    avg_speed = 0
                    if class_name in self.avg_speeds and self.avg_speeds[class_name]:
                        avg_speed = sum(self.avg_speeds[class_name]) / len(
                            self.avg_speeds[class_name]
                        )

                    # Calculate percentage
                    percentage = (count / total * 100) if total > 0 else 0

                    tree.insert(
                        "",
                        "end",
                        values=(
                            class_name,
                            count,
                            f"{avg_speed:.1f} km/h",
                            f"{percentage:.1f}%",
                        ),
                    )

            # Schedule next update if window is still open
            try:
                parent.after(1000, update_vehicles)
            except:
                pass  # Window was closed

        # Start update loop
        parent.after(1000, update_vehicles)

    def setup_temporal_tab(self, parent):
        """Setup the temporal analysis dashboard tab"""
        # Create a grid layout
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top frame for time series
        time_frame = ttk.LabelFrame(frame, text="Traffic Volume Over Time")
        time_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # Create time series chart
        fig1 = Figure(figsize=(8, 4), dpi=100)
        fig1.patch.set_facecolor(self.plot_bg)
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor(self.plot_bg)
        ax1.tick_params(colors=self.plot_fg)
        ax1.set_xlabel("Time (s)", color=self.plot_fg)
        ax1.set_ylabel("Count", color=self.plot_fg)

        if self.timestamps and len(self.timestamps) > 1:
            # Plot individual vehicle classes
            for class_id in self.vehicle_classes:
                if class_id in self.class_names:
                    class_name = self.class_names[class_id]
                    if (
                        class_name in self.time_series_data
                        and len(self.time_series_data[class_name]) > 1
                    ):
                        ax1.plot(
                            self.timestamps[: len(self.time_series_data[class_name])],
                            self.time_series_data[class_name],
                            label=class_name,
                        )

            ax1.legend()
        else:
            ax1.text(
                0.5,
                0.5,
                "Insufficient time series data",
                ha="center",
                va="center",
                color=self.plot_fg,
            )
            ax1.set_xticks([])
            ax1.set_yticks([])

        canvas1 = FigureCanvasTkAgg(fig1, master=time_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom left frame for peak times
        peak_frame = ttk.LabelFrame(frame, text="Peak Activity")
        peak_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Create a text widget for peak times
        peak_text = tk.Text(
            peak_frame, height=10, width=30, bg=self.plot_bg, fg=self.plot_fg
        )
        peak_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Update peak text with data
        if self.timestamps and len(self.timestamps) > 1:
            # Calculate combined vehicle count over time
            combined_data = []
            min_length = min(
                len(self.timestamps),
                min(len(data) for data in self.time_series_data.values() if data),
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
                sorted_data = sorted(combined_data, key=lambda x: x[1], reverse=True)

                peak_text.insert(tk.END, "Top 5 Peak Activity Times:\n\n")

                for i, (timestamp, count) in enumerate(sorted_data[:5]):
                    mins = int(timestamp // 60)
                    secs = int(timestamp % 60)
                    peak_text.insert(
                        tk.END,
                        f"{i+1}. Time {mins:02d}:{secs:02d} - {count} vehicles\n",
                    )
            else:
                peak_text.insert(tk.END, "Insufficient data for peak analysis")
        else:
            peak_text.insert(tk.END, "Insufficient data for peak analysis")

        peak_text.config(state=tk.DISABLED)  # Make read-only

        # Bottom right frame for distribution
        dist_frame = ttk.LabelFrame(frame, text="Traffic Distribution")
        dist_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # Create distribution chart
        fig2 = Figure(figsize=(4, 4), dpi=100)
        fig2.patch.set_facecolor(self.plot_bg)
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor(self.plot_bg)
        ax2.tick_params(colors=self.plot_fg)

        if self.timestamps and len(self.timestamps) > 10:
            # Calculate binned data for histogram
            combined_data = []
            min_length = min(
                len(self.timestamps),
                min(len(data) for data in self.time_series_data.values() if data),
            )

            if min_length > 0:
                for i in range(min_length):
                    total = sum(
                        data[i]
                        for data in self.time_series_data.values()
                        if i < len(data)
                    )
                    combined_data.append(total)

                ax2.hist(combined_data, bins=10, color=self.accent_color, alpha=0.7)
                ax2.set_xlabel("Vehicle Count", color=self.plot_fg)
                ax2.set_ylabel("Frequency", color=self.plot_fg)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "Insufficient data for distribution analysis",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax2.set_xticks([])
                ax2.set_yticks([])
        else:
            ax2.text(
                0.5,
                0.5,
                "Insufficient data for distribution analysis",
                ha="center",
                va="center",
                color=self.plot_fg,
            )
            ax2.set_xticks([])
            ax2.set_yticks([])

        canvas2 = FigureCanvasTkAgg(fig2, master=dist_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=2)
        frame.rowconfigure(1, weight=1)

        # Update function for real-time updates
        def update_temporal():
            # Update time series chart
            ax1.clear()
            ax1.set_facecolor(self.plot_bg)
            ax1.tick_params(colors=self.plot_fg)
            ax1.set_xlabel("Time (s)", color=self.plot_fg)
            ax1.set_ylabel("Count", color=self.plot_fg)

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
                                len(self.timestamps),
                                len(self.time_series_data[class_name]),
                            )
                            ax1.plot(
                                self.timestamps[:data_len],
                                self.time_series_data[class_name][:data_len],
                                label=class_name,
                            )

                ax1.legend()
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "Insufficient time series data",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax1.set_xticks([])
                ax1.set_yticks([])

            canvas1.draw()

            # Update peak text
            peak_text.config(state=tk.NORMAL)
            peak_text.delete(1.0, tk.END)

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

                        peak_text.insert(tk.END, "Top 5 Peak Activity Times:\n\n")

                        for i, (timestamp, count) in enumerate(sorted_data[:5]):
                            mins = int(timestamp // 60)
                            secs = int(timestamp % 60)
                            peak_text.insert(
                                tk.END,
                                f"{i+1}. Time {mins:02d}:{secs:02d} - {count} vehicles\n",
                            )
                    else:
                        peak_text.insert(tk.END, "Insufficient data for peak analysis")
                else:
                    peak_text.insert(tk.END, "Insufficient data for peak analysis")
            else:
                peak_text.insert(tk.END, "Insufficient data for peak analysis")

            peak_text.config(state=tk.DISABLED)

            # Update distribution chart
            ax2.clear()
            ax2.set_facecolor(self.plot_bg)
            ax2.tick_params(colors=self.plot_fg)

            if self.timestamps and len(self.timestamps) > 10:
                # Calculate binned data for histogram
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
                            combined_data.append(total)

                        ax2.hist(
                            combined_data,
                            bins=min(10, len(combined_data) // 2),
                            color=self.accent_color,
                            alpha=0.7,
                        )
                        ax2.set_xlabel("Vehicle Count", color=self.plot_fg)
                        ax2.set_ylabel("Frequency", color=self.plot_fg)
                    else:
                        ax2.text(
                            0.5,
                            0.5,
                            "Insufficient data for distribution analysis",
                            ha="center",
                            va="center",
                            color=self.plot_fg,
                        )
                        ax2.set_xticks([])
                        ax2.set_yticks([])
                else:
                    ax2.text(
                        0.5,
                        0.5,
                        "Insufficient data for distribution analysis",
                        ha="center",
                        va="center",
                        color=self.plot_fg,
                    )
                    ax2.set_xticks([])
                    ax2.set_yticks([])
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "Insufficient data for distribution analysis",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax2.set_xticks([])
                ax2.set_yticks([])

            canvas2.draw()

            # Schedule next update if window is still open
            try:
                parent.after(1000, update_temporal)
            except:
                pass  # Window was closed

        # Start update loop
        parent.after(1000, update_temporal)

    def setup_spatial_tab(self, parent):
        """Setup the spatial analysis dashboard tab"""
        # Create a grid layout
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top frame for heatmap
        heatmap_frame = ttk.LabelFrame(frame, text="Traffic Density Heatmap")
        heatmap_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")

        # Create heatmap visualization
        fig1 = Figure(figsize=(8, 4), dpi=100)
        fig1.patch.set_facecolor(self.plot_bg)
        ax1 = fig1.add_subplot(111)

        if np.sum(self.heatmap_data) > 0:
            # Normalize heatmap
            norm_heatmap = self.heatmap_data.copy()
            norm_heatmap = (
                norm_heatmap / np.max(norm_heatmap)
                if np.max(norm_heatmap) > 0
                else norm_heatmap
            )

            # Display heatmap
            im = ax1.imshow(norm_heatmap, cmap="jet", interpolation="gaussian")

            # Add colorbar
            cbar = fig1.colorbar(im, ax=ax1)
            cbar.set_label("Normalized Density", color=self.plot_fg)
            cbar.ax.tick_params(colors=self.plot_fg)

            # Hide axis ticks
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_title("Traffic Density Heatmap", color=self.plot_fg)
        else:
            ax1.text(
                0.5,
                0.5,
                "Insufficient data for heatmap",
                ha="center",
                va="center",
                color=self.plot_fg,
            )
            ax1.set_xticks([])
            ax1.set_yticks([])

        canvas1 = FigureCanvasTkAgg(fig1, master=heatmap_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom left frame for counting lines
        lines_frame = ttk.LabelFrame(frame, text="Counting Line Statistics")
        lines_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Create a table of counting line statistics
        columns = ("Line", "Count", "% of Total")
        line_tree = ttk.Treeview(lines_frame, columns=columns, show="headings")

        for col in columns:
            line_tree.heading(col, text=col)
            line_tree.column(col, width=100, anchor="center")

        # Calculate total for percentage
        total_line_count = sum(line["count"] for line in self.counting_lines)

        # Add rows for each counting line
        for line in self.counting_lines:
            name = line["name"]
            count = line["count"]

            # Calculate percentage
            percentage = (count / total_line_count * 100) if total_line_count > 0 else 0

            line_tree.insert("", "end", values=(name, count, f"{percentage:.1f}%"))

        # Add scrollbar
        line_scrollbar = ttk.Scrollbar(
            lines_frame, orient="vertical", command=line_tree.yview
        )
        line_tree.configure(yscrollcommand=line_scrollbar.set)
        line_scrollbar.pack(side="right", fill="y")
        line_tree.pack(fill="both", expand=True)

        # Bottom right frame for direction statistics
        dir_frame = ttk.LabelFrame(frame, text="Direction Analysis")
        dir_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        # Create a pie chart for direction statistics
        fig2 = Figure(figsize=(4, 4), dpi=100)
        fig2.patch.set_facecolor(self.plot_bg)
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor(self.plot_bg)

        # Plot direction data
        if sum(self.direction_counts.values()) > 0:
            # Filter out zero counts
            labels = []
            sizes = []
            for direction, count in self.direction_counts.items():
                if count > 0:
                    labels.append(direction.capitalize())
                    sizes.append(count)

            if sizes:
                # Create pie chart
                colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
                ax2.pie(
                    sizes,
                    labels=labels,
                    colors=colors,
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"color": self.plot_fg},
                )
                ax2.set_title("Traffic Direction", color=self.plot_fg)
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No direction data",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax2.set_xticks([])
                ax2.set_yticks([])
        else:
            ax2.text(
                0.5,
                0.5,
                "No direction data",
                ha="center",
                va="center",
                color=self.plot_fg,
            )
            ax2.set_xticks([])
            ax2.set_yticks([])

        canvas2 = FigureCanvasTkAgg(fig2, master=dir_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Configure grid weights
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=2)
        frame.rowconfigure(1, weight=1)

        # Update function for real-time updates
        def update_spatial():
            # Update heatmap
            ax1.clear()

            if np.sum(self.heatmap_data) > 0:
                # Normalize heatmap
                norm_heatmap = self.heatmap_data.copy()
                norm_heatmap = (
                    norm_heatmap / np.max(norm_heatmap)
                    if np.max(norm_heatmap) > 0
                    else norm_heatmap
                )

                # Display heatmap
                im = ax1.imshow(norm_heatmap, cmap="jet", interpolation="gaussian")

                # Add colorbar if not already present
                if not hasattr(fig1, "colorbar"):
                    cbar = fig1.colorbar(im, ax=ax1)
                    cbar.set_label("Normalized Density", color=self.plot_fg)
                    cbar.ax.tick_params(colors=self.plot_fg)
                    fig1.colorbar = cbar

                # Hide axis ticks
                ax1.set_xticks([])
                ax1.set_yticks([])
                ax1.set_title("Traffic Density Heatmap", color=self.plot_fg)
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "Insufficient data for heatmap",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax1.set_xticks([])
                ax1.set_yticks([])

            canvas1.draw()

            # Update counting line statistics
            # Clear existing rows
            for item in line_tree.get_children():
                line_tree.delete(item)

            # Calculate total for percentage
            total_line_count = sum(line["count"] for line in self.counting_lines)

            # Add updated rows
            for line in self.counting_lines:
                name = line["name"]
                count = line["count"]

                # Calculate percentage
                percentage = (
                    (count / total_line_count * 100) if total_line_count > 0 else 0
                )

                line_tree.insert("", "end", values=(name, count, f"{percentage:.1f}%"))

            # Update direction pie chart
            ax2.clear()
            if sum(self.direction_counts.values()) > 0:
                # Filter out zero counts
                labels = []
                sizes = []
                for direction, count in self.direction_counts.items():
                    if count > 0:
                        labels.append(direction.capitalize())
                        sizes.append(count)

                if sizes:
                    # Create pie chart
                    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
                    ax2.pie(
                        sizes,
                        labels=labels,
                        colors=colors,
                        autopct="%1.1f%%",
                        startangle=90,
                        textprops={"color": self.plot_fg},
                    )
                    ax2.set_title("Traffic Direction", color=self.plot_fg)
                else:
                    ax2.text(
                        0.5,
                        0.5,
                        "No direction data",
                        ha="center",
                        va="center",
                        color=self.plot_fg,
                    )
                    ax2.set_xticks([])
                    ax2.set_yticks([])
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "No direction data",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax2.set_xticks([])
                ax2.set_yticks([])

            canvas2.draw()

            # Schedule next update if window is still open
            try:
                parent.after(1000, update_spatial)
            except:
                pass  # Window was closed

        # Start update loop
        parent.after(1000, update_spatial)

    def setup_speed_tab(self, parent):
        """Setup the speed analysis dashboard tab"""
        # Create a grid layout
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Top frame for speed chart
        speed_frame = ttk.LabelFrame(frame, text="Average Speed by Vehicle Type")
        speed_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # Create horizontal bar chart for speeds
        fig1 = Figure(figsize=(6, 4), dpi=100)
        fig1.patch.set_facecolor(self.plot_bg)
        ax1 = fig1.add_subplot(111)
        ax1.set_facecolor(self.plot_bg)
        ax1.tick_params(colors=self.plot_fg)

        # Check if we have speed data
        if any(len(speeds) > 0 for speeds in self.avg_speeds.values()):
            # Compute average speeds for each vehicle class
            avg_speed_data = {}
            for class_name, speeds in self.avg_speeds.items():
                if speeds:
                    avg_speed_data[class_name] = sum(speeds) / len(speeds)

            if avg_speed_data:
                # Sort by speed
                sorted_speeds = sorted(
                    avg_speed_data.items(), key=lambda x: x[1], reverse=True
                )

                classes = [item[0] for item in sorted_speeds]
                speeds = [item[1] for item in sorted_speeds]

                # Create horizontal bar chart
                bars = ax1.barh(classes, speeds, color=self.accent_color)

                # Add speed values
                for i, bar in enumerate(bars):
                    width = bar.get_width()
                    ax1.text(
                        width + 0.5,
                        bar.get_y() + bar.get_height() / 2,
                        f"{width:.1f} km/h",
                        va="center",
                        color=self.plot_fg,
                    )

                ax1.set_xlabel("Speed (km/h)", color=self.plot_fg)
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "Processing speed data...",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax1.set_xticks([])
                ax1.set_yticks([])
        else:
            ax1.text(
                0.5,
                0.5,
                "Insufficient data for speed analysis",
                ha="center",
                va="center",
                color=self.plot_fg,
            )
            ax1.set_xticks([])
            ax1.set_yticks([])

        canvas1 = FigureCanvasTkAgg(fig1, master=speed_frame)
        canvas1.draw()
        canvas1.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Bottom frame for speed distribution
        dist_frame = ttk.LabelFrame(frame, text="Speed Distribution")
        dist_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        # Create histogram for speed distribution
        fig2 = Figure(figsize=(6, 4), dpi=100)
        fig2.patch.set_facecolor(self.plot_bg)
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor(self.plot_bg)
        ax2.tick_params(colors=self.plot_fg)

        # Combine all speed data
        all_speeds = []
        for speeds in self.avg_speeds.values():
            all_speeds.extend(speeds)

        if all_speeds:
            # Create histogram
            ax2.hist(
                all_speeds,
                bins=min(20, len(all_speeds) // 5 + 1),
                color=self.accent_color,
                alpha=0.7,
            )
            ax2.set_xlabel("Speed (km/h)", color=self.plot_fg)
            ax2.set_ylabel("Frequency", color=self.plot_fg)

            # Add vertical line for average speed
            avg_speed = sum(all_speeds) / len(all_speeds)
            ax2.axvline(x=avg_speed, color="red", linestyle="--", linewidth=2)
            ax2.text(
                avg_speed + 0.5,
                ax2.get_ylim()[1] * 0.9,
                f"Avg: {avg_speed:.1f} km/h",
                color="red",
            )
        else:
            ax2.text(
                0.5,
                0.5,
                "Insufficient data for speed distribution",
                ha="center",
                va="center",
                color=self.plot_fg,
            )
            ax2.set_xticks([])
            ax2.set_yticks([])

        canvas2 = FigureCanvasTkAgg(fig2, master=dist_frame)
        canvas2.draw()
        canvas2.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Right frame for speed statistics
        stats_frame = ttk.LabelFrame(frame, text="Speed Statistics")
        stats_frame.grid(row=0, column=1, rowspan=2, padx=5, pady=5, sticky="nsew")

        # Create text widget for stats
        stats_text = tk.Text(stats_frame, width=30, bg=self.plot_bg, fg=self.plot_fg)
        stats_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Calculate and display speed statistics
        if all_speeds:
            avg_speed = sum(all_speeds) / len(all_speeds)
            median_speed = sorted(all_speeds)[len(all_speeds) // 2] if all_speeds else 0
            max_speed = max(all_speeds) if all_speeds else 0
            min_speed = min(all_speeds) if all_speeds else 0

            # Calculate percentiles
            p85 = (
                sorted(all_speeds)[int(len(all_speeds) * 0.85)]
                if len(all_speeds) > 1
                else 0
            )
            p95 = (
                sorted(all_speeds)[int(len(all_speeds) * 0.95)]
                if len(all_speeds) > 1
                else 0
            )

            # Calculate std deviation
            if len(all_speeds) > 1:
                std_dev = math.sqrt(
                    sum((x - avg_speed) ** 2 for x in all_speeds) / len(all_speeds)
                )
            else:
                std_dev = 0

            stats_text.insert(tk.END, "Speed Statistics:\n\n")
            stats_text.insert(tk.END, f"Number of measurements: {len(all_speeds)}\n\n")
            stats_text.insert(tk.END, f"Average speed: {avg_speed:.1f} km/h\n")
            stats_text.insert(tk.END, f"Median speed: {median_speed:.1f} km/h\n")
            stats_text.insert(tk.END, f"Maximum speed: {max_speed:.1f} km/h\n")
            stats_text.insert(tk.END, f"Minimum speed: {min_speed:.1f} km/h\n\n")
            stats_text.insert(tk.END, f"Standard deviation: {std_dev:.2f} km/h\n")
            stats_text.insert(tk.END, f"85th percentile: {p85:.1f} km/h\n")
            stats_text.insert(tk.END, f"95th percentile: {p95:.1f} km/h\n\n")

            # Add speed classification
            stats_text.insert(tk.END, "Speed Classification:\n\n")

            # Count vehicles in different speed ranges
            speed_ranges = {
                "Very slow (0-20 km/h)": len([s for s in all_speeds if 0 <= s < 20]),
                "Slow (20-40 km/h)": len([s for s in all_speeds if 20 <= s < 40]),
                "Medium (40-60 km/h)": len([s for s in all_speeds if 40 <= s < 60]),
                "Fast (60-80 km/h)": len([s for s in all_speeds if 60 <= s < 80]),
                "Very fast (80+ km/h)": len([s for s in all_speeds if s >= 80]),
            }

            for range_name, count in speed_ranges.items():
                percentage = (count / len(all_speeds) * 100) if all_speeds else 0
                stats_text.insert(
                    tk.END, f"{range_name}: {count} ({percentage:.1f}%)\n"
                )
        else:
            stats_text.insert(tk.END, "Insufficient data for speed statistics")

        stats_text.config(state=tk.DISABLED)  # Make read-only

        # Configure grid weights
        frame.columnconfigure(0, weight=2)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        # Update function for real-time updates
        def update_speed():
            # Update speed chart
            ax1.clear()
            ax1.set_facecolor(self.plot_bg)
            ax1.tick_params(colors=self.plot_fg)

            # Check if we have speed data
            if any(len(speeds) > 0 for speeds in self.avg_speeds.values()):
                # Compute average speeds for each vehicle class
                avg_speed_data = {}
                for class_name, speeds in self.avg_speeds.items():
                    if speeds:
                        avg_speed_data[class_name] = sum(speeds) / len(speeds)

                if avg_speed_data:
                    # Sort by speed
                    sorted_speeds = sorted(
                        avg_speed_data.items(), key=lambda x: x[1], reverse=True
                    )

                    classes = [item[0] for item in sorted_speeds]
                    speeds = [item[1] for item in sorted_speeds]

                    # Create horizontal bar chart
                    bars = ax1.barh(classes, speeds, color=self.accent_color)

                    # Add speed values
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax1.text(
                            width + 0.5,
                            bar.get_y() + bar.get_height() / 2,
                            f"{width:.1f} km/h",
                            va="center",
                            color=self.plot_fg,
                        )

                    ax1.set_xlabel("Speed (km/h)", color=self.plot_fg)
                else:
                    ax1.text(
                        0.5,
                        0.5,
                        "Processing speed data...",
                        ha="center",
                        va="center",
                        color=self.plot_fg,
                    )
                    ax1.set_xticks([])
                    ax1.set_yticks([])
            else:
                ax1.text(
                    0.5,
                    0.5,
                    "Insufficient data for speed analysis",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax1.set_xticks([])
                ax1.set_yticks([])

            canvas1.draw()

            # Update speed distribution
            ax2.clear()
            ax2.set_facecolor(self.plot_bg)
            ax2.tick_params(colors=self.plot_fg)

            # Combine all speed data
            all_speeds = []
            for speeds in self.avg_speeds.values():
                all_speeds.extend(speeds)

            if all_speeds:
                # Create histogram
                bins = min(20, max(5, len(all_speeds) // 5 + 1))
                ax2.hist(all_speeds, bins=bins, color=self.accent_color, alpha=0.7)
                ax2.set_xlabel("Speed (km/h)", color=self.plot_fg)
                ax2.set_ylabel("Frequency", color=self.plot_fg)

                # Add vertical line for average speed
                avg_speed = sum(all_speeds) / len(all_speeds)
                ax2.axvline(x=avg_speed, color="red", linestyle="--", linewidth=2)
                ax2.text(
                    avg_speed + 0.5,
                    ax2.get_ylim()[1] * 0.9,
                    f"Avg: {avg_speed:.1f} km/h",
                    color="red",
                )
            else:
                ax2.text(
                    0.5,
                    0.5,
                    "Insufficient data for speed distribution",
                    ha="center",
                    va="center",
                    color=self.plot_fg,
                )
                ax2.set_xticks([])
                ax2.set_yticks([])

            canvas2.draw()

            # Update speed statistics
            stats_text.config(state=tk.NORMAL)
            stats_text.delete(1.0, tk.END)

            if all_speeds:
                avg_speed = sum(all_speeds) / len(all_speeds)
                median_speed = (
                    sorted(all_speeds)[len(all_speeds) // 2] if all_speeds else 0
                )
                max_speed = max(all_speeds) if all_speeds else 0
                min_speed = min(all_speeds) if all_speeds else 0

                # Calculate percentiles
                p85 = (
                    sorted(all_speeds)[int(len(all_speeds) * 0.85)]
                    if len(all_speeds) > 1
                    else 0
                )
                p95 = (
                    sorted(all_speeds)[int(len(all_speeds) * 0.95)]
                    if len(all_speeds) > 1
                    else 0
                )

                # Calculate std deviation
                if len(all_speeds) > 1:
                    std_dev = math.sqrt(
                        sum((x - avg_speed) ** 2 for x in all_speeds) / len(all_speeds)
                    )
                else:
                    std_dev = 0

                stats_text.insert(tk.END, "Speed Statistics:\n\n")
                stats_text.insert(
                    tk.END, f"Number of measurements: {len(all_speeds)}\n\n"
                )
                stats_text.insert(tk.END, f"Average speed: {avg_speed:.1f} km/h\n")
                stats_text.insert(tk.END, f"Median speed: {median_speed:.1f} km/h\n")
                stats_text.insert(tk.END, f"Maximum speed: {max_speed:.1f} km/h\n")
                stats_text.insert(tk.END, f"Minimum speed: {min_speed:.1f} km/h\n\n")
                stats_text.insert(tk.END, f"Standard deviation: {std_dev:.2f} km/h\n")
                stats_text.insert(tk.END, f"85th percentile: {p85:.1f} km/h\n")
                stats_text.insert(tk.END, f"95th percentile: {p95:.1f} km/h\n\n")

                # Add speed classification
                stats_text.insert(tk.END, "Speed Classification:\n\n")

                # Count vehicles in different speed ranges
                speed_ranges = {
                    "Very slow (0-20 km/h)": len(
                        [s for s in all_speeds if 0 <= s < 20]
                    ),
                    "Slow (20-40 km/h)": len([s for s in all_speeds if 20 <= s < 40]),
                    "Medium (40-60 km/h)": len([s for s in all_speeds if 40 <= s < 60]),
                    "Fast (60-80 km/h)": len([s for s in all_speeds if 60 <= s < 80]),
                    "Very fast (80+ km/h)": len([s for s in all_speeds if s >= 80]),
                }

                for range_name, count in speed_ranges.items():
                    percentage = (count / len(all_speeds) * 100) if all_speeds else 0
                    stats_text.insert(
                        tk.END, f"{range_name}: {count} ({percentage:.1f}%)\n"
                    )
            else:
                stats_text.insert(tk.END, "Insufficient data for speed statistics")

            stats_text.config(state=tk.DISABLED)

            # Schedule next update if window is still open
            try:
                parent.after(1000, update_speed)
            except:
                pass  # Window was closed

        # Start update loop
        parent.after(1000, update_speed)

    def show_speed_analysis(self):
        """Show dedicated speed analysis window"""
        self.show_dashboard()  # For now, just show the dashboard
        self.viz_notebook.select(self.speed_tab)

    def show_density_analysis(self):
        """Show dedicated density analysis window"""
        self.show_dashboard()  # For now, just show the dashboard
        self.viz_notebook.select(self.spatial_tab)

    def show_direction_analysis(self):
        """Show dedicated direction analysis window"""
        self.show_dashboard()  # For now, just show the dashboard
        self.viz_notebook.select(self.direction_tab)

    def show_time_analysis(self):
        """Show dedicated temporal analysis window"""
        self.show_dashboard()  # For now, just show the dashboard
        self.viz_notebook.select(self.temporal_tab)

    def show_instructions(self):
        """Show application instructions"""
        instructions = """
        Traffic Analysis Application Instructions
        
        Getting Started:
        1. Load a YOLO model using the "Load Model" button or from the File menu
        2. Select a video file for analysis
        3. Press "Play" to start the analysis
        
        Analysis Zones:
        • Add counting lines to track vehicles crossing specific boundaries
        • Click and drag on the video to position counting lines
        
        Display Options:
        • Use the View menu to toggle different visualization elements
        • Switch between dark and light mode for better visibility
        
        Performance Settings:
        • Adjust frame skip to balance between performance and smoothness
        • Change detection threshold to control sensitivity
        • Select different model sizes based on your computer's capabilities
        
        Analytics:
        • View real-time analytics in the visualization tabs
        • Generate comprehensive reports for detailed analysis
        • Export data for further processing in other applications
        
        Tips for CPU Optimization:
        • Use YOLOv8 Nano model for better performance
        • Increase frame skip on slower systems
        • Reduce input video resolution for faster processing
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
        Advanced Traffic Analysis System
        Version 1.0
        
        A comprehensive traffic monitoring and analysis application 
        using YOLOv8 for real-time vehicle detection and tracking.
        
        Features:
        • Real-time vehicle detection and classification
        • Traffic flow analysis with counting lines
        • Speed estimation and direction tracking
        • Heatmap visualization of traffic density
        • Comprehensive analytics dashboard
        • Data export and reporting
        
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
    app = EnhancedTrafficAnalysisApp(root)

    # Set up closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start the main loop
    root.mainloop()
