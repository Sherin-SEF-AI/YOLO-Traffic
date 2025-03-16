import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import time
import threading
from collections import defaultdict, deque
import torch
from ultralytics import YOLO


class TrafficAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Analysis with YOLOv8")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)  # Set minimum window size

        # YOLO model and settings
        self.model = None
        self.model_loaded = False
        self.class_names = {}
        self.detection_threshold = 0.5

        # Define vehicle classes early to avoid attribute errors
        self.vehicle_classes = [2, 3, 5, 7]  # car, truck, bus, motorcycle

        # Video processing variables
        self.video_path = None
        self.cap = None
        self.is_playing = False
        self.processing_thread = None
        self.update_id = None  # For tracking Tkinter's after() calls

        # Analysis results
        self.detected_objects = defaultdict(int)
        self.time_series_data = defaultdict(list)
        self.timestamps = []
        self.frame_count = 0
        self.start_time = 0

        # History for tracking
        self.object_history = {}
        self.history_window = 30
        self.next_object_id = 0

        # For vehicle counting
        self.count_line_y = 400
        self.counted_objects = set()
        self.vehicle_count = 0

        # GUI update control
        self.last_gui_update = 0
        self.gui_update_interval = 0.03  # Seconds between GUI updates (33ms ≈ 30fps)

        # Buffer for processed frames
        self.processed_frames = deque(maxlen=5)

        # CPU optimization settings
        self.frame_skip = 2  # Process every other frame for better performance
        self.current_skip_count = 0
        self.downscale_factor = 1.0  # Can set to 0.5 to further reduce processing load

        # Fixed dimensions for video and visualization
        self.video_width = 640
        self.video_height = 480
        self.viz_width = 480
        self.viz_height = 640

        # Setup UI
        self.setup_ui()

        # Initialize plotting
        self.setup_plots()

        # Debug flag
        self.debug = True

    def setup_ui(self):
        # Main layout frames
        self.control_frame = ttk.Frame(self.root, padding=10)
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        self.content_frame = ttk.Frame(self.root)
        self.content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left panel for video with fixed size
        self.video_frame = ttk.LabelFrame(
            self.content_frame, text="Video Feed", padding=10
        )
        self.video_frame.pack(side=tk.LEFT, padx=5, pady=5)

        # Fixed size for video canvas
        self.video_canvas = tk.Canvas(
            self.video_frame,
            bg="black",
            width=self.video_width,
            height=self.video_height,
        )
        self.video_canvas.pack()

        # Bind click event for setting count line
        self.video_canvas.bind("<Button-1>", self.set_count_line)

        # Right panel for visualizations with fixed size
        self.viz_frame = ttk.LabelFrame(
            self.content_frame, text="Visualizations", padding=10
        )
        self.viz_frame.pack(side=tk.RIGHT, padx=5, pady=5)

        # Control buttons
        ttk.Button(self.control_frame, text="Load Model", command=self.load_model).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(
            self.control_frame, text="Select Video", command=self.select_video
        ).pack(side=tk.LEFT, padx=5)

        self.play_button = ttk.Button(
            self.control_frame, text="Play", command=self.toggle_play, state=tk.DISABLED
        )
        self.play_button.pack(side=tk.LEFT, padx=5)

        # Add a button to show a single frame for debugging
        ttk.Button(
            self.control_frame, text="Show Frame", command=self.show_single_frame
        ).pack(side=tk.LEFT, padx=5)

        # CPU optimization controls
        ttk.Label(self.control_frame, text="Frame Skip:").pack(side=tk.LEFT, padx=5)
        self.frame_skip_var = tk.IntVar(value=self.frame_skip)
        frame_skip_spinbox = ttk.Spinbox(
            self.control_frame,
            from_=1,
            to=10,
            width=5,
            textvariable=self.frame_skip_var,
            command=self.update_frame_skip,
        )
        frame_skip_spinbox.pack(side=tk.LEFT)

        ttk.Label(self.control_frame, text="Detection Threshold:").pack(
            side=tk.LEFT, padx=5
        )
        self.threshold_var = tk.DoubleVar(value=0.5)
        threshold_slider = ttk.Scale(
            self.control_frame,
            from_=0.1,
            to=0.9,
            variable=self.threshold_var,
            orient=tk.HORIZONTAL,
            length=100,
            command=self.update_threshold,
        )
        threshold_slider.pack(side=tk.LEFT, padx=5)

        self.threshold_label = ttk.Label(self.control_frame, text="0.50")
        self.threshold_label.pack(side=tk.LEFT, padx=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready. Please load a YOLO model.")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # Object count display
        self.count_frame = ttk.LabelFrame(self.control_frame, text="Vehicle Count")
        self.count_frame.pack(side=tk.RIGHT, padx=10)
        self.count_var = tk.StringVar(value="0")
        count_label = ttk.Label(
            self.count_frame, textvariable=self.count_var, font=("Arial", 16, "bold")
        )
        count_label.pack(padx=10, pady=5)

    def update_frame_skip(self):
        try:
            self.frame_skip = self.frame_skip_var.get()
            self.status_var.set(
                f"Frame skip set to {self.frame_skip} (higher = faster but less smooth)"
            )
        except Exception as e:
            if self.debug:
                print(f"Error updating frame skip: {str(e)}")

    def setup_plots(self):
        # Create matplotlib figure with fixed size and proper padding
        plt.rcParams.update({"figure.autolayout": True})  # Enable auto layout
        self.fig = plt.figure(
            figsize=(self.viz_width / 100, self.viz_height / 100), dpi=100
        )

        # Create subplots with proper spacing
        self.ax1 = self.fig.add_subplot(211)  # Top subplot
        self.ax2 = self.fig.add_subplot(212)  # Bottom subplot

        # Add more padding
        self.fig.subplots_adjust(
            left=0.15, right=0.95, bottom=0.1, top=0.95, hspace=0.3
        )

        # Object count bar chart
        self.ax1.set_title("Detected Objects")
        self.ax1.set_xlabel("Class")
        self.ax1.set_ylabel("Count")

        # Time series plot
        self.ax2.set_title("Objects Over Time")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Count")

        # Embed in tkinter with fixed size
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.viz_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.config(width=self.viz_width, height=self.viz_height)
        self.canvas_widget.pack()

    def load_model(self):
        try:
            self.status_var.set("Loading YOLOv8 model (may take a moment)...")
            self.root.update()

            # For CPU optimization, use the nano model instead of small
            self.model = YOLO("yolov8n.pt")  # Smaller, faster model for CPU

            # Get class names
            self.class_names = self.model.names

            self.model_loaded = True

            # Check if using CPU or GPU (will be CPU for your system)
            device = "GPU" if torch.cuda.is_available() else "CPU"
            self.status_var.set(
                f"Model loaded successfully. Using {device}. Ready to analyze videos."
            )

        except Exception as e:
            self.status_var.set(f"Error loading model: {str(e)}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def select_video(self):
        video_path = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mov"), ("All files", "*.*")]
        )

        if video_path:
            # Close any previously opened video
            if self.cap is not None:
                self.cap.release()
                self.cap = None

            self.video_path = video_path
            self.status_var.set(f"Video selected: {os.path.basename(video_path)}")

            # Reset analysis data
            self.reset_analysis_data()

            # Enable play button if model is loaded
            if self.model_loaded:
                self.play_button.config(state=tk.NORMAL)

            # Try to open the video to show the first frame
            try:
                self.cap = cv2.VideoCapture(self.video_path)
                if not self.cap.isOpened():
                    self.status_var.set(
                        f"Error: Could not open video file: {self.video_path}"
                    )
                    return

                ret, frame = self.cap.read()
                if ret:
                    # Set counting line at middle of the frame by default
                    self.count_line_y = self.video_height // 2

                    # Resize frame to fit canvas (fixed size)
                    frame = self.resize_frame(frame)
                    self.show_frame(frame)
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
        self.vehicle_count = 0
        self.counted_objects = set()
        self.object_history = {}
        self.next_object_id = 0
        self.count_var.set("0")

    def resize_frame(self, frame):
        if frame is None:
            return None

        try:
            # Always resize to fixed dimensions for the video canvas
            return cv2.resize(frame, (self.video_width, self.video_height))
        except Exception as e:
            self.status_var.set(f"Error resizing frame: {str(e)}")
            if self.debug:
                import traceback

                traceback.print_exc()
            return frame  # Return original frame if resize fails

    def show_frame(self, frame):
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

            # Create image centered in the canvas
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.video_canvas.image = img_tk  # Keep a reference

            # Draw counting line
            self.video_canvas.create_line(
                0,
                self.count_line_y,
                self.video_width,
                self.count_line_y,
                fill="red",
                width=2,
                dash=(5, 5),
            )

            # Add label for the counting line
            self.video_canvas.create_text(
                10,
                self.count_line_y - 10,
                text="Counting Line (click to reposition)",
                anchor=tk.W,
                fill="white",
                tags="line_label",
            )
        except Exception as e:
            self.status_var.set(f"Error displaying frame: {str(e)}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def show_single_frame(self):
        """Debug function to show a single frame from the video"""
        if self.cap is None or not self.cap.isOpened():
            self.status_var.set("No video loaded or video cannot be read.")
            return

        try:
            # Read a frame
            ret, frame = self.cap.read()
            if not ret:
                # Try to reopen the video if we reached the end
                self.cap.release()
                self.cap = cv2.VideoCapture(self.video_path)
                ret, frame = self.cap.read()
                if not ret:
                    self.status_var.set("Could not read frame from video.")
                    return

            # Process frame if model is loaded
            if self.model_loaded:
                processed_frame = self.process_frame(frame)
            else:
                processed_frame = frame

            # Display the frame
            resized_frame = self.resize_frame(processed_frame)
            self.show_frame(resized_frame)
            self.status_var.set(f"Showing frame {self.frame_count}")

        except Exception as e:
            self.status_var.set(f"Error showing single frame: {str(e)}")
            if self.debug:
                import traceback

                traceback.print_exc()

    def set_count_line(self, event):
        # Set new position for counting line
        self.count_line_y = event.y

        # Redraw the current frame if available
        if self.cap is not None and not self.is_playing:
            try:
                # Store current position
                current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                # Go back one frame
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)

                # Read and show frame
                ret, frame = self.cap.read()
                if ret:
                    if self.model_loaded:
                        frame = self.process_frame(frame)
                    frame = self.resize_frame(frame)
                    self.show_frame(frame)

                    # Restore position
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            except Exception as e:
                self.status_var.set(f"Error updating count line: {str(e)}")
                if self.debug:
                    import traceback

                    traceback.print_exc()

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
            self.start_time = time.time() - (self.frame_count / 30)  # Assuming 30 fps
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
        self.detection_threshold = float(value)
        self.threshold_label.config(text=f"{self.detection_threshold:.2f}")

    def process_video(self):
        """Thread function for processing video frames"""
        try:
            while self.is_playing and self.cap is not None and self.cap.isOpened():
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
                    continue

                # Optional: Downscale frame for faster processing
                if self.downscale_factor != 1.0:
                    h, w = frame.shape[:2]
                    new_h, new_w = int(h * self.downscale_factor), int(
                        w * self.downscale_factor
                    )
                    small_frame = cv2.resize(frame, (new_w, new_h))
                    processed_frame = self.process_frame(small_frame)
                    # Resize back to original size for display
                    processed_frame = cv2.resize(processed_frame, (w, h))
                else:
                    # Process frame with YOLO at original size
                    processed_frame = self.process_frame(frame)

                # Add to our frame queue for the GUI thread to pick up
                self.processed_frames.append(processed_frame)

                # Small sleep to reduce CPU usage
                time.sleep(0.05)  # Increased sleep time for CPU systems

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
                    resized_frame = self.resize_frame(frame)
                    self.show_frame(resized_frame)

                # Update plots occasionally (less frequently for CPU systems)
                if (
                    self.frame_count % 30 == 0
                ):  # Reduced from every 10 frames to every 30
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

    def process_frame(self, frame):
        if frame is None or not self.model_loaded:
            return frame

        try:
            # Increment frame counter
            self.frame_count += 1

            # Get current timestamp
            current_time = time.time() - self.start_time

            # Create a copy for drawing
            output_frame = frame.copy()

            # Calculate count line position relative to frame height
            frame_height = frame.shape[0]
            frame_width = frame.shape[1]
            # Scale the count line position if the frame is not the same size as the canvas
            relative_y = int(self.count_line_y * (frame_height / self.video_height))

            # Draw counting line
            cv2.line(
                output_frame,
                (0, relative_y),
                (frame_width, relative_y),
                (0, 0, 255),
                2,
                cv2.LINE_AA,
                0,
            )

            # Run detection with YOLOv8 - use CPU-optimized settings
            results = self.model(
                frame, conf=self.detection_threshold, verbose=False
            )  # Disable verbose output

            # Process each detection for this frame
            current_detections = []

            # YOLOv8 results structure
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    try:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                        # Get class and confidence
                        class_id = int(box.cls[0].item())
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

                        # Draw bounding box and label
                        color = (0, 255, 0)  # Green for vehicles
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

                        # Create label
                        label = f"{class_name}: {confidence:.2f}"

                        # Draw label background
                        text_size, _ = cv2.getTextSize(
                            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                        )
                        cv2.rectangle(
                            output_frame,
                            (x1, y1 - text_size[1] - 5),
                            (x1 + text_size[0], y1),
                            color,
                            -1,
                        )

                        # Draw label text
                        cv2.putText(
                            output_frame,
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

            # Track objects across frames - use relative counting line position
            self.track_objects(current_detections, current_time, relative_y)

            # Update status less frequently to reduce CPU load
            if (
                self.frame_count % 30 == 0
            ):  # Update every 30 frames instead of every frame
                self.status_var.set(
                    f"Processing frame {self.frame_count} | "
                    f"Time: {current_time:.1f}s | "
                    f"Vehicle count: {self.vehicle_count}"
                )

            # Update the count display
            self.count_var.set(str(self.vehicle_count))

            return output_frame

        except Exception as e:
            self.status_var.set(f"Error processing frame: {str(e)}")
            if self.debug:
                import traceback

                traceback.print_exc()
            return frame  # Return original frame if processing fails

    def track_objects(self, current_detections, current_time, count_line_y):
        try:
            # If no history yet, initialize with current detections
            if not self.object_history:
                for det in current_detections:
                    if det["class_id"] in self.vehicle_classes:  # Only track vehicles
                        obj_id = self.next_object_id
                        self.next_object_id += 1
                        self.object_history[obj_id] = {
                            "positions": deque(maxlen=self.history_window),
                            "class_id": det["class_id"],
                            "class_name": det["class_name"],
                            "counted": False,
                        }
                        self.object_history[obj_id]["positions"].append(det["center"])
                return

            # Match current detections with existing tracked objects
            matched_indices = set()

            for det in current_detections:
                if det["class_id"] not in self.vehicle_classes:
                    continue

                center = det["center"]
                min_dist = float("inf")
                best_match = None

                # Find the closest tracked object
                for obj_id, obj_data in self.object_history.items():
                    if obj_id in matched_indices:
                        continue

                    if obj_data["class_id"] != det["class_id"]:
                        continue

                    if not obj_data["positions"]:
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
                    matched_indices.add(best_match)

                    # Check if object crossed the counting line
                    positions = self.object_history[best_match]["positions"]
                    if len(positions) >= 2:
                        prev_y = positions[-2][1]
                        curr_y = positions[-1][1]

                        # Check if object crossed the line from top to bottom
                        if prev_y <= count_line_y and curr_y > count_line_y:
                            if not self.object_history[best_match]["counted"]:
                                self.vehicle_count += 1
                                self.object_history[best_match]["counted"] = True
                else:
                    # Create a new tracked object
                    obj_id = self.next_object_id
                    self.next_object_id += 1
                    self.object_history[obj_id] = {
                        "positions": deque(maxlen=self.history_window),
                        "class_id": det["class_id"],
                        "class_name": det["class_name"],
                        "counted": False,
                    }
                    self.object_history[obj_id]["positions"].append(center)

            # Record data for time series plot
            self.timestamps.append(current_time)

            # Count visible objects of each class for the time series
            class_counts = defaultdict(int)
            for obj_id, obj_data in self.object_history.items():
                if len(obj_data["positions"]) > 0:  # Object is currently visible
                    class_counts[obj_data["class_name"]] += 1

            # Update time series data for all classes we're tracking
            for class_id in self.vehicle_classes:
                if class_id in self.class_names:
                    class_name = self.class_names[class_id]
                    self.time_series_data[class_name].append(class_counts[class_name])

            # Prune history to remove objects not seen recently
            to_remove = []
            for obj_id in self.object_history:
                if len(self.object_history[obj_id]["positions"]) == 0:
                    to_remove.append(obj_id)

            for obj_id in to_remove:
                del self.object_history[obj_id]

        except Exception as e:
            if self.debug:
                print(f"Error in object tracking: {str(e)}")
                import traceback

                traceback.print_exc()

    def update_plots(self):
        try:
            # Clear the axes
            self.ax1.clear()
            self.ax2.clear()

            # Bar chart of detected objects (top 5)
            sorted_objects = sorted(
                self.detected_objects.items(), key=lambda x: x[1], reverse=True
            )[:5]

            if sorted_objects:  # Only plot if we have data
                labels = [item[0] for item in sorted_objects]
                values = [item[1] for item in sorted_objects]

                self.ax1.bar(labels, values, color="skyblue")
                self.ax1.set_title("Top 5 Detected Objects")
                self.ax1.set_ylabel("Count")

                # Rotate labels if they're long
                if any(len(label) > 6 for label in labels):
                    self.ax1.set_xticklabels(labels, rotation=45, ha="right")

            # Time series plot (only vehicles)
            vehicle_classes = []
            for class_id in self.vehicle_classes:
                if class_id in self.class_names:
                    vehicle_classes.append(self.class_names[class_id])

            # Make sure we have data
            if self.timestamps:
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
                            self.ax2.plot(timestamps, values, label=class_name)

            self.ax2.set_title("Vehicles Over Time")
            self.ax2.set_xlabel("Time (s)")
            self.ax2.set_ylabel("Count")

            # Only add legend if we have data
            if any(
                len(self.time_series_data.get(class_name, [])) > 0
                for class_name in vehicle_classes
            ):
                self.ax2.legend(loc="upper left")

            # Update the canvas - avoid using tight_layout() which was causing problems
            self.canvas.draw()

        except Exception as e:
            if self.debug:
                print(f"Error updating plots: {str(e)}")
                import traceback

                traceback.print_exc()

    def on_closing(self):
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

    # Create the app
    app = TrafficAnalysisApp(root)

    # Set up closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    # Start the main loop
    root.mainloop()
