import argparse
import os
from typing import List, Tuple, Dict
import cv2
import matplotlib
import numpy as np
import torch
from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from enum import Enum

matplotlib.use("TkAgg")

# SAM2 imports
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# FIXME: Nidhish: Import SegType & SEGMENTATION_PALETTE from mt/nn
class SegType(int, Enum):
    """
    Base class for SegType enums.
    """
    BACKGROUND = 0
    HARD_FLOOR = 1
    CARPET = 2
    WIRE = 3
    CH = 4
    BAG_STRAP = 5
    SHOE_LACE = 6
    TASSELS = 7
    PUKE = 8
    POOP = 9
    SPILL = 10

@dataclass
class ObjectTracker:
    id: int
    label: str
    color: Tuple[float, float, float]
    masks: Dict[int, np.ndarray]  # frame_idx -> mask
    points: Dict[int, List[Tuple[float, float, int]]]  # frame_idx -> [(x, y, label)]

class MultiObjectTrackingGUI:   
    def __init__(self, image_dir: Path, mask_dir: Path, max_points_per_frame: int = 50):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.max_points_per_frame = max_points_per_frame
        
        # Initialize predefined labels
        self.seg_types = {item.value: item.name.lower() for item in SegType}
        
        # Initialize tracking state
        self.objects: Dict[int, ObjectTracker] = {}
        self.next_object_id = 1
        self.current_object_id = None
        self.current_index = 0
        self.dirty = False
        
        # Load images and scan for existing masks
        self.setup_image_paths()
        self.load_existing_masks()
        
        # Load SAM2 model
        self.setup_sam2()
        
        # Initialize display
        self.setup_display()
        self.setup_gui_controls()
        
        # Show first frame
        self.show_image(0)

        # Create initial objects for all SegTypes if they don't exist
        self.initialize_seg_type_objects()

        # # update dropdown
        self.update_object_dropdown()

    def initialize_seg_type_objects(self):
        """Initialize objects for all SegTypes if they don't exist"""

        SEGMENTATION_PALETTE = {
            "invalid": (0, 0, 0),
            "background": (50, 50, 50),
            "hard_floor": (252, 194, 10),
            "carpet": (135, 17, 8),
            "wire": (0, 255, 255),
            "ch": (255, 0, 0),
            "bag_strap": (30, 100, 190),
            "shoe_lace": (211, 100, 60),
            "tassels": (255, 105, 180),
            "puke": (128, 128, 0),
            "poop": (121, 72, 47),
            "spill": (0, 145, 100),
        }

        for seg_id, seg_label in self.seg_types.items():
            if seg_id not in self.objects:
                color =  SEGMENTATION_PALETTE.get(seg_label, (np.random.random(3)*255))
                breakpoint()
                color = np.array(color) / 255.0 # If dict query was successful above we normalize to [0, 1]
                self.objects[seg_id] = ObjectTracker(
                    id=seg_id,
                    label=seg_label,
                    color=tuple(color),
                    masks={},
                    points={}
                )
        
        # Update next_object_id to be higher than all predefined IDs
        self.next_object_id = max(self.seg_types.keys()) + 1

    def load_existing_masks(self):
        """Scan mask directory and load all existing objects and their masks"""
        print("[INFO] Scanning for existing mask files...")
        
        # Track unique objects across all mask files
        object_info = {}  # obj_id -> (label, color)
        
        # Scan all NPZ files
        mask_files = sorted(self.mask_dir.glob("*.npz"))
        for mask_path in mask_files:
            try:
                mask_data = np.load(mask_path)
                frame_idx = int(mask_path.stem)  # Assuming filenames are frame indices
                
                # breakpoint()
                for key in mask_data.keys():
                    # if not key.startswith("mask_"):
                    #     continue
                        
                    # Parse object info from key (format: "mask_label_id")
                    parts = key.split("_")
                    # if len(parts) < 2:
                    #     continue
                        
                    obj_id = int(parts[-1])
                    if key.startswith("mask_"):
                        label = "_".join(parts[1:-1])  # Handle labels with underscores
                    else:
                        label = "_".join(parts[0:-1])  # Handle labels with underscores
                    
                    # Record object info if new
                    if obj_id not in object_info:
                        color = np.random.random(3)
                        object_info[obj_id] = (label, tuple(color))
                        
                        # Create new object tracker
                        self.objects[obj_id] = ObjectTracker(
                            id=obj_id,
                            label=label,
                            color=tuple(color),
                            masks={},
                            points={}
                        )
                    
                    # Store mask for this frame
                    self.objects[obj_id].masks[frame_idx] = mask_data[key]
                    
            except Exception as e:
                print(f"[WARNING] Error loading {mask_path}: {e}")
        
        # Update next_object_id
        if self.objects:
            self.next_object_id = max(self.objects.keys()) + 1
            print(f"[INFO] Loaded {len(self.objects)} existing objects")
            for obj_id, obj in self.objects.items():
                print(f"  - ID {obj_id}: {obj.label} ({len(obj.masks)} frames)")
        else:
            print("[INFO] No existing objects found")

    def setup_sam2(self):
        """Initialize SAM2 model and predictor"""
        SAM2_CHECKPOINT = "./checkpoints/sam2.1_hiera_large.pt"
        SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        
        print("[INFO] Loading SAM2 model...")
        # breakpoint()
        self.sam2_model = build_sam2(SAM2_MODEL_CONFIG, SAM2_CHECKPOINT, device=DEVICE)
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        print("[INFO] SAM2 model loaded successfully.")

    def setup_image_paths(self):
        """Load all image paths and validate"""
        self.image_paths = sorted(
            img for ext in ["*.png", "*.PNG", "*.jpg", "*.JPG"] 
            for img in self.image_dir.glob(ext)
        )
        if len(self.image_paths) == 0:
            raise ValueError(f"No image files found in {self.image_dir}")
        self.num_frames = len(self.image_paths)

    def setup_display(self):
        """Setup matplotlib display"""
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)
        
        # Create frame selector list
        self.root = tk.Tk()
        self.root.title("Multi-Object Tracking")
        
        # Frame navigation
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Frame slider
        self.frame_slider = ttk.Scale(
            nav_frame,
            from_=0,
            to=self.num_frames - 1,
            orient=tk.HORIZONTAL,
            command=self.on_slider_change
        )
        self.frame_slider.pack(fill=tk.X, expand=True)
        
        # Frame list
        frame_list = tk.Frame(self.root)
        frame_list.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(frame_list, orient=tk.VERTICAL)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.listbox = tk.Listbox(frame_list, yscrollcommand=scrollbar.set,
                                 selectmode=tk.SINGLE, activestyle='none')
        for i, path in enumerate(self.image_paths):
            self.listbox.insert(tk.END, f"{i:04d} - {path.name}")
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.listbox.yview)
        
        # Object controls - Top row
        obj_frame_top = tk.Frame(self.root)
        obj_frame_top.pack(fill=tk.X, padx=5, pady=2)
        
        # Object selection dropdown
        tk.Label(obj_frame_top, text="Current Object:").pack(side=tk.LEFT)
        self.object_var = tk.StringVar()
        self.object_dropdown = ttk.Combobox(
            obj_frame_top,
            textvariable=self.object_var,
            state='readonly',
            width=30
        )
        self.object_dropdown.pack(side=tk.LEFT, padx=5)
        self.object_dropdown.bind('<<ComboboxSelected>>', self.on_object_selected)
        
        # Object controls - Bottom row
        obj_frame = tk.Frame(self.root)
        obj_frame.pack(fill=tk.X, padx=5, pady=2)
        
        # Object ID entry
        tk.Label(obj_frame, text="New Object ID:").pack(side=tk.LEFT)
        self.obj_id_entry = tk.Entry(obj_frame, width=5)
        self.obj_id_entry.pack(side=tk.LEFT, padx=2)
        self.obj_id_entry.insert(0, str(self.next_object_id))
        
        # Object label entry
        tk.Label(obj_frame, text="Label:").pack(side=tk.LEFT, padx=2)
        self.obj_label_entry = tk.Entry(obj_frame, width=20)
        self.obj_label_entry.pack(side=tk.LEFT, padx=2)
        
        # Create new object button
        tk.Button(obj_frame, text="New Object", 
                 command=self.create_new_object).pack(side=tk.LEFT, padx=5)

    def setup_gui_controls(self):
        """Setup GUI controls and buttons"""
        # Save button
        save_ax = plt.axes([0.8, 0.05, 0.15, 0.075])
        self.save_button = Button(save_ax, "Save")
        self.save_button.on_clicked(self.on_save_click)
        
        # Connect event handlers
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.listbox.bind("<<ListboxSelect>>", self.on_list_select)

    def update_object_dropdown(self):
        """Update the object dropdown with current objects"""
        objects = [(obj.id, obj.label) for obj in self.objects.values()]
        objects.sort()  # Sort by ID
        
        # Update dropdown items
        self.object_dropdown['values'] = [
            f"{obj_id}: {label}" for obj_id, label in objects
        ]
        
        # Select current object if exists
        if self.current_object_id is not None:
            current_idx = next(
                (i for i, (obj_id, _) in enumerate(objects)
                 if obj_id == self.current_object_id),
                None
            )
            if current_idx is not None:
                self.object_dropdown.current(current_idx)

    def create_new_object(self):
        """Create a new object tracker"""
        try:
            obj_id = int(self.obj_id_entry.get())
            label = self.obj_label_entry.get() or f"object_{obj_id}"
            label = label.replace("_", "-")
            
            if obj_id in self.objects:
                print(f"[WARNING] Object ID {obj_id} already exists!")
                return
            
            # Don't allow overriding predefined SegType IDs
            if obj_id in self.seg_types:
                print(f"[WARNING] Cannot create object with reserved ID {obj_id}!")
                return
                
            # Generate random color for this object which isn't in SegType
            color = np.random.random(3)
            
            self.objects[obj_id] = ObjectTracker(
                id=obj_id,
                label=label,
                color=tuple(color),
                masks={},
                points={}
            )
            
            self.current_object_id = obj_id
            self.next_object_id = max(self.objects.keys()) + 1
            self.obj_id_entry.delete(0, tk.END)
            self.obj_id_entry.insert(0, str(self.next_object_id))
            
            # Update object dropdown
            self.update_object_dropdown()
            
            print(f"[INFO] Created new object: ID={obj_id}, Label={label}")
            
        except ValueError:
            print("[ERROR] Invalid object ID!")

    def on_object_selected(self, event):
        """Handle object selection from dropdown"""
        selection = self.object_dropdown.get()
        if selection:
            obj_id = int(selection.split(':')[0])
            self.current_object_id = obj_id
            self.redraw_overlay()

    def on_slider_change(self, value):
        """Handle frame slider movement"""
        idx = int(float(value))
        if idx != self.current_index:
            self.show_image(idx)
            # Highlight current frame in listbox
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(idx)
            self.listbox.see(idx)  # Ensure the selected item is visible

    def predict_mask_from_points(self, points: List[Tuple[float, float, int]]) -> np.ndarray:
        """Predict mask from list of points with labels"""
        if not points:
            return None
            
        # Split points into coordinates and labels
        coords = np.array([[x, y] for x, y, _ in points], dtype=np.float32)
        labels = np.array([label for _, _, label in points], dtype=np.int64)
        
        masks, scores, logits = self.sam2_predictor.predict(
            point_coords=coords,
            point_labels=labels,
            box=None,
            mask_input=None,
            multimask_output=False,
            return_logits=False,
            normalize_coords=True,
        )
        return masks[0].astype(bool)
    
    def add_point(self, x: float, y: float, label: int):
        """Add a point to current object's points list for current frame"""
        # breakpoint()
        if self.current_object_id is None:
            return
            
        obj = self.objects[self.current_object_id]
        
        # Initialize points list for current frame if needed
        if self.current_index not in obj.points:
            obj.points[self.current_index] = []
            
        # Add new point
        obj.points[self.current_index].append((x, y, label))
        
        # Implement FIFO if buffer is too large
        if len(obj.points[self.current_index]) > self.max_points_per_frame:
            obj.points[self.current_index].pop(0)  # Remove oldest point
        
        # Update mask using all points
        new_mask = self.predict_mask_from_points(obj.points[self.current_index])
        if new_mask is not None:
            obj.masks[self.current_index] = new_mask
            
        self.dirty = True
        # breakpoint()
        self.update_save_button()
        self.redraw_overlay()

    # def predict_mask_from_click(self, x: float, y: float) -> np.ndarray:
    #     """Predict mask from click coordinates"""
    #     point_coords = np.array([[x, y]], dtype=np.float32)
    #     point_labels = np.array([1], dtype=np.int64)
    #     masks, scores, logits = self.sam2_predictor.predict(
    #         point_coords=point_coords,
    #         point_labels=point_labels,
    #         box=None,
    #         mask_input=None,
    #         multimask_output=False,
    #         return_logits=False,
    #         normalize_coords=True,
    #     )
    #     return masks[0].astype(bool)

    def load_image_and_mask(self, idx: int):
        """Load image, existing masks, and points for frame idx"""
        # Load image
        bgr = cv2.imread(str(self.image_paths[idx]))
        if bgr is None:
            raise ValueError(f"Could not load image: {self.image_paths[idx]}")
            
        self.current_image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Set image in predictor (ensure no negative strides)
        bgr2rgb = bgr[..., ::-1].copy()
        self.sam2_predictor.set_image(bgr2rgb)
        
        # Load masks and points from NPZ if they exist
        mask_path = self.mask_dir / f"{self.image_paths[idx].stem}.npz"
        if mask_path.exists():
            data = np.load(mask_path, allow_pickle=True)
            for obj_id in self.objects:
                obj = self.objects[obj_id]
                
                # Load mask if it exists
                mask_key = f"mask_{obj.label}_{obj_id}"
                if mask_key in data:
                    obj.masks[idx] = data[mask_key]
                    
                # Load points if they exist
                points_key = f"points_{obj.label}_{obj_id}"
                if points_key in data:
                    points_array = data[points_key]
                    if len(points_array) > 0:  # Check if there are any points
                        obj.points[idx] = [tuple(p) for p in points_array]

    # def load_image_and_mask(self, idx: int):
    #     """Load image and existing masks for frame idx"""
    #     # Load image
    #     bgr = cv2.imread(str(self.image_paths[idx]))
    #     if bgr is None:
    #         raise ValueError(f"Could not load image: {self.image_paths[idx]}")
            
    #     self.current_image_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
    #     # Set image in predictor (ensure no negative strides)
    #     bgr2rgb = bgr[..., ::-1].copy()
    #     self.sam2_predictor.set_image(bgr2rgb)
        
    #     # Load masks from NPZ if they exist
    #     mask_path = self.mask_dir / f"{self.image_paths[idx].stem}.npz"
    #     if mask_path.exists():
    #         mask_data = np.load(mask_path)
    #         for obj_id in self.objects:
    #             if f"mask_{self.objects[obj_id].label}_{obj_id}" in mask_data:
    #                 self.objects[obj_id].masks[idx] = mask_data[f"mask_{self.objects[obj_id].label}_{obj_id}"]
    #             if f"mask_{obj_id}" in mask_data:
    #                 self.objects[obj_id].masks[idx] = mask_data[f"mask_{obj_id}"]

    def redraw_overlay(self):
        """Redraw the image with all object masks and points overlaid"""
        self.ax.clear()
        
        # Start with the base image
        overlaid = self.current_image_rgb.copy()
        
        # Add each object's mask with its color
        for obj in self.objects.values():
            if self.current_index in obj.masks:
                mask = obj.masks[self.current_index]
                overlaid[mask] = (1 - 0.5) * overlaid[mask] + 0.5 * np.array(obj.color)
                
            # Draw points
            if self.current_index in obj.points:
                for x, y, label in obj.points[self.current_index]:
                    color = 'g' if label == 1 else 'r'
                    self.ax.plot(x, y, color + 'o', markersize=5)
        
        fname = self.image_paths[self.current_index].name
        points_info = ""
        if self.current_object_id is not None and self.current_index in self.objects[self.current_object_id].points:
            num_points = len(self.objects[self.current_object_id].points[self.current_index])
            points_info = f" Points: {num_points}/{self.max_points_per_frame}"
            
        self.ax.set_title(
            f"{fname} [{self.current_index+1}/{self.num_frames}] "
            f"Current Object: {self.current_object_id}{points_info}"
        )
        self.ax.imshow(overlaid)
        self.ax.axis("off")
        self.fig.canvas.draw_idle()
        print("Redrew overlay")
    
    # def redraw_overlay(self):
    #     """Redraw the image with all object masks overlaid"""
    #     self.ax.clear()
        
    #     # Start with the base image
    #     overlaid = self.current_image_rgb.copy()
        
    #     # Add each object's mask with its color
    #     for obj in self.objects.values():
    #         if self.current_index in obj.masks:
    #             mask = obj.masks[self.current_index]
    #             overlaid[mask] = (1 - 0.5) * overlaid[mask] + 0.5 * np.array(obj.color)
        
    #     fname = self.image_paths[self.current_index].name
    #     self.ax.set_title(
    #         f"{fname} [{self.current_index+1}/{self.num_frames}] "
    #         f"Current Object: {self.current_object_id}"
    #     )
    #     self.ax.imshow(overlaid)
    #     self.ax.axis("off")
    #     self.fig.canvas.draw_idle()

    def show_image(self, idx: int):
        """Load and display frame idx"""
        self.current_index = idx
        self.load_image_and_mask(idx)
        self.dirty = False
        self.update_save_button()
        self.redraw_overlay()
        
        # Update slider position
        self.frame_slider.set(idx)
        
        # Highlight current frame in listbox
        self.listbox.selection_clear(0, tk.END)
        self.listbox.selection_set(idx)
        self.listbox.see(idx)  # Ensure the selected item is visible

    def update_save_button(self):
        """Update save button appearance"""
        if self.dirty:
            self.save_button.ax.set_facecolor("#ffcccc")
            self.save_button.label.set_text("Save *")
        else:
            self.save_button.ax.set_facecolor("#cccccc")
            self.save_button.label.set_text("Save")

    def on_click(self, event):
        """Handle mouse clicks for mask refinement"""
        # breakpoint()
        if event.inaxes != self.ax or self.current_object_id is None:
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return
            
        # Left click (1) = foreground, Right click (3) = background
        label = 1 if event.button == 1 else 0
        self.add_point(x, y, label)

    # def on_click(self, event):
    #     """Handle mouse clicks for mask creation/modification"""
    #     if event.inaxes != self.ax or self.current_object_id is None:
    #         return

    #     x, y = event.xdata, event.ydata
    #     if x is None or y is None:
    #         return

    #     obj = self.objects[self.current_object_id]
    #     new_mask = self.predict_mask_from_click(x, y)
        
    #     # Get existing mask or create empty
    #     old_mask = obj.masks.get(self.current_index, 
    #                             np.zeros_like(new_mask))
        
    #     if event.button == 1:  # Left click = union
    #         obj.masks[self.current_index] = old_mask | new_mask
    #     elif event.button == 3:  # Right click = difference
    #         obj.masks[self.current_index] = old_mask & (~new_mask)
            
    #     self.dirty = True
    #     self.update_save_button()
    #     self.redraw_overlay()

    def on_key(self, event):
        """Handle keyboard navigation"""
        if event.key == "right":
            self.show_image((self.current_index + 1) % self.num_frames)
        elif event.key == "left":
            self.show_image((self.current_index - 1) % self.num_frames)
        elif event.key == "escape":
            plt.close(self.fig)

    def on_save_click(self, event):
        """Save final masks for current frame"""
        if not self.dirty:
            return
            
        # Create dictionary of masks for this frame
        masks_dict = {
            f"mask_{obj.label}_{obj_id}": obj.masks[self.current_index]
            for obj_id, obj in self.objects.items()
            if self.current_index in obj.masks
        }
        
        # Save points data for potential future refinement
        points_dict = {
            f"points_{obj.label}_{obj_id}": np.array(obj.points.get(self.current_index, []))
            for obj_id, obj in self.objects.items()
            if self.current_index in obj.points and obj.points[self.current_index]
        }
        
        # Combine masks and points in one NPZ file
        save_dict = {**masks_dict, **points_dict}
        
        # Save to NPZ
        out_path = self.mask_dir / f"{self.image_paths[self.current_index].stem}.npz"
        np.savez_compressed(out_path, **save_dict)
        print(f"[INFO] Saved masks and points to {out_path}")
        
        self.dirty = False
        self.update_save_button()
    
    # def on_save_click(self, event):
    #     """Save masks for current frame"""
    #     if not self.dirty:
    #         return
            
    #     # Create dictionary of masks for this frame
    #     masks_dict = {
    #         f"mask_{obj.label}_{obj_id}": obj.masks[self.current_index]
    #         for obj_id, obj in self.objects.items()
    #         if self.current_index in obj.masks
    #     }
        
    #     # Save to NPZ
    #     out_path = self.mask_dir / f"{self.image_paths[self.current_index].stem}.npz"
    #     np.savez_compressed(out_path, **masks_dict)
    #     print(f"[INFO] Saved masks to {out_path}")
        
    #     self.dirty = False
    #     self.update_save_button()

    def on_list_select(self, evt):
        """Handle frame selection from listbox"""
        w = evt.widget
        index_list = w.curselection()
        if not index_list:
            return
        idx = index_list[0]
        self.show_image(idx)

    def run(self):
        """Start the GUI"""
        print("\nInstructions:")
        print(" - Create new objects with ID and optional label")
        print(" - Left/Right Arrow: navigate frames")
        print(" - Left-click: add to mask for current object")
        print(" - Right-click: remove from mask for current object")
        print(" - Save button: save all masks for current frame")
        print(" - Ctrl+C or x button to close the window after saving your masks")
        
        plt.show(block=False)
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(
        description="Multi-object tracking annotation tool"
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Path to directory containing images"
    )
    parser.add_argument(
        "--mask-dir",
        type=Path,
        required=True,
        help="Path to directory for saving masks"
    )
    
    args = parser.parse_args()
    
    # Create mask directory if it doesn't exist
    args.mask_dir.mkdir(parents=True, exist_ok=True)
    
    # Start GUI
    gui = MultiObjectTrackingGUI(args.image_dir, args.mask_dir)
    gui.run()


if __name__ == "__main__":
    main()