import argparse
import os
from typing import List, Tuple, Dict, Union
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
from tqdm import tqdm
import supervision as sv

matplotlib.use("TkAgg")

# SAM2 imports
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.sam2_video_predictor import SAM2VideoPredictorVOS

import torch._dynamo
torch._dynamo.config.suppress_errors = True


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
    CHAIR_LEG_FLAT = 11
    DESK_LEG_FLAT = 12
    DOOR_STOPPER = 13
    PET_TRAY = 14
    HEATING_VENT = 15
    LAPTOP = 16
    SILL = 17
    WEIGHING_SCALE = 18
    FOAM_MAT = 19
    PERSON = 20

@dataclass
class ObjectTracker:
    id: int
    label: str # object label/ name
    color: Tuple[float, float, float]
    masks: Dict[int, np.ndarray]  # frame_idx -> mask
    points: Dict[int, List[Tuple[float, float, int]]]  # frame_idx -> [(x, y, label)] (label: 0=bg, 1=fg)

# @dataclass
# class PromptedFrames:
#     frames: Dict[int, str]  # frame_idx -> file path as str

class MultiObjectTrackingGUI:   
    def __init__(self, image_dir: Path, mask_dir: Path, max_points_per_frame: int = 100):
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
            "chair_leg_flat": (255, 0, 255), # magenta
            "desk_leg_flat": (0, 255, 0), # green
            "door_stopper": (150, 75, 0), # brown
            "pet_tray": (0, 0, 255), # blue
            "heating_vent": (255, 255, 255), # white
            "laptop": (0, 0, 128), # navy
            "sill": (255, 255, 0), # yellow
            "weighing_scale": (0, 128, 128), # teal
            "foam_mat": (128, 128, 128), # gray
            "person": (128, 0, 0), # maroon
        }

        for seg_id, seg_label in self.seg_types.items():
            if seg_id not in self.objects:
                color =  SEGMENTATION_PALETTE.get(seg_label, (np.random.random(3)*255))
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
                frame_idx = self.file_stem_to_idx[mask_path.stem]
                
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
        self.sam2_VOS_model = build_sam2_video_predictor(config_file=SAM2_MODEL_CONFIG, ckpt_path=SAM2_CHECKPOINT, 
                                                     vos_optimized=True, device=DEVICE)
        # self.sam2_inference_state = None
        torch.compiler.cudagraph_mark_step_begin()
        self.sam2_inference_state = self.sam2_VOS_model.init_state(str(self.image_dir))
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
        self.file_stem_to_idx = {path.stem: idx for idx, path in enumerate(self.image_paths)} # use to match on mask stem

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
            # if obj.masks.get(self.current_index) is not None:
            #     obj.masks[self.current_index] = obj.masks[self.current_index] | new_mask
            # else:
            obj.masks[self.current_index] = new_mask
            
        self.dirty = True
        # breakpoint()
        self.update_save_button()
        self.redraw_overlay()

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

        # get a new inference state for SAM2 Video Pred with just this image
        current_img_path = str(self.image_paths[idx])
        img_paths = []
        # add all the prompts from before to img_paths
        for obj_id in self.objects:
            obj = self.objects[obj_id]
            for prompt_img_idx in obj.points.keys():
                img_paths.append(str(self.image_paths[prompt_img_idx]))
        if current_img_path in img_paths:
            return # if this image is already in the prompt list, then we don't need to infer again
        # img_paths.append(current_img_path)
        if len(img_paths) < 1:
            return # "It makes sense to infer on SAM2 VOS only if we have at least 1 prompt image & 1 inference image"
        print(f"Inferring SAM2 VOS on {current_img_path}")
        # since same img can be added twice; we are also assuming that the items in img_paths are in the same dir.
        img_paths = sorted(set(img_paths))
        if self.sam2_inference_state is not None:
            self.sam2_VOS_model.reset_state(self.sam2_inference_state)
            # torch.compiler.cudagraph_mark_step_begin()
        # self.sam2_inference_state = self.sam2_VOS_model.init_state(self.image_dir, img_paths=img_paths)
        print(f"[INFO] Inference state initialized for SAM2 VOS") # for {len(img_paths)} images")
        # add all the prompts from before
        prompt_dict = {} # frame_idx -> {obj_id: {(x,y): label}}
        for obj_id in tqdm(self.objects):
            # if obj_id == 9:
            #     breakpoint()
            obj = self.objects[obj_id]
            if not obj.masks:
                continue
            for prompt_img_idx , points_list in obj.points.items():
                for x, y, label in points_list:
                    prompt_dict[prompt_img_idx] = {obj_id: {(x,y): label}}
                    
            for prompt_img_idx in prompt_dict.keys():
                for obj_id in prompt_dict[prompt_img_idx].keys():
                    xy_list = list(prompt_dict[prompt_img_idx][obj_id].keys())
                    label_list = list(prompt_dict[prompt_img_idx][obj_id].values())
                    _, _, _, _ = self.sam2_VOS_model.add_new_points_or_box(self.sam2_inference_state, 
                                                            # since idx in self.image_paths will be different than the video slice (subset of all image frames) that we sent to SAM2
                                                            img_paths.index(str(self.image_paths[prompt_img_idx])), # TODO: Do this a better way.
                                                            obj_id, 
                                                            points=xy_list, 
                                                            labels=label_list, 
                                                            )
        if prompt_dict:
            print(f"[INFO] {len(prompt_dict.keys())} Prompts added")
        else:
            print("[INFO] No prompts to add")
            breakpoint()
        # predict on this image if prompt dict is not empty
        # inferred_segmentation_on_img = {}
        if prompt_dict: # only run inference if there were prompts to begin with
            # out_frame_idxs, out_obj_ids, out_mask_logits = self.sam2_VOS_model.propagate_in_video(self.sam2_inference_state)
            for out_frame_idx, out_obj_ids, out_mask_logits in self.sam2_VOS_model.propagate_in_video(
                                                                            self.sam2_inference_state,
                                                                            start_frame_idx=idx,
                                                                            max_frame_num_to_track=2,):
                
                # if isinstance(out_frame_idxs, int):
                #     out_frame_idxs = [out_frame_idxs]
                # ONLY INDEX 0 is coming out as the inference frame so used yield
                # print(f"[INFO] Inferred on {len(out_frame_idxs)} frames")
                # for out_frame_idx in out_frame_idxs:
                print("[INFO] Inference done")
                if out_frame_idx != idx:
                    print(f"out_frame_idx: {out_frame_idx} != inferred image idx: {idx}")
                    continue # this would be a prompted frame
                for i, obj_id in enumerate(out_obj_ids):
                    obj = self.objects[obj_id]
                    obj.masks[idx] = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                    # breakpoint()
                break # dont need to process remaining prompted frames
                
                # if out_frame_idx == idx:
                #     inferred_segmentation_on_img[out_frame_idx] = {
                #         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                #         for i, out_obj_id in enumerate(out_obj_ids)
                #     }

                # # add the inferred mask to all objects' masks for this frame
                # for obj_id in self.objects:
                #     obj = self.objects[obj_id]
                #     obj.masks[idx] = inferred_segmentation_on_img[] # idx is the idx in self.image_paths

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
        # print("Redrew overlay")

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
        "-i",
        "--image-dir",
        type=Path,
        required=True,
        help="Path to directory containing images or video frames (list of images)"
    )
    parser.add_argument(
        "-m",
        "--mask-dir",
        type=Path,
        required=True,
        help="Path to directory for saving masks"
    )
    parser.add_argument('-vp', '--videopath', type=str, default=None, help='Path to video file (if its an mp4 and not list of images) for inference')
    
    args = parser.parse_args()
    image_dir = args.image_dir
    mask_dir = args.mask_dir

    if args.videopath:
        """
        Custom video input directly using video files
        """
        VIDEO_PATH = args.videopath
        video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
        print(video_info)
        frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

        # saving video to frames
        source_frames = Path(image_dir)
        source_frames.mkdir(parents=True, exist_ok=True)

        with sv.ImageSink(
            target_dir_path=source_frames, 
            overwrite=True, 
            image_name_pattern="{:05d}.jpg"
        ) as sink:
            for frame in tqdm(frame_generator, desc="Saving Video Frames"):
                sink.save_image(frame)
    
    # Create mask directory if it doesn't exist
    args.mask_dir.mkdir(parents=True, exist_ok=True)

    
    # Start GUI
    gui = MultiObjectTrackingGUI(image_dir, mask_dir)
    gui.run()


if __name__ == "__main__":
    main()