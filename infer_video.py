import os
import cv2
import torch
import numpy as np
import supervision as sv
from torchvision.ops import box_convert
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor, build_sam2_video_predictor_legacy
from typing import Dict, Any
import argparse
import subprocess

# Add argument parser
parser = argparse.ArgumentParser(description='Video frame processing script with SAM2')
parser.add_argument('-v', '--output-video', type=str, 
                    default=None, #"./shaf_home_glass_door_rotten_juice_rear1_local.mp4"
                    help='Path for output video file')
parser.add_argument('-fps' ,'--fps', type=int, 
                    default=25,
                    help='FPS for output video file')
parser.add_argument('-i', '--source-frames', type=str,
                    default="./assets/kanyest_home_rgb_subset",
                    help='Directory containing source video frames')
parser.add_argument('-o', '--tracking-results', type=str,
                    default=None,
                    help='Directory to save tracking results')
parser.add_argument('-p', '--prompt-img-dir', type=str,
                    default=None, 
                    help='Directory containing prompt images for which masks are provided')
parser.add_argument('-m', '--mask-dir', type=str,
                    default="./masks/masks_tassels/all_mask_npz", #"./assets/amy_floor_habitual_handball/perception_rgb/front0_masks"
                    help='Directory containing NPZ mask files')
parser.add_argument('-ms', '--model-size', type=str, default="tiny", help='Model size for SAM2')
parser.add_argument('-vp', '--videopath', type=str, default=None, help='Path to video file (if its an mp4 and not list of images) for inference')
# parser.add_argument('--sam-checkpoint', type=str,
#                     default="./checkpoints/sam2.1_hiera_tiny.pt",
#                     help='Path to SAM2 checkpoint file')
# parser.add_argument('--model-config', type=str,
#                     default="configs/sam2.1/sam2.1_hiera_t.yaml",
#                     help='Path to model configuration file')

args = parser.parse_args()

# Set variables with argument values and print if using defaults
SOURCE_VIDEO_FRAME_DIR = args.source_frames
if SOURCE_VIDEO_FRAME_DIR == "./assets/kanyest_home_rgb_subset":
    print("Using default source frames directory:", SOURCE_VIDEO_FRAME_DIR)

prompt_images_path = args.prompt_img_dir
if prompt_images_path is None:
    prompt_images_path = SOURCE_VIDEO_FRAME_DIR
    print("Using default prompt images directory:", prompt_images_path)

mask_dir = args.mask_dir
if mask_dir == "./masks/masks_tassels/all_mask_npz":
    print("Using default mask directory:", mask_dir)

SAVE_TRACKING_RESULTS_DIR = args.tracking_results
if SAVE_TRACKING_RESULTS_DIR is None:
    SAVE_TRACKING_RESULTS_DIR = f"{SOURCE_VIDEO_FRAME_DIR}/tracking_results"
    print("Using default tracking results directory:", SAVE_TRACKING_RESULTS_DIR)

OUTPUT_VIDEO_PATH = args.output_video
if OUTPUT_VIDEO_PATH is None:
    OUTPUT_VIDEO_PATH = f"{SAVE_TRACKING_RESULTS_DIR}/seg_output.mp4"
    print("Using default output video path:", OUTPUT_VIDEO_PATH)

fps = args.fps

model_size = args.model_size
assert model_size in ["tiny", "small", "base_plus", "large"], "Invalid model size. Choose from: tiny, small, base_plus, large."
print("Using model size:", model_size)
sam2_checkpoint = f"./checkpoints/sam2.1_hiera_{model_size}.pt"
model_cfg = f"configs/sam2.1/sam2.1_hiera_{model_size[0]}.yaml"
if model_size == "base_plus":
    model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

# sam2_checkpoint = args.sam_checkpoint
# if sam2_checkpoint == "./checkpoints/sam2.1_hiera_tiny.pt":
#     print("Using default SAM2 checkpoint:", sam2_checkpoint)

# model_cfg = args.model_config
# if model_cfg == "configs/sam2.1/sam2.1_hiera_t.yaml":
#     print("Using default model configuration:", model_cfg)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# torch.cuda.set_device(6)

os.makedirs(SAVE_TRACKING_RESULTS_DIR, exist_ok=True)

"""
Step 1: Environment settings and model initialization for SAM 2
"""

# init sam image predictor and video predictor model
sam2_checkpoint = "./checkpoints/sam2.1_hiera_tiny.pt" #"./checkpoints/sam2.1_hiera_large.pt" #"./checkpoints/sam2.1_hiera_base_plus.pt" # #"./checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml" #"configs/sam2.1/sam2.1_hiera_l.yaml" #"configs/sam2.1/sam2.1_hiera_b+.yaml" # #"configs/sam2.1/sam2.1_hiera_t.yaml"

if args.videopath:
    """
    Custom video input directly using video files
    """
    VIDEO_PATH = args.videopath
    video_info = sv.VideoInfo.from_video_path(VIDEO_PATH)  # get video info
    print(video_info)
    frame_generator = sv.get_video_frames_generator(VIDEO_PATH, stride=1, start=0, end=None)

    # saving video to frames
    source_frames = Path(SOURCE_VIDEO_FRAME_DIR)
    source_frames.mkdir(parents=True, exist_ok=True)

    with sv.ImageSink(
        target_dir_path=source_frames, 
        overwrite=True, 
        image_name_pattern="{:05d}.jpg"
    ) as sink:
        for frame in tqdm(frame_generator, desc="Saving Video Frames"):
            sink.save_image(frame)


# scan all the JPEG/PNG frame names in this directory
frame_names = [
    p for p in os.listdir(prompt_images_path)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

infer_frame_names = [
    p for p in os.listdir(SOURCE_VIDEO_FRAME_DIR)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
]
infer_frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

# video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint) 
video_predictor = build_sam2_video_predictor_legacy(model_cfg, sam2_checkpoint)

# init video predictor state
# inference_state = video_predictor.init_state(SOURCE_VIDEO_FRAME_DIR)
inference_state = video_predictor.init_state(SOURCE_VIDEO_FRAME_DIR, prompt_images_path)

"""
Step 2: Carry over torch possibly relevant stuff from grounding dino's script
"""

# FIXME: figure how does this influence the G-DINO model
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

"""
Step 3: Load masks and initialize tracking for each object
"""
def load_object_masks(mask_path: str) -> dict:
    """Load masks for all objects from NPZ file"""
    mask_data = np.load(mask_path)
    objects = {}
    
    # Extract object IDs from mask keys (format: "mask_1", "mask_2", etc.)
    for key in mask_data.keys():
        if key.startswith("mask_"):
            obj_label = key.split("_")[1]
            obj_id = int(key.split("_")[-1])
            #objects[f"{obj_label}_{obj_id}"] = mask_data[key]
            objects[f"{obj_id}"] = mask_data[key]
            
    return objects

# Find all frames with mask annotations and record object IDs
annotated_frames = {}  # frame_idx -> {obj_id -> mask}
all_object_ids = set()

print("Loading mask annotations...")
for idx, frame_name in enumerate(frame_names):
    mask_path = os.path.join(mask_dir, f"{os.path.splitext(frame_name)[0]}.npz")
    if os.path.exists(mask_path):
        frame_objects = load_object_masks(mask_path)
        if frame_objects:
            annotated_frames[idx] = frame_objects
            all_object_ids.update(frame_objects.keys())

# breakpoint()

if not annotated_frames:
    print("No mask files found! Please create masks using the GUI first.")
    # raise ValueError("No mask files found! Please create masks using the GUI first.")

print(f"Found annotations for {len(annotated_frames)} frames")
print(f"Found {len(all_object_ids)} unique objects")

# Initialize object labels
object_labels = {obj_label_id.split('_')[-1]: obj_label_id for obj_label_id in all_object_ids}

# Register all objects with their masks in temporal order
for frame_idx in sorted(annotated_frames.keys()):
    frame_objects = annotated_frames[frame_idx]
    
    for obj_id, mask in frame_objects.items():
        print(f"Registering mask for object {obj_id} at frame {frame_idx}")
        _, _out_obj_ids, _out_mask_logits = video_predictor.add_new_mask(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            mask=mask
        )

"""
Step 4: Propagate the video predictor to get the segmentation results for each frame
"""
# at this stage we added masks for prompted images & passed source (inferred) images to the video predictor
# so safe to extend frame_names to include all.
frame_names.extend(infer_frame_names)

video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# breakpoint() # check unique values in video_segments

"""
Step 5: Visualize the segment results across the video and save them
"""

def save_mask_with_object_ids(frame_names,
                            frame_idx: int, 
                            segments: Dict[str, Any],
                            output_dir: str) -> None:
    """
    Convert multiple binary masks into a single mask where pixel values represent object IDs.
    Save the resulting mask as a PNG file.
    
    Args:
        frame_names: List of frame names from source frames
        frame_idx: Frame index
        segments: Dictionary containing object segments
        frame_shape: Shape of the original frame (height, width)
        output_dir: Directory to save the output PNG files
    """
    # Initialize empty mask with zeros
    # combined_mask = np.zeros(frame_shape[:2], dtype=np.int32)
    combined_mask = np.zeros(list(segments.values())[0].shape[-2:], dtype=np.uint8)
    
    # Process each segment
    for key, mask in segments.items():
        # Extract object ID from key
        if isinstance(key, int):
            obj_id = key
        else:
            try:
                obj_id = int(key.split('_')[-1])
            except (ValueError, IndexError):
                print(f"WARNING: Skipping invalid key format: {key}")
                continue
        
        # Check for invalid object ID
        if obj_id == 0:
            print(f"WARNING: Found object ID 0 in frame {frame_idx}. Object IDs should not be 0. Skipping")
            continue
            
        # Convert mask to boolean
        binary_mask = np.squeeze(mask > 0)
        #breakpoint()
        
        # Check for overlapping masks
        overlap = np.logical_and(combined_mask > 0, binary_mask)
        if np.any(overlap):
            overlapping_indices = np.nonzero(overlap) # tuple of size ndim in overlap
            print(f"WARNING: Overlapping masks found for object ID {obj_id} in frame {frame_idx} at {overlapping_indices[0].size} indices or {overlapping_indices[0].size *100/ binary_mask.size}% of image.")
        
        # Add object ID to the combined mask
        combined_mask[binary_mask] = obj_id
    
    # Save the combined mask as PNG
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, frame_names[frame_idx])
    img = Image.fromarray(combined_mask)
    img.save(output_path[:-3] + "png") # if frame_names are in jpg format
    # cv2.imwrite(output_path, combined_mask)

# Visualize and save frames
for frame_idx, segments in video_segments.items():
    # Load frame
    # breakpoint()
    img = cv2.imread(os.path.join(SOURCE_VIDEO_FRAME_DIR, frame_names[frame_idx]))
    if img is None:
        img = cv2.imread(os.path.join(prompt_images_path, frame_names[frame_idx]))

    # Prepare detections for this frame
    object_ids = list(key.split('_')[-1] if type(key)!=int else key for key in segments.keys())
    masks = list(segments.values())


    for key in segments.keys():
        if type(key) == int:
            all_object_ids.add(key)
    
    if masks:  # Check if we have any masks
        masks = np.concatenate(masks, axis=0) #np.stack(masks)
        
        # Calculate xyxy for each mask separately
        xyxy_boxes = []
        for mask in masks:
            # Get binary mask coordinates
            binary_mask = mask > 0  # Ensure boolean mask
            y_indices, x_indices = np.nonzero(binary_mask)
            #assert np.all(z_ == 0), "Only 2D masks are supported"
            
            if len(y_indices) > 0 and len(x_indices) > 0:
                x1, y1 = x_indices.min(), y_indices.min()
                x2, y2 = x_indices.max(), y_indices.max()
                xyxy_boxes.append([x1, y1, x2, y2])
            else:
                # If mask is empty, add a dummy box
                xyxy_boxes.append([0, 0, 1, 1])
        
        xyxy_boxes = np.array(xyxy_boxes)
        
        # Create detections object
        detections = sv.Detections(
            xyxy=xyxy_boxes,
            mask=np.squeeze(masks) if masks.ndim == 4 else masks,
            class_id=np.array(object_ids, dtype=np.int32),
        )
        
        custom_colors = [
            (0, 0, 0),
            (50, 50, 50),
            (252, 194, 10),
            (135, 17, 8),
            (0, 255, 255),
            (255, 0, 0),
            (30, 100, 190),
            (211, 100, 60),
            (255, 105, 180),
            (128, 128, 0),
            (121, 72, 47),
            (0, 145, 100),
            (255, 0, 255), # magenta
            (0, 255, 0), # green
            (150, 75, 0), # brown
            (0, 0, 255), # blue
            (255, 255, 255), # white
            (0, 0, 128), # navy
            (255, 255, 0), # yellow
            (0, 128, 128), # teal
            (128, 0, 0), # maroon
        ]


        # Create annotators with default settings
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator(color=sv.ColorPalette([sv.Color(color[0], color[1], color[2]) for color in custom_colors]))
        label_annotator = sv.LabelAnnotator()
        
        # Apply annotations
        annotated_frame = img.copy()
        annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)
        
        # uncomment line below to get box annotations as well.
        # annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
        
        # Add labels with object IDs
        # breakpoint() # check object_label keys
        labels  = []
        for obj_id in object_ids:
            try:
                labels.append(object_labels[obj_id])
            except KeyError:
                pass

        # labels = [object_labels[obj_id] for obj_id in object_ids]
        # breakpoint()
        
        # comment this block if anno are causing errors
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        save_mask_with_object_ids(frame_names + infer_frame_names, frame_idx, segments, SAVE_TRACKING_RESULTS_DIR + "/png_masks")

    else:
        annotated_frame = img
    
    # Save frame
    output_path = os.path.join(SAVE_TRACKING_RESULTS_DIR, f"tracked_frame_{frame_idx:05d}.jpg")
    cv2.imwrite(output_path, annotated_frame)
    print(f"Saved frame {frame_idx}")

"""
Step 6: Convert the annotated frames to video
"""

def create_video_from_images(image_folder, prompt_images_path, output_video_path, frame_rate=25):
    """
    Create a video from images using FFmpeg, combining prompt images and inferred images
    """
    valid_extensions = [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]
    
    # Get and sort image files from both directories
    image_files = [f for f in os.listdir(image_folder) 
                   if os.path.splitext(f)[1] in valid_extensions]
    prompt_files = [f for f in os.listdir(prompt_images_path)
                     if os.path.splitext(f)[1] in valid_extensions]
    image_files.sort()
    prompt_files.sort()
    print("Prompted images:", prompt_files)
    print("Inferred Images:", image_files)
    
    if not image_files and not prompt_files:
        raise ValueError("No valid image files found in the specified folders.")
    
    # Create a temporary file listing all images in order
    with open('temp_file_list.txt', 'w') as f:
        # Write prompt images
        # for img in prompt_files:
        #     f.write(f"file '{os.path.join(os.path.abspath(prompt_images_path), img)}'\n")
        #     f.write(f"duration {1/frame_rate}\n")
        
        # Write inferred images
        for img in image_files:
            f.write(f"file '{os.path.join(os.path.abspath(image_folder), img)}'\n")
            f.write(f"duration {1/frame_rate}\n")
        
        # Write the last image one more time (required by FFmpeg)
        if image_files:
            f.write(f"file '{os.path.join(os.path.abspath(image_folder), image_files[-1])}'\n")
        elif prompt_files:
            f.write(f"file '{os.path.join(os.path.abspath(prompt_images_path), prompt_files[-1])}'\n")

    # FFmpeg command
    ffmpeg_cmd = [
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'temp_file_list.txt',
        '-vsync', 'vfr',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-preset', 'medium',
        '-crf', '23',
        '-y',
        output_video_path
    ]
    
    try:
        # Execute FFmpeg command
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"Video saved at {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
    finally:
        # Clean up temporary file
        if os.path.exists('temp_file_list.txt'):
            os.remove('temp_file_list.txt')

create_video_from_images(SAVE_TRACKING_RESULTS_DIR, prompt_images_path, OUTPUT_VIDEO_PATH, fps)
