import torch
import onnxruntime
import numpy as np
from onnx_conv_using_wrapper import ONNXWrapper, images, video_dir
import os
import cv2
from datetime import datetime

def find_first_png(path):
    """
    Find the first .png file in the provided directory path.

    Args:
        path (str): The directory to search.

    Returns:
        str: The full path of the first .png file found, or None if no .png files exist.
    """
    try:
        # List all files in the directory
        files = os.listdir(path)

        # Iterate through the files
        for file in files:
            if file.lower().endswith('.png'):  # Check if the file has a .png extension
                return cv2.imread(os.path.join(path, file))

        return None  # Return None if no .png files are found
    except Exception as e:
        print(f"Error looking for png: {e}")
        return None

def load_onnx_model(onnx_path):
    # Create ONNX Runtime session
    session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    return session

def run_parity_check(pytorch_model, onnx_session, input_tensor, actual_image):
    # Get PyTorch output
    pytorch_model.eval()
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor)
    
    # Prepare input for ONNX model
    # Convert PyTorch tensor to numpy array
    input_numpy = input_tensor.cpu().numpy()
    
    # Get input name from ONNX model
    input_name = onnx_session.get_inputs()[0].name
    
    # Run ONNX model
    onnx_output = onnx_session.run(None, {input_name: input_numpy})[0]
    
    # Convert outputs to numpy for comparison
    pytorch_output_np = pytorch_output.cpu().numpy()
    
    # use np all close
    if np.allclose(onnx_output, pytorch_output_np, atol=1e-5):
        print("Parity check passed: ONNX and PyTorch outputs are close!")

    if np.allclose(onnx_output, pytorch_output_np, atol=1e-5, equal_nan=True):
        print("If NaNs are ignored, parity check passed.")
    
    # Calculate differences
    max_diff = np.max(np.abs(pytorch_output_np - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output_np - onnx_output))
    
    print(f"Maximum absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # You might want to set a threshold based on your requirements
    THRESHOLD = 1e-4
    is_similar = max_diff < THRESHOLD
    
    print(f"Models are {'similar' if is_similar else 'different'} (threshold: {THRESHOLD})")
    
    if actual_image is not None:
        pt_mask = (pytorch_output_np.squeeze() > 0.0).T # 560 x 680
        ox_mask = (onnx_output.squeeze() > 0.0).T
        mask_overlay_color = np.array([0, 0, 255], dtype=np.uint8) # seems to be BGR
        pt_img_rgb = actual_image.copy()
        pt_img_rgb[pt_mask] = mask_overlay_color
        ox_img_rgb = actual_image.copy()
        ox_img_rgb[ox_mask] = mask_overlay_color
        now = datetime.now().strftime("%m%d%H%M")
        cv2.imwrite(f"pytorch_seg_{now}.png", pt_img_rgb)
        cv2.imwrite(f"onnx_seg_{now}.png", ox_img_rgb)
        print(f"Segmented images for PT & Onnx saved to cwd.")
        print(f"Are the masks exactly the same? {np.all(pt_mask == ox_mask)}")

    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'is_similar': is_similar,
        'pytorch_output': pytorch_output_np,
        'onnx_output': onnx_output
    }

if __name__ == "__main__":

    first_png = find_first_png(video_dir) # assumes video_dir is a dir of imgs

    # Use your existing model and input
    model = ONNXWrapper()
    onnx_session = load_onnx_model("sam2_hiera_tiny_expt.onnx")

    # Run parity check
    results = run_parity_check(model, onnx_session, images, first_png)

    # Optional: Detailed analysis of differences
    if not results['is_similar']:
        print("\nDetailed analysis of differences:")
        diff = np.abs(results['pytorch_output'] - results['onnx_output'])
        print(f"Shape of outputs: {results['pytorch_output'].shape}")
        print(f"Number of elements with diff > 1e-4: {np.sum(diff > 1e-4)}")
        print(f"Locations of largest differences:")
        # Get indices of top 5 differences
        flat_indices = np.argsort(diff.flatten())[-5:]
        for idx in flat_indices:
            multi_idx = np.unravel_index(idx, diff.shape)
            print(f"At index {multi_idx}:")
            print(f"  PyTorch: {results['pytorch_output'][multi_idx]}")
            print(f"  ONNX: {results['onnx_output'][multi_idx]}")
            print(f"  Difference: {diff[multi_idx]}")