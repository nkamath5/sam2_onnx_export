import torch
import numpy as np
import torch
from sam2.utils.misc import load_video_frames_from_jpg_images

# torch.set_default_device('cuda')
device='cuda'



from sam2.build_sam import build_sam2_video_predictor

model_size = "tiny"
sam2_checkpoint = f"../checkpoints/sam2.1_hiera_{model_size}.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
video_dir = "../assets/front0"

images, video_height, video_width = load_video_frames_from_jpg_images(
    video_dir,
    image_size=1024, # this seems to be the image size set for the model from config file
    offload_video_to_cpu=True,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
)

class ONNXWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

    def forward(self, images):
        inference_state = self.predictor.init_state(images)
        # predictor.reset_state(inference_state)
        # add fixed prompts since we need at least 1
        inference_state_2, _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            points= np.array([[10, 10]], dtype=np.float32), # [10,10] is the original (x,y) point given to the onnx model            labels=np.array([1], np.int32), 
        ) # garbage values


        video_segments = {}  # video_segments contains the per-frame segmentation results
        # for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
        #     video_segments[out_frame_idx] = {
        #         out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        #         for i, out_obj_id in enumerate(out_obj_ids)
        #     }
        out_frame_idx, out_obj_ids, out_mask_logits = self.predictor.propagate_in_video(inference_state_2)
        video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
        return out_mask_logits # last frame logits
    


if __name__ == "__main__":
    model = ONNXWrapper()
    # print(model.forward(images))
    torch.onnx.export(
        model,
        images,
        f"sam2_hiera_{model_size}_expt.onnx",
        input_names=["input image"],
        dynamo=True,
    )