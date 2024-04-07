import torch
import imageio
import os
import shutil


def convert_pt_to_gif(folder_path: str):
    """
    Convert .pt files within specified folder_path into gifs, optimized to avoid unnecessary disk I/O
    """
    video_names = os.listdir(folder_path)

    for video_name in video_names:
        if not video_name.endswith('.pt'):  # Skip non-.pt files
            continue

        # Construct the full path for the video file
        vid_path = os.path.join(folder_path, video_name)

        # Load the tensor
        tensor = torch.load(vid_path)
        frames = tensor.shape[1]

        # Directly create a list of images from tensor
        images = [tensor[0, i].permute(1, 2, 0).numpy().astype('uint8') for i in range(frames)]

        # Save directly to GIF without intermediate frame files
        gif_path = f'{vid_path}.gif'
        imageio.mimsave(gif_path, images, fps=100)

        os.remove(vid_path)
