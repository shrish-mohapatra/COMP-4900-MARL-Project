import torch
import imageio
import os
import shutil


def convert_pt_to_gif(folder_path: str):
    """
    Convert .pt files within specified folder_path into gifs
    """
    # TODO: detect number of pt files to convert
    num_videos = len(os.listdir(folder_path))

    for i in range(num_videos):
        VID_NAME = f'{folder_path}/eval_video_{i}'

        folder = 'frames'
        os.makedirs(folder, exist_ok=True)

        tensor = torch.load(f'./{VID_NAME}.pt')
        frames = tensor.shape[1]

        for i in range(frames):
            frame = tensor[0, i].permute(1, 2, 0).numpy().astype('uint8')
            imageio.imwrite(f'./frames/frame_{i}.png', frame)

        images = []
        for i in range(frames):
            images.append(imageio.imread(f'./frames/frame_{i}.png'))

        imageio.mimsave(f'{VID_NAME}.gif', images, fps=100)
        shutil.rmtree(folder)
