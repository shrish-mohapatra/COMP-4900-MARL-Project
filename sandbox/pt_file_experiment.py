import torch
import imageio
import os
import shutil

# run the sandbox_benchmarl.py with logger as csv. In the videos folder grab the .pt files and make sure they are in the same directory as this file

VID_NAME = 'ssl2'

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
