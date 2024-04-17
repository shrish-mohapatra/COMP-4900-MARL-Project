# PoPoRL
> Multi agent reinforcement learning for emergency response systems

![image](https://github.com/shrish-mohapatra/PoPoRL/assets/51547897/14c44529-0163-40fd-9aaf-eb48d2021b4f)

## Development
```
/utils
    scripts to help convert stuff
    pt_to_gif.py
    ...

/src
    files relevant to our experiment
    complex_speaker_listener.py

/results
    contain execution results from experiments

/experiments
    shrish/
        baseline.ipynb
    josh/
        baseline.py
    jovin/

gitignore
readme
requirements

venv
```


## Setup
```shell
# Create venv
python -m venv venv

# Activate venv
venv/Scripts/activate  # for windows

# Install requirements
pip install -r requirements.txt

# Create experiment folder
mkdir sandbox/experiments

# Run your experiment
python -m experiments.shrish.test
```

## Resources
- [VMAS github](https://github.com/proroklab/VectorizedMultiAgentSimulator)
- [BenchMARL github](https://github.com/facebookresearch/BenchMARL)
- [BenchMARL + VMAS colab example](https://colab.research.google.com/github/facebookresearch/BenchMARL/blob/main/notebooks/run.ipynb#scrollTo=4f32b88e)


## Development

### Changing number of generated videos
With our current setup each .pt file used to create video is around 100mb. To reduce this we need to change the experiment_config settings as follows:
```py
# how many frames to collect a batch
experiment_config.on_policy_collected_frames_per_batch = 1_000

# how many frames to evaluate to create a video
experiment_config.evaluation_interval = 50_000

# how many iterations to run
experiment_config.max_n_iters = 100

# how many videos will be created?
# 1.000 frames x 100 iterations / 50,000 eval interval = 2
```

### Running with CUDA
We want to use our GPU to run experiments faster. Since the library uses PyTorch follow these instructions.
1. install [Anaconda](https://www.anaconda.com/download)
    - this is a tool to manage python environments
2. create an Anaconda environment (similar to creating a venv)
3. use the following command to install cuda & pytorch
```shell
# https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
4. install project dependencies using pip
```shell
pip install benchmarl vmas wandb
```
5. run the following code to make sure GPU is enabled
```py
import torch
torch.cuda.is_available()
# should be True
```
