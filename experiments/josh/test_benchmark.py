from benchmarl.algorithms import MappoConfig, MaddpgConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from torch.multiprocessing import freeze_support

from src.BaselineVmasTask import BaselineVmasTask
from src.Ext1VmasTask import Ext1VmasTask
from src.Ext2VmasTask import Ext2VmasTask
from src.Ext3VmasTask import Ext3VmasTask
from src.Ext4VmasTask import Ext4VmasTask
from src.lstm_model import LSTMMlpConfig
from src.custom_benchmark import Benchmark
from src.MappoDecoder import MappoDecoderConfig

from utils.pt_export import convert_pt_to_gif
import torch
import os
import time

start_time = time.time()

seeds = { 69 }
# Loads from "benchmarl/conf/experiment/base_experiment.yaml"
experiment_config = ExperimentConfig.get_from_yaml()

# Loads from "benchmarl/conf/task/vmas/balance.yaml"
# task = Ext1VmasTask.SIMPLE_SPEAKER_LISTENER.get_from_yaml()
tasks = [
    BaselineVmasTask.SIMPLE_SPEAKER_LISTENER.get_from_yaml(), 
    Ext1VmasTask.SIMPLE_SPEAKER_LISTENER.get_from_yaml(), 
    Ext2VmasTask.SIMPLE_SPEAKER_LISTENER.get_from_yaml(), 
    Ext3VmasTask.SIMPLE_SPEAKER_LISTENER.get_from_yaml(), 
    Ext4VmasTask.SIMPLE_SPEAKER_LISTENER.get_from_yaml()
]
# task = Ext1VmasTask.SIMPLE_SPEAKER_LISTENER.get_from_yaml()
# print(task)

# Loads from "benchmarl/conf/algorithm/mappo.yaml"
algorithm_config_path = os.path.join('src', 'mappodecoder.yaml')
# algorithm_config = MappoDecoderConfig.get_from_yaml(algorithm_config_path)
# algorithm_config = MappoConfig.get_from_yaml()
algorithm_configs = [MappoConfig.get_from_yaml(), MappoDecoderConfig.get_from_yaml(algorithm_config_path)]
# algorithm_config = MaddpgConfig.get_from_yaml()

# Loads from "benchmarl/conf/model/layers/mlp.yaml"
# model_config = MlpConfig.get_from_yaml()
model_config_path = os.path.join('src', 'lstmmlp.yaml')
# model_config = LSTMMlpConfig.get_from_yaml(model_config_path)
# critic_model_config = MlpConfig.get_from_yaml()
model_configs = [MlpConfig.get_from_yaml(), LSTMMlpConfig.get_from_yaml(model_config_path)]

# THIS IS VERY IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
experiment_config.on_policy_collected_frames_per_batch = 1_000
experiment_config.off_policy_collected_frames_per_batch = 1_000
experiment_config.evaluation = True
experiment_config.render = True
# experiment_config.evaluation_interval = 12_000
experiment_config.evaluation_interval = 150_000
experiment_config.evaluation_episodes = 10

experiment_config.max_n_iters = 1500 # epoch
experiment_config.max_n_frames = 8_000_000
experiment_config.loggers = ["csv"]
experiment_config.create_json = True
experiment_config.save_folder = "results"
if torch.cuda.is_available():
    print('running cuda')
    experiment_config.sampling_device = 'cuda'
    experiment_config.train_device = 'cuda'
experiment_config.off_policy_train_batch_size = 400
experiment_config.on_policy_minibatch_size = 600

# Create the experiment
# experiment = Experiment(
#     task=task,
#     algorithm_config=algorithm_config,
#     model_config=model_config,
#     critic_model_config=critic_model_config,
#     seed=0,
#     config=experiment_config,
# )



# Log experiment folder path
# print(experiment.name)

# Run experiment
# experiment.run()

if __name__ == "__main__":
    freeze_support()
    benchmark = Benchmark(
        algorithm_configs=algorithm_configs,
        model_configs=model_configs,
        tasks=tasks,
        seeds=seeds,
        experiment_config=experiment_config,
        num_process=4,
    )

    # benchmark.run_sequential()
    benchmark.run_parallel()