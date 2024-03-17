from benchmarl.algorithms import MappoConfig, MaddpgConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig

from src.BaselineVmasTask import BaselineVmasTask
from utils.pt_export import convert_pt_to_gif

# Loads from "benchmarl/conf/experiment/base_experiment.yaml"
experiment_config = ExperimentConfig.get_from_yaml()

# Loads from "benchmarl/conf/task/vmas/balance.yaml"
# task = VmasTask.SIMPLE_SPEAKER_LISTENER.get_from_yaml()
task = BaselineVmasTask.SIMPLE_SPEAKER_LISTENER.get_from_yaml()
print(task)

# Loads from "benchmarl/conf/algorithm/mappo.yaml"
# algorithm_config = MappoConfig.get_from_yaml()
algorithm_config = MaddpgConfig.get_from_yaml()

# Loads from "benchmarl/conf/model/layers/mlp.yaml"
model_config = MlpConfig.get_from_yaml()
critic_model_config = MlpConfig.get_from_yaml()

# THIS IS VERY IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
experiment_config.on_policy_collected_frames_per_batch = 1_000
experiment_config.off_policy_collected_frames_per_batch = 1_000
experiment_config.evaluation = True
experiment_config.render = True
# experiment_config.evaluation_interval = 12_000
experiment_config.evaluation_interval = 50_000
experiment_config.evaluation_episodes = 10

experiment_config.max_n_iters = 100
experiment_config.loggers = ["csv"]
experiment_config.create_json = True
experiment_config.save_folder = "results"

# Create the experiment
experiment = Experiment(
    task=task,
    algorithm_config=algorithm_config,
    model_config=model_config,
    critic_model_config=critic_model_config,
    seed=0,
    config=experiment_config,
)

# Log experiment folder path
print(experiment.name)

# Run experiment
experiment.run()

# Convert pt files to gifs
video_file_path = f'{experiment_config.save_folder}/{experiment.name}/{experiment.name}/videos'
convert_pt_to_gif(video_file_path)