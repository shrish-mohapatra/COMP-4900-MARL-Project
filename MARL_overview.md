# MARL Overview
> Overview of python libraries

## VMAS
- vectorized physics engine for MARL
- contains tasks: simple_speaker_listener.py
    - [source code](venv\Lib\site-packages\vmas\scenarios\mpe\simple_speaker_listener.py)

## TorchRL
- represent RL tasks/algos using tensors

## BenchMARL
- easier way to test MARL algos and tasks?

involves creating an `Experiment` instance

- task
    - `VmasTask` instance
    - contains enums corresponding to VMAS tasks (ex. `SIMPLE_SPEAKER_LISTENER`)
    - [source code](venv\Lib\site-packages\benchmarl\environments\vmas\common.py)
    - `get_env_fun()` used to retrive Vmas environment
        - `VmasEnv` instance
        - [source code](venv\Lib\site-packages\torchrl\envs\libs\vmas.py)

- algorithm_config
    - `AlgorithmConfig` instance
    - [source code](venv\Lib\site-packages\benchmarl\algorithms\common.py)
    - example: `MappoConfig`

- model_config
    - `MlpConfig` for NN arch

- critic_model_config
    - `MlpConfig` for NN arch

- config
    - `ExperimentConfig` instance
    - parameters for rendering experiments
        ```py
        # example config
        experiment_config = ExperimentConfig.get_from_yaml()
        experiment_config.on_policy_collected_frames_per_batch = 6000
        experiment_config.evaluation = True
        experiment_config.render = True
        experiment_config.evaluation_interval = 6_000
        experiment_config.evaluation_episodes = 10

        experiment_config.max_n_iters = 2
        experiment_config.loggers = ["wandb"]
        experiment_config.create_json = True
        experiment_config.save_folder = "experiments"
        ```
    - [source code](venv\Lib\site-packages\benchmarl\experiment\experiment.py)
    - [info on config params](https://github.com/facebookresearch/BenchMARL/blob/main/benchmarl/conf/experiment/base_experiment.yaml)

### WandDB
- web app for logging experiment results
- cursed `Mailbox` error



---

1. create `complex_speaker_listener.py`
    - inherit from vmas....BaseScenario
2. create a new BenchMarl Task
    - inherit from `VmasTask`
    - override `get_env_fun()` so that it uses `complex_speaker_listener` class
    
