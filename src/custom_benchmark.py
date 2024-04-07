#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import time
import json
import torch.multiprocessing as mp
import torch
from typing import Iterator, Optional, Sequence, Set
from multiprocessing import Process

from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.environments import Task
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.common import ModelConfig

from src.CustomExperiment import CustomExperiment
from src.BaselineVmasTask import BaselineVmasTask
from utils.pt_export import convert_pt_to_gif


class Benchmark:
    """A benchmark.

    Benchmarks are collections of experiments to compare.

    Args:
        algorithm_configs (list of AlgorithmConfig): the algorithms to benchmark
        model_config (ModelConfig): the config of the policy model
        tasks (list of Task):  the tasks to benchmark
        seeds (set of int): the seeds for the benchmark
        experiment_config (ExperimentConfig): the experiment config
        critic_model_config (ModelConfig, optional): the config of the critic model. Defaults to model_config
        num_process (int): number of processes to run at a time for parallelization
    """

    def __init__(
        self,
        algorithm_configs: Sequence[AlgorithmConfig],
        model_configs: Sequence[ModelConfig],
        tasks: Sequence[Task],
        seeds: Set[int],
        experiment_config: ExperimentConfig,
        critic_model_config: Optional[ModelConfig] = None,
        num_process=1,
    ):
        self.algorithm_configs = algorithm_configs
        self.tasks = tasks
        self.seeds = seeds

        self.model_configs = model_configs
        self.critic_model_config = (
            critic_model_config if critic_model_config is not None else model_configs[0]
        )
        self.experiment_config = experiment_config
        self.num_process = num_process

        print(f"Created benchmark with {self.n_experiments} experiments.")

    @property
    def n_experiments(self):
        """The number of experiments in the benchmark."""
        return len(self.model_configs) * len(self.algorithm_configs) * len(self.tasks) * len(self.seeds) + 1

    def get_experiments(self) -> Iterator[Experiment]:
        """Yields one experiment at a time"""
        for model_config in self.model_configs:
            for algorithm_config in self.algorithm_configs:
                for task in self.tasks:
                    for seed in self.seeds:
                        yield Experiment(
                            task=task,
                            algorithm_config=algorithm_config,
                            seed=seed,
                            model_config=model_config,
                            critic_model_config=self.critic_model_config,
                            config=self.experiment_config,
                        )

    def get_experiments_kwargs(self) -> Iterator[dict]:
        for model_config in self.model_configs:
            for algorithm_config in self.algorithm_configs:
                for task in self.tasks:
                    for seed in self.seeds:
                        yield {
                            "task": task,
                            "algorithm_config": algorithm_config,
                            "seed": seed,
                            "model_config": model_config,
                            "critic_model_config": self.critic_model_config,
                            "config": self.experiment_config,
                        }

        # Curriculum learning for baseline only
        # ex. freeze_timeline = {
        #     100: ["civilian", "policeHQ"],
        #     1100: ["listener"],
        #     2100: None,
        # } for 3000 epochs total
        epochs = self.experiment_config.max_n_iters
        speaker_freeze = int(0.03 * epochs)
        listener_freeze = speaker_freeze + int(0.3 * epochs)
        no_freeze = listener_freeze + int(0.3 * epochs)
        freeze_timeline = {
            speaker_freeze: ["civilian", "policeHQ"],
            listener_freeze: ["listener"],
            no_freeze: None,
        }
        print(f"curriculum freeze={freeze_timeline}")

        yield {
            "freeze_timeline": freeze_timeline,
            "task": self.tasks[0],
            "algorithm_config": self.algorithm_configs[0],
            "seed": list(self.seeds)[0],
            "model_config": self.model_configs[0],
            "critic_model_config": self.critic_model_config,
            "config": self.experiment_config,
        }

    def display_experiment(self, experiment: Experiment, experiment_index: int):
        print(
            json.dumps({
                "msg": f"Running experiment {experiment_index+1}/{self.n_experiments}",
                "task": str(type(experiment.task)),
                "algorithm": str(type(experiment.algorithm_config)),
                "model": str(type(experiment.model_config)),
            }, indent=2)
        )

    def run_parallel(self):
        start_time = time.time()
        experiment_queue = mp.Queue(maxsize=self.num_process)

        # Start worker processes
        processes = []
        for _ in range(self.num_process):
            p = mp.Process(target=self._worker, args=(experiment_queue,))
            processes.append(p)
            p.start()

        # Feed experiments into queue
        for i, experiment_kwargs in enumerate(self.get_experiments_kwargs()):
            experiment_queue.put((i, experiment_kwargs))

        # Signal workers to stop by adding 'None' to the queue for each worker
        for i in range(self.num_process):
            experiment_queue.put((i, None))

        # Wait for all workers to finish
        for p in processes:
            p.join()

        print(f'benchmark took {time.time() - start_time} seconds in total')

    def _worker(self, experiment_queue: mp.Queue):
        """Worker function to process experiments from queue"""
        while True:
            experiment_index, experiment_kwargs = experiment_queue.get()
            if experiment_kwargs is None:
                break

            experiment = CustomExperiment(**experiment_kwargs)
            self.display_experiment(experiment, experiment_index)
            self.run_experiment(experiment)

    def run_experiment(self, experiment: Experiment):
        try:
            experiment.run()

            convert_start_time = time.time()
            video_file_path = f'{experiment.config.save_folder}/{experiment.name}/{experiment.name}/videos'
            convert_pt_to_gif(video_file_path)
            conversion_time = time.time() - convert_start_time
            print(f'video conversion took {conversion_time} seconds')
        except KeyboardInterrupt as interrupt:
            print("\n\nBenchmark was closed gracefully\n\n")
            experiment.close()
            raise interrupt

    def run_sequential(self):
        """Run all the experiments in the benchmark in a sequence."""
        for i, experiment in enumerate(self.get_experiments()):
            self.display_experiment(experiment, i)
            self.run_experiment(experiment)
