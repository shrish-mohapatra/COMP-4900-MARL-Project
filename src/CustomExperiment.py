import copy
import time
import torch

from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.environments import Task
from benchmarl.algorithms.common import AlgorithmConfig
from benchmarl.models.common import ModelConfig
from benchmarl.experiment.callback import Callback

from torchrl.collectors import SyncDataCollector
from torchrl.envs import SerialEnv, TransformedEnv
from torchrl.envs.transforms import Compose
from tensordict.nn import TensorDictSequential
from tensordict import TensorDictBase

from tqdm import tqdm


# freeze_timeline = {
#     100: ["civilian", "policeHQ"],
#     1100: ["listener"],
#     2100: None,
# }

class CustomExperiment(Experiment):
    def __init__(self, freeze_timeline, task: Task, algorithm_config: AlgorithmConfig, model_config: ModelConfig, seed: int, config: ExperimentConfig, critic_model_config: ModelConfig | None = None, callbacks: torch.List[Callback] | None = None):
        super().__init__(task, algorithm_config, model_config, seed, config, critic_model_config, callbacks)
        self.freeze_timeline = freeze_timeline

    def _collection_loop(self):
        pbar = tqdm(
            initial=self.n_iters_performed,
            total=self.config.get_max_n_iters(self.on_policy),
        )
        sampling_start = time.time()

        # print(f"DEBUG self.collector len={len(self.collector)}")

        # Training/collection iterations
        for batch in self.collector:
            # Logging collection
            collection_time = time.time() - sampling_start
            current_frames = batch.numel()
            self.total_frames += current_frames
            self.mean_return = self.logger.log_collection(
                batch,
                total_frames=self.total_frames,
                task=self.task,
                step=self.n_iters_performed,
            )
            pbar.set_description(
                f"mean return = {self.mean_return}", refresh=False)

            # Callback
            self._on_batch_collected(batch)

            # Loop over groups
            training_start = time.time()

            # TODO: check eval agents before trianing lil bro
            # self.n_iters_performed

            # group_map isn't reset every loop
            if self.n_iters_performed in self.freeze_timeline:
                self.train_group_map = copy.deepcopy(self.group_map)

                agents_to_freeze = self.freeze_timeline[self.n_iters_performed]
                if agents_to_freeze is not None:
                    for agent in agents_to_freeze:
                        self.train_group_map.pop(agent)

            # print(f"DEBUG train_group_map={self.train_group_map}")
            for group in self.train_group_map.keys():
                # print(f"DEBUG training group={group}")
                group_batch = batch.exclude(*self._get_excluded_keys(group))
                group_batch = self.algorithm.process_batch(group, group_batch)
                group_batch = group_batch.reshape(-1)
                self.replay_buffers[group].extend(group_batch)

                training_tds = []
                for _ in range(self.config.n_optimizer_steps(self.on_policy)):
                    for _ in range(
                        self.config.train_batch_size(self.on_policy)
                        // self.config.train_minibatch_size(self.on_policy)
                    ):
                        training_tds.append(self._optimizer_loop(group))
                training_td = torch.stack(training_tds)
                self.logger.log_training(
                    group, training_td, step=self.n_iters_performed
                )

                # Callback
                self._on_train_end(training_td, group)

                # Exploration update
                if isinstance(self.group_policies[group], TensorDictSequential):
                    explore_layer = self.group_policies[group][-1]
                else:
                    explore_layer = self.group_policies[group]
                if hasattr(explore_layer, "step"):  # Step exploration annealing
                    explore_layer.step(current_frames)

            # Update policy in collector
            self.collector.update_policy_weights_()

            # Timers
            training_time = time.time() - training_start
            iteration_time = collection_time + training_time
            self.total_time += iteration_time
            self.logger.log(
                {
                    "timers/collection_time": collection_time,
                    "timers/training_time": training_time,
                    "timers/iteration_time": iteration_time,
                    "timers/total_time": self.total_time,
                    "counters/current_frames": current_frames,
                    "counters/total_frames": self.total_frames,
                    "counters/iter": self.n_iters_performed,
                },
                step=self.n_iters_performed,
            )

            # Evaluation
            if (
                self.config.evaluation
                and (self.total_frames % self.config.evaluation_interval == 0)
                and (len(self.config.loggers) or self.config.create_json)
            ):
                self._evaluation_loop()

            # End of step
            self.n_iters_performed += 1
            self.logger.commit()
            if (
                self.config.checkpoint_interval > 0
                and self.total_frames % self.config.checkpoint_interval == 0
            ):
                self._save_experiment()
            pbar.update()
            sampling_start = time.time()

        self.close()
