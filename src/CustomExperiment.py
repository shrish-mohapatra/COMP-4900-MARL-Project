import copy
import time
import torch

from benchmarl.experiment import Experiment, ExperimentConfig
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

freeze_timeline = {}


class CustomExperiment(Experiment):

    def _optimizer_loop(self, group: str) -> TensorDictBase:
        subdata = self.replay_buffers[group].sample()
        loss_vals = self.losses[group](subdata)
        training_td = loss_vals.detach()
        loss_vals = self.algorithm.process_loss_vals(group, loss_vals)

        for loss_name, loss_value in loss_vals.items():
            # print(f"DEBUG loss_name={loss_name} loss_value={loss_value}")
            if loss_name in self.optimizers[group].keys():
                optimizer = self.optimizers[group][loss_name]

                loss_value.backward()

                grad_norm = self._grad_clip(optimizer)

                training_td.set(
                    f"grad_norm_{loss_name}",
                    torch.tensor(grad_norm, device=self.config.train_device),
                )

                optimizer.step()
                optimizer.zero_grad()
        self.replay_buffers[group].update_tensordict_priority(subdata)
        if self.target_updaters[group] is not None:
            self.target_updaters[group].step()

        callback_loss = self._on_train_step(subdata, group)
        if callback_loss is not None:
            training_td.update(callback_loss)

        return training_td

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
            if self.n_iters_performed in freeze_timeline:
                self.train_group_map = copy.deepcopy(self.group_map)

                agents_to_freeze = freeze_timeline[self.n_iters_performed]
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

    # def _setup_task(self):
    #     test_env = self.model_config.process_env_fun(
    #         self.task.get_env_fun(
    #             num_envs=self.config.evaluation_episodes,
    #             continuous_actions=self.continuous_actions,
    #             seed=self.seed,
    #             device=self.config.sampling_device,
    #         )
    #     )()
    #     env_func = self.model_config.process_env_fun(
    #         self.task.get_env_fun(
    #             num_envs=self.config.n_envs_per_worker(self.on_policy),
    #             continuous_actions=self.continuous_actions,
    #             seed=self.seed,
    #             device=self.config.sampling_device,
    #         )
    #     )
    #     self.eval_groups = ["listener"]

    #     self.observation_spec = self.task.observation_spec(test_env)
    #     self.info_spec = self.task.info_spec(test_env)
    #     self.state_spec = self.task.state_spec(test_env)
    #     self.action_mask_spec = self.task.action_mask_spec(test_env)
    #     self.action_spec = self.task.action_spec(test_env)
    #     self.group_map = self.task.group_map(test_env)
    #     self.train_group_map = copy.deepcopy(self.group_map)
    #     print(f"DEBUG train_group_map={self.train_group_map}")
    #     for key in self.eval_groups:
    #         self.train_group_map.pop(key)
    #     print(f"DEBUG train_group_map={self.train_group_map}")
    #     self.max_steps = self.task.max_steps(test_env)
    #     transforms = [self.task.get_reward_sum_transform(test_env)]
    #     transform = Compose(*transforms)

    #     if test_env.batch_size == ():
    #         self.env_func = lambda: TransformedEnv(
    #             SerialEnv(self.config.n_envs_per_worker(self.on_policy), env_func),
    #             transform.clone(),
    #         )
    #     else:
    #         self.env_func = lambda: TransformedEnv(env_func(), transform.clone())

    #     self.test_env = test_env.to(self.config.sampling_device)

    # def _setup_collector(self):
    #     self.policy = self.algorithm.get_policy_for_collection()

    #     self.group_policies = {}
    #     eval_groups = ["listener"]
    #     for group in self.group_map.keys():
    #         print(f"DEBUG group={group}")
    #         group_policy = self.policy.select_subsequence(out_keys=[(group, "action")])
    #         assert len(group_policy) == 1

    #         if group in eval_groups:
    #             print(f"DEBUG eval mode for={group}")
    #             group_policy.eval()

    #         self.group_policies.update({group: group_policy[0]})

    #     self.collector = SyncDataCollector(
    #         self.env_func,
    #         self.policy,
    #         device=self.config.sampling_device,
    #         storing_device=self.config.train_device,
    #         frames_per_batch=self.config.collected_frames_per_batch(self.on_policy),
    #         total_frames=self.config.get_max_n_frames(self.on_policy),
    #         init_random_frames=self.config.off_policy_init_random_frames
    #         if not self.on_policy
    #         else 0,
    #     )
