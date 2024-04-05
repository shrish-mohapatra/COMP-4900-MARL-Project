#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
import numpy as np

from vmas.simulator.core import World, Agent, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        """
        Define agents (speaker & listener) and landmar

        agents = [listener_0, speaker_1, speaker_2]
        - listener_0 "police responder", trying to reach landmark
        - speaker_1 "civilian"
        - speaker_2 "policeHQ"
        """
        self.moving_target = False
        if "moving_target" in kwargs:
            self.moving_target = kwargs["moving_target"]

        world = World(batch_dim=batch_dim, device=device, dim_c=6)

        self.MIN_DISTANCE = 0.01

        # Add agents ---

        # Add listener agent
        name = f"listener_0"
        agent = Agent(
            name=name,
            collide=False,
            movable=True,
            silent=True,
            shape=Sphere(radius=0.075),
        )
        world.add_agent(agent)

        ###########################################
        name = f"civilian_1"
        agent = Agent(
            name=name,
            collide=False,
            movable=False,
            silent=False,
            shape=Sphere(radius=0.075),
        )
        world.add_agent(agent)

        name = f"policeHQ_2"
        agent = Agent(
            name=name,
            collide=False,
            movable=False,
            silent=False,
            shape=Sphere(radius=0.075),
        )
        world.add_agent(agent)
        ###############################################

        # Create map from agent name to agent
        self.agent_map = {}
        for agent in world.agents:
            self.agent_map[agent.name] = agent

        landmark = Landmark(
            name=f"target",
            collide=False,
            shape=Sphere(radius=0.04),
            movable=self.moving_target,
        )
        world.add_landmark(landmark)

        # created attribute for easier access
        world.target = landmark
        return world

    def reset_world_at(self, env_index: int = None):
        # set random initial states
        for agent in self.world.agents:
            agent.set_pos(
                torch.zeros(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )
        for landmark in self.world.landmarks:
            landmark.set_pos(
                torch.zeros(
                    (1, self.world.dim_p)
                    if env_index is not None
                    else (self.world.batch_dim, self.world.dim_p),
                    device=self.world.device,
                    dtype=torch.float32,
                ).uniform_(
                    -1.0,
                    1.0,
                ),
                batch_index=env_index,
            )

        if self.moving_target:
            # Choose random direction for target to move in
            self.moving_target_vel = torch.zeros(
                (1, self.world.dim_p),
                device=self.world.device,
                dtype=torch.float32,
            ).uniform_(
                -0.05,
                0.05,
            )

        self.rew = torch.zeros(self.world.batch_dim, device=self.world.device)

    def process_action(self, agent: Agent):
        if self.moving_target:
            # Move target to assigned random direction at each step
            self.world.target.set_vel(
                self.moving_target_vel,
                batch_index=None,
            )

    def reward(self, agent: Agent):
        # squared distance from listener to landmark

        # TODO: maybe change this to cummulative
        self.rew -= torch.sqrt(
            torch.sum(
                torch.square(self.world.target.state.pos - self.agent_map["listener_0"].state.pos), dim=-1
            )
        ) / 100

        cur_distance = torch.linalg.vector_norm(
            (self.agent_map["listener_0"].state.pos -
             self.world.target.state.pos),
            dim=1
        )
        updates = cur_distance <= self.MIN_DISTANCE
        self.rew[updates] += 20000

        return self.rew

    def done(self):
        cur_distance = torch.linalg.vector_norm(
            (self.agent_map["listener_0"].state.pos -
             self.world.target.state.pos),
            dim=1
        )
        result = cur_distance <= self.MIN_DISTANCE
        # if result.any():
        #     print(f"cur_distance={cur_distance}")
        return result
        # listener.pos - goal.pos < threshold -> done :D
        # [ True, False, False, ... n_envs]

    def observation(self, agent: Agent):
        """Compute observation tensor for specific agent"""
        # change this if we increase number of agents
        if agent.name == "listener_0":
            # Aggregate communication states of speaker agents
            comm = [
                other_agent.state.c
                for other_agent in self.world.agents
                if other_agent.state.c is not None
            ]
            obs = torch.cat([*comm], dim=-1)
            # print('listener:', obs)
            return obs

        elif agent.name == "civilian_1":
            # return its distance from itself to the target
            # obs_pos = torch.sub(self.world.target.state.pos, agent.state.pos)
            # obs_pad = torch.zeros(
            #     [self.world.batch_dim, 2], device=self.world.device, dtype=torch.float32)
            # obs = torch.cat([
            #     obs_pos,
            #     obs_pad
            # ], dim=1)

            obs = torch.sub(self.world.target.state.pos, agent.state.pos)
            # print('civilian_1:', obs)
            return obs

        elif agent.name == "policeHQ_2":
            # return its distance from itself to the listener, and speaker_0 to itself
            # [ x, y, x2, y2 ]
            obs = torch.cat([
                self.agent_map["listener_0"].state.pos - agent.state.pos,
                self.agent_map["civilian_1"].state.pos - agent.state.pos,
            ], dim=-1)
            # print('policeHQ_2:', obs)
            return obs

        else:
            raise Exception(f"Unsupported agent {agent.name}")
