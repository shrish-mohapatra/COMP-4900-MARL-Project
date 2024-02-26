#  Copyright (c) 2022-2023.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch
import numpy as np

from vmas.simulator.core import World, Agent, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario


class Scenario(BaseScenario):
    def make_world(self, batch_dim: int, device: torch.device, **kwargs):
        world = World(batch_dim=batch_dim, device=device, dim_c=3)
        # set any world properties first
        num_agents = 3
        num_landmarks = 1

        # Add agents
        for i in range(num_agents):
            speaker = False if i == num_agents-1 else True
            name = f"listener_{i}" if not speaker else f"speaker_{i}"
            agent = Agent(
                name=name,
                collide=False,
                movable=False if speaker else True,
                silent=False if speaker else True,
                shape=Sphere(radius=0.075),
            )
            world.add_agent(agent)
        # Add landmarks
        for i in range(num_landmarks):
            landmark = Landmark(
                name=f"landmark {i}", collide=False, shape=Sphere(radius=0.04)
            )
            world.add_landmark(landmark)

        return world

    def reset_world_at(self, env_index: int = None):
        if env_index is None:
            # # assign goals to agents
            # for agent in self.world.agents:
            #     agent.goal_a = None
            #     agent.goal_b = None
            # # want listener to go to the goal landmark
            # self.world.agents[0].goal_a = self.world.agents[1]
            # self.world.agents[0].goal_b = self.world.landmarks[
            #     torch.randint(0, len(self.world.landmarks), (1,)).item()
            # ]

            # assign goals to agents
            for agent in self.world.agents:
                agent.goal_a = self.world.agents[len(self.world.agents)-1]
                agent.goal_b = self.world.landmarks[
                    torch.randint(0, len(self.world.landmarks), (1,)).item()
                ]
            # want listener to go to the goal landmark
            self.world.agents[len(self.world.agents)-1].goal_a = None
            self.world.agents[len(self.world.agents)-1].goal_b = None

            # random properties for agents
            blue_guy = False
            for i, agent in enumerate(self.world.agents):
                rgb = np.random.rand(3)
                if agent.silent:
                    agent.color = torch.tensor(
                        [rgb[0]/4, 0.85, rgb[2]/4], device=self.world.device, dtype=torch.float32
                    )
                    if not blue_guy:
                        agent.color = torch.tensor(
                            [rgb[0]/4, rgb[1]/4, 0.85], device=self.world.device, dtype=torch.float32
                        )
                        blue_guy = True
                else:
                    agent.color = torch.tensor(
                        [rgb[0]/4, rgb[1]/4, 0.85], device=self.world.device, dtype=torch.float32
                    )

            # random properties for landmarks
            for i, landmark in enumerate(self.world.landmarks):
                rgb = np.random.rand(3)
                landmark.color = torch.tensor(
                    [rgb[0], rgb[1], rgb[2]], device=self.world.device, dtype=torch.float32
                )
            # self.world.landmarks[0].color = torch.tensor(
            #     [0.65, 0.15, 0.15], device=self.world.device, dtype=torch.float32
            # )
            # self.world.landmarks[1].color = torch.tensor(
            #     [0.15, 0.65, 0.15], device=self.world.device, dtype=torch.float32
            # )
            # self.world.landmarks[2].color = torch.tensor(
            #     [0.15, 0.15, 0.65], device=self.world.device, dtype=torch.float32
            # )
            # self.world.landmarks[3].color = torch.tensor(
            #     [0.15, 0.15, 0.65], device=self.world.device, dtype=torch.float32
            # )
            # special colors for goals
            # self.world.agents[0].goal_a.color = self.world.agents[
            #     0
            # ].goal_b.color + torch.tensor(
            #     [0.45, 0.45, 0.45], device=self.world.device, dtype=torch.float32
            # )
            for agent in self.world.agents:
                if not agent.silent:
                    agent.goal_a.color = agent.goal_b.color + torch.tensor(
                        [0.45, 0.45, 0.45], device=self.world.device, dtype=torch.float32
                    )

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

    def reward(self, agent: Agent):
        # squared distance from listener to landmark
        is_first = agent == self.world.agents[0]
        if is_first:
            self.rew = torch.zeros(self.world.batch_dim, device=self.world.device)
            for _ in self.world.agents:
                a = self.world.agents[0]
                self.rew += -torch.sqrt(
                    torch.sum(
                        torch.square(a.goal_a.state.pos - a.goal_b.state.pos), dim=-1
                    )
                )
        return self.rew

    def observation(self, agent):
        # goal color
        goal_color = torch.zeros(3, device=self.world.device, dtype=torch.float32)
        if agent.goal_b is not None:
            goal_color = agent.goal_b.color

        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in self.world.landmarks:
            entity_pos.append(entity.state.pos - agent.state.pos)

        # communication of all other agents
        comm = []
        for other in self.world.agents:
            if other is agent or (other.state.c is None):
                continue
            comm.append(other.state.c)

        # speaker
        if not agent.movable:
            return goal_color.repeat(self.world.batch_dim, 1)
        # listener
        if agent.silent:
            return torch.cat([agent.state.vel, *entity_pos, *comm], dim=-1)
            # return torch.cat([*comm], dim=-1)
