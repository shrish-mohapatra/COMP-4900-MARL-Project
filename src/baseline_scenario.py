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
        world = World(batch_dim=batch_dim, device=device, dim_c=6)

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

        # Add speaker agents
        num_speakers = 2
        # for i in range(num_speakers):
        #     name = f"speaker_{i+1}"
        #     agent = Agent(
        #         name=name,
        #         collide=False,
        #         movable=False,
        #         silent=False,
        #         shape=Sphere(radius=0.075),
        #     )
        #     world.add_agent(agent)

        # Create map from agent name to agent
        self.agent_map = {}
        for agent in world.agents:
            self.agent_map[agent.name] = agent

        landmark = Landmark(
            name=f"target", collide=False, shape=Sphere(radius=0.04)
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

    def reward(self, agent: Agent):
        # squared distance from listener to landmark

        # TODO: maybe change this to cummulative
        self.rew = -torch.sqrt(
            torch.sum(
                torch.square(self.world.target.state.pos - self.agent_map["listener_0"].state.pos), dim=-1
            )
        )

        return self.rew

    def observation(self, agent):
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
