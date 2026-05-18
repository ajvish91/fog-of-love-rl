from __future__ import annotations

import datetime
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from pprint import pprint
from typing import List, Optional, Tuple, Type

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from tqdm import tqdm

from fol_env import FoLEnvironment
from fol_training.affinity_condition import check_affinity_condition
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer


@dataclass
class FolTrainingConfig:
    """Unified configuration for former arg_main / basic_main / mul_main / vanilla_main drivers."""

    variant: str
    gpu: str
    debug: bool
    abrl: bool
    lambda_: float
    affinity_p1: List[int]
    affinity_p2: List[int]
    abrl_reg_inputs: Optional[Tuple[Tuple[int, float], Tuple[int, float]]] = None


def _env_truthy(name: str) -> bool:
    """True if env var is set to a common truthy string (used for CI/pytest hooks)."""
    v = os.environ.get(name, "").strip().lower()
    return v in ("1", "true", "yes", "on")


def _maddpg_class(variant: str) -> Type:
    if variant in ("local_abrl", "multi_local_abrl"):
        from agilerl.algorithms.maddpg_local_abrl import MADDPG

        return MADDPG
    if variant == "standard_maddpg":
        from agilerl.algorithms.maddpg import MADDPG

        return MADDPG
    if variant == "maddpg_abrl":
        from agilerl.algorithms.maddpg_abrl import MADDPG

        return MADDPG
    raise ValueError(f"Unknown training variant: {variant!r}")


class Run:
    def __init__(self, check_affinity_condition, config: FolTrainingConfig):
        v = config.variant
        gpu = config.gpu
        debug = config.debug
        is_abrl = config.abrl
        MADDPG = _maddpg_class(v)

        self._variant = v
        self.device = torch.device("cuda:" + gpu if torch.cuda.is_available() else "cpu")
        self.env = FoLEnvironment()
        dt = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S')
        self.debug = debug
        self.abrl = is_abrl
        self.figure_file = "plots/maddpg1_" + dt
        self.test_figure_file = "plots/test_maddpg1_" + dt

        self.log_test_trajectories = v == "local_abrl"
        self.pareto_mode = self.log_test_trajectories

        self.get_dims()

        self.channels_last = False  # Swap image channels dimension from last to first [H, W, C] -> [C, H, W]
        self.n_agents = self.env.num_agents
        self.agent_ids = [agent_id for agent_id in self.env.agents]
        self.field_names = ["state", "action", "reward", "next_state", "done"]
        self.memory = MultiAgentReplayBuffer(
            memory_size=1_000_000,
            field_names=self.field_names,
            agent_ids=self.agent_ids,
            device=self.device,
        )
        print(sum(sum(x) for x in self.state_dim))
        NET_CONFIG = {
            'arch': 'mlp',
            'hidden_size': [1024, 1024, 1024],
        }

        self.lr_actor = 1e-5
        self.lr_critic = 1e-2

        if debug:
            self.epochs = 1
        else:
            self.epochs = 10

        if is_abrl and v == "maddpg_abrl":
            assert config.abrl_reg_inputs is not None
            self.lambda_ = 5.0
            inputs1, inputs2 = config.abrl_reg_inputs
            player_1_params = [
                inputs1[1] if x == inputs1[0] else (1 - inputs1[1]) / (len(inputs1))
                for x in range(3)
            ]
            player_2_params = [
                inputs2[1] if x == inputs2[0] else (1 - inputs2[1]) / (len(inputs2))
                for x in range(3)
            ]
            self.reg_params = {"player_1": player_1_params, "player_2": player_2_params}
            if self.debug:
                print(self.reg_params)
            self.affinity_indices = {"player_1": [20], "player_2": [20]}
            self.agent = MADDPG(
                state_dims=self.state_dim,
                action_dims=self.action_dim,
                one_hot=self.one_hot,
                n_agents=self.n_agents,
                agent_ids=self.agent_ids,
                max_action=self.max_action,
                min_action=self.min_action,
                discrete_actions=self.discrete_actions,
                lr_actor=self.lr_actor,
                lr_critic=self.lr_critic,
                reg_params=self.reg_params,
                lambda_=self.lambda_,
                device=self.device,
                net_config=NET_CONFIG,
            )
            self.project = 'fol-abrl-agile-implement'
        elif is_abrl:
            if v == "local_abrl":
                self.lambda_ = config.lambda_
            else:
                self.lambda_ = 5.0
            self.reg_params = {"player_1": [0.2, 0.6, 0.2], "player_2": [0.2, 0.6, 0.2]}
            self.affinity_indices = {"player_1": list(config.affinity_p1), "player_2": list(config.affinity_p2)}
            self.affinity_condition = check_affinity_condition
            self.agent = MADDPG(
                state_dims=self.state_dim,
                action_dims=self.action_dim,
                one_hot=self.one_hot,
                n_agents=self.n_agents,
                agent_ids=self.agent_ids,
                max_action=self.max_action,
                min_action=self.min_action,
                affinity_indices=self.affinity_indices,
                affinity_condition=self.affinity_condition,
                discrete_actions=self.discrete_actions,
                lr_actor=self.lr_actor,
                lr_critic=self.lr_critic,
                reg_params=self.reg_params,
                lambda_=self.lambda_,
                device=self.device,
                net_config=NET_CONFIG,
            )
            if v == "multi_local_abrl":
                self.project = 'fol-multi-local-abrl-agile-implement'
            else:
                self.project = 'fol-local-abrl-agile-implement'
        else:
            self.agent = MADDPG(
                state_dims=self.state_dim,
                action_dims=self.action_dim,
                one_hot=self.one_hot,
                n_agents=self.n_agents,
                agent_ids=self.agent_ids,
                max_action=self.max_action,
                min_action=self.min_action,
                discrete_actions=self.discrete_actions,
                lr_actor=self.lr_actor,
                lr_critic=self.lr_critic,
                device=self.device,
                net_config=NET_CONFIG,
            )
            self.project = 'fol-agile-implement'

        if not debug:
            if is_abrl and v == "maddpg_abrl":
                self.wandy = wandb.init(
                    project=self.project,
                    config={
                        "model_name": "maddpg",
                        "lr_actor": self.lr_actor,
                        "lr_critic": self.lr_critic,
                        "epochs": self.epochs,
                        "network": NET_CONFIG["hidden_size"],
                        "reg_params": self.reg_params,
                        "lambda_": self.lambda_,
                    },
                )
            elif is_abrl:
                self.wandy = wandb.init(
                    project=self.project,
                    config={
                        "model_name": "maddpg",
                        "lr_actor": self.lr_actor,
                        "lr_critic": self.lr_critic,
                        "epochs": self.epochs,
                        "network": NET_CONFIG["hidden_size"],
                        "reg_params": self.reg_params,
                        "lambda_": self.lambda_,
                        "affinity_indices": self.affinity_indices,
                    },
                )
            else:
                self.wandy = wandb.init(
                    project=self.project,
                    config={
                        "model_name": "maddpg",
                        "lr_actor": self.lr_actor,
                        "lr_critic": self.lr_critic,
                        "epochs": self.epochs,
                        "network": NET_CONFIG["hidden_size"],
                    },
                )

        if v == "maddpg_abrl":
            self.episodes = 2 if debug else 25000
        else:
            self.episodes = 20 if debug else 25000

        self.avg_scores = []
        self.losses = []
        self.max_steps = 100
        self.epsilon = 1.0
        self.eps_end = 0.1
        self.eps_decay = 0.995

        self.test_env = FoLEnvironment()
        if v == "local_abrl":
            self.test_num_episodes = 40 if debug else 20
        elif v == "maddpg_abrl":
            self.test_num_episodes = 2 if debug else 200
        else:
            self.test_num_episodes = 40 if debug else 200

        if debug and _env_truthy("FOL_TEST_MINIMAL"):
            # Pytest / CI: keep debug=1 (no wandb, single training epoch) but shorten env rollouts.
            self.episodes = 1
            self.test_num_episodes = min(self.test_num_episodes, 4)

        self.test_scores = []
        self.avg_test_scores = []

        self.virtues = [[agent_id + "_" + name for name in self.test_env.personality_names] for agent_id in self.agent_ids]
        self.satisfactions = [[agent_id + "_" + self.test_env.satisfaction_name] for agent_id in self.agent_ids]
        self.goals = [[agent_id + "_goal_" + name for name in self.test_env.goal_names] for agent_id in self.agent_ids]
        self.agent_reward = {agent_id: 0 for agent_id in self.test_env.agents}
        self.init_state_info = []
        self.final_state_info = []
        self.test_actions = []
        self.test_actions_history = []
        self.test_observations_history = []




    def get_dims(self, ):
        try:
            self.state_dim = [self.env.observation_space(agent).n for agent in self.env.agents]
            self.one_hot = False
        except TypeError:
            self.state_dim = []
            for i, agent in enumerate(self.env.agents):
                self.state_dim.append([len(self.env.observation_space[agent])])
            self.one_hot = False
        except Exception:
            self.state_dim = [self.env.observation_space(agent).shape for agent in self.env.agents]
            self.one_hot = False
        try:
            self.action_dim = [self.env.action_space(agent).n for agent in self.env.agents]
            self.discrete_actions = True
            self.max_action = None
            self.min_action = None
        except TypeError:
            self.action_dim = []
            self.max_action = []
            self.min_action = []
            for i, agent in enumerate(self.env.agents):
                self.action_dim.append(self.env.action_space[agent].n)
                self.max_action.append([self.env.action_space[agent].n - 1])
                self.min_action.append([0])
            self.discrete_actions = True
        except Exception:
            self.action_dim = [self.env.action_space(agent).shape[0] for agent in self.env.agents]
            self.discrete_actions = False
            self.max_action = [self.env.action_space(agent).high for agent in self.env.agents]
            self.min_action = [self.env.action_space(agent).low for agent in self.env.agents]

        print(self.state_dim, self.one_hot, self.action_dim, self.discrete_actions, self.max_action, self.min_action)


    def train(self, ):
        for ep in tqdm(range(self.episodes)):
            state, info  = self.env.reset() # Reset environment at start of episode
            #env.reset()
            agent_reward = {agent_id: 0 for agent_id in self.env.agents}
            if self.channels_last:
                state = {agent_id: np.moveaxis(np.expand_dims(s, 0), [3], [1]) for agent_id, s in self.env.state.items()}
            state = {key: np.array([value[sub_key] for sub_key in value]) for key, value in state.items()}
            #print(state)
            for _ in range(self.max_steps):
                #if self.debug:
                #    print("time step", _)
                agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
                env_defined_actions = (
                    info["env_defined_actions"]
                    if "env_defined_actions" in info.keys()
                    else None
                )
                #print(np.array(state))
                # Get next action from agent
                cont_actions, discrete_action = self.agent.get_action(
                    state, #epsilon, agent_mask, env_defined_actions
                )
                #print(cont_actions, discrete_action)
                if self.agent.discrete_actions:
                    #print("discrete")
                    action = discrete_action
                else:
                    #print("cont")
                    action = cont_actions

                #print(action)
                actions = {pl: act[0] for pl, act in action.items()}
                next_state, reward, termination, truncation, info = self.env.step(
                    actions
                )  # Act in environment

                # Define the number of possible actions
                num_actions = 3

                # Create an empty dictionary for storing the one-hot encoded arrays
                one_hot_dict = {}

                # Iterate over the items in the action dictionary
                for key, value in discrete_action.items():
                    # Create a tensor of zeros with shape (num_actions,)
                    one_hot_action = np.zeros(num_actions)
                    # Set the value of the one-hot encoding for the action to 1
                    one_hot_action[value] = 1
                    # Add the one-hot encoded array to the result dictionary
                    one_hot_dict[key] = one_hot_action

                #print("discrete", discrete_action, "one_hot_dict", one_hot_dict)
                # Save experiences to replay buffer
                next_state = {key: np.array([value[sub_key] for sub_key in value]) for key, value in next_state.items()}
                self.memory.save_to_memory_single_env(state, one_hot_dict, reward, next_state, termination)

                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                # Learn according to learning frequency
                if (self.memory.counter % self.agent.learn_step == 0) and (len(
                        self.memory) >= self.agent.batch_size):
                    #print("sample memory", memory.sample(2))
                    experiences = self.memory.sample(self.agent.batch_size) # Sample replay buffer
                    #print("experiences", experiences)
                    #states, actions, rewards, next_states, dones = experiences
                    # Convert actions list (a NumPy array) to a PyTorch tensor
                    #action_values = list(actions.values())
                    #print(len(action_values))
                    #print(states.values())
                    # Convert states (a list of tensors) to a tensor list and move them to the correct device
                    #input_combined = torch.cat(list(states.values()) + action_values, 1)

                    #print("action_values", action_values[1].shape)
                    #print("input_combined", input_combined.shape)
                    #q_value = critic(input_combined)
                    for i in range(self.epochs):
                        #if self.debug:
                            #print("Epoch ", i)
                        loss_dict = self.agent.learn(experiences) # Learn according to agent's RL algorithm
                        self.losses.append(loss_dict)
                # Update the state
                if self.channels_last:
                    next_state = {agent_id: np.expand_dims(ns,0) for agent_id, ns in next_state.items()}


                state = next_state
                #print(truncation, termination)
                # Stop episode if any agents have terminated
                if any(value for value in truncation.values()) or any(value for value in termination.values()):
                    break

            # Save the total episode reward
            score = sum(agent_reward.values())
            #print(score)
            self.agent.scores.append(score)

            if not self.debug:
                self.wandy.log({'player_1_score': agent_reward["player_1"]})
                self.wandy.log({'player_2_score': agent_reward["player_2"]})
                self.wandy.log({'average_episode_score': np.mean(self.agent.scores[-100:])})
                self.wandy.log({'average_p1_actor_loss': np.mean([loss["player_1"][0] for loss in self.losses[-100:]])})
                self.wandy.log({'average_p2_actor_loss': np.mean([loss["player_2"][0] for loss in self.losses[-100:]])})
                self.wandy.log({'average_p1_critic_loss': np.mean([loss["player_1"][1] for loss in self.losses[-100:]])})
                self.wandy.log({'average_p2_critic_loss': np.mean([loss["player_2"][1] for loss in self.losses[-100:]])})
                #print(self.agent.L)
                if self.abrl:
                    self.wandy.log({'average_p1_affinity_loss': np.mean([loss for loss in self.agent.L["player_1"][-100:]])})
                    self.wandy.log({'average_p2_affinity_loss': np.mean([loss for loss in self.agent.L["player_2"][-100:]])})
            # Update epsilon for exploration
            #self.epsilon = max(eps_end, epsilon * eps_decay)

            self.avg_scores.append(np.mean(self.agent.scores[-100:]))


    def test(self, ):
        init_state, info  = self.test_env.reset() # Reset environment at start of episode
        #pprint(state)
        indices = {}
        for agent_id in self.agent_ids:
            #print(state[agent_id])
            indices[agent_id] = {}
            for id, agent_state in enumerate(init_state[agent_id]):
                #print(agent_id, agent_state)
                indices[agent_id][agent_state] = id
        for ep in tqdm(range(self.test_num_episodes)):
            state, info  = self.test_env.reset() # Reset environment at start of episode
            agent_reward = {agent_id: 0 for agent_id in self.env.agents}
            #pprint(state)
            state = {key: np.array([value[sub_key] for sub_key in value]) for key, value in state.items()}
            test_score = 0
            player_stats = {}
            episode_observations = [] if self.log_test_trajectories else None
            episode_actions = [] if self.log_test_trajectories else None
            #print(state)
            for ind, agent_id in enumerate(self.agent_ids):
                #print(state[agent_id])
                for virtue in self.virtues[ind]:
                    #print(indices)
                    #print(indices[agent_id])
                    #print(virtue)
                    #print(indices[agent_id][virtue])
                    player_stats[virtue] = state[agent_id][indices[agent_id][virtue]]
                for goal in self.goals[ind]:
                    player_stats[goal] = state[agent_id][indices[agent_id][goal]]
                for satisfaction in self.satisfactions[ind]:
                    player_stats[satisfaction] = state[agent_id][indices[agent_id][satisfaction]]
            self.init_state_info.append(player_stats)
            #print(len(init_state_info))
            for _ in range(self.max_steps):
                test_env_defined_actions = (
                        info["env_defined_actions"]
                        if "env_defined_actions" in info.keys()
                        else None
                    )
                #pprint(state)
                # Get next action from agent
                cont_actions, discrete_action = self.agent.get_action(
                    state,# epsilon, agent_mask, env_defined_actions
                )
                #pprint(discrete_action)
                if self.agent.discrete_actions:
                    action = discrete_action
                else:
                    action = cont_actions

                actions = {pl: act[0] for pl, act in action.items()}
                self.test_actions.append(action)
                if self.log_test_trajectories:
                    episode_actions.append(action)
                    episode_observations.append(state)
                next_state, reward, termination, truncation, info = self.test_env.step(
                    actions
                )  # Act in environment

                next_state = {key: np.array([value[sub_key] for sub_key in value]) for key, value in next_state.items()}

                for agent_id, r in reward.items():
                    agent_reward[agent_id] += r

                state = next_state
                if any(value for value in truncation.values()) or any(value for value in termination.values()):
                    break

            if self.log_test_trajectories:
                self.test_actions_history.append(episode_actions)
                self.test_observations_history.append(episode_observations)

            player_stats = {}
            for ind, agent_id in enumerate(self.agent_ids):
                for virtue in self.virtues[ind]:
                    player_stats[virtue] = state[agent_id][indices[agent_id][virtue]]
                for satisfaction in self.satisfactions[ind]:
                    player_stats[satisfaction] = state[agent_id][indices[agent_id][satisfaction]]
            self.final_state_info.append(player_stats)
            # Save the total episode reward
            test_score = sum(agent_reward.values())
            #print(score)

            self.test_scores.append(test_score)
            if not self.debug:
                self.wandy.log({'player_1_test_score': agent_reward["player_1"]})
                self.wandy.log({'player_2_test_score': agent_reward["player_2"]})
                self.wandy.log({'average_test_episode_score': np.mean(self.test_scores[-10:])})

            self.avg_test_scores.append(np.mean(self.test_scores[-10:]))
        if not self.debug and self.log_test_trajectories:
            self.wandy.log({"test_actions": self.test_actions_history})
            self.wandy.log({"test_observations": self.test_observations_history})


    def plot_analysis(self,):
        fig, ax = plt.subplots()
        ax.bar(range(0, len(self.test_scores)), self.test_scores)
        plt.savefig(self.test_figure_file + "_avg_score.png")
        if not self.debug:
            self.wandy.log({"test_scores": wandb.Image(self.test_figure_file + "_avg_score.png", caption="Average Test Scores")})

        plt.clf()
        dict_action = {
            "00": 0,
            "01": 0,
            "02": 0,
            "10": 0,
            "11": 0,
            "12": 0,
            "20": 0,
            "21": 0,
            "22": 0,
        }
        for test_action in self.test_actions:
            if test_action["player_1"] == test_action["player_2"]:
                if test_action["player_1"] == 0:
                    dict_action["00"] += 1
                elif test_action["player_1"] == 1:
                    dict_action["11"] += 1
                else:
                    dict_action["22"] += 1
            else:
                if test_action["player_1"] == 0 and test_action["player_2"] == 1:
                    dict_action["01"] += 1
                elif test_action["player_1"] == 1 and test_action["player_2"] == 0:
                    dict_action["10"] += 1
                elif test_action["player_1"] == 1 and test_action["player_2"] == 2:
                    dict_action["12"] += 1
                elif test_action["player_1"] == 2 and test_action["player_2"] == 1:
                    dict_action["21"] += 1
                elif test_action["player_1"] == 0 and test_action["player_2"] == 2:
                    dict_action["02"] += 1
                else:
                    dict_action["20"] += 1

        plt.bar(dict_action.keys(), dict_action.values())
        plt.savefig(self.test_figure_file + "_action_dist.png")
        mean_value = {}
        if sum(list(dict_action.values())) != 0:
            for key, val in dict_action.items():
                mean_value[key] = val/sum(list(dict_action.values()))
            print(mean_value)

            if not self.debug:
                self.wandy.log({"probability_" + key: value for key, value in mean_value.items()})
        if not self.debug:
            self.wandy.log({"act_dist": wandb.Image(self.test_figure_file + "_action_dist.png", caption="Action Distribution")})
            self.wandy.log({"act_count": dict_action})
        plt.clf()


    def analyze_virtues(self, ):

        goals_to_accomplish = []
        for state_info in self.init_state_info:
            episode_goal = {}
            for key in state_info.keys():
                if "goal" in key and state_info[key] != 0:
                    episode_goal[key.replace("_goal", "")] = state_info[key]
            goals_to_accomplish.append(episode_goal)

        success_rate = []
        total_success_rate = 0
        for i, final_state in enumerate(self.final_state_info):
            goal_dict = {
                "satisfied": {},
                "unsatisfied": {}
            }
            for key in final_state.keys():
                if key in goals_to_accomplish[i].keys():
                    if goals_to_accomplish[i][key] > 0:
                        if final_state[key] >= goals_to_accomplish[i][key]:
                            goal_dict["satisfied"][key] = (goals_to_accomplish[i][key], final_state[key])
                        else:
                            goal_dict["unsatisfied"][key] = (goals_to_accomplish[i][key], final_state[key])
                    else:
                        if final_state[key] <= goals_to_accomplish[i][key]:
                            goal_dict["satisfied"][key] = (goals_to_accomplish[i][key], final_state[key])
                        else:
                            goal_dict["unsatisfied"][key] = (goals_to_accomplish[i][key], final_state[key])
                if (len(goal_dict["satisfied"]) + len(goal_dict["unsatisfied"])) != 0:
                    goal_dict["success_rate"] = len(goal_dict["satisfied"]) / (len(goal_dict["satisfied"]) + len(goal_dict["unsatisfied"]))
                else:
                    goal_dict["success_rate"] = 0
            total_success_rate += goal_dict["success_rate"]
            success_rate.append(goal_dict)
        average_success_rate = total_success_rate/self.test_num_episodes

        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.title('Episode-wise success rate')

        plt.bar(np.arange(self.test_num_episodes), [success["success_rate"] for success in success_rate])
        plt.savefig(self.test_figure_file + "_success_rate.png")
        if not self.debug:
            self.wandy.log({"success_rate": wandb.Image(self.test_figure_file + "_success_rate.png", caption="Success Rate")})
            self.wandy.log({"success_rate_dict": success_rate})
            self.wandy.log({"average_success_rate": average_success_rate})
        plt.clf()
        pprint(success_rate)
        pprint(average_success_rate)
        return success_rate

    def player_wise_virtue_analysis(self, ):

        p1_goals = []
        p2_goals = []
        for state_info in self.init_state_info:
            p1_episode_goal = {}
            p2_episode_goal = {}
            for key in state_info.keys():
                if "player_1_goal" in key and state_info[key] != 0:
                    p1_episode_goal[key.replace("_goal", "")] = state_info[key]
                if "player_2_goal" in key and state_info[key] != 0:
                    p2_episode_goal[key.replace("_goal", "")] = state_info[key]
            p1_goals.append(p1_episode_goal)
            p2_goals.append(p2_episode_goal)


        p1_success_rate = []
        p2_success_rate = []

        for i, final_state in enumerate(self.final_state_info):
            p1_goal_dict = {
                "satisfied": {},
                "unsatisfied": {}
            }
            p2_goal_dict = {
                "satisfied": {},
                "unsatisfied": {}
            }
            for key in final_state.keys():
                if key in p1_goals[i].keys():
                    if p1_goals[i][key] > 0:
                        if final_state[key] >= p1_goals[i][key]:
                            p1_goal_dict["satisfied"][key] = (p1_goals[i][key], final_state[key])
                        else:
                            p1_goal_dict["unsatisfied"][key] = (p1_goals[i][key], final_state[key])
                    else:
                        if final_state[key] <= p1_goals[i][key]:
                            p1_goal_dict["satisfied"][key] = (p1_goals[i][key], final_state[key])
                        else:
                            p1_goal_dict["unsatisfied"][key] = (p1_goals[i][key], final_state[key])
                if key in p2_goals[i].keys():
                    if p2_goals[i][key] > 0:
                        if final_state[key] >= p2_goals[i][key]:
                            p2_goal_dict["satisfied"][key] = (p2_goals[i][key], final_state[key])
                        else:
                            p2_goal_dict["unsatisfied"][key] = (p2_goals[i][key], final_state[key])
                    else:
                        if final_state[key] <= p2_goals[i][key]:
                            p2_goal_dict["satisfied"][key] = (p2_goals[i][key], final_state[key])
                        else:
                            p2_goal_dict["unsatisfied"][key] = (p2_goals[i][key], final_state[key])
                if (len(p1_goal_dict["satisfied"]) + len(p1_goal_dict["unsatisfied"])) != 0:
                    p1_goal_dict["success_rate"] = len(p1_goal_dict["satisfied"]) / (len(p1_goal_dict["satisfied"]) + len(p1_goal_dict["unsatisfied"]))
                else:
                    p1_goal_dict["success_rate"] = 0
                if (len(p2_goal_dict["satisfied"]) + len(p2_goal_dict["unsatisfied"])) != 0:
                    p2_goal_dict["success_rate"] = len(p2_goal_dict["satisfied"]) / (len(p2_goal_dict["satisfied"]) + len(p2_goal_dict["unsatisfied"]))
                else:
                    p2_goal_dict["success_rate"] = 0
            p1_success_rate.append(p1_goal_dict)
            p2_success_rate.append(p2_goal_dict)



        # Width of each bar
        bar_width = 0.45

        # Adjust the x positions for each bar
        x_pos1 = [x for x in np.arange(self.test_num_episodes)]
        x_pos2 = [x + bar_width for x in x_pos1]
        plt.clf()
        plt.bar(x_pos1, [success["success_rate"] for success in p1_success_rate], width=0.5, label="Player 1")
        plt.bar(x_pos2, [success["success_rate"] for success in p2_success_rate], width=0.5, label="Player 2")
        plt.xlabel('Success Rate')
        plt.ylabel('Episode')
        plt.title('Success rate per episode')
        plt.xticks([x + bar_width/2 for x in x_pos1], x_pos1)
        plt.legend()
        plt.savefig(self.test_figure_file + "_player_success_rate.png")

        if not self.debug:
            self.wandy.log({"player_success_rate": wandb.Image(self.test_figure_file + "_player_success_rate.png", caption="Player Success Rate"),
                "avg_p1_success_rate": np.mean(np.array([success["success_rate"] for success in p1_success_rate])),
                "avg_p2_success_rate": np.mean(np.array([success["success_rate"] for success in p2_success_rate]))})

        pprint(p1_success_rate)

        return p1_success_rate

    def get_affinity_success_rate(self, success_rate):
        init_state, _ = self.test_env.reset()
        affinity_keys = []
        for key in self.affinity_indices:
            for index in range(14, 21):
                affinity_keys.append(list(init_state[key].keys())[index])
        if self.debug:
            print(affinity_keys)
        check_affinity = [affinity_key.replace("_goal", "") for affinity_key in affinity_keys]

        affinity_success_count = {affinity_key: 0 for affinity_key in check_affinity}
        affinity_failure_count = {affinity_key: 0 for affinity_key in check_affinity}
        for success in success_rate:
            for affinity in check_affinity:
                if affinity in success["satisfied"].keys():
                    if self.debug:
                        print("success", affinity, success["satisfied"].keys())
                    affinity_success_count[affinity] += 1
                if affinity in success["unsatisfied"].keys():
                    if self.debug:
                        print("failure", affinity, success["unsatisfied"].keys())
                    affinity_failure_count[affinity] += 1
        if not self.debug:
            for (affinity_success_key, affinity_success_value), (affinity_failure_key, affinity_failure_value) in zip(affinity_success_count.items(), affinity_failure_count.items()):
                if affinity_success_value + affinity_failure_value != 0:
                    self.wandy.log({affinity_success_key + "_success_rate": affinity_success_value/(affinity_success_value + affinity_failure_value)})
        return affinity_success_count, affinity_failure_count

    def get_ep_success_rate(self, success_rate):
        coin_errors = []
        for ep in success_rate:
            error = 0
            for value in ep["unsatisfied"].values():
                error += abs(value[0] - value[1])
            coin_errors.append(error)
        mean_coin_error = np.mean(np.array(coin_errors))
        plt.clf()
        plt.bar(np.arange(len(coin_errors)), coin_errors)
        plt.xlabel("Test Episode")
        plt.ylabel("Error")
        plt.plot(mean_coin_error*np.ones(len(coin_errors)), color="black", label="Mean")
        plt.xticks(np.arange(len(coin_errors)))
        plt.legend()
        plt.savefig(self.test_figure_file + "_error.png")
        if not self.debug:
            self.wandy.log({"episode_error": wandb.Image(self.test_figure_file + "_error.png", caption="Episodewise error"),
                "mean_error": mean_coin_error})
        np.std(np.array(coin_errors))
        return mean_coin_error


COMPAT_MODES = frozenset({"arg_main", "basic_main", "vanilla_main", "mul_main"})


def _split_mul_lists(numbers, delimiter):
    try:
        delimiter_index = numbers.index(delimiter)
        return numbers[:delimiter_index], numbers[delimiter_index + 1:]
    except ValueError as exc:
        raise ValueError(f"Delimiter {delimiter!r} not found in the list") from exc


def _validate_mul_list_elements(lst, min_val, max_val):
    for item in lst:
        if not (min_val <= item <= max_val):
            raise ValueError(f"Element {item} in the list is not between {min_val} and {max_val}")


def _parse_bool01(name, raw):
    if raw not in ("0", "1"):
        print(f"Invalid input for {name}: {raw}")
        sys.exit(1)
    return raw == "1"


def run_training_pipeline(run, abrl):
    run.train()
    run.test()
    run.plot_analysis()
    success_rate = run.analyze_virtues()
    if abrl:
        affinity_success, affinity_failure = run.get_affinity_success_rate(success_rate)
        print(affinity_success, affinity_failure)
    mean_coin_error = run.get_ep_success_rate(success_rate)
    print(mean_coin_error)


def main(argv=None):
    argv = list(argv if argv is not None else sys.argv)
    if len(argv) >= 3 and argv[1] in ("--config", "-c"):
        from fol_train_config import argv_suffix_from_yaml

        cfg_path = Path(argv[2])
        try:
            suffix = argv_suffix_from_yaml(cfg_path)
        except Exception as e:
            print(f"Config error: {e}", file=sys.stderr)
            sys.exit(1)
        argv = [argv[0]] + suffix

    print(argv)
    if len(argv) < 2 or argv[1] not in COMPAT_MODES:
        print(
            "Usage: python -m fol_training.run_fol <arg_main|basic_main|vanilla_main|mul_main> ...\n"
            "   or: python -m fol_training.run_fol --config configs/smoke_arg_localized.yaml\n"
            "  arg_main:    <debug 0|1> <abrl 0|1> <gpu 0-11> <aff1 14-20> <aff2 14-20> <lambda float>\n"
            "  basic_main:  <debug> <abrl> <gpu 0-11> <aff1 14-20> <aff2 14-20>\n"
            "  vanilla_main:<debug> <abrl> <gpu 0-11> <reg1 0-2> <reg2 0-2>\n"
            "  mul_main:    <debug> <abrl> <gpu 0-12> <idx> ... '_' ... <idx>  (lists 14-20, '_' separates players)"
        )
        sys.exit(1)

    mode = argv[1]
    rest = argv[2:]

    if mode == "arg_main":
        if len(rest) != 6:
            print("arg_main expects 6 args after mode: debug abrl gpu aff1 aff2 lambda")
            sys.exit(1)
        debug_s, abrl_s, gpu, a1s, a2s, lam_s = rest
        debug = _parse_bool01("debug", debug_s)
        abrl = _parse_bool01("abrl", abrl_s)
        if int(gpu) < 0 or int(gpu) > 11:
            print("Invalid input for gpu" + gpu)
            sys.exit(1)
        aff_idx_1 = int(a1s)
        aff_idx_2 = int(a2s)
        if not (14 <= aff_idx_1 <= 20 and 14 <= aff_idx_2 <= 20):
            print("Invalid affinity index (expected 14-20)")
            sys.exit(1)
        try:
            lambda_ = float(lam_s)
        except ValueError:
            print("lambda_ must be a float value")
            sys.exit(1)
        cfg = FolTrainingConfig(
            variant="local_abrl",
            gpu=str(gpu),
            debug=debug,
            abrl=abrl,
            lambda_=lambda_,
            affinity_p1=[aff_idx_1],
            affinity_p2=[aff_idx_2],
        )
        run = Run(check_affinity_condition, cfg)
        run_training_pipeline(run, abrl)
        return

    if mode == "basic_main":
        if len(rest) != 5:
            print("basic_main expects 5 args after mode: debug abrl gpu aff1 aff2")
            sys.exit(1)
        debug_s, abrl_s, gpu, a1s, a2s = rest
        debug = _parse_bool01("debug", debug_s)
        abrl = _parse_bool01("abrl", abrl_s)
        if abrl:
            print(
                "basic_main with abrl=1 is not supported in the unified driver (stock MADDPG is built without "
                "affinity arguments). Use: python -m fol_training.run_fol arg_main … for localized ABRL.",
                file=sys.stderr,
            )
            sys.exit(1)
        if int(gpu) < 0 or int(gpu) > 11:
            print("Invalid input for gpu" + gpu)
            sys.exit(1)
        aff_idx_1 = int(a1s)
        aff_idx_2 = int(a2s)
        if not (14 <= aff_idx_1 <= 20 and 14 <= aff_idx_2 <= 20):
            print("Invalid affinity index (expected 14-20)")
            sys.exit(1)
        cfg = FolTrainingConfig(
            variant="standard_maddpg",
            gpu=str(gpu),
            debug=debug,
            abrl=abrl,
            lambda_=5.0,
            affinity_p1=[aff_idx_1],
            affinity_p2=[aff_idx_2],
        )
        run = Run(check_affinity_condition, cfg)
        run_training_pipeline(run, abrl)
        return

    if mode == "vanilla_main":
        if len(rest) != 5:
            print("vanilla_main expects 5 args after mode: debug abrl gpu reg1 reg2")
            sys.exit(1)
        debug_s, abrl_s, gpu, r1s, r2s = rest
        debug = _parse_bool01("debug", debug_s)
        abrl = _parse_bool01("abrl", abrl_s)
        if int(gpu) < 0 or int(gpu) > 11:
            print("Invalid input for gpu" + gpu)
            sys.exit(1)
        r1 = int(r1s)
        r2 = int(r2s)
        if not (0 <= r1 <= 2 and 0 <= r2 <= 2):
            print("Invalid reg index (expected 0-2)")
            sys.exit(1)
        cfg = FolTrainingConfig(
            variant="maddpg_abrl",
            gpu=str(gpu),
            debug=debug,
            abrl=abrl,
            lambda_=5.0,
            affinity_p1=[],
            affinity_p2=[],
            abrl_reg_inputs=((r1, 1 / 3), (r2, 1 / 3)),
        )
        run = Run(check_affinity_condition, cfg)
        run_training_pipeline(run, abrl)
        return

    if mode == "mul_main":
        if len(rest) < 4:
            print("mul_main expects at least: debug abrl gpu ...indices... with '_' between player lists")
            sys.exit(1)
        debug_s, abrl_s, gpu, *number_tokens = rest
        debug = _parse_bool01("debug", debug_s)
        abrl = _parse_bool01("abrl", abrl_s)
        gi = int(gpu)
        if gi < 0 or gi > 12:
            print("Invalid input for gpu" + gpu)
            sys.exit(1)
        numbers_str = " ".join(number_tokens)
        numbers = numbers_str.split()
        numbers = [int(x) if x.isdigit() else x for x in numbers]
        try:
            aff_indices_1, aff_indices_2 = _split_mul_lists(numbers, "_")
            _validate_mul_list_elements(aff_indices_1, 14, 20)
            _validate_mul_list_elements(aff_indices_2, 14, 20)
        except ValueError as e:
            print(e)
            sys.exit(1)
        print("Initial Numbers:", debug, abrl, gpu)
        print("List 1:", aff_indices_1)
        print("List 2:", aff_indices_2)
        cfg = FolTrainingConfig(
            variant="multi_local_abrl",
            gpu=str(gpu),
            debug=debug,
            abrl=abrl,
            lambda_=5.0,
            affinity_p1=list(aff_indices_1),
            affinity_p2=list(aff_indices_2),
        )
        run = Run(check_affinity_condition, cfg)
        run_training_pipeline(run, abrl)
        return

    print(f"Unknown mode {mode!r}")
    sys.exit(1)


if __name__ == "__main__":
    main()
