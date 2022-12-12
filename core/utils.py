"""
This file implements some helper functions.

-----

2022-2023 fall quarter, CS269 Seminar 5: Reinforcement Learning.
Department of Computer Science at University of California, Los Angeles.
Course Instructor: Professor Bolei ZHOU.
Assignment Author: Zhenghao PENG.
"""
import copy
import json
import time
from collections import deque

import gym
import numpy as np
import torch
import yaml

import glob
import os
from metadrive import (
    MultiAgentMetaDrive, MultiAgentTollgateEnv, MultiAgentBottleneckEnv, MultiAgentIntersectionEnv,
    MultiAgentRoundaboutEnv, MultiAgentParkingLotEnv
)

def verify_log_dir(log_dir, *others):
    if others:
        log_dir = os.path.join(log_dir, *others)
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)
    return os.path.abspath(log_dir)


def step_envs(cpu_actions, envs, episode_rewards, reward_recorder, success_recorder, total_steps, total_episodes,
              device):
    """Step the vectorized environments for one step. Process the reward
    recording and terminal states."""
    obs, reward, done, info = envs.step(cpu_actions)
    episode_rewards += reward.reshape(episode_rewards.shape)
    episode_rewards_old_shape = episode_rewards.shape
    if not np.isscalar(done[0]):
        done = np.all(done, axis=1)
    for idx, d in enumerate(done):
        if d:  # the episode is done
            # Record the reward of the terminated episode to
            reward_recorder.append(episode_rewards[idx].copy())
            if "arrive_dest" in info[idx]:
                success_recorder.append(info[idx]["arrive_dest"])
            total_episodes += 1
    masks = 1. - done.astype(np.float32)
    episode_rewards *= masks.reshape(-1, 1)
    assert episode_rewards.shape == episode_rewards_old_shape
    total_steps += obs[0].shape[0] if isinstance(obs, tuple) else obs.shape[0]
    masks = torch.from_numpy(masks).to(device).view(-1, 1)
    return obs, reward, done, info, masks, total_episodes, total_steps, episode_rewards


def flatten_dict(dt, delimiter="/"):
    dt = copy.deepcopy(dt)
    while any(isinstance(v, dict) for v in dt.values()):
        remove = []
        add = {}
        for key, value in dt.items():
            if isinstance(value, dict):
                for subkey, v in value.items():
                    add[delimiter.join([key, subkey])] = v
                remove.append(key)
        dt.update(add)
        for k in remove:
            del dt[k]
    return dt


def evaluate(trainer, eval_envs, num_episodes=10, seed=0):
    """This function evaluate the given policy and return the mean episode
    reward.
    :param policy: a function whose input is the observation
    :param env: an environment instance
    :param num_episodes: number of episodes you wish to run
    :param seed: the random seed
    :return: the averaged episode reward of the given policy.
    """

    def get_action(obs):
        with torch.no_grad():
            act = trainer.compute_action(obs, deterministic=True)[1]
        if trainer.discrete:
            act = act.view(-1).cpu().numpy()
        else:
            act = act.cpu().numpy()
        return act

    reward_recorder = []
    episode_length_recorder = []
    episode_rewards = np.zeros([eval_envs.num_envs, 1], dtype=np.float)
    total_steps = 0
    total_episodes = 0
    eval_envs.seed(seed)
    obs = eval_envs.reset()
    while True:
        obs, reward, done, info, masks, total_episodes, total_steps, \
        episode_rewards = step_envs(
            get_action(obs), eval_envs, episode_rewards, reward_recorder, episode_length_recorder,
            total_steps, total_episodes, trainer.device)
        if total_episodes >= num_episodes:
            break
    return reward_recorder, episode_length_recorder


class Timer:
    def __init__(self, interval=10):
        self.value = 0.0
        self.start = time.time()
        self.buffer = deque(maxlen=interval)

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.value = time.time() - self.start
        self.buffer.append(self.value)

    @property
    def now(self):
        """Return the seconds elapsed since initializing this class"""
        return time.time() - self.start

    @property
    def avg(self):
        return np.mean(self.buffer, dtype=float)


def pretty_print(result):
    result = result.copy()
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v
    cleaned = json.dumps(out)
    print(yaml.safe_dump(json.loads(cleaned), default_flow_style=False))


def register_metadrive():
    from gym.envs.registration import register
    try:
        from metadrive.envs import MetaDriveEnv, SafeMetaDriveEnv
        from metadrive.utils.config import merge_config_with_unknown_keys
    except ImportError as e:
        print("Please install MetaDrive through: pip install git+https://github.com/decisionforce/metadrive")
        raise e

    env_names = []
    try:
        env_name = "MetaDrive-Tut-Easy-v0"
        make_env = lambda config=None: MetaDriveEnv(merge_config_with_unknown_keys(dict(
            map="S",
            environment_num=1,
            horizon=200,
            start_seed=1000,
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        # This environment is actually identical to the test environment!
        env_name = "MetaDrive-Tut-Hard-v0"
        make_env = lambda config=None: MetaDriveEnv(merge_config_with_unknown_keys(dict(
            start_seed=1000,
            environment_num=20,
            horizon=1000,
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        env_name = "MetaDrive-Tut-Traffic-v0"
        make_env = lambda config=None: MetaDriveEnv(merge_config_with_unknown_keys(dict(
            start_seed=1000,
            environment_num=20,
            random_traffic=True,
            horizon=1000,
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        env_name = "MetaDrive-Gen-v0" #general environment
        make_env = lambda config=None: MetaDriveEnv(merge_config_with_unknown_keys(dict(
            start_seed=0,
            environment_num=500,
            horizon=1000,
            random_lane_width= True, 
            random_lane_num= True,
            traffic_density = 0.2
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        env_name = "MetaDrive-Gen-v1" #speed environment
        make_env = lambda config=None: SafeMetaDriveEnv(merge_config_with_unknown_keys(dict(
            start_seed=1000,
            environment_num=500,
            horizon=1000,
            random_traffic=True,
            speed_reward = 0.4
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        env_name = "MetaDrive-Gen-v2" #clean driving environment
        make_env = lambda config=None: MetaDriveEnv(merge_config_with_unknown_keys(dict(
            start_seed=2000,
            environment_num=500,
            horizon=1000,
            random_lane_width= True, 
            random_lane_num= True,
            driving_reward = 1.3,
            use_lateral_reward = 0.5,
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        env_name = "MetaDrive-Gen-v3" #high traffic
        make_env = lambda config=None: MetaDriveEnv(merge_config_with_unknown_keys(dict(
            start_seed=3000,
            environment_num=500,
            horizon=1000,
            random_traffic=True,
            traffic_density=0.3,
            driving_reward = 1.2
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)


        ##
        env_name = "MetaDrive-Gen-Next-v0" #safe environment
        make_env = lambda config=None: SafeMetaDriveEnv(merge_config_with_unknown_keys(dict(
            start_seed=2000,
            environment_num=500,
            horizon=1000,
            random_lane_width= True, 
            random_lane_num= True,
            driving_reward = 1.3,
            use_lateral_reward = 0.5,
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        env_name = "MetaDrive-Gen-Next-v1" #general with traffic environment
        make_env = lambda config=None: MetaDriveEnv(merge_config_with_unknown_keys(dict(
            start_seed=0,
            environment_num=500,
            horizon=1000,
            random_lane_width= True, 
            random_lane_num= True,
            traffic_density = 0.3
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        env_name = "MetaDrive-Gen-Next-v2" #speed environment
        make_env = lambda config=None: SafeMetaDriveEnv(merge_config_with_unknown_keys(dict(
            start_seed=1000,
            environment_num=500,
            horizon=1000,
            random_traffic=True,
            speed_reward = 0.4
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)


        env_name = "MetaDrive-Alt-Next-v0" #safe environment
        make_env = lambda config=None: SafeMetaDriveEnv(merge_config_with_unknown_keys(dict(
            start_seed=2000,
            environment_num=500,
            horizon=2000,
            random_lane_width= True, 
            random_lane_num= True,
            driving_reward = 1.5,
            use_lateral_reward = 0.9
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        env_name = "MetaDrive-Alt-Next-v1" #speed improvement
        make_env = lambda config=None: SafeMetaDriveEnv(merge_config_with_unknown_keys(dict(
            start_seed=0,
            environment_num=500,
            horizon=2000,
            random_lane_width= True, 
            random_lane_num= True,
            traffic_density = 0.2,
            speed_reward = 0.5,
            driving_reward = 1.5
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        env_name = "MetaDrive-Alt-Next-v2" #add general traffic layer again
        make_env = lambda config=None: MetaDriveEnv(merge_config_with_unknown_keys(dict(
            start_seed=1000,
            environment_num=500,
            horizon=2000,
            random_traffic=True,
            random_lane_width=True,
            random_lane_num = True,
            speed_reward = 0.2
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        ##Dummy MARL env to get lidar config
        env_name = "MetaDrive-MARL-Dummy-v0"
        make_env = lambda config=None: MultiAgentMetaDrive(merge_config_with_unknown_keys(dict(
            use_render=False
        ), config or {}))
        register(id=env_name, entry_point=make_env)

        dummy_env = gym.make("MetaDrive-MARL-Dummy-v0")
        vehicle_config = dummy_env.config["vehicle_config"]
        dummy_env.close()

        dummy_env = gym.make("MetaDrive-Tut-Hard-v0")
        vehicle_config["lidar"] = dummy_env.config["vehicle_config"]["lidar"]
        # vehicle_config["use_special_color"]=True
        dummy_env.close()

        env_name = "MetaDrive-MARL-v0"
        make_env = lambda config=None: MultiAgentMetaDrive(merge_config_with_unknown_keys(dict(
            start_seed=1000,
            environment_num=50,
            horizon=1000,
            num_agents=7,
            use_render=False,
            vehicle_config = vehicle_config
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        env_name = "MetaDrive-MARL-v1"
        make_env = lambda config=None: MultiAgentMetaDrive(merge_config_with_unknown_keys(dict(
            start_seed=1000,
            environment_num=50,
            horizon=500,
            num_agents=5,
            use_render=False,
            vehicle_config = vehicle_config
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        env_name = "MetaDrive-MARL-v2"
        make_env = lambda config=None: MultiAgentMetaDrive(merge_config_with_unknown_keys(dict(
            start_seed=1000,
            environment_num=50,
            horizon=500,
            num_agents=15,
            use_render=False,
            vehicle_config = vehicle_config
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        env_name = "MetaDrive-MARL-v3"
        make_env = lambda config=None: MultiAgentMetaDrive(merge_config_with_unknown_keys(dict(
            start_seed=1000,
            environment_num=50,
            horizon=500,
            num_agents=25,
            use_render=False,
            vehicle_config = vehicle_config
        ), config or {}))
        register(id=env_name, entry_point=make_env)
        env_names.append(env_name)

        # env_name = "MetaDrive-New-Next-v0" #safe environment
        # make_env = lambda config=None: SafeMetaDriveEnv(merge_config_with_unknown_keys(dict(
        #     start_seed=2000,
        #     environment_num=500,
        #     horizon=1000,
        #     random_lane_width= True, 
        #     random_lane_num= True,
        #     driving_reward = 1.5,
        #     crash_vehicle_cost = 1.5,
        #     use_lateral_reward = 1.0,
        # ), config or {}))
        # register(id=env_name, entry_point=make_env)
        # env_names.append(env_name)

        # env_name = "MetaDrive-New-Next-v1" #general with traffic environment
        # make_env = lambda config=None: MetaDriveEnv(merge_config_with_unknown_keys(dict(
        #     start_seed=0,
        #     environment_num=500,
        #     horizon=1500,
        #     random_lane_width= True, 
        #     random_lane_num= True,
        #     traffic_density = 0.4
        # ), config or {}))
        # register(id=env_name, entry_point=make_env)
        # env_names.append(env_name)

        # env_name = "MetaDrive-New-Next-v2" #speed environment
        # make_env = lambda config=None: SafeMetaDriveEnv(merge_config_with_unknown_keys(dict(
        #     start_seed=1000,
        #     environment_num=500,
        #     horizon=1000,
        #     random_traffic=True,
        #     speed_reward = 0.4
        # ), config or {}))
        # register(id=env_name, entry_point=make_env)
        # env_names.append(env_name)


        # env_name = "MetaDrive-Gen-v0"
        # make_env = lambda config=None: MetaDriveEnv(merge_config_with_unknown_keys(dict(
        #     start_seed=1000,
        #     environment_num=100,
        #     horizon=1500,
        #     random_lane_width= True, 
        #     random_lane_num= True,
        #     traffic_density = 0.2
        # ), config or {}))
        # register(id=env_name, entry_point=make_env)
        # env_names.append(env_name)

        # env_name = "MetaDrive-Gen-v0"
        # make_env = lambda config=None: MetaDriveEnv(merge_config_with_unknown_keys(dict(
        #     start_seed=1000,
        #     environment_num=100,
        #     horizon=1500,
        #     random_lane_width= True, 
        #     random_lane_num= True,
        #     traffic_density = 0.2
        # ), config or {}))
        # register(id=env_name, entry_point=make_env)
        # env_names.append(env_name)

    except gym.error.Error:
        pass
    else:
        print("Successfully registered MetaDrive environments: ", env_names)


if __name__ == '__main__':
    # Test
    register_metadrive()
    env = gym.make("MetaDrive-Tut-Easy-v0", config={'use_render': True})
    env.reset()
