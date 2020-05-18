import os
from datetime import datetime

import numpy as np
import retro
import tensorflow as tf
from gym import wrappers
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv

from util import get_valid_filename


def make_vec_env(env_id, n_envs, filename):
    # stable-baselines built-in make_vec_env converted to work with retro, and
    # modified to use gym's default Monitor which provides video recording
    def make_env(rank):
        def _init():
            env = retro.make(env_id)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_dir = f"./videos/Learn Rate/{filename}"
            os.makedirs(monitor_dir, exist_ok=True)
            env = wrappers.Monitor(env, directory=monitor_dir, uid=str(rank),
                                   force=True, mode="evaluation")
            return env
        
        return _init
    
    return SubprocVecEnv([make_env(i) for i in range(n_envs)])


def main(game_name, num_envs, num_timesteps, num_episodes, policy, discount,
         batch_size, learn_rate):
    log_dir = f"./logs/Learn Rate/{learn_rate}"
    env = make_vec_env(game_name, num_envs, learn_rate)
    env.seed(309)
    obs = env.reset()
    
    tr_log_dir = f"{log_dir}-training"
    os.makedirs(tr_log_dir, exist_ok=True)
    os.makedirs("models/Learn Rate", exist_ok=True)
    model_filename = "models/Learn Rate/" + get_valid_filename(
        f"ppo2-{game_name}-{policy}-{discount}-{batch_size}-{learn_rate}")
    if os.path.exists(f"{model_filename}.zip"):
        model = PPO2.load(model_filename, env=env)
        model.tensorboard_log = tr_log_dir
    else:
        model = PPO2(policy=policy, env=env, gamma=discount,
                     n_steps=batch_size, learning_rate=learn_rate, verbose=1,
                     tensorboard_log=tr_log_dir)
    model.learn(total_timesteps=num_timesteps)
    model.save(model_filename)
    
    dones = np.array([[False] * num_envs] * num_episodes)
    ep_rewards = [[0 for _ in range(num_envs)] for _ in range(num_episodes)]
    curr_rewards = [0] * num_envs
    while not dones.all():
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)
        env.render(mode="human")
        for i in range(num_envs):
            if dones[-1][i]:
                continue
            else:
                for j in range(num_episodes):
                    if not dones[j][i]:
                        if reward[i] != 0:
                            curr_rewards[i] += reward[i]
                        if done[i]:
                            ep_rewards[j][i] = curr_rewards[i]
                            curr_rewards[i] = 0
                            dones[j][i] = True
                        break
    env.close()
    
    rewards = list()
    for i in range(num_episodes):
        for j in range(num_envs):
            rewards.append(ep_rewards[i][j])
    mean = np.mean(rewards)
    std_dev = np.std(rewards)
    # Outliers: outside of 3 standard deviations
    outlier_threshold_upper = mean + 3 * std_dev
    outlier_threshold_lower = mean - 3 * std_dev
    trimmed_rewards = [
        rew for rew in rewards
        if outlier_threshold_lower <= rew <= outlier_threshold_upper
    ]
    avg_reward = np.mean(trimmed_rewards)
    print(f"Average score over {num_envs * num_episodes} games: "
          f"{avg_reward:.2f}")
    
    os.makedirs(log_dir, exist_ok=True)
    summary_writer = tf.summary.FileWriter(log_dir)
    sess = tf.Session()
    rew_var = tf.Variable(0, dtype=tf.int64)
    rew_val = tf.summary.scalar(f"Reward / Episode ({learn_rate})", rew_var)
    for i in range(num_envs * num_episodes):
        rew = rewards[i]
        sess.run(rew_var.assign(rew))
        summary_writer.add_summary(sess.run(rew_val), i)
        
    avg_var = tf.Variable(0.0, dtype=tf.float64)
    avg_val = tf.summary.scalar(f"Trimmed Average ({learn_rate})", avg_var)
    sess.run(avg_var.assign(avg_reward))
    summary_writer.add_summary(sess.run(avg_val), 0)
    summary_writer.flush()
    summary_writer.close()
    sess.close()


if __name__ == "__main__":
    game = "SpaceInvaders-Snes"
    
    sizes = [1024]
    for i in range(2, 11):
        lr = i * 0.00025
        retry = True
        while retry:
            try:
                main(game, num_envs=4, num_timesteps=25000, num_episodes=25,
                     policy="CnnPolicy", discount=0.97, batch_size=512, 
                     learn_rate=lr)
                retry = False
            except EOFError:
                pass
