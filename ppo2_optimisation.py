import os

import numpy as np
import retro
import tensorflow as tf
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv

from util import get_valid_filename


def make_vec_env(env_id, num_envs):
    # stable-baselines built-in make_vec_env converted to work with retro
    def make_env():
        def _init():
            return retro.make(env_id)
        return _init
    
    return SubprocVecEnv([make_env() for _ in range(num_envs)])


def main(game, num_timesteps, num_episodes, dir_name, model_name, policy,
         discount=0.99, batch_size=128, learn_rate=0.00025, num_envs=4):
    dir_name = get_valid_filename(dir_name)
    model_name = get_valid_filename(model_name)
    
    eval_log_dir = f"logs/{dir_name}/{model_name}"
    tr_log_dir = f"{eval_log_dir}-training"
    model_dir = f"models/{dir_name}"
    os.makedirs(eval_log_dir, exist_ok=True)
    os.makedirs(tr_log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    env = make_vec_env(game, num_envs)
    env.seed(309)
    
    model = PPO2(policy=policy, env=env, gamma=discount, n_steps=batch_size,
                 learning_rate=learn_rate, verbose=1, seed=309,
                 tensorboard_log=tr_log_dir, n_cpu_tf_sess=1)
    model.learn(total_timesteps=num_timesteps)
    model.save(f"{model_dir}/{model_name}.zip")

    eps_done = 0
    ep_rewards = np.array([0] * num_episodes)
    curr_rewards = [0] * num_envs
    obs = env.reset()
    while eps_done != num_episodes:
        if eps_done % 10 == 0:
            print(f"Episodes completed: {eps_done} / {num_episodes}", end="\r")
        # For vectorised environments, they are automatically reset when done,
        # so returned obs would be the start state of next episode
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        env.render(mode="human")
        
        for i in range(num_envs):
            curr_rewards[i] += reward[i]
            if done[i]:
                ep_rewards[eps_done] = curr_rewards[i]
                curr_rewards[i] = 0
                eps_done += 1
    print("All episodes completed")
    env.close()
    
    mean = ep_rewards.mean()
    std_dev = ep_rewards.std()
    # Outliers: outside of 3 standard deviations
    outlier_threshold_upper = mean + 3 * std_dev
    outlier_threshold_lower = mean - 3 * std_dev
    trimmed_rewards = np.array([
        rew for rew in ep_rewards
        if outlier_threshold_lower <= rew <= outlier_threshold_upper
    ])
    avg_reward = trimmed_rewards.mean()
    print(f"Average score over {num_episodes} games: {avg_reward:.2f}")
    
    summary_writer = tf.summary.FileWriter(eval_log_dir)
    sess = tf.Session()
    rew_var = tf.Variable(0, dtype=tf.int64)
    rew_val = tf.summary.scalar(f"Reward / Episode ({model_name})", rew_var)
    for i in range(num_episodes):
        rew = ep_rewards[i]
        sess.run(rew_var.assign(rew))
        summary_writer.add_summary(sess.run(rew_val), i)
    
    avg_var = tf.Variable(0.0, dtype=tf.float64)
    avg_val = tf.summary.scalar(f"Trimmed Average ({model_name})", avg_var)
    sess.run(avg_var.assign(avg_reward))
    summary_writer.add_summary(sess.run(avg_val), 0)
    
    summary_writer.flush()
    summary_writer.close()
    sess.close()


if __name__ == "__main__":
    game_name = "SpaceInvaders-Snes"
    policy_name = "MlpLnLstmPolicy"
    dct_factors = [0.1, 0.8]
    batch = 128
    rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    for dct in dct_factors:
        for lr in rates:
            retry = True
            while retry:
                try:
                    main(game=game_name, num_timesteps=100000,
                         num_episodes=100, dir_name="ppo2-lr",
                         model_name=f"{policy_name}-{dct}-{batch}-{lr}",
                         policy=policy_name, discount=dct, batch_size=batch,
                         learn_rate=lr)
                    retry = False
                except EOFError:
                    pass
