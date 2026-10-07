import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pickle
import torch
import tyro

from cleanrl.ppo import make_env

def eval_run(filename:str, envs:any):

    if isinstance(envs.single_action_space, gym.spaces.Box):
        from cleanrl.ppo_continuous_action import Agent
    else:
        from cleanrl.ppo import Agent
    # Load agent
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(filename, weights_only=True))
    agent.eval()
    #print(agent)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # rollout and collect return
    returns = torch.zeros(args.num_envs).to(device)
    episodes_per_env = torch.zeros(args.num_envs).to(device)
    for i in range(args.eval_iterations):
        action, _, _, _ = agent.get_action_and_value(next_obs)

        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        next_done = np.logical_or(terminations, truncations)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
        
        if "final_info" in infos:
            for i, info in enumerate(infos["final_info"]):
                if info is not None:
                    episodes_per_env[i] = episodes_per_env[i] + 1
                    returns[i] = returns[i] + info["episode"]["r"]        
           

    # log returns
    log = lambda x: f"{(x).mean().item():.2f} +- {(x).var().item()**0.5:.2f}"

    print(f"run: {filename}")
    print(f"total returns: {log(returns)}")
    print(f"reward per step: {log(returns/args.eval_iterations)}")
    print(f"reward per episode: {log(returns/episodes_per_env)}")
    print(f"avg. episode length: {log(args.eval_iterations/episodes_per_env)}")

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    num_envs: int = 4
    """the number of parallel game environments"""
    eval_iterations: int = 20000
    """ssnumber of eval iterations"""
    filename: str|None = None
    """filename in single filename mode"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )

    if args.filename is not None:
        eval_run(args.filename, envs)
    else:
        run_folder = "runs"
        subfolders = os.listdir(run_folder)

        for sub in reversed(subfolders):
            filename = os.path.join(run_folder, sub, "policy/agent.pkl")
            eval_run(filename, envs)