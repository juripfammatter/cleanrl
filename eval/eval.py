import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pickle
import torch
import tyro
from torch.distributions.categorical import Categorical

from cleanrl.ppo import Agent, make_env

def eval_run(filename:str):
    # Load agent
    agent = torch.load(filename, weights_only=False)
    agent.eval()
    #print(agent)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # rollout and collect return
    returns = torch.zeros(args.num_envs).to(device)
    episodes_per_env = torch.zeros(args.num_envs).to(device)
    for i in range(args.eval_iterations):
        with torch.no_grad():
            logits = agent(next_obs)
            probs = Categorical(logits=logits)
            action = probs.sample()

        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        next_done = np.logical_or(terminations, truncations)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
        #TODO: mask with dones?
        episodes_per_env += next_done
        returns += reward        

    # log returns
    log = lambda x: f"{(x).mean().item():.2f} +- {(x).var().item()**0.5:.2f}"

    print(f"run: {filename}")
    print(f"total returns: {log(returns)}")
    print(f"reward per step: {log(returns/args.eval_iterations)}")
    print(f"reward per episode: {log(returns/episodes_per_env)}")
    print(f"avg. episode length: {log(args.eval_iterations/episodes_per_env)}")

    db = {
        "returns": returns,
        "max_iter": args.eval_iterations,
        "episodes_per_env": episodes_per_env
    }


    result_file = os.path.join(os.path.dirname(filename), "results.pkl")
    with open(result_file, "wb") as f:
        pickle.dump(db, f)

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
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    run_folder = "runs"
    subfolders = os.listdir(run_folder)

    for sub in reversed(subfolders):
        #filename = "runs/CartPole-v0__ppo__1__1791292737/policy/policy.pkl"
        filename = os.path.join(run_folder, sub, "policy/policy.pkl")
        eval_run(filename)