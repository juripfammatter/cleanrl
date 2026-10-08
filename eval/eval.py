import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from scipy import stats
import pickle
import torch
import tyro

def eval_run(filename:str, envs:any):
    if isinstance(envs.single_action_space, gym.spaces.Box):
        from cleanrl.ppo_continuous_action import Agent
    else:
        from cleanrl.ppo import Agent
    # Load agent
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(filename, weights_only=True, map_location=device))
    agent.eval()
    #print(agent)

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # rollout and collect return
    returns = []
    successes = []

    for i in range(args.eval_iterations):
        action, _, _, _ = agent.get_action_and_value(next_obs, deterministic=True)
        
        next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
        next_done = np.logical_or(terminations, truncations)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    returns.append(info["episode"]["r"])
                    if hasattr(envs.envs[0].env, "_max_episode_steps"):
                        if info["episode"]["l"] == envs.envs[0].env._max_episode_steps:
                            successes.append(1)
                        else:
                            successes.append(0)

    # log returns
    # display mean + std
    log_std = lambda x: f"{np.array(x).mean().item():>7.2f} +- {np.array(x).var().item()**0.5:.2f}"
    log_se = lambda x: f"{np.array(x).mean().item():>7.2f} +- {np.array(x).var().item()**0.5/np.sqrt(len(x)):.2f}"
    
    print(f"run: {filename}")
    print(f"{15*'-'} std {15*'-'}")
    print(f"{'avg returns':<20}{log_std(returns):}")
    print(f"{'success rate':<20}{log_std(successes):}")
    print(f"{15*'-'} se {15*'-'}")
    print(f"{'avg returns':<20}{log_se(returns):}")
    print(f"{'success rate':<20}{log_se(successes):}")

    db = {
        "returns": returns,
        "mean_return": np.array(returns).mean().item(),
        "success": successes,
        "mean_success": np.array(successes).mean().item()
    }

    result_file = os.path.join(os.path.dirname(filename), "results.pkl")
    with open(result_file, "wb") as f:
        pickle.dump(db, f)

    return db

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
    gamma: float = 0.99
    """the discount factor gamma"""

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    if isinstance(gym.make(args.env_id).env.action_space, gym.spaces.Box):
        from cleanrl.ppo_continuous_action import make_env
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
        )
    else:
        from cleanrl.ppo import make_env
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
        )

    data = []

    if args.filename is not None:
        data.append(eval_run(args.filename, envs))
    else:
        run_folder = "runs"
        subfolders = os.listdir(run_folder)

        for sub in reversed(subfolders):
            filename = os.path.join(run_folder, sub, "policy/agent.pkl")
            data.append(eval_run(filename, envs))

    envs.close()

    # aggregate results

    def metrics(x):
        mean = np.array(x).mean().item()
        std = np.array(x).std()
        se = std/np.sqrt(len(x))
        return {
            "mean": mean,
            "std": std,
            "se": se
        }
    
    returns = []
    success_rates = []
    for run in data:
        returns.append(run["mean_return"])
        success_rates.append(run["mean_success"])

    print(f"{15*'-'} aggregate {15*'-'}")
    print(f"{'avg episodic returns':<30}{metrics(returns)}")
    print(f"{'success rate':<30}{metrics(success_rates)}")