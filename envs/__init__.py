import gymnasium as gym
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
import torch

class ShapedRewardWrapper(gym.Wrapper):
    def __init__(
            self, 
            env, 
            cart_pos_weight:float=0.0,
            cart_vel_weight:float=0.0,
            pole_angle_weight:float=0.0,
            pole_vel_weight:float=0.0,
            termination_weight:float=0.0,
            vel_penalty_delay:int|None=None,
        ):
        super().__init__(env)
        self.cart_pos_weight = cart_pos_weight
        self.cart_vel_weight = cart_vel_weight
        self.pole_angle_weight = pole_angle_weight
        self.pole_vel_weight = pole_vel_weight
        self.vel_penalty_delay = vel_penalty_delay
        self.termination_weight = termination_weight

        self.total_steps:int = 0

    def step(self, action):
        # TODO: normalize with num envs
        self.total_steps += 1
        
        obs, r, term, trunc, info = self.env.step(action)
        x, x_dot, theta, theta_dot = obs

        cart_pos_pen = -self.cart_pos_weight * torch.norm(torch.tensor(theta), p=1)
        cart_vel_pen = - self.cart_vel_weight * torch.norm(torch.tensor(x_dot), p=2)
        pole_angle_pen = - self.pole_angle_weight * torch.norm(torch.tensor(theta), p=1)
        pole_vel_pen = -self.pole_vel_weight * torch.norm(torch.tensor(theta_dot), p=2)
        term_pen = -self.termination_weight * term

        if self.vel_penalty_delay is not None:
            mask = self.total_steps > self.vel_penalty_delay
            cart_vel_pen = cart_vel_pen * mask
            pole_vel_pen = pole_vel_pen * mask
        r_shaped = r + cart_pos_pen + cart_vel_pen + pole_angle_pen + pole_vel_pen + term_pen
        return obs, r_shaped.item(), term, trunc, info


def make_shaped_cartpole(
        cart_pos_weight:float=0.0,
        cart_vel_weight:float=0.0,
        pole_angle_weight:float=0.0,
        pole_vel_weight:float=0.0,
        termination_weight:float=0.0,
        vel_penalty_delay:int|None = None,
        **kwargs,
    ):
    # kwargs carries things like render_mode from gym.make
    env = CartPoleEnv(**kwargs)
    return ShapedRewardWrapper(
        env, 
        cart_pos_weight=cart_pos_weight,
        cart_vel_weight=cart_vel_weight,
        pole_angle_weight=pole_angle_weight,
        pole_vel_weight=pole_vel_weight,
        vel_penalty_delay=vel_penalty_delay,
        termination_weight=termination_weight,
        )

gym.register(
    id='CartPoleShaped-v0',
    entry_point=make_shaped_cartpole,
    max_episode_steps=200,
    kwargs={
        "cart_pos_weight": 1.0,
        "cart_vel_weight": 0.0,
        "pole_angle_weight": 5.0,
        "pole_vel_weight": 0.0,
        "termination_weight": 0.0,
        #"vel_penalty_delay": 5_000, # multiplied with num envs
        },
)