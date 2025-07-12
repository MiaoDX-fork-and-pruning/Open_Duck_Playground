"""Runs training and evaluation loop for Open Duck Mini V2."""

import argparse
import os

from playground.common import randomize
from playground.common.runner import BaseRunner
from playground.open_duck_mini_v2 import joystick, standing
from playground.open_duck_mini_v2 import newton_joystick, newton_standing


class OpenDuckMiniV2Runner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
        available_envs = {
            "joystick": (joystick, joystick.Joystick),
            "standing": (standing, standing.Standing),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        self.env_config = self.env_file[0].default_config()
        
        # Create environments - passing backend if available
        backend = getattr(args, 'backend', 'mjx')
        
        if args.backend == "newton":
            try:
                import newton
            except ImportError as e:
                raise ImportError("Newton backend requested but not available. Please install required packages.") from e
        
        if args.env == "joystick":
            env_class = newton_joystick.NewtonJoystick if args.backend == "newton" else self.env_file[1]
        elif args.env == "standing":
            env_class = newton_standing.NewtonStanding if args.backend == "newton" else self.env_file[1]
        else:
            raise ValueError(f"Unknown env {args.env}")
        
        self.env = env_class(task=args.task, backend=backend)
        self.eval_env = env_class(task=args.task, backend=backend)
        
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )  # 0: state 1: privileged_state
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size}")
        
        # Print backend info if available
        if hasattr(self.env, 'backend_name'):
            print(f"Physics backend: {self.env.backend_name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Duck Mini Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    # parser.add_argument("--num_timesteps", type=int, default=300000000)
    parser.add_argument("--num_timesteps", type=int, default=150000000)
    parser.add_argument("--env", type=str, default="joystick", help="env")
    parser.add_argument("--task", type=str, default="flat_terrain", help="Task to run")
    parser.add_argument(
        "--restore_checkpoint_path",
        type=str,
        default=None,
        help="Resume training from this checkpoint",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="mjx",
        choices=["mjx", "newton"],
        help="Physics backend to use (mjx or newton)",
    )
    # parser.add_argument(
    #     "--debug", action="store_true", help="Run in debug mode with minimal parameters"
    # )
    args = parser.parse_args()
    
    # Backend can also be set via environment variable
    if "OPEN_DUCK_BACKEND" in os.environ:
        args.backend = os.environ["OPEN_DUCK_BACKEND"]
        print(f"Using backend from environment: {args.backend}")

    runner = OpenDuckMiniV2Runner(args)

    runner.train()


if __name__ == "__main__":
    main()
