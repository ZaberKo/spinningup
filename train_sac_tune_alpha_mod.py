import argparse
import multiprocessing as mp
import gym
from spinup.algos.pytorch.sac import sac_tune_alpha_mod
import torch

def main(args):
    from spinup.utils.run_utils import setup_logger_kwargs

    def run(seed):
        logger_kwargs = setup_logger_kwargs(args.exp_name, seed)

        cpus=mp.cpu_count()//args.num_tests
        print(f"seed: {seed} cpu_usage: {cpus}")
        torch.set_num_threads(cpus)
        sac_tune_alpha_mod.sac(lambda: gym.make(args.env), actor_critic=sac_tune_alpha_mod.core.MLPActorCritic2,
                ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
                gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                logger_kwargs=logger_kwargs)

    processes = [
        mp.Process(target=run, args=(args.seed+i,))
        for i in range(args.num_tests)
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--num_tests', type=int, default=1)
    args = parser.parse_args()

    main(args)

