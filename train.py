import argparse
import multiprocessing as mp
import gym
from spinup.algos.pytorch.sac import (
    sac,
    sac_fix_alpha_mod,
    sac_tune_alpha_mod,
    sac_tune_alpha_ori,
    sac_tune_alpha,
    core as sac_core
)
from spinup.algos.pytorch.td3 import (
    td3,
    td3_mod,
    core as td3_core
)
import torch
from datetime import datetime
from dataclasses import dataclass
from typing import Callable, Type


@dataclass
class Algo_Spec:
    trainer_func: Callable
    actor_critic_cls: Type


algo_dict = {
    "sac": Algo_Spec(sac.sac, sac_core.MLPActorCritic),
    "sac_fix_alpha_mod": Algo_Spec(sac_fix_alpha_mod.sac, sac_core.MLPActorCritic),
    "sac_tune_alpha_mod": Algo_Spec(sac_tune_alpha_mod.sac, sac_core.MLPActorCritic2),
    "sac_tune_alpha_ori": Algo_Spec(sac_tune_alpha_ori.sac, sac_core.MLPActorCritic2),
    "sac_tune_alpha": Algo_Spec(sac_tune_alpha.sac, sac_core.MLPActorCritic2),
    "td3": Algo_Spec(td3.td3, td3_core.MLPActorCritic),
    "td3_mod": Algo_Spec(td3_mod.td3, td3_core.MLPActorCritic)
}


def main(args):
    from spinup.utils.run_utils import setup_logger_kwargs

    if args.exp_name is None:
        exp_name = f"{args.algo}_{args.env}_" + \
            datetime.today().strftime("%Y_%m_%d_%H_%M_%S")
    else:
        exp_name = args.exp_name
    
    env=args.env

    if args.cpu is None:
        cpus = mp.cpu_count()//args.num_tests
    else:
        cpus = args.cpu

    algo_spec = algo_dict[args.algo]

    def run(seed):
        logger_kwargs = setup_logger_kwargs(exp_name, seed)

        print(f"seed: {seed} cpu_usage: {cpus}")
        torch.set_num_threads(cpus)
        algo_spec.trainer_func(
            lambda: gym.make(env),
            actor_critic=algo_spec.actor_critic_cls,
            ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
            gamma=args.gamma,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.bs,
            logger_kwargs=logger_kwargs
        )

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
    parser.add_argument('--algo', type=str, required=True)
    parser.add_argument('--env', type=str, required=True)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--num_tests', type=int, default=1)
    parser.add_argument('--bs', type=int, default=256)
    parser.add_argument('--cpu', type=int)
    args = parser.parse_args()

    main(args)
