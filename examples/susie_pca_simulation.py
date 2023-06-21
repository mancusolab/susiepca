#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse as ap
import sys
from csv import writer
from time import time

import numpy as np
import procrustes

from jax.config import config

import susiepca as sp


def main(args):
    """
    run the simulation with user-specific setting and
    produce summary results in args.output path. Currently each simulation
    will expect to take ~10 seconds to converge on CPU, and ~1.5 seconds on GPU.
    """
    argp = ap.ArgumentParser(description="")
    argp.add_argument("--n-sim", default=1, type=int, help="number of simulations")
    argp.add_argument("--n-dim", default=1000, type=int, help="Number of samples")
    argp.add_argument("--p-dim", default=6000, type=int, help="Number of features")
    argp.add_argument("--z-dim", default=4, type=int, help="Number of latent factors")
    argp.add_argument(
        "--real-z-dim", default=4, type=int, help="Number of real latent factors"
    )
    argp.add_argument("--l-dim", default=40, type=int, help="Number of single effects")
    argp.add_argument(
        "--real-l-dim", default=40, type=int, help="Number of real single effects"
    )
    argp.add_argument(
        "--tol", default=1e-1, type=float, help="Delta ELBO threshold to stop VI"
    )
    argp.add_argument(
        "--max-iter", default=150, type=int, help="Maximum number of iterations for VI"
    )
    argp.add_argument(
        "-o",
        "--output",
        type=str,
        default="/home1/dongyuan/SuSiEPCA/simulation_results/test.csv",
    )
    argp.add_argument(
        "--platform",
        type=str,
        default="cpu",
        help="Choose the platform to run inference",
    )

    args = argp.parse_args(args)

    config.update("jax_enable_x64", True)
    config.update("jax_platform_name", args.platform)

    for sim in range(args.n_sim):
        # simulate data
        Z, W, X = sp.sim.generate_sim(
            seed=sim,
            l_dim=args.real_l_dim,
            n_dim=args.n_dim,
            p_dim=args.p_dim,
            z_dim=args.real_z_dim,
            effect_size=1,
        )

        # start inference
        print(f"Begin inference at simulation {sim}")
        start_susie = time()
        # run inference
        results = sp.infer.susie_pca(
            X,
            z_dim=args.z_dim,
            l_dim=args.l_dim,
            tau=10,
            tol=args.tol,
            max_iter=args.max_iter,
            verbose=False,
        )
        end_susie = time()
        run_susie = end_susie - start_susie

        # calculate procruste error for W
        W_hat = results.W
        proc_trans_susie = procrustes.orthogonal(
            np.asarray(W_hat.T), np.asarray(W.T), scale=True
        )
        error_susie = proc_trans_susie.error

        # compute the rrmse
        X_hat = results.params.mu_z @ W_hat
        rrmse_susie = sp.metrics.mse(X, X_hat)

        # calculate procruste error for Z
        Z_hat = results.params.mu_z
        proc_trans_susie_z = procrustes.orthogonal(
            np.asarray(Z_hat), np.asarray(Z), scale=True
        )
        error_z_susie = proc_trans_susie_z.error

        # summarize results
        summary = [
            sim,
            args.n_dim,
            args.p_dim,
            args.z_dim,
            args.l_dim,
            error_susie,
            rrmse_susie,
            error_z_susie,
            run_susie,
        ]

        with open(f"{args.output}", "a", newline="") as fd:
            writer_object = writer(fd)
            writer_object.writerow(summary)
            fd.close()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
