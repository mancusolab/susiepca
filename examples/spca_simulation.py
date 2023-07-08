#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse as ap
import sys

from csv import writer
from time import time

import numpy as np
import procrustes

from sklearn.decomposition import SparsePCA

import susiepca


def mySparsePCA(X, n_components, seed):
    spca = SparsePCA(n_components=n_components, random_state=seed)
    spca_z = spca.fit_transform(X)
    spca_weights = spca.components_

    return spca_z, spca_weights


def main(args):
    argp = ap.ArgumentParser(description="")
    argp.add_argument("--n-sim", default=100, type=int, help="number of simulations")
    argp.add_argument("--n-dim", default=600, type=int, help="Number of samples")
    argp.add_argument("--p-dim", default=6, type=int, help="Number of features")
    argp.add_argument("--z-dim", default=4, type=int, help="Number of latent factors")
    argp.add_argument(
        "--real-z-dim", default=4, type=int, help="Number of real latent factors"
    )
    argp.add_argument(
        "--real-l-dim", default=40, type=int, help="Number of real single effects"
    )
    argp.add_argument(
        "-o",
        "--output",
        type=str,
        default="/Users/dong/Documents/Project/SuSiE/SusiEPCA/test.csv",
    )

    args = argp.parse_args(args)

    for sim in range(args.n_sim):
        # simulate data
        Z, W, X = susiepca.sim.generate_sim(
            seed=sim,
            l_dim=args.real_l_dim,
            n_dim=args.n_dim,
            p_dim=args.p_dim,
            z_dim=args.real_z_dim,
            effect_size=1,
        )

        # start inference
        print(f"Begin inference at simulation {sim}")
        start = time()
        # run inference
        spca_z, spca_weights = mySparsePCA(X, args.z_dim, args.seed)
        end = time()
        run_time = end - start

        # calculate procruste error
        # Procruste transformation
        proc_trans_spca = procrustes.orthogonal(
            np.asarray(spca_weights.T), np.asarray(W.T), scale=True
        )

        err = proc_trans_spca.error

        reconstruct_spca_X = spca_z @ spca_weights
        rrmse = susiepca.metrics.mse(X, reconstruct_spca_X)
        # summarize results
        summary = [
            sim,
            args.n_dim,
            args.p_dim,
            args.z_dim,
            args.real_z_dim,
            args.real_l_dim,
            err,
            rrmse,
            run_time,
        ]

        with open(f"{args.output}", "a", newline="") as fd:
            writer_object = writer(fd)
            writer_object.writerow(summary)
            fd.close()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
