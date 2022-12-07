import pickle

import numpy as np

from susiepca.infer import SuSiEPCAResults

# from jax import jit

__all__ = [
    "save_results",
]


def save_results(results: SuSiEPCAResults, path: str):
    """Create a function to save SuSiE PCA results returned by function
    infer.susiepca

    Args:
        results: results object returned by infer.susiepca
        path: local path to save the results subject


    """
    print("Save results from SuSiE PCA")

    np.savetxt(f"{path}/W.txt", results.W)
    np.savetxt(f"{path}/pip.txt", results.pip)
    np.savetxt(f"{path}/pve.txt", results.pve)

    params_file = open(f"{path}/params_file.pkl", "wb")
    pickle.dump(results.params, params_file)
    params_file.close()

    print(f"Results saved successfully at {path}")

    return
