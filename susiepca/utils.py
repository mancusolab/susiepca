import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import jax.numpy as jnp

__all__ = ["pip_analysis", "pve_plot", "z_cor_plot"]

# Common functions import from infer.py


# small function to generate factor name
def fac_name(prefix, n):
    return list(prefix + str(i) for i in range(n))


# analysis of number of genes with specifi PIP cutoff
def pip_analysis(pip: jnp.ndarray, rho=0.9, rho_prime=0.05):
    """Create a function to give a quick summary of PIPs

    Args:
        pip:the pip matrix, a ndarray from results object returned by
        infer.susiepca

    """
    z_dim, p_dim = pip.shape
    results = []

    print(f"Of {p_dim} features from the data, SuSiE PCA identifies:")
    for k in range(z_dim):
        num_signal = jnp.where(pip[k, :] >= rho)[0].shape[0]
        num_zero = jnp.where(pip[k, :] < rho_prime)[0].shape[0]
        print(
            f"Component {k} has {num_signal} features with pip>{rho}; "
            f"and {num_zero} features with pip<{rho_prime}"
        )
        results.append([num_signal, num_zero])

    df = pd.DataFrame(results, columns=["num_signal", "num_zero"])

    # Calculate and print mean and standard deviation for each column
    mean_signal = df["num_signal"].mean()
    std_signal = df["num_signal"].std()
    mean_zero = df["num_zero"].mean()
    std_zero = df["num_zero"].std()

    print(f"Mean and standard deviation for num_signal: {mean_signal}, {std_signal}")
    print(f"Mean and standard deviation for num_zero: {mean_zero}, {std_zero}")

    return df


# build function to draw percent of variance explained
def pve_plot(pve: jnp.ndarray):
    """Create function to draw the barplot of percent of
    variance explained (PVE) across factors

    Args:
        pve: percent of variance, a ndarray from results object returned by
        infer.susiepca

    """
    # create sorted dataframe of PVE acording to its values
    df_pve = pd.DataFrame(pve * 100, columns=["pve"], index=fac_name("z", pve.shape[0]))
    df_pve_sorted = df_pve.sort_values(by=["pve"], ascending=False)
    print(f"Total PVE is {pve.sum()*100}%.")

    # assign new factor name based on PVE in descending order
    df_pve_sorted["component"] = fac_name("z", pve.shape[0])

    pve_plot = sns.barplot(
        x=df_pve_sorted.component,
        y=df_pve_sorted.pve,
        data=df_pve_sorted,
        color="deepskyblue",
    )
    pve_plot.set_xticklabels(pve_plot.get_xticklabels(), rotation=270)
    pve_plot.set(xlabel="Factors", ylabel="Percent of Variance Explained (%)")

    return pve_plot


# plot posterior weights/pips
def plot_weight(factor: int, gene_name: pd.Series, n: int, W: jnp.ndarray):
    df_W = pd.DataFrame(
        {"weight": W[factor], "name": gene_name, "abs_weight": jnp.abs(W[factor])}
    )
    df_W["rank"] = df_W["weight"].rank(ascending=True)
    df_W["abs_rank"] = df_W["abs_weight"].rank(ascending=False)
    large = df_W.query("abs_rank < " + str(n + 1))
    plt.figure(figsize=(10, 13))
    plt.scatter(df_W["rank"], df_W["weight"], alpha=0.4, color="grey")
    plt.xlabel("Rank", fontsize=15)
    plt.ylabel("Weights", fontsize=15)
    plt.scatter(large["rank"], large["weight"], color="red")
    for i in range(large.shape[0]):
        plt.annotate(
            large["name"].tolist()[i],
            (large["rank"].tolist()[i], large["weight"].tolist()[i]),
        )
    plt.tight_layout()
    return df_W


# Generate Z correlation plot
def z_cor_plot(mu_z: jnp.ndarray):
    z_cor = np.corrcoef(mu_z.T)
    # Generate a mask for the lower triangle
    mask = np.zeros_like(z_cor, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    # generate the plot
    cor_plot = sns.heatmap(z_cor, annot=True, cmap="coolwarm", mask=mask)

    return cor_plot
