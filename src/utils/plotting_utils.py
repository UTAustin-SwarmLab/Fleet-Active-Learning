import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from statannot import add_stat_annotation
import matplotlib.patches
import pylab


# Creates L2-Norm Plots for the simulations
def plot_accs(
    accs: list,
    names: list,
    save_loc: str,
    linestyles=["--", "-", "-.", ":", "--", "-", "-."],
    markers=["P", "o", "s", "D", "^", "v", "X"],
    legend: bool = False,
) -> None:
    """
    :param accs: Accuracy values
    :param names: Names of the algorithms
    :param save_loc: Location to save the plot
    :return: None
    """

    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 7), dpi=600)

    for i in range(len(accs)):
        data = pd.DataFrame(accs[i].reshape(-1, 1), columns=["Accuracy"])
        data["Round"] = [i for i in range(accs[i].shape[1])] * accs[i].shape[0]
        sns.lineplot(
            data=data,
            x="Round",
            y="Accuracy",
            label=names[i],
            linestyle=linestyles[i],
            marker=markers[i],
            linewidth=3,
            markersize=10,
        )

    plt.grid(linestyle="--", linewidth=2)
    plt.xlabel("Round $r$", fontweight="bold", fontsize=24)
    plt.ylabel("Accuracy", fontweight="bold", fontsize=24)

    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.linewidth"] = 2
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(prop=dict(size=20, weight="bold"))
    plt.tight_layout()
    if not legend:
        plt.legend([], [], frameon=False)
    plt.savefig(save_loc)


def plot_accs_wrt_n_samples(
    accs: list, names: list, n_samples: list, save_loc: str
) -> None:
    """
    :param accs: Accuracy values
    :param names: Names of the algorithms
    :param n_samples: Number of samples used for each round
    :param save_loc: Location to save the plot
    :return: None
    """

    plt.close("all")
    fig, ax = plt.subplots(figsize=(14, 7), dpi=600)

    for i in range(len(accs)):
        data = pd.DataFrame(accs[i].reshape(-1, 1), columns=["Accuracy"])
        data["N. of Samples"] = n_samples * accs[i].shape[0]
        sns.lineplot(
            data=data, x="N. of Samples", y="Accuracy", label=names[i], linewidth=3
        )

    plt.grid(linestyle="--", linewidth=2)
    plt.xlabel("Number of Samples", fontweight="bold", fontsize=24)
    plt.ylabel("Accuracy", fontweight="bold", fontsize=24)

    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.linewidth"] = 2
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    # plt.legend(prop=dict(size=16,weight='bold'),bbox_to_anchor=(1,1))
    plt.tight_layout()
    plt.legend([], [], frameon=False)
    plt.savefig(save_loc)


def make_legend(save_loc="./"):
    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(18.5, 10.5)
    gs = fig.add_gridspec(1, 1)

    acc = np.ones((7, 2))

    all_data_df = pd.DataFrame(acc, columns=["Iterations", "Real Grad Norm"])
    all_data_df["method"] = [
        "Distributed",
        "Centralized",
        "Interactive",
        "Random",
        "Entropy",
        "Least Confidence",
        "BvSB",
    ]

    labels = [
        "Distributed",
        "Centralized",
        "Interactive",
        "Random",
        "Entropy",
        "Least Confidence",
        "BvSB",
    ]
    ax00 = fig.add_subplot(gs[0])
    markers = ["P", "o", "s", "D", "^", "v", "X"]
    linestyles = ["--", "-", "-.", ":", "--", "-", "-."]

    for i in range(7):
        sns.lineplot(
            x=[1],
            y=[2],
            marker=markers[i],
            markersize=9,
            linestyle=linestyles[i],
            ax=ax00,
            linewidth=3,
            label=labels[i],
        )
    handles, labels = ax00.get_legend_handles_labels()

    figLegend = pylab.figure(figsize=(12, 0.3))
    pylab.figlegend(
        *ax00.get_legend_handles_labels(),
        loc="upper left",
        mode="expand",
        ncol=7,
        prop={"weight": "bold", "size": 12},
        borderaxespad=0,
        frameon=False,
    )
    figLegend.savefig(os.path.join(save_loc, f"lengend.png"), dpi=600)


# Creates L2-Norm Plots for the simulations
def plot_values(
    values: list, names: list, save_loc: str, y_label: str, linestyles=["--", "-", "-."]
) -> None:
    """
    :param accs: Accuracy values
    :param names: Names of the algorithms
    :param save_loc: Location to save the plot
    :return: None
    """

    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 7), dpi=600)

    for i in range(len(values)):
        data = pd.DataFrame(values[i].reshape(-1, 1), columns=[y_label])
        data["Round"] = [i for i in range(values[i].shape[1])] * values[i].shape[0]
        sns.lineplot(
            data=data,
            x="Round",
            y=y_label,
            label=names[i],
            linewidth=3,
            linestyle=linestyles[i],
        )

    plt.grid(linestyle="--", linewidth=2)
    plt.xlabel("Round $r$", fontweight="bold", fontsize=24)
    plt.ylabel(y_label, fontweight="bold", fontsize=24)

    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.linewidth"] = 2
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(prop=dict(size=20, weight="bold"))
    plt.tight_layout()
    plt.legend([], [], frameon=False)
    plt.savefig(save_loc)


# Creates Accuracy plots for the simulations
def plot_boxplot_values(values, names, save_loc, y_label, plt_int=False):
    vals = np.concatenate(values)
    df = pd.DataFrame(vals, columns=[y_label])

    n_sim = len(values[0])

    df_names = []
    for name in names:
        df_names.extend([name] * n_sim)
    df["Policy"] = df_names

    sns.set_theme(style="darkgrid")
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 7), dpi=600)
    gg = sns.boxplot(data=df, x="Policy", y=y_label, order=names)

    boxes = ax.findobj(matplotlib.patches.PathPatch)

    colors = ["gray", "tab:blue", "tab:orange", "tab:green"]

    for color, box in zip(colors, boxes):
        box.set_facecolor(color)

    for i in range(len(names)):
        boxes[i].set_label(names[i])

    if n_sim >= 2:
        f = add_stat_annotation(
            gg,
            data=df,
            x="Policy",
            y=y_label,
            order=names,
            box_pairs=[((names[1]), (names[3]))],
            test="Wilcoxon",
            text_format="full",
            loc="outside",
            verbose=0,
            fontsize=20,
        )
    else:
        f = add_stat_annotation(
            gg,
            data=df,
            x="Policy",
            y=y_label,
            order=names,
            box_pairs=[((names[1]), (names[1]))],
            test="Mann-Whitney",
            text_format="full",
            loc="outside",
            verbose=0,
            fontsize=20,
        )

    ax.set_xlabel(ax.get_xlabel(), fontdict={"weight": "bold", "size": 24})
    plt.ylabel(y_label, fontweight="bold", fontsize=24)
    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.linewidth"] = 2
    ax.xaxis.set_tick_params(labelsize=14, width=2)
    ax.yaxis.set_tick_params(labelsize=14)
    locs, labels = plt.xticks()
    plt.xticks(locs, labels, weight="bold")
    plt.tight_layout()
    plt.legend([], [], frameon=False)
    plt.savefig(save_loc)


def plot_tsne(train_embs, method_embs, save_loc):
    fig, ax = plt.subplots(figsize=(8, 7), dpi=600)
    # plot the distribution of the data points with gray color

    sns.color_palette("tab10")

    data = np.concatenate(
        (train_embs, method_embs[0], method_embs[1], method_embs[2]), axis=0
    )
    data = pd.DataFrame(data, columns=["x", "y"])
    data["Cluster"] = (
        ["Training Data"] * len(train_embs)
        + ["Distributed"] * len(method_embs[0])
        + ["Centralized"] * len(method_embs[1])
        + ["Interactive"] * len(method_embs[2])
    )

    # sns.scatterplot(x="x",y="y",data=data,hue="Cluster",palette=["gray","tab:blue","tab:orange","tab:green"],alpha=0.5,legend=False,ax=ax)

    plt.scatter(
        train_embs[:, 0],
        train_embs[:, 1],
        color="tab:gray",
        label="Training Data",
        alpha=0.1,
    )

    plt.scatter(
        method_embs[0][:, 0],
        method_embs[0][:, 1],
        marker="x",
        label="Distributed",
        linewidths=3,
        alpha=1,
        color="tab:blue",
        s=200,
    )
    plt.scatter(
        method_embs[1][:, 0],
        method_embs[1][:, 1],
        marker="+",
        label="Centralized",
        alpha=1,
        linewidths=3,
        color="tab:orange",
        s=200,
    )
    plt.scatter(
        method_embs[2][:, 0],
        method_embs[2][:, 1],
        marker="o",
        facecolors="none",
        alpha=1,
        label="Interactive",
        linewidths=3,
        color="tab:green",
        s=200,
    )

    plt.rcParams["font.size"] = 15
    plt.rcParams["axes.linewidth"] = 2
    plt.rcParams["legend.labelspacing"] = 0.4

    handles = [
        plt.scatter([], [], marker="o", color="tab:gray", label="Training Data", s=150),
        plt.scatter(
            [],
            [],
            marker="x",
            color="tab:blue",
            label="Distributed",
            s=200,
            linewidths=3,
        ),
        plt.scatter(
            [],
            [],
            marker="+",
            color="tab:orange",
            label="Centralized",
            s=200,
            linewidths=3,
        ),
        plt.scatter(
            [],
            [],
            marker="o",
            facecolors="none",
            color="tab:green",
            s=200,
            label="Interactive",
            linewidths=3,
        ),
    ]
    # legend = plt.legend(handles=handles,frameon=True,loc='lower left',bbox_to_anchor=(-0.1, 1.1))
    legend = plt.legend(handles=handles, frameon=True)

    for text in legend.get_texts():
        text.set_weight("bold")

    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)

    plt.tick_params(
        axis="both",
        left=False,
        top=False,
        right=False,
        bottom=False,
        labelleft=False,
        labeltop=False,
        labelright=False,
        labelbottom=False,
    )

    plt.axis("off")
    plt.tight_layout()

    plt.savefig(save_loc)


def make_federated_learning_legend(save_loc="./"):
    fig = plt.figure(constrained_layout=False)
    fig.set_size_inches(18.5, 10.5)
    gs = fig.add_gridspec(1, 1)

    acc = np.ones((7, 2))

    all_data_df = pd.DataFrame(acc, columns=["Iterations", "Real Grad Norm"])
    all_data_df["method"] = [
        "Distributed",
        "Centralized",
        "Interactive",
        "Random",
        "Entropy",
        "Least Confidence",
        "BvSB",
    ]

    labels = [
        r"1$\times$",
        r"2$\times$",
        r"5$\times$",
        r"10$\times$",
        r"20x",
        r"50$\times$",
    ]
    ax00 = fig.add_subplot(gs[0])
    markers = ["P", "o", "s", "D", "^", "v", "X"]
    # linestyles=["--","-","-.",":","--","-","-."]
    linestyles = ["-.", "-", "--", ":", "-.", "-", "--", ":", "-.", "-"]

    sns.lineplot(
        x=[1],
        y=[2],
        color=sns.color_palette("PuRd", 8)[2],
        linestyle=":",
        ax=ax00,
        linewidth=3,
        label=r"1$\times$ FL",
    )
    sns.lineplot(
        x=[1],
        y=[2],
        color=sns.color_palette("PuRd", 8)[3],
        linestyle="--",
        ax=ax00,
        linewidth=3,
        label=r"2$\times$ FL",
    )
    sns.lineplot(
        x=[1],
        y=[2],
        color=sns.color_palette("PuRd", 8)[4],
        linestyle="-",
        ax=ax00,
        linewidth=3,
        label=r"5$\times$ FL",
    )
    sns.lineplot(
        x=[1],
        y=[2],
        color=sns.color_palette("PuRd", 8)[5],
        linestyle="-.",
        ax=ax00,
        linewidth=3,
        label=r"10$\times$ FL",
    )
    sns.lineplot(
        x=[1],
        y=[2],
        color=sns.color_palette("PuRd", 8)[6],
        linestyle=":",
        ax=ax00,
        linewidth=3,
        label=r"20$\times$ FL",
    )
    sns.lineplot(
        x=[1],
        y=[2],
        color=sns.color_palette("PuRd", 8)[7],
        linestyle="--",
        ax=ax00,
        linewidth=3,
        label=r"50$\times$ FL",
    )
    sns.lineplot(
        x=[1],
        y=[2],
        color="tab:green",
        marker="s",
        markersize=7,
        linestyle="-.",
        ax=ax00,
        linewidth=3,
        label="Interactive",
    )

    # sns_plot = sns.lineplot(x="Iterations", y="Real Grad Norm", data=all_data_df, hue="method", style="method", markers=["P","o","s","D","^","v","X"],linestyles=["--","-","-.",":","--","-","-."], ax=ax00,linewidth=10)
    handles, labels = ax00.get_legend_handles_labels()

    figLegend = pylab.figure(figsize=(10, 0.3))
    # print(ax00.get_legend_handles_labels())
    pylab.figlegend(
        *ax00.get_legend_handles_labels(),
        loc="upper left",
        mode="expand",
        ncol=7,
        prop={"weight": "bold", "size": 12},
        borderaxespad=0,
        frameon=False,
    )
    figLegend.savefig(os.path.join(save_loc, f"FLlengend.png"), dpi=600)


# Creates L2-Norm Plots for the simulations
def plot_FL_accs(
    accs: list,
    names: list,
    save_loc: str,
    linestyles=[":", "--", "-", "-.", ":", "--", "-", "-."],
    legend: bool = False,
) -> None:
    """
    :param accs: Accuracy values
    :param names: Names of the algorithms
    :param save_loc: Location to save the plot
    :return: None
    """

    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 7), dpi=600)

    colors = sns.color_palette("PuRd", 8)

    for i in range(len(accs)):
        data = pd.DataFrame(accs[i].reshape(-1, 1), columns=["Accuracy"])
        data["Round"] = [i for i in range(accs[i].shape[1])] * accs[i].shape[0]
        sns.lineplot(
            data=data,
            x="Round",
            y="Accuracy",
            label=names[i],
            linestyle=linestyles[i],
            color=colors[i + 2],
            linewidth=3,
        )

    plt.grid(linestyle="--", linewidth=2)
    plt.xlabel("Round $r$", fontweight="bold", fontsize=24)
    plt.ylabel("Accuracy", fontweight="bold", fontsize=24)

    plt.rcParams["font.size"] = 18
    plt.rcParams["axes.linewidth"] = 2
    ax.xaxis.set_tick_params(labelsize=14)
    ax.yaxis.set_tick_params(labelsize=14)
    plt.legend(prop=dict(size=20, weight="bold"))
    plt.tight_layout()
    if not legend:
        plt.legend([], [], frameon=False)
    plt.savefig(save_loc)
