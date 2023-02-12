import os

import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sps
import torch
from matplotlib import patches

sys.path.append("../..")
from neuroprob import utils


def graphical_model(fig, X, Y):
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        left=0.0 + X,
        right=0.35 + X,
        bottom=-0.8 + Y,
        top=1.0 + Y,
    )

    ax = fig.add_subplot(spec[0, 0])

    Xoff = -0.3
    Yoff = -0.6

    # Instantiate the PGM
    pgm = utils.plots.daft_init_figax(fig, ax, shape=(2.5, 3.0), node_unit=0.7)

    pgm.add_node("f", r"$f_{cnt}$", 1.0 + Xoff, 2.6 + Yoff, plot_params={"fc": "white"})
    pgm.add_node(
        "x",
        r"$\mathbf{x}_{t}$",
        0.775 + Xoff,
        3.2 + Yoff,
        observed=True,
        plot_params={"ec": "tab:red"},
        label_params={"c": "tab:red"},
    )
    pgm.add_node("y", r"$y_{nt}$", 1.0 + Xoff, 1.475 + Yoff, observed=True)
    pgm.add_node(
        "z",
        r"$\mathbf{z}_{t}$",
        1.225 + Xoff,
        3.2 + Yoff,
        plot_params={"fc": "white", "ec": "tab:green"},
        label_params={"c": "tab:green"},
    )
    pgm.add_node(
        "p",
        r"$\mathbf{\pi}_{nt}$",
        1.0 + Xoff,
        2.0 + Yoff,
        plot_params={"fc": "white", "ec": "tab:blue"},
        label_params={"c": "tab:blue"},
    )

    # deterministic variables
    pgm.add_node(
        "pr",
        r"$\theta^{\,\mathrm{pr}}$",
        1.8 + Xoff,
        3.2 + Yoff,
        shape="rectangle",
        plot_params={"fc": "white"},
    )
    pgm.add_node(
        "gp",
        r"$\theta^{\,\mathrm{GP}}_{cn}$",
        1.8 + Xoff,
        2.6 + Yoff,
        shape="rectangle",
        plot_params={"fc": "white"},
    )
    pgm.add_node("pr_", "", 1.805 + Xoff, 3.2 + Yoff, plot_params={"alpha": 0.0})
    pgm.add_node("gp_", "", 1.805 + Xoff, 2.6 + Yoff, plot_params={"alpha": 0.0})

    pgm.add_node(
        "w",
        r"$W_n$",
        1.8 + Xoff,
        2.0 + Yoff,
        shape="rectangle",
        plot_params={"fc": "white"},
    )
    pgm.add_node(
        "b",
        r"$\mathbf{b}_n$",
        1.8 + Xoff,
        1.6 + Yoff,
        shape="rectangle",
        plot_params={"fc": "white"},
    )
    pgm.add_node("w_", "", 1.805 + Xoff, 2.0 + Yoff, plot_params={"alpha": 0.0})
    pgm.add_node("b_", "", 1.78 + Xoff, 1.6 + Yoff, plot_params={"alpha": 0.0})

    pgm.add_node(
        "ph",
        r"$\phi$",
        0.4 + Xoff,
        2.0 + Yoff,
        fixed=True,
        fontsize=14,
        offset=(0, 3.0),
    )

    pgm.add_edge("x", "f")
    pgm.add_edge("z", "f")
    pgm.add_edge("f", "p")
    pgm.add_edge("p", "y")

    pgm.add_edge("w_", "p")
    pgm.add_edge("b_", "p")
    pgm.add_edge("ph", "p")
    pgm.add_edge("pr_", "z")
    pgm.add_edge("gp_", "f")

    pgm.add_plate(
        [0.65 + Xoff, 1.15 + Yoff, 1.65, 1.8],
        label=r"$N$",
        position="bottom right",
        shift=0.1,
    )
    pgm.add_plate(
        [0.525 + Xoff, 0.85 + Yoff, 0.95, 2.6],
        label=r"$T$",
        position="bottom right",
        shift=0.1,
    )
    pgm.add_plate(
        [0.75 + Xoff, 2.26 + Yoff, 1.45, 0.58],
        label=r"$C$",
        position="bottom right",
        shift=0.1,
    )

    utils.plots.daft_render(pgm)


def count_dists(fig):
    # probability model and toy data for visualization
    K = 11
    MC = 1000
    beta = 0.1

    X = np.arange(100)

    mu = X * 0.05
    std = 0.5 / (X * 0.01 + 1)

    s = np.log(
        np.exp(
            np.random.randn(MC)[:, None, None] * std[None, None, :] + mu[None, None, :]
        )
        + 1
    )

    # MC, K, X
    a = np.exp(
        -s
        + np.log(s) * np.arange(K)[None, :, None]
        - sps.gammaln(np.arange(1, K + 1)[None, :, None])
    )
    P = a / a.sum(1, keepdims=True)

    m = (np.arange(K)[None, :, None] * P).mean(1)
    s = ((np.arange(K) ** 2)[None, :, None] * P).mean(1)
    v = s - m**2
    ff = v / m

    cov = X * 0.01

    # count distributions with uncertainties
    pP_l, pP_m, pP_u = utils.stats.percentiles_from_samples(torch.from_numpy(P).float())

    widths = [1]
    heights = [1]

    X = -0.02
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        hspace=0.0,
        height_ratios=heights,
        top=0.35,
        bottom=-0.575,
        left=0.415 + X,
        right=0.78 + X,
    )

    ax = fig.add_subplot(spec[0, 0])
    utils.plots.decorate_ax(ax, spines=[True, True, True, True])
    ax.set_xlabel(r"$x$", fontsize=12, labelpad=2, color="tab:red")
    ax.set_ylabel(
        r"$z$", fontsize=12, labelpad=6, color="tab:green", rotation=0, va="center"
    )

    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        top=0.47,
        bottom=-0.695,
        left=0.758,
        right=0.85,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.axis("off")

    ax.set_ylim([0, 1])
    ax.plot(
        np.linspace(0.0, 1.0, 2), np.linspace(0.9, 0.925, 2), "lightgray", linewidth=1
    )
    ax.plot(
        np.linspace(0.0, 1.0, 2), np.linspace(0.1, 0.075, 2), "lightgray", linewidth=1
    )

    ax.plot(
        np.linspace(0.0, 1.0, 2), np.linspace(0.1, 0.6, 2), "lightgray", linewidth=1
    )

    b = 7.5
    db = 1
    ax.plot(
        np.linspace(0.0, b / 10.0, 2),
        np.linspace(0.9, 0.9 - 0.5 * b / 10, 2),
        "lightgray",
        linewidth=1,
    )
    ax.plot(
        np.linspace((b + db) / 10.0, 1.0, 2),
        np.linspace(0.9 - 0.5 * (b + db) / 10, 0.4, 2),
        "lightgray",
        linewidth=1,
    )

    Xoff = -0.02
    X = 0.11
    Y = 0.27
    xxs = [2, 30, 20, 40, 99, 40, 15, 30, 15]
    for k in range(3):
        for l in range(3):
            widths = [1]
            heights = [1]

            spec = fig.add_gridspec(
                ncols=len(widths),
                nrows=len(heights),
                width_ratios=widths,
                hspace=0.0,
                height_ratios=heights,
                top=-0.225 + Y * l,
                bottom=-0.425 + Y * l,
                left=0.45 + X * k + Xoff,
                right=0.55 + X * k + Xoff,
            )

            ax = fig.add_subplot(spec[0, 0])

            xx = xxs[k + 3 * l]
            for pp in range(K):
                XX = np.linspace(pp - 0.5, pp + 0.5, 2)
                YY = np.ones(2) * np.array(pP_m)[pp, xx]
                YY_l = np.ones(2) * np.array(pP_l)[pp, xx]
                YY_u = np.ones(2) * np.array(pP_u)[pp, xx]
                (line,) = ax.plot(XX, YY, c="tab:blue", label=l, alpha=1.0)
                ax.fill_between(XX, YY_l, YY_u, color=line.get_color(), alpha=0.3)

            ax.set_xlim([-0.5, K - 1 + 0.5])
            ax.set_ylim(0)
            ax.set_yticks([])

            if k == 0 and l == 0:
                ax.set_xticks([0, K - 1])
                ax.set_xticklabels([0, "K"])
                ax.set_ylabel("prob.", fontsize=10, labelpad=4)
                ax.set_xlabel("# spikes", labelpad=-9, fontsize=10)
            else:
                ax.set_xticks([])


def latent_traj(fig):
    T = 6001
    ts = np.arange(T) * 0.001

    Z_mean = np.sin(ts * 3 * (1 + np.exp(-((ts - ts[T // 2]) ** 2)))) * np.cos(ts / 2.6)
    Z_s = np.ones_like(ts) * 0.3

    X = -0.02
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        left=0.415 + X,
        right=1.02 + X,
        bottom=0.615,
        top=0.915,
    )

    ax = fig.add_subplot(spec[0, 0])
    (line,) = ax.plot(ts, Z_mean[:T], color="tab:green", alpha=1.0)
    ax.fill_between(
        ts,
        Z_mean[:T] - Z_s[:T],
        Z_mean[:T] + Z_s[:T],
        color=line.get_color(),
        alpha=0.3,
    )
    ax.set_ylabel(
        r"$z$", labelpad=6, fontsize=10, color="tab:green", rotation=0, va="center"
    )
    ax.set_xlabel("time", labelpad=5, fontsize=10)
    ax.set_xlim([0, ts[-1]])
    ax.set_xticks([])
    ax.set_yticks([])


def tunings(fig):
    # 2D heatmaps
    grid_n = [50, 50]
    grid_size = [[0, 1], [0, 1]]
    grid_mesh = np.meshgrid(
        np.linspace(grid_size[0][0], grid_size[0][1], grid_n[0]),
        np.linspace(grid_size[1][0], grid_size[1][1], grid_n[1]),
    )

    # mean tunings
    mu = np.array([0.6, 0.6])[:, None, None]
    mean_field = 5 * np.exp(-((grid_mesh - mu) ** 2).sum(0) * 10)

    X = 0.05
    Y = 0.0
    widths = [1, 0.6]
    heights = [0.6, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.2,
        hspace=0.8,
        left=0.87 + X,
        right=0.96 + X,
        bottom=0.0 + Y,
        top=0.375 + Y,
    )

    ax = fig.add_subplot(spec[1, 0])

    rate = mean_field
    rm = rate.max()
    ax.set_title("{:.1f} Hz".format(rate.max()), fontsize=10, pad=3)
    im = ax.imshow(
        rate.T, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=rm
    )
    utils.plots.decorate_ax(ax, spines=[True, True, True, True])
    ax.set_ylabel(
        r"$z$", fontsize=10, labelpad=6, color="tab:green", rotation=0, va="center"
    )
    ax.set_xlabel(r"$x$", labelpad=3, fontsize=10, color="tab:red")

    fig.text(0.95, 0.41, "firing rate", ha="center", fontweight="bold")
    ax = fig.add_subplot(spec[0, 0])
    ts = np.linspace(0, 1, 50)
    ys = mean_field.mean(1)
    (line,) = ax.plot(ts, ys, color="tab:gray", alpha=1.0)
    ax.fill_between(ts, ys * 0.8, ys * 1.2, color=line.get_color(), alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim(0)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(spec[1, 1])
    ts = np.linspace(0, 1, 50)
    ys = mean_field.mean(0)
    (line,) = ax.plot(ys, ts, color="tab:gray", alpha=1.0)
    ax.fill_betweenx(ts, ys * 0.8, ys * 1.2, color=line.get_color(), alpha=0.3)
    ax.set_ylim([0, 1])
    ax.set_xlim(0)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(spec[0, 1])
    ax.axis("off")
    ax.text(0.2, 0.25, r"TI$_x$ (rate)", color="tab:red")
    ax.text(1.25, -0.85, r"TI$_z$ (rate)", color="tab:green")

    # colorbars
    X = -0.05
    Y = -0.0
    cspec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        left=0.94 + X,
        right=0.945 + X,
        bottom=0.1 + Y,
        top=0.275 + Y,
    )
    ax = fig.add_subplot(cspec[0, 0])
    utils.plots.add_colorbar(
        (fig, ax),
        im,
        ticktitle="",
        ticks=[0, rm],
        ticklabels=["0", "max"],
        cbar_pad=15,
        cbar_fontsize=10,
        cbar_format=None,
        cbar_ori="vertical",
    )
    ax.yaxis.set_ticks_position("left")

    # annotations
    X = 0.05
    Y = 0.0
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=1.6,
        hspace=0.8,
        left=0.8 + X,
        right=0.96 + X,
        bottom=0.0 + Y,
        top=0.375 + Y,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.annotate(
        "",
        xy=(0.8, 1.0),
        zorder=1,
        xytext=(0.8, 0.7),
        arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color="gray"),
    )

    ax.annotate(
        "",
        xy=(0.8, 0.55),
        zorder=1,
        xytext=(1.0, 0.55),
        arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color="gray"),
    )
    ax.axis("off")

    white = "#ffffff"
    black = "#000000"
    red = "#ff0000"
    blue = "#0000ff"
    weight_map = utils.plots.make_cmap([blue, white, red], "weight_map")

    # FF tunings
    mu = np.array([0.4, 0.4])[:, None, None]
    ff_field = 1.5 * np.exp(-((grid_mesh - mu) ** 2).sum(0) * 2)

    X = 0.05
    Y = -0.62
    widths = [1, 0.6]
    heights = [0.6, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.2,
        hspace=0.8,
        left=0.87 + X,
        right=0.96 + X,
        bottom=0.0 + Y,
        top=0.375 + Y,
    )

    ax = fig.add_subplot(spec[1, 0])

    bp = 4 / 5 * (grid_size[0][1] - grid_size[0][0]) + grid_size[0][0]
    rp = 1 / 5 * (grid_size[0][1] - grid_size[0][0]) + grid_size[0][0]
    py = 1.1 * grid_size[1][1]

    rate = np.log(ff_field)
    g = max(-rate.min(), rate.max())
    ax.text(
        bp,
        py,
        "{:.1f}".format(np.exp(g)),
        ha="center",
        fontsize=10,
        color="red",
        transform=ax.transAxes,
    )
    ax.text(
        rp,
        py,
        "{:.1f}".format(np.exp(-g)),
        ha="center",
        fontsize=10,
        color="blue",
        transform=ax.transAxes,
    )

    im = ax.imshow(
        rate.T, origin="lower", aspect="auto", cmap=weight_map, vmin=-g, vmax=g
    )
    utils.plots.decorate_ax(ax, spines=[True, True, True, True])
    ax.set_ylabel(
        r"$z$", fontsize=10, labelpad=6, color="tab:green", rotation=0, va="center"
    )
    ax.set_xlabel(r"$x$", labelpad=3, fontsize=10, color="tab:red")

    fig.text(0.95, -0.2, "Fano factor", ha="center", fontweight="bold")
    ax = fig.add_subplot(spec[0, 0])
    ts = np.linspace(0, 1, 50)
    ys = ff_field.mean(1)
    (line,) = ax.plot(ts, ys, color="tab:gray", alpha=1.0)
    ax.fill_between(ts, ys * 0.7, ys * 1.3, color=line.get_color(), alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim(0)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(spec[1, 1])
    ts = np.linspace(0, 1, 50)
    ys = ff_field.mean(0)
    (line,) = ax.plot(ys, ts, color="tab:gray", alpha=1.0)
    ax.fill_betweenx(ts, ys * 0.7, ys * 1.3, color=line.get_color(), alpha=0.3)
    ax.set_ylim([0, 1])
    ax.set_xlim(0)
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(spec[0, 1])
    ax.axis("off")
    ax.text(0.2, 0.4, r"TI$_x$ (FF)", color="tab:red")
    ax.text(1.25, -0.85, r"TI$_z$ (FF)", color="tab:green")

    # colorbars
    X = -0.05
    Y = -0.62
    cspec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        left=0.94 + X,
        right=0.945 + X,
        bottom=0.1 + Y,
        top=0.275 + Y,
    )
    ax = fig.add_subplot(cspec[0, 0])
    cbar = utils.plots.add_colorbar(
        (fig, ax),
        im,
        ticktitle="",
        ticks=[-g, 0, g],
        ticklabels=["min", "1", "max"],
        cbar_pad=15,
        cbar_fontsize=10,
        cbar_format=None,
        cbar_ori="vertical",
    )
    ax.yaxis.set_ticks_position("left")

    # annotations
    X = 0.05
    Y = -0.62
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=1.6,
        hspace=0.8,
        left=0.8 + X,
        right=0.96 + X,
        bottom=0.0 + Y,
        top=0.375 + Y,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.annotate(
        "",
        xy=(0.8, 1.0),
        zorder=1,
        xytext=(0.8, 0.8),
        arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color="gray"),
    )

    ax.annotate(
        "",
        xy=(0.85, 0.55),
        zorder=1,
        xytext=(1.0, 0.55),
        arrowprops=dict(arrowstyle="|-|,widthA=0.2,widthB=0.2", color="gray"),
    )
    ax.axis("off")


def main():
    save_dir = "../output/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use(["paper.mplstyle"])

    # plot
    fig = plt.figure(figsize=(8, 2))

    ### components ###
    graphical_model(fig, 0.02, 0.0)
    count_dists(fig)
    latent_traj(fig)
    tunings(fig)

    ### connections ###
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        left=0.25,
        right=1.0,
        bottom=-0.5,
        top=0.8,
    )
    ax = fig.add_subplot(spec[0, 0])
    ax.axis("off")

    style = "simple, head_length=8.4, head_width=6.2"
    kw = dict(arrowstyle=style, color="gray")
    a = patches.FancyArrowPatch(
        (0.08, 0.425), (0.145, 0.425), connectionstyle="arc3,rad={}".format(0), **kw
    )
    ax.add_patch(a)

    style = "simple, head_length=8.4, head_width=6.2"
    kw = dict(arrowstyle=style, color="gray")
    a = patches.FancyArrowPatch(
        (0.03, 0.95), (0.145, 0.95), connectionstyle="arc3,rad={}".format(0), **kw
    )
    ax.add_patch(a)

    plt.savefig(save_dir + "plot_schem.pdf")


if __name__ == "__main__":
    main()
