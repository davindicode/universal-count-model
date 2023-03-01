import os
import pickle

import sys

import matplotlib.pyplot as plt
import numpy as np

import torch

from neuroprob import utils


def regression_scores(fig, RG_dict):
    region_edge = region_edge[
        0
    ]  # boundary separating PoS (lower) and ANT (higher or equal)

    # scores
    widths = [2, 1]
    heights = [1, 3, 2, 3, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        wspace=0.9,
        hspace=0.0,
        height_ratios=heights,
        top=1.0,
        bottom=0.75,
        left=0.0,
        right=0.35,
    )

    eps = 0.4
    Ncases = PLL_rg_ll.shape[0] - 1
    order = [1, 2, 0]

    # RG
    ax = fig.add_subplot(spec[1:4, 0])

    fact = 1e3
    scores = PLL_rg_ll
    rel_score = (scores[order, :] - scores[order, :][0:1, :]) / fact
    ax.errorbar(
        np.arange(scores.shape[0])[1:],
        rel_score.mean(-1)[1:],
        linestyle="",
        marker="+",
        markersize=4,
        capsize=3,
        yerr=rel_score.std(-1, ddof=1)[1:] / np.sqrt(rel_score.shape[-1]),
        c="k",
    )

    ax.plot(np.linspace(-eps, Ncases + eps, 2), np.zeros(2), "gray")
    ax.plot(
        np.arange(scores.shape[0])[:, None].repeat(scores.shape[1], axis=1),
        rel_score,
        color="gray",
        marker=".",
        markersize=4,
        alpha=0.5,
    )
    ax.set_xlim(-eps, Ncases + eps)
    ax.set_xticks(np.arange(PLL_rg_ll.shape[0]))
    ax.set_xticklabels(["Poisson", "hNB", "Universal"])
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_ylabel(r"$\Delta$cvLL ($10^3$)", fontsize=10, labelpad=5)

    # KS
    ax = fig.add_subplot(spec[:2, 1])
    fact = 10 ** (-2)
    ax.set_xlim(-eps, Ncases + eps)
    for en, r in enumerate(T_KS_ll.mean(0)[order] / fact):
        ax.scatter(
            en * np.ones(len(r)) + np.random.rand(len(r)) * eps / 2 - eps / 4,
            r,
            color="gray",
            marker="+",
        )

    xl, xu = ax.get_xlim()
    ax.fill_between(
        np.linspace(xl, xu, 2), 0, np.ones(2) * sign_KS / fact, color="k", alpha=0.2
    )
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_ylabel(r"$T_{KS}$ ($10^{-2}$)", fontsize=10, labelpad=5)
    ax.set_xticks(np.arange(scores.shape[0]))
    ax.set_xticklabels([])

    # DS
    ax = fig.add_subplot(spec[3:, 1])
    ax.set_xlim(-eps, Ncases + eps)
    for en, r in enumerate(T_DS_ll.mean(0)[order]):
        ax.scatter(
            en * np.ones(len(r)) + np.random.rand(len(r)) * eps / 2 - eps / 4,
            r,
            color="gray",
            marker="+",
        )

    xl, xu = ax.get_xlim()
    ax.fill_between(
        np.linspace(xl, xu, 2),
        -np.ones(2) * sign_DS,
        np.ones(2) * sign_DS,
        color="k",
        alpha=0.2,
    )
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_ylabel(r"$T_{DS}$", fontsize=10, labelpad=0)
    ax.set_xticks(np.arange(scores.shape[0]))
    ax.set_xticklabels(["P", "hNB", "U"])


def binning_stats(fig):
    skip = 20
    BINS = 20

    X = 0.0
    Y = 0.0
    widths = np.ones(len(show_neuron))
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.2,
        left=0.47 + X,
        right=0.65 + X,
        bottom=0.8 + Y,
        top=0.97 + Y,
    )

    for k, ne in enumerate(show_neuron):
        ax = fig.add_subplot(spec[0, k])
        if ne < region_edge:
            c = poscol
        else:
            c = antcol
        ax.set_title("cell {}".format(ne + 1), fontsize=12, color=c)
        if k == 0:
            ax.set_ylabel("Fano factor", fontsize=10, labelpad=5)
        ax.scatter(
            avg_models[1][ne, ::skip] / tbin,
            ff_models[1][ne, ::skip],
            marker=".",
            alpha=0.3,
        )
        ax.set_xlim(0)
        ax.set_ylim(0, 2.5)
        ax.set_yticks([0, 1, 2])
        if k > 0:
            ax.set_yticklabels([])

        xlims = ax.get_xlim()
        xx = np.linspace(xlims[0], xlims[1])
        ax.plot(xx, np.ones_like(xx), "k", alpha=0.5)

        # linear regression
        A = avg_models[1][ne, :] / tbin
        B = ff_models[1][ne, :]
        a = ((A * B).mean() - A.mean() * B.mean()) / A.var()
        b = (B.mean() * (A**2).mean() - (A * B).mean() * A.mean()) / A.var()
        ax.plot(xx, a * xx + b, "r")

    fig.text(0.55 + X, 0.71 + Y, "firing rate (Hz)", fontsize=10, ha="center")


def regression_stats(fig, RG_dict):
    # histograms
    FF = np.array([ff_models[b].mean(-1) for b in [1]])
    spec = fig.add_gridspec(
        ncols=1,
        nrows=3,
        width_ratios=[1],
        height_ratios=[1, 1, 2],
        hspace=0.2,
        left=0.68 + X,
        right=0.78 + X,
        bottom=0.8 + Y,
        top=1.0 + Y,
    )
    ax = fig.add_subplot(spec[-1, 0])

    xx = np.arange(FF.shape[0])[:, None].repeat(len(pick_neuron) - region_edge, axis=-1)
    xxrnd = np.random.rand(*xx.shape) * 0.2 - 0.1
    ax.scatter(FF[:, region_edge:], xx + xxrnd + 0.4, marker=".", s=4, c=antcol)
    xx = np.arange(FF.shape[0])[:, None].repeat(region_edge, axis=-1)
    xxrnd = np.random.rand(*xx.shape) * 0.2 - 0.1
    ax.scatter(FF[:, :region_edge], xx + xxrnd - 0.4, marker=".", s=4, c=poscol)

    ax.set_ylim([-1.0, 1.0])
    ax.set_yticks([])  # np.arange(lFF.shape[0]))
    ax.set_xlabel("average FF", fontsize=10, labelpad=5)
    bins = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], BINS)
    XLIM1 = ax.get_xlim()

    ax = fig.add_subplot(spec[1, 0])
    ax.hist(FF[0, :region_edge], bins=bins, alpha=0.5, density=True, color=poscol)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)

    ax = fig.add_subplot(spec[0, 0])
    ax.hist(FF[0, region_edge:], bins=bins, alpha=0.5, density=True, color=antcol)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)

    pr = np.array([p[0] for p in Pearson_ff])
    spec = fig.add_gridspec(
        ncols=1,
        nrows=3,
        width_ratios=[1],
        height_ratios=[1, 1, 2],
        hspace=0.2,
        left=0.815 + X,
        right=0.915 + X,
        bottom=0.8 + Y,
        top=1.0 + Y,
    )
    ax = fig.add_subplot(spec[-1, 0])

    xx = pr[region_edge:]
    xxrnd = np.random.rand(*xx.shape) * 0.2 - 0.1
    ax.scatter(xx, xxrnd + 0.4, marker=".", s=4, c=antcol, label="ANT")
    xx = pr[:region_edge]
    xxrnd = np.random.rand(*xx.shape) * 0.2 - 0.1
    ax.scatter(xx, xxrnd - 0.4, marker=".", s=4, c=poscol, label="PoS")

    ax.set_ylim([-1.0, 1.0])
    ax.set_yticks([])
    ax.set_xlim([-1, 1])
    ax.set_xlabel("FF-mean corr.", fontsize=10, labelpad=5)
    lgnd = ax.legend(handletextpad=0.0, bbox_to_anchor=(0.85, 1.5))
    lgnd.legendHandles[0]._sizes = [50]
    lgnd.legendHandles[1]._sizes = [50]

    bins = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], BINS)
    XLIM2 = ax.get_xlim()

    ax = fig.add_subplot(spec[1, 0])
    ax.hist(pr[:region_edge], bins=bins, alpha=0.5, density=True, color=poscol)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)

    ax = fig.add_subplot(spec[0, 0])
    ax.hist(pr[region_edge:], bins=bins, alpha=0.5, density=True, color=antcol)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)

    # lines
    spec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        hspace=0.2,
        left=0.68 + X,
        right=0.78 + X,
        bottom=0.8 + Y,
        top=1.0 + Y,
    )
    ax = fig.add_subplot(spec[0, 0])
    ax.set_xlim(XLIM1)
    ax.plot(1 * np.ones(2), np.linspace(0, 1, 2), "gray", alpha=0.5)
    ax.axis("off")

    spec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        hspace=0.2,
        left=0.815 + X,
        right=0.915 + X,
        bottom=0.8 + Y,
        top=1.0 + Y,
    )
    ax = fig.add_subplot(spec[0, 0])
    ax.set_xlim(XLIM2)
    ax.plot(0 * np.ones(2), np.linspace(0, 1, 2), "gray", alpha=0.5)
    ax.axis("off")


def tunings(fig, RG_dict):
    # lines
    X = 0.0
    Y = 0.0
    for tuns in range(4):
        spec = fig.add_gridspec(
            ncols=1,
            nrows=1,
            width_ratios=[1],
            height_ratios=[1],
            left=0.17 + X + tuns * 0.21,
            right=0.18 + X + tuns * 0.21,
            bottom=0.05 + Y,
            top=0.55 + Y,
        )
        ax = fig.add_subplot(spec[0, 0])
        ax.plot(np.zeros(2), np.linspace(0, 1, 2), "k", alpha=1.0, linewidth=0.9)
        ax.axis("off")

    # head direction
    X = 0.0
    Y = 0.0
    widths = np.ones(len(show_neuron))
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.8,
        hspace=0.4,
        left=0.0 + X,
        right=0.15 + X,
        bottom=0.3 + Y,
        top=0.55 + Y,
    )

    for k, ne in enumerate(show_neuron):
        ax = fig.add_subplot(spec[0, k])
        if ne < region_edge:
            c = poscol
        else:
            c = antcol
        ax.set_title("cell {}".format(ne + 1), fontsize=12, pad=10, color=c)
        if k == 0:
            ax.set_ylabel("rate (HZ)", fontsize=10, labelpad=5)

        (line,) = ax.plot(covariates_hd, mean_hd[ne] / tbin)
        ax.fill_between(
            covariates_hd,
            lower_hd[ne] / tbin,
            upper_hd[ne] / tbin,
            color=line.get_color(),
            alpha=0.5,
        )

        ax.set_ylim(0)
        ax.set_xlim(0, 2 * np.pi)
        ax.set_xticks([0, 2 * np.pi])
        ax.set_xticklabels([])

    for k, ne in enumerate(show_neuron):
        ax = fig.add_subplot(spec[1, k])
        ax.set_xticks([0, 2 * np.pi])
        if k == 0:
            ax.set_ylabel("FF", fontsize=10)
            ax.set_xlim(0, 2 * np.pi)
            ax.set_xticklabels([r"$0$", r"$2\pi$"])
        else:
            ax.set_xticklabels([])
        #    ax.set_yticklabels([])

        (line,) = ax.plot(covariates_hd, ffmean_hd[ne])
        ax.fill_between(
            covariates_hd,
            fflower_hd[ne],
            ffupper_hd[ne],
            color=line.get_color(),
            alpha=0.5,
        )

        ax.set_ylim(0.0, 1.7)
        ax.set_yticks([0, 1])
        ax.set_xlim(0, 2 * np.pi)

    fig.text(0.075 + X, 0.2 + Y, r"head direction", fontsize=10, ha="center")

    spec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        wspace=0.2,
        hspace=0.2,
        left=0.0 + X,
        right=0.15 + X,
        bottom=0.05 + Y,
        top=0.17 + Y,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(hd_mean_tf[:region_edge], hd_ff_tf[:region_edge], c=poscol, marker=".")
    ax.scatter(hd_mean_tf[region_edge:], hd_ff_tf[region_edge:], c=antcol, marker=".")
    ax.set_aspect(1)
    ax.set_xlabel("TI (rate)", fontsize=10, labelpad=2)
    ax.set_ylabel("TI (FF)", fontsize=10, labelpad=1)
    ax.plot(
        np.linspace(0.0, 1.0, 2), np.linspace(0.0, 1.0, 2), "k", linewidth=1, alpha=0.3
    )
    ax.set_xlim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 1])

    # position tuning
    X = 0.21
    Y = 0.0
    widths = np.ones(len(show_neuron))
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        hspace=0.5,
        wspace=0.1,
        left=0.0 + X,
        right=0.13 + X,
        bottom=0.3 + Y,
        top=Y + 0.55,
    )

    for k, ne in enumerate(show_neuron):
        ax = fig.add_subplot(spec[0, k])

        rate = field_pos[ne] / tbin  # [8:-8, 8:-8]
        ax.set_title("{:.1f}".format(rate.max()), fontsize=10, pad=3)
        im = utils.plot.visualize_field(
            (fig, ax), rate.T, grid_shape_pos, cbar=False, aspect="equal"
        )
        utils.plot.decorate_ax(ax, spines=[True, True, True, True])
        rm = rate.max()

    bp = 4 / 5 * (grid_shape_pos[0][1] - grid_shape_pos[0][0]) + grid_shape_pos[0][0]
    rp = 1 / 5 * (grid_shape_pos[0][1] - grid_shape_pos[0][0]) + grid_shape_pos[0][0]
    py = 1.1 * grid_shape_pos[1][1]
    for k, ne in enumerate(show_neuron):
        ax = fig.add_subplot(spec[1, k])

        FF = ff_pos[ne][5:-5, 5:-5]
        rate = np.log(FF)
        g = max(-rate.min(), rate.max())
        ax.text(
            bp, py, "{:.1f}".format(np.exp(g)), ha="center", fontsize=10, color="red"
        )
        ax.text(
            rp, py, "{:.1f}".format(np.exp(-g)), ha="center", fontsize=10, color="blue"
        )
        im2 = utils.plot.visualize_field(
            (fig, ax),
            rate.T,
            grid_shape_pos,
            cbar=False,
            aspect="equal",
            vmin=-g,
            vmax=g,
            cmap=weight_map,
        )
        utils.plot.decorate_ax(ax, spines=[True, True, True, True])

    fig.text(
        -0.02 + X, 0.425 + Y, r"$y$ position", rotation=90, fontsize=10, va="center"
    )
    fig.text(0.075 + X, 0.2 + Y, r"$x$ position", fontsize=10, ha="center")

    spec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        wspace=0.2,
        hspace=0.2,
        left=0.0 + X,
        right=0.15 + X,
        bottom=0.05 + Y,
        top=0.17 + Y,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(pos_mean_tf[:region_edge], pos_ff_tf[:region_edge], c=poscol, marker=".")
    ax.scatter(pos_mean_tf[region_edge:], pos_ff_tf[region_edge:], c=antcol, marker=".")
    ax.set_aspect(1)
    ax.plot(
        np.linspace(0.0, 1.0, 2), np.linspace(0.0, 1.0, 2), "k", linewidth=1, alpha=0.3
    )
    ax.set_xlim([0, 1])
    ax.set_xticks([0, 1])
    ax.set_ylim([0, 1])
    ax.set_yticks([0, 1])

    # colorbars
    X = 0.21
    Y = 0.0
    cspec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        left=0.135 + X,
        right=0.14 + X,
        bottom=0.45 + Y,
        top=0.525 + Y,
    )
    ax = fig.add_subplot(cspec[0, 0])
    ax.set_title("     max", fontsize=10, pad=1)
    utils.plot.add_colorbar(
        (fig, ax),
        im,
        ticktitle="",
        ticks=[0, rm],
        ticklabels=["0", ""],
        cbar_format=None,
        cbar_ori="vertical",
    )

    cspec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        left=0.135 + X,
        right=0.14 + X,
        bottom=0.325 + Y,
        top=0.4 + Y,
    )
    ax = fig.add_subplot(cspec[0, 0])
    # ax.set_title('Fano factor', fontsize=10)
    utils.plot.add_colorbar(
        (fig, ax),
        im2,
        ticktitle="",
        ticks=[0],
        ticklabels=["1"],
        cbar_format=None,
        cbar_ori="vertical",
    )

    # lengthbar
    cspec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        left=0.05 + X,
        right=0.07 + X,
        bottom=0.21 + Y,
        top=0.32 + Y,
    )
    ax = fig.add_subplot(cspec[0, 0])
    ax.plot(np.linspace(0, 1, 2), np.zeros(2), "gray")
    ax.plot(np.zeros(2), np.linspace(0, 1, 2), "gray")
    ax.set_aspect(1)
    ax.set_xlim(0, 1)
    ax.axis("off")
    ax.text(0.3, 0.3, "{} cm".format(int(W // 3)), ha="left", color="gray")

    # omega
    X = 0.42
    Y = 0.0
    widths = np.ones(len(show_neuron))
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.8,
        hspace=0.4,
        left=0.0 + X,
        right=0.15 + X,
        bottom=0.3 + Y,
        top=0.55 + Y,
    )

    for k, ne in enumerate(show_neuron):
        ax = fig.add_subplot(spec[0, k])

        (line,) = ax.plot(covariates_w, mean_w[ne] / tbin)
        ax.fill_between(
            covariates_w,
            lower_w[ne] / tbin,
            upper_w[ne] / tbin,
            color=line.get_color(),
            alpha=0.5,
        )

        if k == 1:
            ax.set_ylim(0, 5 / tbin)
        else:
            ax.set_ylim(0, 1.5 / tbin)

        ax.set_xlim(covariates_w.min(), covariates_w.max())
        ax.set_xticks([int(covariates_w.min()), 0, int(covariates_w.max())])
        ax.set_xticklabels([])

    for k, ne in enumerate(show_neuron):
        ax = fig.add_subplot(spec[1, k])

        (line,) = ax.plot(covariates_w, ffmean_w[ne])
        ax.fill_between(
            covariates_w,
            fflower_w[ne],
            ffupper_w[ne],
            color=line.get_color(),
            alpha=0.5,
        )

        ax.set_ylim(0, 1.5)
        ax.set_yticks([0, 1])
        ax.set_xlim(covariates_w.min(), covariates_w.max())
        ax.set_xticks([int(covariates_w.min()), 0, int(covariates_w.max())])
        if k == 0:
            ax.set_xticklabels([int(covariates_w.min()), 0, int(covariates_w.max())])
        else:
            ax.set_xticklabels([])
        #    ax.set_yticklabels([])

    fig.text(0.075 + X, 0.2 + Y, r"AHV (rad/s)", fontsize=10, ha="center")

    spec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        wspace=0.2,
        hspace=0.2,
        left=0.0 + X,
        right=0.15 + X,
        bottom=0.05 + Y,
        top=0.17 + Y,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(w_mean_tf[:region_edge], w_ff_tf[:region_edge], c=poscol, marker=".")
    ax.scatter(w_mean_tf[region_edge:], w_ff_tf[region_edge:], c=antcol, marker=".")
    ax.set_aspect(1)
    ax.plot(
        np.linspace(0.0, 0.6, 2), np.linspace(0.0, 0.6, 2), "k", linewidth=1, alpha=0.3
    )
    ax.set_xlim([0, 0.6])
    ax.set_xticks([0, 0.6])
    ax.set_ylim([0, 0.6])
    ax.set_yticks([0, 0.6])

    # speed modulation
    X = 0.63
    Y = 0.0
    widths = np.ones(len(show_neuron))
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.8,
        hspace=0.4,
        left=0.0 + X,
        right=0.15 + X,
        bottom=0.3 + Y,
        top=Y + 0.55,
    )

    for k, ne in enumerate(show_neuron):
        ax = fig.add_subplot(spec[0, k])

        (line,) = ax.plot(covariates_s, mean_s[ne] / tbin)
        ax.fill_between(
            covariates_s,
            lower_s[ne] / tbin,
            upper_s[ne] / tbin,
            color=line.get_color(),
            alpha=0.5,
        )
        ax.set_ylim(0)
        ax.set_xlim(covariates_s[0], covariates_s[-1])
        ax.set_xticks([covariates_s[0], covariates_s[-1]])
        ax.set_xticklabels([])

    for k, ne in enumerate(show_neuron):
        ax = fig.add_subplot(spec[1, k])

        (line,) = ax.plot(covariates_s, ffmean_s[ne])
        ax.fill_between(
            covariates_s,
            fflower_s[ne],
            ffupper_s[ne],
            color=line.get_color(),
            alpha=0.5,
        )

        ax.set_ylim(0, 1.5)
        ax.set_yticks([0, 1])
        ax.set_xlim(covariates_s[0], covariates_s[-1])
        ax.set_xticks([covariates_s[0], covariates_s[-1]])
        if k == 0:
            ax.set_xticklabels([int(covariates_s[0]), int(covariates_s[-1] / 10.0)])
        else:
            ax.set_xticklabels([])
        #    ax.set_yticklabels([])

    fig.text(0.075 + X, 0.2 + Y, r"speed (cm/s)", fontsize=10, ha="center")

    spec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        wspace=0.2,
        hspace=0.2,
        left=0.0 + X,
        right=0.15 + X,
        bottom=0.05 + Y,
        top=0.17 + Y,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(s_mean_tf[:region_edge], s_ff_tf[:region_edge], c=poscol, marker=".")
    ax.scatter(s_mean_tf[region_edge:], s_ff_tf[region_edge:], c=antcol, marker=".")
    ax.plot(
        np.linspace(0.0, 0.2, 2), np.linspace(0.0, 0.2, 2), "k", linewidth=1, alpha=0.3
    )
    ax.set_aspect(1)
    ax.set_xlim([0, 0.2])
    ax.set_xticks([0, 0.2])
    ax.set_ylim([0, 0.2])
    ax.set_yticks([0, 0.2])

    ### time ###
    X = 0.84
    Y = 0.0
    widths = np.ones(len(show_neuron))
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.8,
        hspace=0.4,
        left=0.0 + X,
        right=0.15 + X,
        bottom=0.3 + Y,
        top=Y + 0.55,
    )

    for k, ne in enumerate(show_neuron):
        ax = fig.add_subplot(spec[0, k])

        (line,) = ax.plot(covariates_t, mean_t[ne] / tbin)
        ax.fill_between(
            covariates_t,
            lower_t[ne] / tbin,
            upper_t[ne] / tbin,
            color=line.get_color(),
            alpha=0.5,
        )
        if k == 1:
            ax.set_ylim(0, 5 / tbin)
        else:
            ax.set_ylim(0, 1.5 / tbin)
        ax.set_xlim(covariates_t[0], covariates_t[-1])
        ax.set_xticks([covariates_t[0], covariates_t[-1]])
        ax.set_xticklabels([])

    for k, ne in enumerate(show_neuron):
        ax = fig.add_subplot(spec[1, k])

        (line,) = ax.plot(covariates_t, ffmean_t[ne])
        ax.fill_between(
            covariates_t,
            fflower_t[ne],
            ffupper_t[ne],
            color=line.get_color(),
            alpha=0.5,
        )

        ax.set_ylim(0, 1.5)
        ax.set_yticks([0, 1])
        ax.set_xlim(covariates_t[0], covariates_t[-1])
        ax.set_xticks([covariates_t[0], covariates_t[-1] // 60 * 60])
        if k == 0:
            ax.set_xticklabels([int(covariates_t[0]), int(covariates_t[-1] // 60)])
        else:
            ax.set_xticklabels([])
        #    ax.set_yticklabels([])

    fig.text(0.075 + X, 0.2 + Y, r"time (min)", fontsize=10, ha="center")

    spec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        wspace=0.2,
        hspace=0.2,
        left=0.0 + X,
        right=0.15 + X,
        bottom=0.05 + Y,
        top=0.17 + Y,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(t_mean_tf[:region_edge], t_ff_tf[:region_edge], c=poscol, marker=".")
    ax.scatter(t_mean_tf[region_edge:], t_ff_tf[region_edge:], c=antcol, marker=".")
    ax.plot(
        np.linspace(0.0, 0.2, 2), np.linspace(0.0, 0.2, 2), "k", linewidth=1, alpha=0.3
    )
    ax.set_aspect(1)


def noise_correlation_scores(fig, NC_dict):
    Ncases = fisher_z.shape[1]
    eps = 0.4

    X = 0.0
    Y = -0.05
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.4,
        left=0.0 + X,
        right=0.2 + X,
        bottom=-0.32 + Y,
        top=-0.15 + Y,
    )

    fact = 1e3
    c = cv_pll * (
        len(pick_neuron) / 5
    )  # .reshape(cv_pll.shape[0], -1) # rescale to be comparable to A
    rel_c = (c.mean(-1) - c.mean(-1)[0:1, :]) / fact

    ax = fig.add_subplot(spec[0, 0])
    ax.set_xlim(-eps, Ncases + eps - 1)
    ax.errorbar(
        np.arange(5)[1:],
        rel_c.mean(1)[1:],
        yerr=rel_c.std(1, ddof=1)[1:] / np.sqrt(rel_c.shape[1]),
        linestyle="",
        capsize=2,
        markersize=4,
        marker="+",
        color="k",
    )
    ax.plot(
        np.arange(5)[:, None].repeat(c.shape[1], axis=1),
        rel_c,
        marker="+",
        color="gray",
    )

    ax.set_xlabel(r"$D_z$", labelpad=5, fontsize=10)
    ax.set_xticks(np.arange(5))
    for d in ax.xaxis.get_majorticklabels():
        d.set_y(-1.05)
    ax.set_ylabel(r"$\Delta$cvLL ($10^3$)", labelpad=5, fontsize=10)

    kcv_ind = 0
    widths = np.ones(Ncases)
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.4,
        left=0.0 + X,
        right=0.2 + X,
        bottom=-0.505 + Y,
        top=-0.355 + Y,
    )

    for en, r in enumerate(fisher_z[kcv_ind]):
        ax = fig.add_subplot(spec[0, en])

        ax.hist(
            r,
            bins=np.linspace(-5, 5, 20),
            density=True,
            orientation="horizontal",
            color="gray",
        )
        ax.set_xticks([])
        if en == 0:
            ax.set_ylabel(r"Fisher $Z$", labelpad=4, fontsize=10)
        else:
            ax.set_yticks([])

        xx = np.linspace(-5, 5, 100)
        yy = scstats.norm.pdf(xx)
        ax.plot(yy, xx, "r")
        ax.set_ylim(-5, 5)
        # ax.scatter(en*np.ones(len(r))+np.random.rand(len(r))*eps/2-eps/4, r, color='gray', marker='+')
    # ax.set_ylabel(r'Fisher $Z$', labelpad=5, fontsize=10)


def noise_correlations_mats(fig, NC_dict):
    X = -0.035
    Y = -0.05
    widths = [1, 1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        hspace=0.5,
        left=0.3 + X,
        right=0.5 + X,
        bottom=-0.35 + Y,
        top=-0.1 + Y,
    )

    order = list(np.argsort(pref_hdw[:region_edge, 10])) + list(
        np.argsort(pref_hdw[region_edge:, 10]) + region_edge
    )
    datas = [
        R_mat_spt[0, kcv_ind][order, :][:, order],
        R_mat_spt[2, kcv_ind][order, :][:, order],
    ]
    for d in datas:
        d[np.arange(len(order)), np.arange(len(order))] = 0
    g = max(-np.stack(datas).min(), np.stack(datas).max()) * 0.3
    for en, data in enumerate(datas):  # show correlations
        ax = fig.add_subplot(spec[0, en])
        if en == 0:
            ax.set_title(r"$D_z=0$", fontsize=10, pad=1)
        else:
            ax.set_title(r"$D_z=2$", fontsize=10, pad=1)

        time_steps = len(avg_models[0][0])
        # data = 0.5*np.log((1+data)/(1-data))*np.sqrt(time_steps-3)
        im = utils.plot.draw_2d(
            (fig, ax),
            data,
            origin="lower",
            cmap=weight_map,
            vmin=-g,
            vmax=g,
            aspect="equal",
        )

        ax.set_xlim(-1.0, len(pick_neuron) + 1.0)
        ax.set_ylim(-1.0, len(pick_neuron) + 1.0)

        # borders
        ax.plot(
            np.linspace(0, region_edge, 2), len(pick_neuron) * np.ones(2), color="k"
        )
        ax.plot(
            0 * np.ones(2), np.linspace(region_edge, len(pick_neuron), 2), color="k"
        )
        ax.plot(
            np.linspace(region_edge, len(pick_neuron), 2), 0 * np.ones(2), color="k"
        )
        ax.plot(
            len(pick_neuron) * np.ones(2), np.linspace(0, region_edge, 2), color="k"
        )

        ax.plot(np.linspace(0, region_edge, 2), np.zeros(2), color=poscol)
        ax.plot(np.zeros(2), np.linspace(0, region_edge, 2), color=poscol)
        ax.plot(
            np.linspace(region_edge, len(pick_neuron), 2),
            len(pick_neuron) * np.ones(2),
            color=antcol,
        )
        ax.plot(
            len(pick_neuron) * np.ones(2),
            np.linspace(region_edge, len(pick_neuron), 2),
            color=antcol,
        )

        ax.plot(
            np.linspace(0, region_edge, 2),
            region_edge * np.ones(2),
            color=poscol,
            alpha=1.0,
        )
        ax.plot(
            region_edge * np.ones(2),
            np.linspace(0, region_edge, 2),
            color=poscol,
            alpha=1.0,
        )
        ax.plot(
            np.linspace(region_edge, len(pick_neuron), 2),
            region_edge * np.ones(2),
            color=antcol,
            alpha=1.0,
        )
        ax.plot(
            region_edge * np.ones(2),
            np.linspace(region_edge, len(pick_neuron), 2),
            color=antcol,
            alpha=1.0,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        utils.plot.decorate_ax(ax, spines=[False, False, False, False])
        if en == 0:
            # ax.set_xlabel(r'neuron $i$', fontsize=10, labelpad=10)
            ax.set_ylabel(r"neuron $i$", fontsize=10, labelpad=2)

    fig.text(0.4 + X, -0.35 + Y, r"neuron $j$", fontsize=10, ha="center")

    cspec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        left=0.51 + X,
        right=0.513 + X,
        bottom=-0.28 + Y,
        top=-0.17 + Y,
    )
    ax = fig.add_subplot(cspec[0, 0])
    ax.set_title(r"  $r_{ij}$", fontsize=12, pad=10)
    utils.plot.add_colorbar(
        (fig, ax),
        im,
        ticktitle=r"",
        ticks=[-0.05, 0, 0.05],
        ticklabels=["-0.05", "0", "0.05"],
        cbar_format=None,
        cbar_ori="vertical",
    )


def noise_correlations_single_neuron_variability(fig, NC_dict):
    X = -0.02
    Y = -0.05
    widths = [1, 1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        hspace=0.8,
        left=0.3 + X,
        right=0.55 + X,
        bottom=-0.52 + Y,
        top=-0.42 + Y,
    )
    ax = fig.add_subplot(spec[0, 0])
    ff1 = np.array(ff_models_z).mean(-1)[0]
    ff2 = np.array(ff_models_z).mean(-1)[2]
    ax.scatter(ff1[:region_edge], ff2[:region_edge], marker=".", color=poscol)
    ax.scatter(ff1[region_edge:], ff2[region_edge:], marker=".", color=antcol)
    ax.plot(np.linspace(0.7, 1.3, 2), np.linspace(0.7, 1.3, 2), "gray")
    ax.plot(np.ones(2), np.linspace(0.7, 1.3, 2), "gray", alpha=0.5)
    ax.plot(np.linspace(0.7, 1.3, 2), np.ones(2), "gray", alpha=0.5)
    ax.set_xlim(0.7, 1.3)
    ax.set_ylim(0.7, 1.3)
    ax.set_xticks([0.8, 1.0, 1.2])
    ax.set_yticks([0.8, 1.0, 1.2])
    ax.set_xticklabels([0.8, "", 1.2])
    ax.set_yticklabels([0.8, "", 1.2])
    ax.set_aspect(1)
    ax.set_ylabel(r"$D_z=2$", fontsize=10, labelpad=5)
    ax.set_title("average FF", fontsize=10, pad=5)

    ax = fig.add_subplot(spec[0, 1])
    # ff1 = [p[0] for p in Pearson_ffz[0]]
    # ff2 = [p[0] for p in Pearson_ffz[2]]
    ddata1 = fisher_z[
        kcv_ind, 0, :
    ]  # np.concatenate([datas[0][k, k+1:] for k in range(len(pick_neuron))])
    ddata2 = fisher_z[
        kcv_ind, 2, :
    ]  # np.concatenate([datas[1][k, k+1:] for k in range(len(pick_neuron))])
    for en in range(len(ddata1)):
        m, n_ = model_utils.ind_to_pair(en, len(pick_neuron))
        n = len(pick_neuron) - n_ - 1
        if m < region_edge and n < region_edge:
            c = poscol
        elif m >= region_edge and n >= region_edge:
            c = antcol
        else:
            c = "k"
        ax.scatter(ddata1[en], ddata2[en], marker=".", color=c, s=1)

    ax.set_aspect(1)
    xlim_m = 7  # max(-ax.get_xlim()[0], ax.get_xlim()[1])
    xlim = [-xlim_m, xlim_m]
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-5, 0, 5])
    ax.plot(np.linspace(xlim[0], xlim[1], 2), np.linspace(xlim[0], xlim[1], 2), "gray")
    ax.plot(np.zeros(2), np.linspace(xlim[0], xlim[1], 2), "gray", alpha=0.5)
    ax.plot(np.linspace(xlim[0], xlim[1], 2), np.zeros(2), "gray", alpha=0.5)
    ax.set_xlim(xlim)
    ax.set_ylim(xlim)
    ax.set_title(r"Fisher $Z$", fontsize=10, pad=5)
    # ax.set_title(r'FF-mean corr.', fontsize=10, pad=5)
    fig.text(0.425 + X, -0.6 + Y, r"$D_z=0$", fontsize=10, ha="center")


def latent_trajs(fig, NC_dict):
    ### data ###

    ### plot ###

    # trajectories
    X = -0.03
    Y = -0.05
    widths = [1, 0.2]
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.1,
        hspace=0.4,
        left=0.63 + X,
        right=0.75 + X,
        bottom=-0.5 + Y,
        top=-0.12 + Y,
    )

    T_plot = 251
    tbin = 0.04
    ts = np.arange(T_plot) * tbin

    ax = fig.add_subplot(spec[0, 0])
    (line,) = ax.plot(ts, X_c[:T_plot, 0], color="k", alpha=0.5)
    ax.fill_between(
        ts,
        X_c[:T_plot, 0] - X_s[:T_plot, 0],
        X_c[:T_plot, 0] + X_s[:T_plot, 0],
        color=line.get_color(),
        alpha=0.3,
    )
    ax.set_ylabel(r"$z_1$", fontsize=10, labelpad=2)
    ax.set_ylim([-0.2, 0.2])
    ax.set_xlim(ts[0], ts[-1])
    ax.set_xticks([])
    ax.set_yticks([])

    ax = fig.add_subplot(spec[0, 1])
    ax.hist(X_c[:, 0], density=True, orientation="horizontal", alpha=0.3, color="k")
    ax.set_ylim([-0.2, 0.2])
    ax.set_yticks([])
    ax.set_xticks([])

    ax = fig.add_subplot(spec[1, 0])
    ax.set_ylim([-0.2, 0.2])
    (line,) = ax.plot(ts, X_c[:T_plot, 1], color="k", alpha=0.5)
    ax.fill_between(
        ts,
        X_c[:T_plot, 1] - X_s[:T_plot, 1],
        X_c[:T_plot, 1] + X_s[:T_plot, 1],
        color=line.get_color(),
        alpha=0.3,
    )
    ax.set_ylabel(r"$z_2$", fontsize=10, labelpad=2)
    ax.set_xlabel("time (s)", fontsize=10, labelpad=0)
    ax.set_xlim(ts[0], ts[-1])
    ax.set_xticks([ts[0], ts[-1]])
    ax.set_yticks([])

    ax = fig.add_subplot(spec[1, 1])
    ax.hist(X_c[:, 1], density=True, orientation="horizontal", alpha=0.3, color="k")
    ax.set_ylim([-0.2, 0.2])
    ax.set_yticks([])
    ax.set_xticks([])

    X = 0.0
    Y = -0.05
    widths = [1]
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        hspace=0.6,
        left=0.78 + X,
        right=0.83 + X,
        bottom=-0.5 + Y,
        top=-0.12 + Y,
    )

    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(z1_mean_tf[:region_edge], z1_ff_tf[:region_edge], c=poscol, marker=".")
    ax.scatter(z1_mean_tf[region_edge:], z1_ff_tf[region_edge:], c=antcol, marker=".")
    ax.set_aspect(1)
    ax.plot(
        np.linspace(0.0, 1.0, 2), np.linspace(0.0, 1.0, 2), "k", linewidth=1, alpha=0.3
    )
    ax.set_xlim([0, 1.0])
    ax.set_xticks([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.set_yticks([0, 1.0])

    ax = fig.add_subplot(spec[1, 0])
    ax.scatter(z2_mean_tf[:region_edge], z2_ff_tf[:region_edge], c=poscol, marker=".")
    ax.scatter(z2_mean_tf[region_edge:], z2_ff_tf[region_edge:], c=antcol, marker=".")
    ax.set_aspect(1)
    ax.plot(
        np.linspace(0.0, 1.0, 2), np.linspace(0.0, 1.0, 2), "k", linewidth=1, alpha=0.3
    )
    ax.set_xlim([0, 1.0])
    ax.set_xticks([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.set_yticks([0, 1.0])
    ax.set_xlabel("TI (rate)", fontsize=10, labelpad=2)
    ax.set_ylabel("TI (FF)", fontsize=10, labelpad=1)


def timescales(fig, RG_dict, NC_dict):
    ### data ###

    ### plot ###
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        hspace=0.6,
        left=0.93 + X,
        right=1.0 + X,
        bottom=-0.6 + Y,
        top=-0.1 + Y,
    )

    ax = fig.add_subplot(spec[0, 0])
    tl_min = np.log10(t_lengths.min())
    tbin_l = np.log10(tbin)
    TT = np.log10(timescales)
    names = ["HD", "AHV", "speed", r"pos.", "", r"$z_1$", r"$z_2$"]

    ax.text(0.98, tl_min * 0.85, "min drift")
    ax.plot(np.linspace(0.95, 1.2, 2), np.ones(2) * tl_min, "k")
    ax.text(0.98, tbin_l * 1.25, "time bin")
    ax.plot(np.linspace(0.95, 1.2, 2), np.ones(2) * tbin_l, "k")

    ax.set_xlim(0.95, 1.2)
    ax.set_xticks([])
    ax.set_ylim(-1.9)

    for en, name in enumerate(names):
        if en == 5:
            dd = -0.2
            c = "r"
        elif en == 6:
            dd = 0.2
            c = "r"
        else:
            dd = 0
            c = "k"

        ax.scatter(np.ones(len(timescales))[en], TT[en], marker="+", s=30, color=c)
        ax.text(1.05, TT[en] + dd, name, fontsize=10, va="center", color=c)
    ax.set_ylabel(r"log$_{10}$ $\tau$ (s)", fontsize=10, labelpad=-3)


def main():
    save_dir = "../output/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use(["paper.mplstyle"])

    # load
    RG_results = pickle.load(open(save_dir + "th1_RG_results.p", "rb"))
    NC_results = pickle.load(open(save_dir + "th1_NC_results.p", "rb"))

    # plot
    fig = plt.figure(figsize=(8, 4))

    fig.text(-0.08, 1.02, "A", transform=fig.transFigure, size=15, fontweight="bold")
    fig.text(0.415, 1.02, "B", transform=fig.transFigure, size=15, fontweight="bold")
    fig.text(-0.08, 0.6, "C", transform=fig.transFigure, size=15, fontweight="bold")
    fig.text(-0.08, -0.15, "D", transform=fig.transFigure, size=15, fontweight="bold")
    fig.text(0.23, -0.15, "E", transform=fig.transFigure, size=15, fontweight="bold")
    fig.text(0.565, -0.15, "F", transform=fig.transFigure, size=15, fontweight="bold")
    fig.text(0.87, -0.15, "G", transform=fig.transFigure, size=15, fontweight="bold")

    white = "#ffffff"
    black = "#000000"
    red = "#ff0000"
    blue = "#0000ff"
    weight_map = utils.plot.make_cmap([blue, white, red], "weight_map")

    poscol = "forestgreen"
    antcol = "orange"
    BINS = 20
    S = 2  # marker size

    H = grid_shape_pos[1][1] - grid_shape_pos[1][0]
    W = grid_shape_pos[0][1] - grid_shape_pos[0][0]

    show_neuron = [11, 26]  # PoS and ANT respectively

    ### regression ###
    regression_scores(fig, RG_results)
    binning_stats(fig, RG_results)
    regression_stats(fig, RG_results)
    tunings(fig, RG_results)

    ### noise correlations ###
    noise_correlation_scores(fig, NC_results)
    noise_correlations_mats(fig, NC_results)
    noise_correlations_single_neuron_variability(fig, NC_results)

    ### covariates ###
    latent_trajs(fig, NC_results)
    timescales(fig, NC_results)

    plt.savefig(save_dir + "plot_hdc.pdf")


if __name__ == "__main__":
    main()
