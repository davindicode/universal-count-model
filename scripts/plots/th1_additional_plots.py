import os
import pickle

import sys

import matplotlib.pyplot as plt
import numpy as np

import torch

sys.path.append("..")
from neuroprob import utils


def regressors(fig, RG_dict):
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        wspace=0.4,
        height_ratios=heights,
        top=0.95,
        bottom=0.75,
        left=0.0,
        right=0.2,
    )

    eps = 0.4
    Ncases = PLL_rg_ll.shape[0] - 1
    fact = 10**3

    ax = fig.add_subplot(spec[0, 0])
    ax.set_xlim(-eps, Ncases + eps)

    rel_score = (PLL_rg_cov - PLL_rg_cov[0:1, :]) / fact
    ax.errorbar(
        np.arange(rel_score.shape[0])[1:],
        rel_score.mean(-1)[1:],
        linestyle="",
        marker="+",
        markersize=4,
        capsize=3,
        yerr=rel_score.std(-1, ddof=1)[1:] / np.sqrt(yerr.shape[-1]),
        c="k",
    )
    ax.plot(np.linspace(-eps, Ncases + eps, 2), np.zeros(2), "gray")
    ax.plot(
        np.arange(rel_score.shape[0])[:, None].repeat(rel_score.shape[1], axis=1),
        rel_score,
        color="gray",
        marker=".",
        markersize=4,
        alpha=0.5,
    )
    ax.set_ylabel(r"$\Delta$cvLL ($10^3$)", fontsize=10, labelpad=5)

    ax.set_xticks(np.arange(PLL_rg_cov.shape[0]))
    ax.set_xticklabels(["HD", "HD\nAHV\nspeed\ntime", "HD\nAHV\nspeed\npos.\ntime"])
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))


def bin_sizes(fig, RG_dict):
    BINS = 20
    binnings = [20, 40, 100, 200, 500]

    Xs = [0.0, 0.24, 0.48, 0.0, 0.24]
    Ys = [0.0, 0.0, 0.0, -0.25, -0.25]
    skips = [40, 20, 5, 2, 1]
    b = 0
    for X, Y in zip(Xs, Ys):
        skip = skips[b]
        widths = np.ones(len(show_neuron))
        heights = [1]
        spec = fig.add_gridspec(
            ncols=len(widths),
            nrows=len(heights),
            width_ratios=widths,
            height_ratios=heights,
            wspace=0.6,
            left=0.35 + X,
            right=0.52 + X,
            bottom=0.82 + Y,
            top=0.95 + Y,
        )

        if b == 1:
            addstr = " (main results)"
        else:
            addstr = ""

        fig.text(
            0.425 + X,
            0.97 + Y,
            "{} ms".format(binnings[b]) + addstr,
            ha="center",
            color="gray",
        )
        for k, ne in enumerate(show_neuron):
            ax = fig.add_subplot(spec[0, k])
            if ne < region_edge:
                c = poscol
            else:
                c = antcol

            if b == 0:
                ax.set_title("cell {}".format(ne + 1), fontsize=12, color=c, pad=20)

            if k == 0 and b == 3:
                ax.set_ylabel("Fano factor", fontsize=10, labelpad=5)
            ax.scatter(
                avg_models[b][ne, ::skip] / binnings[b] * 1000,
                ff_models[b][ne, ::skip],
                marker=".",
                alpha=0.3,
            )
            ax.set_xlim(0)
            # ax.set_ylim(0, 3)
            if b == 0:
                if k == 0:
                    ax.set_yticks([0.8, 1.0])
                else:
                    ax.set_yticks([0.6, 1.0])

            xlims = ax.get_xlim()
            xx = np.linspace(xlims[0], xlims[1])
            ax.plot(xx, np.ones_like(xx), "k", alpha=0.5)
        b += 1

    X, Y = Xs[3], Ys[3]
    fig.text(0.425 + X, 0.725 + Y, "firing rate (Hz)", fontsize=10, ha="center")

    X = 0.0
    Y = 0.0
    spec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        wspace=0.2,
        hspace=0.2,
        left=0.83 + X,
        right=1.0 + X,
        bottom=0.55 + Y,
        top=0.725 + Y,
    )
    ax = fig.add_subplot(spec[0, 0])
    lFF = np.log(np.array([ff_models[b].mean(-1) for b in np.arange(5)]))

    xx = np.arange(5)[:, None].repeat(len(pick_neuron) - region_edge, axis=-1)
    xxrnd = np.random.rand(*xx.shape) * 0.2 - 0.1
    ax.scatter(
        xx + xxrnd + 0.2, lFF[:, region_edge:], marker=".", s=4, c=antcol, label="ANT"
    )
    xx = np.arange(5)[:, None].repeat(region_edge, axis=-1)
    xxrnd = np.random.rand(*xx.shape) * 0.2 - 0.1
    ax.scatter(
        xx + xxrnd - 0.2, lFF[:, :region_edge], marker=".", s=4, c=poscol, label="PoS"
    )

    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(binnings)
    ax.set_xlabel("bin size (ms)", fontsize=10, labelpad=5)
    ax.set_ylabel("log average FF", fontsize=10, labelpad=1)

    xlims = ax.get_xlim()
    xx = np.linspace(xlims[0], xlims[1])
    ax.plot(xx, np.zeros_like(xx), "k", alpha=0.5)
    ax.set_xlim(xlims)
    lgnd = ax.legend(handletextpad=0.0, bbox_to_anchor=(0.5, 0.3))
    lgnd.legendHandles[0]._sizes = [50]
    lgnd.legendHandles[1]._sizes = [50]


def ATIs(fig, RG_dict):
    # HD - AHV, ATIs
    R_min = 0.999  # minimum correlation from circular-linear correlation
    X = 0.0
    Y = -0.02
    widths = np.ones(len(show_neuron))
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        height_ratios=heights,
        wspace=0.3,
        left=0.0 + X,
        right=0.2 + X,
        bottom=0.15 + Y,
        top=0.3 + Y,
    )

    for k, ne in enumerate(show_neuron):
        ax = fig.add_subplot(spec[0, k])
        if ne < region_edge:
            c = poscol
        else:
            c = antcol

        fig.text(
            0.05 + X + 0.1 * k,
            0.35 + Y,
            "cell {}".format(ne + 1),
            fontsize=12,
            color=c,
            ha="center",
        )

        rate = field_hdw[ne] / tbin
        ax.set_title("{:.1f} Hz".format(rate.max()), fontsize=10, pad=-5)
        im = utils.plot.visualize_field(
            (fig, ax), rate.T, grid_shape_hdw, cbar=False, aspect="auto"
        )
        utils.plot.decorate_ax(ax, spines=[True, True, True, True])

        if k == 0:
            ax.set_yticks(grid_shape_hdw[1])

    fig.text(-0.06 + X, Y + 0.225, "AHV (rad/s)", rotation=90, fontsize=10, va="center")

    widths = [1]
    heights = [1, 1, 2]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        left=0.29 + X,
        right=0.39 + X,
        bottom=0.2 + Y,
        top=0.3 + Y,
    )

    valid = res_var[:region_edge] < -R_min
    at = ATI[:region_edge][valid] * 1000
    valid = res_var[region_edge:] < -R_min
    at2 = ATI[region_edge:][valid] * 1000

    ax = fig.add_subplot(spec[-1, 0])
    randnbs = np.random.rand(*at.shape) * 0.4 - 0.2
    ax.scatter(at, 0 * np.ones_like(at) + randnbs, marker=".", s=4, c=poscol)
    randnbs = np.random.rand(*at2.shape) * 0.4 - 0.2
    ax.scatter(at2, 1 * np.ones_like(at2) + randnbs, marker=".", s=4, c=antcol)
    ax.set_xlabel("ATI (ms)", labelpad=5, fontsize=10)
    ax.set_ylim(-1, 2)
    ax.set_yticks([])

    bins = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], BINS)
    ax = fig.add_subplot(spec[1, 0])
    ax.hist(at, bins=bins, alpha=0.5, density=True, color=poscol)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)

    ax = fig.add_subplot(spec[0, 0])
    ax.hist(at2, bins=bins, alpha=0.5, density=True, color=antcol)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)


def population_drift(fig, RG_dict):
    X = 0.0
    Y = -0.1
    widths = np.ones(len(show_neuron))
    heights = [1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        wspace=0.3,
        left=0.0 + X,
        right=0.2 + X,
        bottom=-0.05 + Y,
        top=0.1 + Y,
    )

    for k, ne in enumerate(show_neuron):
        ax = fig.add_subplot(spec[k])

        rate = field_hdt[ne] / tbin
        ax.set_title("{:.1f} Hz".format(rate.max()), fontsize=10, pad=-5)
        im = utils.plot.visualize_field(
            (fig, ax), rate.T, grid_shape_hdt, cbar=False, aspect="auto"
        )
        utils.plot.decorate_ax(ax, spines=[True, True, True, True])

        if k == 0:
            ax.set_xticks([0, 2 * np.pi])
            ax.set_xticklabels([r"$0$", r"$2\pi$"])
            ax.set_yticks([grid_shape_hdt[1][0], grid_shape_hdt[1][1] // 60 * 60])
            ax.set_yticklabels([0, 38])

        rm = rate.max()

    fig.text(-0.06 + X, 0.025 + Y, "time (min)", rotation=90, fontsize=10, va="center")
    fig.text(0.1 + X, -0.15 + Y, "head direction", fontsize=10, ha="center")

    widths = [1]
    heights = [1, 1, 2]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        left=0.29 + X,
        right=0.39 + X,
        bottom=0.0 + Y,
        top=0.1 + Y,
    )

    d = drift[:region_edge][res_var_drift[:region_edge] < -R_min] / np.pi * 180.0 * 3600
    d2 = (
        drift[region_edge:][res_var_drift[region_edge:] < -R_min] / np.pi * 180.0 * 3600
    )

    ax = fig.add_subplot(spec[-1, 0])
    randnbs = np.random.rand(*d.shape) * 0.4 - 0.2
    ax.scatter(d, 0 * np.ones_like(d) + randnbs, marker=".", s=4, c=poscol)
    randnbs = np.random.rand(*d2.shape) * 0.4 - 0.2
    ax.scatter(d2, 1 * np.ones_like(d2) + randnbs, marker=".", s=4, c=antcol)
    ax.set_xlabel(r"drift ($^\circ$/hr)", labelpad=5, fontsize=10)
    ax.set_ylim(-1, 2)
    ax.set_yticks([])

    bins = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], BINS)
    ax = fig.add_subplot(spec[1, 0])
    ax.hist(d, bins=bins, alpha=0.5, density=True, color=poscol)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)

    ax = fig.add_subplot(spec[0, 0])
    ax.hist(d2, bins=bins, alpha=0.5, density=True, color=antcol)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["left"].set_visible(False)

    # colorbars
    X = 0.0
    Y = -0.01
    cspec = fig.add_gridspec(
        ncols=1,
        nrows=1,
        width_ratios=[1],
        height_ratios=[1],
        left=0.215 + X,
        right=0.22 + X,
        bottom=-0.05 + Y,
        top=0.2 + Y,
    )
    ax = fig.add_subplot(cspec[0, 0])
    # ax.set_title('     max', fontsize=10, pad=1)
    utils.plot.add_colorbar(
        (fig, ax),
        im,
        ticktitle="firing rate",
        ticks=[0, rm],
        ticklabels=["0", "max"],
        cbar_pad=0,
        cbar_fontsize=10,
        cbar_format=None,
        cbar_ori="vertical",
    )


def latent_variables_scores(fig, LVM_dict):
    cvs = RMS_cv.shape[0]
    fact = 10**3

    X = 0.0
    Y = -0.15
    order = [1, 2, 0]
    widths = [1]
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        hspace=0.3,
        left=0.55 + X,
        right=0.625 + X,
        bottom=0.05 + Y,
        top=0.45 + Y,
    )

    c_ = LVM_cv_ll.transpose(1, 0, 2).mean(-1)[order, :] / fact
    rel_c = c_ - c_[0:1, :]

    ax = fig.add_subplot(spec[0, 0])
    ax.errorbar(
        np.arange(rel_c.shape[0])[1:],
        rel_c.mean(-1)[1:],
        linestyle="",
        marker="+",
        markersize=4,
        capsize=3,
        yerr=rel_c.std(-1, ddof=1)[1:] / np.sqrt(rel_c.shape[-1]),
        c="k",
    )
    ax.plot(np.linspace(-eps, Ncases + eps, 2), np.zeros(2), "gray")
    ax.plot(
        np.arange(rel_c.shape[0])[:, None].repeat(rel_c.shape[1], axis=1),
        rel_c,
        color="gray",
        marker=".",
        markersize=4,
        alpha=0.5,
    )

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels([])

    ax.set_ylabel(r"$\Delta$cvLL ($10^3$)", labelpad=2, fontsize=10)
    ax.set_xlim(-0.5, 0.5 + cvs - 1)

    ax = fig.add_subplot(spec[1, 0])
    ax.set_xlim(-eps, Ncases + eps)
    cvtrials = RMS_cv.shape[1]
    yerr = RMS_cv.std(1, ddof=1) / np.sqrt(cvtrials)
    ax.bar(
        np.arange(cvs),
        RMS_cv.mean(1)[order],
        yerr=yerr[order],
        capsize=3,
        color=[0.5, 0.5, 0.5],
        width=0.5,
    )

    ax.set_ylim(0)
    ax.set_xticks(np.arange(cvs))
    ax.set_xlim(-0.5, 0.5 + cvs - 1)
    ax.set_xticklabels(["Poisson", "hNB", "Universal"], rotation=90)
    ax.set_ylabel("RMSE", labelpad=5, fontsize=10)


def latent_posterior(fig, LVM_dict):
    fig.text(0.865 + X, 0.47 + Y, "Universal", fontsize=12, ha="center")
    widths = [1, 0.5]
    heights = [1, 1]
    spec = fig.add_gridspec(
        ncols=len(widths),
        nrows=len(heights),
        width_ratios=widths,
        height_ratios=heights,
        hspace=0.6,
        wspace=0.6,
        left=0.73 + X,
        right=1.0 + X,
        bottom=0.0 + Y,
        top=0.45 + Y,
    )

    # lat_t, RMS_cv, LVM_cv_ll, drifts_lv
    ax = fig.add_subplot(spec[0, 0])

    T = 300
    T_start = 2500

    ax.set_xlim([0, tbin * T])
    ax.set_xticks([])
    ax.set_xlabel("time", fontsize=10, labelpad=5)
    ax.set_ylabel(r"$z$", fontsize=10, labelpad=0)
    ax.set_ylim([0, 2 * np.pi])
    ax.set_yticks([0, 2 * np.pi])
    ax.set_yticklabels([r"$0$", r"$2\pi$"])

    # ax.set_title(r'posterior $q_{\varphi}(z)$', fontsize=12, pad=7)
    d_m = drifts_lv[0].mean(-1) / np.pi * 180 * 3600
    d_s = drifts_lv[0].std(-1) / np.pi * 180 * 3600 / np.sqrt(drifts_lv.shape[-1] - 1)
    ax.text(
        tbin * T * 1.1,
        -1.0,
        "drift:\n " + r"${:.1f}\pm{:.1f} ^\circ$/hr".format(d_m, d_s),
        color="gray",
    )
    utils.plot.plot_circ_posterior(
        ax,
        tbin * np.arange(T),
        rcov_lvm[0][T_start : T_start + T] % (2 * np.pi),
        None,
        col="k",
        linewidth=1.0,
        step=1,
        l="truth",
    )

    utils.plot.plot_circ_posterior(
        ax,
        tbin * np.arange(T),
        lat_t[T_start : T_start + T],
        lat_t_std[T_start : T_start + T],
        col="tab:blue",
        linewidth=0.7,
        step=1,
        alpha=0.3,
        line_alpha=0.5,
        l_std="var. post.",
    )

    leg = ax.legend(bbox_to_anchor=(1.05, 1.2), handlelength=0.8)
    for l in leg.get_lines()[1:]:
        l.set_linewidth(3)


def latent_delay(fig, LVM_dict):
    ax = fig.add_subplot(spec[1, 0])
    shift_times = 0.1 * (np.arange(delay_RMS.shape[0]) - delay_RMS.shape[0] // 2)
    _arr = delay_RMS.mean(-1)

    m = _arr
    s = _arr.std(-1) / np.sqrt(_arr.shape[-1] - 1)
    (line,) = ax.plot(shift_times, m, marker=".")
    ax.fill_between(shift_times, m - s, m + s, color=line.get_color(), alpha=0.5)
    ax.set_xlim([shift_times[0], shift_times[-1]])
    ax.set_xlabel("behaviour shift (s)", fontsize=10, labelpad=5)
    ax.set_ylabel("RMSE", fontsize=10, labelpad=5)

    # scatter comparison
    ax = fig.add_subplot(spec[1, 1])
    ax.set_aspect(1)
    ax.scatter(rcov_lvm[0], lat_t, marker=".", alpha=0.3)
    ax.set_xlim([0, 2 * np.pi])
    ax.set_xticks([0, 2 * np.pi])
    ax.set_xticklabels([r"$0$", r"$2\pi$"])
    ax.set_ylim([0, 2 * np.pi])
    ax.set_yticks([0, 2 * np.pi])
    ax.set_yticklabels([r"$0$", r"$2\pi$"])
    ax.set_ylabel(r"$z$", labelpad=-4, fontsize=10)
    ax.set_xlabel("head direction", labelpad=0, fontsize=10)


def main():
    save_dir = "../output/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.style.use(["paper.mplstyle"])

    # load
    RG_results = pickle.load(open(save_dir + "th1_RG_results.p", "rb"))
    LVM_results = pickle.load(open(save_dir + "th1_LVM_results.p", "rb"))

    # plot
    fig = plt.figure(figsize=(8, 4))
    fig.text(-0.08, 1.02, "A", transform=fig.transFigure, size=15, fontweight="bold")
    fig.text(0.29, 1.02, "B", transform=fig.transFigure, size=15, fontweight="bold")
    fig.text(-0.08, 0.35, "C", transform=fig.transFigure, size=15, fontweight="bold")
    fig.text(0.475, 0.35, "D", transform=fig.transFigure, size=15, fontweight="bold")

    poscol = "forestgreen"
    antcol = "orange"

    show_neuron = [11, 26]

    ### regression ###
    regressors(fig, RG_results)
    bin_sizes(fig, RG_results)

    ATIs(fig, RG_results)
    population_drift(fig, RG_results)

    ### latent variable models ###
    latent_variables_scores(fig, LVM_results)
    latent_posterior(fig, LVM_results)
    latent_delay(fig, LVM_results)

    plt.savefig(save_dir + "plot_hdc_add.pdf")


if __name__ == "__main__":
    main()
