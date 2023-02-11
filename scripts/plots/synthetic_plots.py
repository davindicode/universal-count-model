import numpy as np
import pickle

import torch

import matplotlib.pyplot as plt

import os
    
import sys
sys.path.append("..")
from neuroprob import utils



def model_icons(fig):
    delX = 0.5
    Xoff = -0.1
    Yoff = 0.4
    for l in range(2):
        widths = [1]
        heights = [1]
        spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                                height_ratios=heights, 
                                left=0.0+delX*l, right=0.3+delX*l, bottom=0.1, top=1.1)

        ax = fig.add_subplot(spec[0, 0])

        pgm = daft.PGM(shape=(2, 2), node_unit=0.7)
        init_figax(pgm, fig, ax)

        pgm.add_node("y", r"$y_n$", .7+Xoff, 0.6+Yoff, observed=True, fontsize=12)
        if l == 0:
            pgm.add_node("x", r"$X$", .7+Xoff, 1.2+Yoff, observed=True, fontsize=12)
            pgm.add_edge("x", "y")
        elif l == 1:
            pgm.add_node("z", r"$Z$", .7+Xoff, 1.2+Yoff, observed=False, fontsize=12)
            pgm.add_edge("z", "y")

        pgm.add_plate([0.3+Xoff, 0.2+Yoff, .8, .75], label=r"$N$", position='bottom right', shift=0.1)

        render(pgm)
        
        
        
def regression_scores(fig):
    widths = [1]
    heights = np.ones(2)
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, hspace=0.4, 
                            height_ratios=heights, top=0.2, bottom=-.4, left=0.07, right=0.17)

    eps = 0.4
    Ncases = T_KS_rg.shape[1]-1

    # RG
    ax = fig.add_subplot(spec[0, 0])
    fact = 10**2
    ax.set_xlim(-eps, Ncases+eps)
    scores = RG_cv_ll
    score_err = scores.std(-1)/np.sqrt(scores.shape[-1]-1)
    rel_score = (scores-scores[0:1, :])/fact
    #ax.plot(np.linspace(-eps, Ncases+eps, 2), np.zeros(2), 'gray', alpha=.5)
    ax.plot(np.arange(scores.shape[0])[:, None].repeat(scores.shape[1], axis=1), rel_score, 
            color='gray', marker='.', markersize=4, alpha=.5)
    ax.errorbar(np.arange(scores.shape[0])[1:], rel_score.mean(-1)[1:], linestyle='', marker='+', markersize=4, capsize=3, 
                yerr=rel_score.std(-1, ddof=1)[1:]/np.sqrt(rel_score.shape[-1]), c='k')
    #ax.set_ylim()

    ax.set_xticks(np.arange(RG_cv_ll.shape[0]))
    ax.set_xticklabels([])
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_ylabel(r'$\Delta$cvLL ($10^2$)', fontsize=10)


    # KS
    ax = fig.add_subplot(spec[1, 0])
    ax.set_xlim(-eps, Ncases+eps)
    for en, r in enumerate(T_KS_rg.mean(0)):
        ax.scatter(en*np.ones(len(r))+np.random.rand(len(r))*eps/2-eps/4, r, color='gray', marker='+')

    xl, xu = ax.get_xlim()
    ax.fill_between(np.linspace(xl, xu, 2), 0, 
                    np.ones(2)*sign_KS, color='k', alpha=0.2)
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_ylabel(r'$T_{KS}$', fontsize=10)
    ax.set_xticks(np.arange(RG_cv_ll.shape[0]))
    ax.set_xticklabels(['Poisson', 'hNB', 'U (GP)', 'U (ANN)'], rotation=90)#, 'GT'])
    
    
    
def count_tuning(fig):
    cx = np.arange(max_count+1)
    plot_cnt = 11

    delx = 0.1
    fig.text(.34, 1.05, 'Universal (GP)', fontsize=12, ha='center')
    for en, n in enumerate([6, 16]):

        widths = [1]
        heights = [0.8, 1, 0.7, 1, 0.4, 1]
        spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, hspace=0.0, 
                                 height_ratios=heights, top=0.85, bottom=-0.45, left=.25+delx*en, right=.33+delx*en)


        ax = fig.add_subplot(spec[0, 0])
        fig.text(.29+delx*en, 0.91, 'neuron {}'.format(use_neuron[n]+1), fontsize=11, ha='center')

        c = ['g', 'b', 'r']
        for enn, hd_n in enumerate(hd):
            if enn == 1:
                continue

            l = 'truth' if enn == 0 else None
            ax.plot(cx[:plot_cnt], ref_prob[enn, n, :plot_cnt], '--', c=c[enn], label=l)
            #for pp in range(plot_cnt):
            #    XX = np.linspace(cx[pp]-0.5, cx[pp]+0.5, 2)
            #    YY = np.ones(2)*ref_prob[enn, n, pp]
            #    ax.plot(XX, YY, '--', c=c[enn], label=l)

            for pp in range(plot_cnt):
                l = 'fit' if (enn == 0 and pp == 0) else None
                XX = np.linspace(cx[pp]-0.5, cx[pp]+0.5, 2)
                YY = np.ones(2)*cmean[n, enn, pp]
                YY_l = np.ones(2)*clower[n, enn, pp]
                YY_u = np.ones(2)*cupper[n, enn, pp]
                line, = ax.plot(XX, YY, c=c[enn], label=l, alpha=0.3)
                ax.fill_between(XX, YY_l, 
                    YY_u, color=line.get_color(), alpha=0.3)

        if en == 1:
            leg = ax.legend(handlelength=1., bbox_to_anchor=(1.5, 1.3), loc="upper right")
            leg.legendHandles[0]._color = 'k'
            leg.legendHandles[1]._color = 'k'
            leg.get_lines()[1].set_linewidth(3)

        ax.set_xticklabels([])
        if en == 0:
            ax.set_ylabel('prob.', fontsize=10)
        ax.set_xlim([-0.5, plot_cnt-1+.5])
        ax.set_ylim(0)
        ax.set_yticks([])
        ax.set_xticks([])#np.arange(plot_cnt))
        #ax.set_xlabel('count')

        ax = fig.add_subplot(spec[1, 0])
        im = utils.plots.draw_2d((fig, ax), P_rg[n, :, :plot_cnt], origin='lower', cmap='gray_r', 
                                vmin=0, vmax=P_rg[n, :, :plot_cnt].max(), interp_method='nearest')

        # arrows
        if en == 0:
            c = ['g', 'b', 'r']
            for enn, hd_n in enumerate(hd):
                #ax.plot(cx[:plot_cnt], hd_n*np.ones(plot_cnt), c=c[en])
                if enn == 1:
                    continue
                ax.annotate(text='', xy=(0., hd_n,), zorder=1, color='tab:blue', va='center', 
                            xytext=(-3., hd_n), arrowprops=dict(arrowstyle='->', color=c[enn]))

        utils.plots.decorate_ax(ax, spines=[False, False, False, False])
        #ax.set_xlim([0, plot_cnt-1])
        ax.set_xticks(np.arange(plot_cnt))

        ax.set_yticklabels([])
        if en == 0:
            ax.set_ylabel(r'$x$', fontsize=10, labelpad=5)
            ax.set_xticklabels([0, '', '', '', '', 5, '', '', '', '', ''])
        else:
            ax.set_xticklabels(['', '', '', '', '', 5, '', '', '', '', 10])


        ax = fig.add_subplot(spec[3, 0])
        line, = ax.plot(covariates[0], avgmean[n, :], color='tab:blue')
        ax.fill_between(covariates[0], avglower[n, :], 
            avgupper[n, :], color=line.get_color(), alpha=0.3)
        ax.plot(covariates[0], grate[n, :]*tbin, 'k--')
        ax.set_ylim(0, 5.)
        if en == 0:
            ax.set_ylabel('mean', fontsize=10)
        else:
            ax.set_yticklabels([])
        ax.set_xlim([0, 2*np.pi])
        ax.set_xticklabels([])
        ax.set_xticks([0, 2*np.pi])


        ax = fig.add_subplot(spec[5, 0])
        line, = ax.plot(covariates[0], ffmean[n, :], color='tab:blue')
        ax.fill_between(covariates[0], fflower[n, :], 
            ffupper[n, :], color=line.get_color(), alpha=0.3)
        ax.set_ylim(0.4, 1.4)
        ax.plot(covariates[0], gFF[n, :], 'k--')
        ax.set_xticks([0, 2*np.pi])
        if en == 0:
            ax.set_ylabel('FF', fontsize=10, labelpad=0)
            ax.set_xticklabels([r'$0$', r'$2\pi$'])
        else:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        ax.set_xlim([0, 2*np.pi])

    fig.text(0.34, 0.25, 'count', ha='center', fontsize=10)
    fig.text(0.34, -0.675, r'head direction $x$', ha='center', fontsize=10)
    
    
    
    
def latent_variables(fig):
    widths = np.ones(1)
    heights = np.ones(1)
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                             height_ratios=heights, top=1.0, bottom=0.7, left=.75, right=0.9)

    ax = fig.add_subplot(spec[0, 0])

    T = 300
    T_start = 0

    ax.set_xlim([0, tbin*T])
    ax.set_xticks([])
    ax.set_xlabel('time', labelpad=5)

    ax.set_ylim([0, 2*np.pi])
    ax.set_yticks([0, 2*np.pi])
    ax.set_yticklabels([r'$0$', r'$2\pi$'])

    ax.set_title(r'posterior $q_{\varphi}(z)$', fontsize=12, pad=7)
    utils.plots.plot_circ_posterior(ax, tbin*np.arange(T), rhd_t[T_start:T_start+T] % (2*np.pi), None, col='k', 
                                   linewidth=1.0, step=3, l='truth')

    utils.plots.plot_circ_posterior(ax, tbin*np.arange(T), lat_t_[0][T_start:T_start+T], 
                                   lat_std_[0][T_start:T_start+T], col='tab:blue', 
                                   linewidth=.7, step=1, alpha=0.5, line_alpha=0.5, l='GP')#, l_std='var. post.')

    utils.plots.plot_circ_posterior(ax, tbin*np.arange(T), lat_t_[1][T_start:T_start+T], 
                                   lat_std_[1][T_start:T_start+T], col='tab:green', 
                                   linewidth=.7, step=1, alpha=0.5, line_alpha=0.5, l='ANN')#, l_std='var. post.')


    leg = ax.legend(bbox_to_anchor=(1.05, 1.2), handlelength=0.8)
    for l in leg.get_lines()[1:]:
        l.set_linewidth(3)
    leg.get_lines()[0].set_linestyle('--')



    widths = np.ones(2)
    heights = [1, 1, 1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, wspace=0.3, hspace=0.3, 
                             height_ratios=heights, top=0.4, bottom=-0.45, left=.75, right=0.95)

    delx = 0.4
    n = 6
    col_ = ['tab:blue', 'tab:green']
    for l in range(2):

        ax = fig.add_subplot(spec[0, l])
        if l == 0:
            ax.set_title('U (GP)', fontsize=11)
        else:
            ax.set_title('U (ANN)', fontsize=11)

        ax.set_aspect(1)
        ax.scatter(rhd_t[:lat_t_[l].shape[0]], lat_t_[l], marker='.', s=1, alpha=0.5, color=col_[l])
        ax.set_xticks([0, 2*np.pi])
        ax.set_xticklabels([])
        ax.set_yticks([0, 2*np.pi])
        if l > 0:
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(['0', r'$2\pi$'])
        if l == 0:
            ax.set_ylabel(r'$z$', fontsize=10)

        ax = fig.add_subplot(spec[1, l])
        lower, mean, upper = comp_avg[l]
        line, = ax.plot(covariates[0], mean[n], color=col_[l])
        ax.fill_between(covariates[0], lower[n], 
            upper[n], color=line.get_color(), alpha=0.3)
        ax.plot(covariates[0], grate[n, :]*tbin, 'k--')
        if l == 0:
            ax.set_ylabel('mean', fontsize=10)
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([0, 5.5])
        ax.set_xticklabels([])
        ax.set_xticks([0, 2*np.pi])
        if l > 0:
            ax.set_yticklabels([])


        ax = fig.add_subplot(spec[2, l])
        lower, mean, upper = comp_ff[l]
        line, = ax.plot(covariates[0], mean[n, :], color=col_[l])
        ax.fill_between(covariates[0], lower[n, :], 
            upper[n, :], color=line.get_color(), alpha=0.3)
        ax.plot(covariates[0], gFF[n, :], 'k--')
        if l == 0:
            ax.set_ylabel('FF', fontsize=10)
        ax.set_xticks([0, 2*np.pi])
        ax.set_xlim([0, 2*np.pi])
        ax.set_ylim([0.4, 2.3])
        if l > 0:
            ax.set_yticklabels([])
            ax.set_xticklabels([])
        else:
            ax.set_xticklabels([r'$0$', r'$2\pi$'])

    fig.text(0.85, -0.675, 'head direction (truth)', ha='center', fontsize=10)
    fig.text(0.97, -0.15, 'neuron {}'.format(n+1), va='center', ha='center', fontsize=11, rotation=90)
    
    
    
def LVM_scores(fig):
    widths = np.ones(1)
    heights = np.ones(2)
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, wspace=0.5, hspace=0.4, 
                             height_ratios=heights, top=.2, bottom=-0.4, left=.565, right=0.665)


    # LVM
    ax = fig.add_subplot(spec[0, 0])
    fact = 10**4
    ax.set_xlim(-eps, Ncases+eps)
    scores = np.transpose(LVM_cv_ll, (1, 0, 2)).mean(-1)#.reshape(LVM_cv_ll.shape[1], -1)
    scores_err = scores.std(-1)/np.sqrt(scores.shape[-1]-1)
    rel_score = (scores-scores[0:1, :])/fact*(len(use_neuron)/5)

    ax.plot(np.arange(scores.shape[0])[:, None].repeat(scores.shape[1], axis=1), rel_score, 
            color='gray', marker='.', markersize=4, alpha=.5)
    ax.errorbar(np.arange(scores.shape[0])[1:], rel_score.mean(-1)[1:], linestyle='', marker='+', markersize=4, capsize=3, 
                yerr=rel_score.std(-1, ddof=1)[1:]/np.sqrt(rel_score.shape[-1]), c='k')


    ax.set_xticks(np.arange(scores.shape[0]))
    ax.set_xticklabels([])
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_ylabel(r'$\Delta$cvLL ($10^4$)', fontsize=10, labelpad=3)


    # RMS
    ax = fig.add_subplot(spec[1, 0])
    ax.set_xlim(-eps, Ncases+eps)
    cvs = RMS_cv.shape[0]
    cvtrials = RMS_cv.shape[1]
    yerr = RMS_cv.std(1, ddof=1)/np.sqrt(cvtrials)
    ax.bar(np.arange(cvs), RMS_cv.mean(1), yerr=yerr, capsize=3, color=[0.5, 0.5, 0.5], width=0.5)

    ax.set_ylim(0)
    ax.set_xticks(np.arange(cvs))
    ax.set_xticklabels(['Poisson', 'hNB', 'U (GP)', 'U (ANN)'], rotation=90)
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_ylabel('RMSE', fontsize=10)
    
    
    
def noise_correlations(fig):
    names = ['Poisson', 'Universal (X)', 'Universal (X,Z)']
    delX = 0.22
    Yoff = -0.25
    Xoff = -0.075
    for l in range(3):
        widths = [1]
        heights = [1]
        spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                                height_ratios=heights, 
                                left=0.0+delX*l, right=0.3+delX*l, bottom=-1.6, top=-0.6)

        ax = fig.add_subplot(spec[0, 0])
        ax.text(1.3+Xoff + (1-l)*0.05, 2.5, names[l], fontsize=12, rotation=0, ha='center')

        pgm = daft.PGM(shape=(2, 2), node_unit=0.7)
        init_figax(pgm, fig, ax)

        pgm.add_node("y", r"$y_n$", .7+Xoff, 0.6+Yoff, observed=True, fontsize=12)
        if l == 0 or l == 1:
            pgm.add_node("x", r"$X$", .7+Xoff, 1.2+Yoff, observed=True, fontsize=12)
            pgm.add_edge("x", "y")
        elif l == 2:
            pgm.add_node("x", r"$X$", .45+Xoff, 1.2+Yoff, observed=True, fontsize=12)
            pgm.add_edge("x", "y")
            pgm.add_node("z", r"$Z$", .95+Xoff, 1.2+Yoff, observed=False, fontsize=12)
            pgm.add_edge("x", "y")
            pgm.add_edge("z", "y")

        pgm.add_plate([0.3+Xoff, 0.2+Yoff, .8, .75], label=r"$N$", position='bottom right', shift=0.1)

        render(pgm)
        
        
    Xoff = 0.03
    aa = [np.argsort(R[0])[-2]]
    delY = 0.5
    delX = 0.22
    J = 10 # jump skip
    for en, a in enumerate(aa):
        for l in range(3): # kcv=2

            widths = [4, 1]
            heights = [1, 4]
            spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                                    height_ratios=heights, left=0.05+delX*l+Xoff, right=0.16+delX*l+Xoff, 
                                    bottom=-2.15-delY*en, top=-1.75-delY*en)

            n, m = model_utils.ind_to_pair(a, len(use_neuron_ll))
            m_here = m+n+1
            L = 4.

            ax = fig.add_subplot(spec[1, 0])
            ax.scatter(Zz[l][n][::J], Zz[l][m_here][::J], marker='.', c='tab:blue', alpha=0.3)
            ax.set_aspect(1)
            ax.set_xlim(-L, L)
            ax.set_ylim(-L, L)
            utils.plots.decorate_ax(ax, spines=[False, False, False, False])
            if l == 0:
                ax.set_xlabel(r'$\xi_1$', labelpad=-1, fontsize=10)
                ax.set_ylabel(r'$\xi_2$', labelpad=-1, fontsize=10)


            ax = fig.add_subplot(spec[0, 0])
            Z = Zz[l][m_here]
            ax.hist(Z, bins=np.linspace(-L, L, 20), density=True, 
                    color='tab:blue')
            xx = np.linspace(-L, L, 100)
            yy = scstats.norm.pdf(xx)
            ax.plot(xx, yy, 'r')
            ax.set_xlim(-L, L)
            ax.set_xticks([])
            ax.set_yticks([])

            ax = fig.add_subplot(spec[1, 1])
            Z = Zz[l][n]
            ax.hist(Z, bins=np.linspace(-L, L, 20), density=True, orientation='horizontal', 
                    color='tab:blue')
            xx = np.linspace(-L, L, 100)
            yy = scstats.norm.pdf(xx)
            ax.plot(yy, xx, 'r')
            ax.set_ylim(-L, L)
            ax.set_xticks([])
            ax.set_yticks([])


            widths = [1]
            heights = [1]
            spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                                    height_ratios=heights, left=0.0+delX*l+Xoff, right=0.025+delX*l+Xoff, 
                                    bottom=-2.15-delY*en, top=-1.75-delY*en)
            L = 0.1
            r = T_DS_[0, l]
            ax = fig.add_subplot(spec[0, 0])
            ax.hist(r, density=True, bins=np.linspace(-L, L, 20), orientation='horizontal', 
                    color='gray')
            ax.set_xticks([])
            if l == 0:
                ax.set_ylabel(r'$T_{DS}$', fontsize=10, labelpad=-4)
            else:
                ax.set_yticks([])

            samples = len(Zz[0][0]) # number of quantiles
            std = np.sqrt(2/(samples-1))
            xx = np.linspace(-L, L, 100)
            yy = scstats.norm.pdf(xx/std)/std
            ax.plot(yy, xx, 'r')
            ax.set_ylim(-L, L)
            
            
    white = '#ffffff'
    lightgray = '#D3D3D3'
    black = '#000000'
    red = '#ff0000'
    blue = '#0000ff'
    weight_map = utils.plots.make_cmap([blue, white, red], 'weight_map')

    Xoff = 0.03
    datas = [R_mat_Xp, R_mat_X, R_mat_XZ]
    g = max(-np.stack(datas).min(), np.stack(datas).max())*1.0
    for en, r in enumerate(fisher_z[:3]):
        widths = [0.25, 1]
        heights = [1]
        spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                                height_ratios=heights, wspace=0.6, 
                                left=0.0+delX*en+Xoff, right=0.14+delX*en+Xoff, bottom=-2.7, top=-2.3)



        ax = fig.add_subplot(spec[0, 0])
        ax.hist(r, bins=np.linspace(-5, 5, 20), density=True, orientation='horizontal', 
                color='gray')
        ax.set_xticks([])
        if en == 0:
            ax.set_ylabel(r'Fisher $Z$', fontsize=10, labelpad=4)
        else:
            ax.set_yticks([])

        xx = np.linspace(-5, 5, 100)
        yy = scstats.norm.pdf(xx)
        ax.plot(yy, xx, 'r')
        ax.set_ylim(-5, 5)

        # show correlations
        ax = fig.add_subplot(spec[0, 1])

        data = datas[en]
        im = utils.plots.draw_2d((fig, ax), data[:, ::-1], origin='lower', cmap=weight_map, vmin=-g, vmax=g, aspect='equal')
        utils.plots.decorate_ax(ax, spines=[False, True, False, True])
        ax.yaxis.set_label_position("right")
        ax.set_xticks([])
        ax.set_yticks([])
        if en == 0:
            ax.set_xlabel(r'neuron $j$', fontsize=10, labelpad=3)
            ax.set_ylabel(r'neuron $i$', fontsize=10, labelpad=3)
            ax.text(-6., 12., r'$r_{ij}$ $(i<j)$', fontsize=12, rotation=45)



    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths,
                            height_ratios=heights,
                            left=0.588+Xoff, right=0.593+Xoff, bottom=-2.65, top=-2.35)
    ax = fig.add_subplot(spec[0, 0])
    ax.set_title(r'  $r_{ij}$', fontsize=12)
    utils.plots.add_colorbar((fig, ax), im, ticktitle='', ticks=[-0.1, 0, 0.1], ticklabels=[-0.1, 0, 0.1])#, 
                            #cbar_format=':.1f')

    # lines
    Yoff = 0.0
    
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, 
                            left=0.205, right=0.445, bottom=-2.7+Yoff, top=-1.0+Yoff)
    ax = fig.add_subplot(spec[0, 0])
    yy = np.linspace(0, 1., 10)
    ax.plot(1.*np.ones_like(yy), yy, 'k', linewidth=.9)
    ax.plot(2.*np.ones_like(yy), yy, 'k', linewidth=.9)
    ax.axis('off')

    
    
    
def latent_observed_tuning(fig):
    ### XZ tuning ###
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, left=0.775, right=0.95, 
                            bottom=-1.3, top=-0.95)

    ax = fig.add_subplot(spec[0, 0])
    fact = 10**3
    scores = cv_pll.mean(-1)#reshape(cv_pll.shape[0], -1)
    scores_err = scores.std(-1)/np.sqrt(scores.shape[-1]-1)
    rel_score = (scores-scores[0:1, :])/fact*(len(use_neuron)/5)
    #ax.plot(np.linspace(-eps, Ncases+eps, 2), np.zeros(2), 'gray', alpha=.5)
    ax.plot(np.arange(scores.shape[0])[:, None].repeat(scores.shape[1], axis=1), rel_score, 
            color='gray', marker='.', markersize=4, alpha=.5)
    ax.errorbar(np.arange(scores.shape[0])[1:], rel_score.mean(-1)[1:], linestyle='', marker='+', markersize=4, capsize=3, 
                yerr=rel_score.std(-1, ddof=1)[1:]/np.sqrt(rel_score.shape[-1]), c='k')

    xlims = ax.get_xlim()
    ax.plot(np.linspace(xlims[0], xlims[1], 2), np.zeros(2), 'gray')
    ax.set_xlim(xlims)

    ax.set_xticks(np.arange(scores.shape[0]))
    ax.set_xticklabels(['Poisson', 'U (X)', 'U (X,Z)'])
    #ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.set_ylabel(r'$\Delta$cvLL ($10^3$)', fontsize=10)



    fig.text(0.86, -1.55, 'Universal (X,Z)', fontsize=12, ha='center')
    widths = [1]
    heights = [1, 1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, left=0.74, right=0.85, hspace=0.3, 
                            bottom=-2.15, top=-1.6)

    ax = fig.add_subplot(spec[0, 0])
    T = 501
    ts = np.arange(T)*tbin_ll
    line, = ax.plot(ts, X_c[:T], alpha=0.5)
    ax.fill_between(ts, X_c[:T]-X_s[:T], 
        X_c[:T]+X_s[:T], color=line.get_color(), alpha=0.3, label='inferred')
    ax.plot(ts, ra_t[:T], 'k--', label='truth')
    ax.set_ylabel(r'$z$', labelpad=5, fontsize=10)
    ax.set_xlim([0, ts[-1]])
    ax.set_xticks([0, ts[-1]])
    ax.set_xticklabels([])
    ax.set_yticks([-1, 0, 1])
    ax.set_yticklabels([-1, '', 1])
    #ax.set_xlabel('{} s'.format(int(ts[-1])), labelpad=2, color='gray')
    leg = ax.legend(handlelength=1., bbox_to_anchor=(2.1, 1.2), loc='upper right')
    for l in leg.get_lines()[1:]:
        l.set_linewidth(3)
    #leg.get_lines()[0].set_linestyle('--')

    T = 501
    ts = np.arange(T)*tbin_ll
    ax = fig.add_subplot(spec[1, 0])
    utils.plots.plot_circ_posterior(ax, tbin*np.arange(T), rhd_t[:T] % (2*np.pi), None, col='k', 
                                   linewidth=.7, step=1)
    ax.set_ylim([0, 2*np.pi])
    ax.set_yticks([0, 2*np.pi])
    ax.set_yticklabels(['0', r'$2\pi$'])
    ax.set_ylabel(r'$x$', labelpad=0, fontsize=10)
    ax.set_xlim([0, ts[-1]])
    ax.set_xticks([0, ts[-1]])
    ax.set_xticklabels([])
    ax.set_xlabel('time ({} s)'.format(int(ts[-1])), labelpad=0, color='k', fontsize=10)


    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, left=0.91, right=0.98, 
                            bottom=-2.2, top=-1.7)

    ax = fig.add_subplot(spec[0, 0])
    ax.scatter(ra_t, X_c, marker='.', alpha=0.5)
    Ra = np.linspace(ra_t.min()-0.3, ra_t.max()+0.3, 2)
    ax.set_xlim(Ra[0], Ra[-1])
    ax.plot(Ra, Ra, 'k')
    ax.set_aspect('equal')
    ax.set_ylabel('inferred', fontsize=10, labelpad=3)
    ax.set_yticks([])
    ax.set_xlabel('truth', fontsize=10, labelpad=3)
    ax.set_xticks([])




    widths = [1, 1, 1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, left=0.74, right=0.97, 
                            bottom=-2.6, top=-2.3)

    for en, n in enumerate([30, 6, 15]):
        ax = fig.add_subplot(spec[0, en])

        line, = ax.plot(covariates_z[1], avgmeanz[n, :])
        ax.fill_between(covariates_z[1], avglowerz[n, :], 
            avgupperz[n, :], color=line.get_color(), alpha=0.3)
        ax.plot(covariates_z[1], gratez[n, :]*tbin, 'k--')
        if en == 0:
            ax.set_ylabel('mean', fontsize=10)

        ax.set_xticks([])
        ax.set_yticks([])
        if en == 1:
            ax.set_xlabel(r'$z$', fontsize=10)
            
            
            
def main():
    if not os.path.exists('./output'):
        os.makedirs('./output')
    plt.style.use(['paper.mplstyle'])
    
    # load
    datarun = pickle.load(open('./saves/hCMP_results.p', 'rb'))

    regression_hCMP = datarun['regression']
    dispersion_hCMP = datarun['dispersion']
    latent_dict_hCMP = datarun['latent']


    datarun = pickle.load(open('./saves/modIP_results.p', 'rb'))

    latent_dict_modIP = datarun['latent']
    correlations_modIP = datarun['correlations']

    # plot
    fig = plt.figure(figsize=(8, 2))
    fig.text(-0.04, 1.1, 'A', fontsize=15, fontweight='bold')
    fig.text(-0.04, -0.95, 'B', fontsize=15, fontweight='bold')
    #fig.text(0.0, -1.75, 'C', fontsize=15)


    rhd_t = rcov[0]
    ra_t = rcov_ll[1]

    ### regression ###
    model_icons(fig)
    regression_scores(fig)
    count_tuning(fig)

    # line
    Yoff = 0.0

    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, 
                            left=0.45, right=0.5, bottom=-0.7+Yoff, top=1.+Yoff)
    ax = fig.add_subplot(spec[0, 0])
    yy = np.linspace(0, 1., 10)
    ax.plot(1.*np.ones_like(yy), yy, 'k', linewidth=.9)
    ax.axis('off')


    ### LVM ###
    latent_variables(fig)
    LVM_scores(fig)

    ### noise correlations ###
    noise_correlations(fig)
    latent_observed_tuning(fig)

    plt.savefig('output/plot_synthetic.pdf')

    
    
    
if __name__ == "__main__":
    main()