import numpy as np
import matplotlib.pyplot as plt

import os
    
import sys
sys.path.append("..")
from neuroprob import utils



def logo(fig):
    
    # scatters and box
    widths = [1]
    heights = [1]
    spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, 
                            height_ratios=heights, top=1.0, bottom=0.0, left=0.0, right=1.0)

    ax = fig.add_subplot(spec[0, 0])
    utils.plots.decorate_ax(ax, spines=[False, False, False, False])

    gcol = np.array([44, 160, 44])/255
    ax.text(0.01, 0.55, r'latent $z$', fontsize=12, rotation=90, color=gcol, va='center')
    weight_map = utils.plots.make_cmap([gcol*0.1 + 0.9*np.ones(3), gcol], 'weight_map')
    utils.plots.cmap_arrow(ax, (0.06, 0.15), (0.06, 0.9), cmap=weight_map, lw=2, 
                          head_size=8, head_length=1., head_width=1.)

    rcol = np.array([214, 39, 40])/255
    ax.text(0.55, 0.015, r'observed $x$', fontsize=12, color=rcol, ha='center')
    weight_map = utils.plots.make_cmap([rcol*0.1 + 0.9*np.ones(3), rcol], 'weight_map')
    utils.plots.cmap_arrow(ax, (0.15, 0.06), (0.9, 0.06), cmap=weight_map, lw=2, 
                          head_size=8, head_length=1., head_width=1.)


    ax.scatter(0.5*np.ones(3), np.arange(3)*0.03+0.9, marker='.', color='gray', s=10)
    ax.scatter(np.arange(3)*0.03+0.9, 0.5*np.ones(3), marker='.', color='gray', s=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)



    # histograms
    X = 0.25
    Y = 0.25
    xxs = [2, 30, 20, 40, 99, 40, 15, 30, 15]
    for k in range(3):
        for l in range(3):
            widths = [1]
            heights = [1]

            spec = fig.add_gridspec(ncols=len(widths), nrows=len(heights), width_ratios=widths, hspace=0.0, 
                                    height_ratios=heights, top=0.38+Y*l, bottom=0.15+Y*l, 
                                    left=0.15+X*k, right=0.38+X*k)


            ax = fig.add_subplot(spec[0, 0])

            xx = xxs[k+3*l]
            for pp in range(K):
                fac = 1.0
                rfac = 0.1 + 0.9*(k+1)/3
                gfac = 0.1 + 0.9*(l+1)/3
                rcx = rcol*rfac + (1-rfac)*np.ones(3)
                gcy = gcol*gfac + (1-gfac)*np.ones(3)
                col = (rcx+gcy)/2.

                XX_ = np.linspace(pp-0.5, pp+0.5, 2)
                YY_ = np.ones(2)*np.array(pP_m)[pp, xx]
                YY_l = np.ones(2)*np.array(pP_l)[pp, xx]
                YY_u = np.ones(2)*np.array(pP_u)[pp, xx]
                line, = ax.plot(XX_, YY_, c=col, label=l, alpha=fac)
                ax.fill_between(XX_, YY_l, 
                    YY_u, color=line.get_color(), alpha=fac*0.3)


            ax.set_xlim([-0.5, K-1+.5])
            ax.set_ylim(0)
            ax.set_yticks([])

            if k == 0 and l == 0:
                ax.set_xticks([0, K-1])
                ax.set_xticklabels([0, 'K'])
                ax.set_ylabel('probability', fontsize=12, labelpad=4)
                ax.set_xlabel('# spikes', labelpad=-9, fontsize=12)
            else:
                ax.set_xticks([])

                
                
def main():
    if not os.path.exists('./output'):
        os.makedirs('./output')
    plt.style.use(['paper.mplstyle'])
    
    # plot
    fig = plt.figure(figsize=(4,4))

    logo(fig)
    
    plt.savefig('output/logo.png', dpi=100)
    
    
    
if __name__ == "__main__":
    main()