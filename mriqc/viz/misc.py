#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
#
# @Author: oesteban
# @Date:   2016-01-05 11:32:01
# @Email:  code@oscaresteban.es
# @Last modified by:   oesteban
""" Helper functions for the figures in the paper """
from __future__ import print_function, division, absolute_import, unicode_literals
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.font_manager import FontProperties
from ..classifier.data import read_dataset, zscore_dataset

def plot_batches(fulldata, out_file=None, excl_columns=None):
    fulldata = fulldata.select_dtypes([np.number]).copy()
    if excl_columns:
        cols = fulldata.columns.ravel().tolist()
        cols = [col for col in cols if col not in excl_columns]
        fulldata = fulldata[cols]

    colmin = fulldata.min()
    fulldata = (fulldata - colmin)
    colmax = fulldata.max()
    fulldata = fulldata / colmax

    fig, ax = plt.subplots(figsize=(20, 10))
    ax.imshow(fulldata.values, cmap=plt.cm.viridis, interpolation='nearest', aspect='auto')

    plt.xticks(range(fulldata.shape[1]), fulldata.columns.ravel().tolist(), rotation='vertical')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(False)
    if out_file is not None:
        fig.savefig(out_file)
    return fig

def plot_roc_curve(true_y, prob_y, out_file=None):
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(true_y, prob_y)

    fig = plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlim([-0.025, 1.025])
    plt.ylim([-0.025, 1.025])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('RoC Curve')
    if out_file is not None:
        fig.savefig(out_file)
    return fig

def fill_matrix(matrix, width, value='n/a'):
    if matrix.shape[0] < width:
        nas = np.chararray((1, 3), itemsize=len(value))
        nas[:] = value
        matrix = np.vstack(tuple([matrix] + [nas] * (width - matrix.shape[0])))
    return matrix

def plot_raters(dataframe, site=None, ax=None, width=101,
                raters=None):
    if raters is None:
        raters = ['rater_1', 'rater_2', 'rater_3']

    if site is not None:
        dataframe = dataframe.loc[dataframe.site == site]

    dataframe = dataframe[raters].sort_values(by=raters, ascending=True)

    matrix = dataframe.as_matrix()
    nsamples = len(dataframe)

    if matrix.shape[0] < width:
        matrix = fill_matrix(matrix, width)

    nblocks = 1
    if matrix.shape[0] > width:
        matrices = []
        nblocks = (matrix.shape[0] // width) + 1

        nas = np.chararray((width, 1), itemsize=3)
        nas[:] = 'n/a'
        for i in range(nblocks):
            if i > 0:
                matrices.append(nas)
            matrices.append(matrix[i * width:(i + 1) * width, ...])

        matrices[-1] = fill_matrix(matrices[-1], width)
        matrix = np.hstack(tuple(matrices))

    palette = {'1.0': 'limegreen', '0.0': 'gold', '-1.0': 'tomato', 'n/a': 'w'}

    ax = ax if ax is not None else plt.gca()

    # ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    size = 0.40

    nrows = ((nsamples - 1) // width) + 1
    xlims = (-14.0, width)
    ylims = (-0.2, nrows * 3.2 + (nrows - 1))

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    for (x, y), w in np.ndenumerate(matrix):
        if str(w) not in list(palette.keys()):
            w = 'n/a'
        color = palette[str(w)]
        rect = plt.Circle([x + 0.5, y + 0.5], size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)


    # text_x = ((nsamples - 1) % width) + 6.5
    text_x = -8.5
    for i, rname in enumerate(raters):
        good = 100 * sum(dataframe[rname] == 1.0) / nsamples
        bad = 100 * sum(dataframe[rname] == -1.0) / nsamples

        text_y = 1.5 * i + (nrows - 1) * 2.0
        ax.text(text_x, text_y, '%2.0f%%' % good,
            color='limegreen', weight=1000, size=16,
            horizontalalignment='right',
            verticalalignment='center',
            transform=ax.transData)
        ax.text(text_x + 3.50, text_y, '%2.0f%%' % max((0.0, 100 - good - bad)),
            color='dimgray', weight=1000, size=16,
            horizontalalignment='right',
            verticalalignment='center',
            transform=ax.transData)
        ax.text(text_x + 7.0, text_y, '%2.0f%%' % bad,
            color='tomato', weight=1000, size=16,
            horizontalalignment='right',
            verticalalignment='center',
            transform=ax.transData)

    # ax.autoscale_view()
    ax.invert_yaxis()
    plt.grid(False)

    # Remove and redefine spines
    for side in ["top", "right", "bottom"]:
        # Toggle the spine objects
        ax.spines[side].set_color('none')
        ax.spines[side].set_visible(False)

    ax.spines["left"].set_linewidth(1.5)
    ax.spines["left"].set_color('dimgray')
    # ax.spines["left"].set_position(('data', xlims[0]))

    ax.set_yticks([0.5 * (ylims[0] + ylims[1])])
    ax.tick_params(axis='y', which='major', pad=15)
    ax.set_yticklabels([site])

    ticks_font = FontProperties(
        family='FreeSans', style='normal', size=20,
        weight='normal', stretch='normal')
    for label in ax.get_yticklabels():
        label.set_fontproperties(ticks_font)

    return ax

def raters_variability_plot(y_path, figsize=(22, 22),
                            width=101, out_file=None):
    rater_types = {'rater_1': float, 'rater_2': float, 'rater_3': float}
    mdata = pd.read_csv(y_path, index_col=False,
                        dtype=rater_types)

    # Swap raters 2 and 3
    cols = mdata.columns.ravel().tolist()
    i, j = cols.index('rater_2'), cols.index('rater_3')
    cols[j], cols[i] = cols[i], cols[j]
    mdata.columns = cols

    sites_list = sorted(set(mdata.site.values.ravel().tolist()))
    sites_len = []
    for site in sites_list:
        sites_len.append(len(mdata.loc[mdata.site == site]))

    sites_len, sites_list = zip(*sorted(zip(sites_len, sites_list)))

    blocks = [(slen - 1) // width + 1 for slen in sites_len]
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(len(sites_list), 1, width_ratios=[1], height_ratios=blocks, hspace=0.05)

    for s, gsel in zip(sites_list, gs):
        ax = plt.subplot(gsel)
        plot_raters(mdata, site=s, ax=ax, width=width)


    # ax.add_line(Line2D([0.0, width], [8.0, 8.0], color='k'))
    # ax.annotate(
    #     '%d images' % width, xy=(0.5 * width, 8), xycoords='data',
    #     xytext=(0.5 * width, 9), fontsize=20, ha='center', va='top',
    #     arrowprops=dict(arrowstyle='-[,widthB=1.0,lengthB=0.2', lw=1.0)
    # )

    # ax.annotate('QC Prevalences', xy=(0.1, -0.15), xytext=(0.5, -0.1), xycoords='axes fraction',
    #         fontsize=20, ha='center', va='top',
    #         arrowprops=dict(arrowstyle='-[, widthB=3.0, lengthB=0.2', lw=1.0))

    newax = plt.axes([0.6, 0.65, .25, .16])
    newax.grid(False)
    newax.set_xticklabels([])
    newax.set_xticks([])
    newax.set_yticklabels([])
    newax.set_yticks([])

    nsamples = len(mdata)
    for i, rater in enumerate(sorted(rater_types.keys())):
        good = 100 * sum(mdata[rater] == 1.0) / nsamples
        bad = 100 * sum(mdata[rater] == -1.0) / nsamples

        text_x = .92
        text_y = .5 - 0.17 * i
        newax.text(text_x - .36, text_y, '%2.1f%%' % good,
            color='limegreen', weight=1000, size=25,
            horizontalalignment='right',
            verticalalignment='center',
            transform=newax.transAxes)
        newax.text(text_x - .18, text_y, '%2.1f%%' % max((0.0, 100 - good - bad)),
            color='dimgray', weight=1000, size=25,
            horizontalalignment='right',
            verticalalignment='center',
            transform=newax.transAxes)
        newax.text(text_x, text_y, '%2.1f%%' % bad,
            color='tomato', weight=1000, size=25,
            horizontalalignment='right',
            verticalalignment='center',
            transform=newax.transAxes)

        newax.text(1 - text_x, text_y, 'Rater %d' % (i + 1),
            color='k', size=25,
            horizontalalignment='left',
            verticalalignment='center',
            transform=newax.transAxes)

    newax.text(0.5, 0.95, 'Imbalance of ratings',
        color='k', size=25,
        horizontalalignment='center',
        verticalalignment='top',
        transform=newax.transAxes)
    newax.text(0.5, 0.85, '(ABIDE, aggregated)',
        color='k', size=25,
        horizontalalignment='center',
        verticalalignment='top',
        transform=newax.transAxes)
    if out_file is None:
        out_file = 'raters.svg'

    fname, ext = op.splitext(out_file)
    if ext[1:] not in ['pdf', 'svg', 'png']:
        ext = '.svg'
        out_file = fname + '.svg'

    fig.savefig(op.abspath(out_file), format=ext[1:],
                bbox_inches='tight', pad_inches=0, dpi=300)
    return fig

def plot_abide_stripplots(inputs, figsize=(15, 2), out_file=None,
                          rating_label='rater_1', dpi=100):
    import seaborn as sn
    sn.set(style="whitegrid")

    mdata = []
    pp_cols = []

    for X, Y, sitename in inputs:
        sitedata, cols = read_dataset(X, Y, rate_label=rating_label,
                                      binarize=False, site_name=sitename)
        sitedata['database'] = [sitename] * len(sitedata)

        if sitename == 'DS030':
            sitedata['site'] = [sitename] * len(sitedata)

        mdata.append(sitedata)
        pp_cols.append(cols)

    mdata = pd.concat(mdata)
    pp_cols = pp_cols[0]

    for col in mdata.columns.ravel().tolist():
        if col.startswith('rater_') and col != rating_label:
            del mdata[col]

    mdata = mdata.loc[mdata[rating_label].notnull()]

    for col in ['size_x', 'size_y', 'size_z', 'spacing_x', 'spacing_y', 'spacing_z']:
        del mdata[col]
        try:
            pp_cols.remove(col)
        except ValueError:
            pass

    zscored = zscore_dataset(mdata, excl_columns=[rating_label])

    sites = list(set(mdata.site.values.ravel()))
    nsites = len(sites)

    # palette = ['dodgerblue', 'darkorange']
    palette = ['limegreen', 'tomato']
    if len(set(mdata[[rating_label]].values.ravel().tolist())) == 3:
        palette = ['tomato', 'gold', 'limegreen']
    # pp_cols = pp_cols[:5]
    nrows = len(pp_cols)

    fig = plt.figure(figsize=(figsize[0], figsize[1] * nrows))
    # ncols = 2 * (nsites - 1) + 2
    gs = GridSpec(nrows, 4, wspace=0.02)
    gs.set_width_ratios([nsites, len(inputs), len(inputs), nsites])

    for i, colname in enumerate(pp_cols):
        ax_nzs = plt.subplot(gs[i, 0])
        axg_nzs = plt.subplot(gs[i, 1])
        axg_zsc = plt.subplot(gs[i, 2])
        ax_zsc = plt.subplot(gs[i, 3])

        # plots
        sn.stripplot(x='site', y=colname, data=mdata, hue=rating_label, jitter=0.18, alpha=.6,
                     split=True, palette=palette, ax=ax_nzs)
        sn.stripplot(x='site', y=colname, data=zscored, hue=rating_label, jitter=0.18, alpha=.6,
                     split=True, palette=palette, ax=ax_zsc)

        sn.stripplot(x='database', y=colname, data=mdata, hue=rating_label, jitter=0.18, alpha=.6,
                     split=True, palette=palette, ax=axg_nzs)
        sn.stripplot(x='database', y=colname, data=zscored, hue=rating_label, jitter=0.18,
                     alpha=.6, split=True, palette=palette, ax=axg_zsc)

        ax_nzs.legend_.remove()
        ax_zsc.legend_.remove()
        axg_nzs.legend_.remove()
        axg_zsc.legend_.remove()

        if i == nrows - 1:
            ax_nzs.set_xticklabels(ax_nzs.xaxis.get_majorticklabels(), rotation=80)
            ax_zsc.set_xticklabels(ax_zsc.xaxis.get_majorticklabels(), rotation=80)
            axg_nzs.set_xticklabels(axg_nzs.xaxis.get_majorticklabels(), rotation=80)
            axg_zsc.set_xticklabels(axg_zsc.xaxis.get_majorticklabels(), rotation=80)
        else:
            ax_nzs.set_xticklabels([])
            ax_zsc.set_xticklabels([])
            axg_nzs.set_xticklabels([])
            axg_zsc.set_xticklabels([])

        ax_nzs.set_xlabel('', visible=False)
        ax_zsc.set_xlabel('', visible=False)
        ax_zsc.set_ylabel('', visible=False)
        ax_zsc.yaxis.tick_right()

        axg_nzs.set_yticklabels([])
        axg_nzs.set_xlabel('', visible=False)
        axg_nzs.set_ylabel('', visible=False)
        axg_zsc.set_yticklabels([])
        axg_zsc.set_xlabel('', visible=False)
        axg_zsc.set_ylabel('', visible=False)


        for yt in ax_nzs.yaxis.get_major_ticks()[1:-1]:
            yt.label1.set_visible(False)

        for yt in axg_nzs.yaxis.get_major_ticks()[1:-1]:
            yt.label1.set_visible(False)

        for yt in zip(ax_zsc.yaxis.get_majorticklabels(), axg_zsc.yaxis.get_majorticklabels()):
            yt[0].set_visible(False)
            yt[1].set_visible(False)

    if out_file is None:
        out_file = 'stripplot.svg'

    fname, ext = op.splitext(out_file)
    if ext[1:] not in ['pdf', 'svg', 'png']:
        ext = '.svg'
        out_file = fname + '.svg'

    fig.savefig(op.abspath(out_file), format=ext[1:],
                bbox_inches='tight', pad_inches=0, dpi=dpi)
    return fig

def plot_corrmat(in_csv, out_file=None):
    import seaborn as sn
    sn.set(style="whitegrid")

    dataframe = pd.read_csv(in_csv, index_col=False, na_values='n/a', na_filter=False)
    colnames = dataframe.columns.ravel().tolist()

    for col in ['subject_id', 'site', 'modality']:
        try:
            colnames.remove(col)
        except ValueError:
            pass

    # Correlation matrix
    corr = dataframe[colnames].corr()
    corr = corr.dropna((0,1), 'all')

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Generate a custom diverging colormap
    cmap = sn.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    corrplot = sn.clustermap(corr, cmap=cmap, center=0., method='average', square=True, linewidths=.5)
    plt.setp(corrplot.ax_heatmap.yaxis.get_ticklabels(), rotation='horizontal')
    # , mask=mask, square=True, linewidths=.5, cbar_kws={"shrink": .5})

    if out_file is None:
        out_file = 'corr_matrix.svg'

    fname, ext = op.splitext(out_file)
    if ext[1:] not in ['pdf', 'svg', 'png']:
        ext = '.svg'
        out_file = fname + '.svg'

    corrplot.savefig(out_file, format=ext[1:], bbox_inches='tight', pad_inches=0, dpi=100)
    return corrplot


def plot_histograms(X, Y, rating_label='rater_1', out_file=None):
    import seaborn as sn
    sn.set(style="whitegrid")

    mdata, pp_cols = read_dataset(X, Y, rate_label=rating_label)
    mdata['rater'] = mdata[[rating_label]].values.ravel()

    for col in mdata.columns.ravel().tolist():
        if col.startswith('rater_'):
            del mdata[col]

    mdata = mdata.loc[mdata.rater.notnull()]
    zscored = zscore_dataset(
            mdata, excl_columns=['rater', 'size_x', 'size_y', 'size_z',
                                 'spacing_x', 'spacing_y', 'spacing_z'])

    colnames = [col for col in sorted(pp_cols)
                if not (col.startswith('spacing') or col.startswith('summary') or col.startswith('size'))]

    nrows = len(colnames)
    # palette = ['dodgerblue', 'darkorange']

    fig = plt.figure(figsize=(18, 2 * nrows))
    gs = GridSpec(nrows, 2, hspace=0.2)

    for i, col in enumerate(sorted(colnames)):
        ax_nzs = plt.subplot(gs[i, 0])
        ax_zsd = plt.subplot(gs[i, 1])

        sn.distplot(mdata.loc[(mdata.rater == 0), col], norm_hist=False,
                    label='Accept', ax=ax_nzs, color='dodgerblue')
        sn.distplot(mdata.loc[(mdata.rater == 1), col], norm_hist=False,
                    label='Reject', ax=ax_nzs, color='darkorange')
        ax_nzs.legend()

        sn.distplot(zscored.loc[(zscored.rater == 0), col], norm_hist=False,
                    label='Accept', ax=ax_zsd, color='dodgerblue')
        sn.distplot(zscored.loc[(zscored.rater == 1), col], norm_hist=False,
                    label='Reject', ax=ax_zsd, color='darkorange')

        alldata = mdata[[col]].values.ravel().tolist()
        minv = np.percentile(alldata, 0.2)
        maxv = np.percentile(alldata, 99.8)
        ax_nzs.set_xlim([minv, maxv])

        alldata = zscored[[col]].values.ravel().tolist()
        minv = np.percentile(alldata, 0.2)
        maxv = np.percentile(alldata, 99.8)
        ax_zsd.set_xlim([minv, maxv])

    if out_file is None:
        out_file = 'histograms.svg'

    fname, ext = op.splitext(out_file)
    if ext[1:] not in ['pdf', 'svg', 'png']:
        ext = '.svg'
        out_file = fname + '.svg'

    fig.savefig(out_file, format=ext[1:], bbox_inches='tight', pad_inches=0, dpi=100)
    return fig


def plot_artifact(image_path, figsize=(20, 20), vmax=None, cut_coords=None, display_mode='ortho',
                  size=None):
    import nilearn.plotting as nplt

    fig = plt.figure(figsize=figsize)
    nplt_disp = nplt.plot_anat(
        image_path, display_mode=display_mode, cut_coords=cut_coords,
        vmax=vmax, figure=fig, annotate=False)

    if size is None:
        size = figsize[0] * 6


    bg_color = 'k'
    fg_color = 'w'
    ax = fig.gca()
    ax.text(
        .1, .95, 'L',
        transform=ax.transAxes,
        horizontalalignment='left',
        verticalalignment='top',
        size=size,
        bbox=dict(boxstyle="square,pad=0", ec=bg_color, fc=bg_color, alpha=1),
        color=fg_color)

    ax.text(
        .9, .95, 'R',
        transform=ax.transAxes,
        horizontalalignment='right',
        verticalalignment='top',
        size=size,
        bbox=dict(boxstyle="square,pad=0", ec=bg_color, fc=bg_color),
        color=fg_color)

    return nplt_disp, ax


def figure1_a(image_path, display_mode='y', vmax=300, cut_coords=None, figsize=(20, 20)):
    import matplotlib.patches as patches

    if cut_coords is None:
        cut_coords = [15]

    disp, ax = plot_artifact(image_path, display_mode=display_mode, vmax=vmax, cut_coords=cut_coords,
                             figsize=figsize)

    ax.add_patch(
        patches.Arrow(
            0.2,            # x
            0.2,            # y
            0.1,            # dx
            0.6,            # dy
            width=.25,
            color='tomato',
            transform=ax.transAxes
        )
    )

    ax.add_patch(
        patches.Arrow(
            0.8,            # x
            0.2,            # y
            -0.1,            # dx
            0.6,            # dy
            width=.25,
            color='tomato',
            transform=ax.transAxes
        )
    )
    return disp


def figure1_b(image_path, display_mode='z', vmax=400, cut_coords=None, figsize=(20, 20)):
    import matplotlib.patches as patches

    if cut_coords is None:
        cut_coords = [-24]

    disp, ax = plot_artifact(image_path, display_mode=display_mode, vmax=vmax, cut_coords=cut_coords,
                             figsize=figsize)

    ax.add_patch(
        patches.Arrow(
            0.02,            # x
            0.55,            # y
            0.1,            # dx
            0.0,            # dy
            width=.10,
            color='tomato',
            transform=ax.transAxes
        )
    )
    ax.add_patch(
        patches.Arrow(
            0.98,            # x
            0.55,            # y
            -0.1,            # dx
            0.0,            # dy
            width=.10,
            color='tomato',
            transform=ax.transAxes
        )
    )

    ax.add_patch(
        patches.Arrow(
            0.02,            # x
            0.80,            # y
            0.1,            # dx
            0.0,            # dy
            width=.10,
            color='limegreen',
            transform=ax.transAxes
        )
    )
    ax.add_patch(
        patches.Arrow(
            0.98,            # x
            0.80,            # y
            -0.1,            # dx
            0.0,            # dy
            width=.10,
            color='limegreen',
            transform=ax.transAxes
        )
    )
    return disp

def figure1(artifact1, artifact2, out_file):
    from .svg import svg2str, combine_svg
    combine_svg([
        svg2str(figure1_b(artifact2)),
        svg2str(figure1_a(artifact1))
        ],
        axis='vertical').save(out_file)

