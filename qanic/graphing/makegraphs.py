# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import h5py

def load_h5py(filename):
    """
    Loads Simulation data from h5py summary file.
    """
    inputstr = ''
    with h5py.File(filename, 'r') as f:
        for key, value in f['Inputs'].attrs.items():
            inputstr += (str(key) + ': ' + str(value) + '\n')
        # these are inputs or "coordinates"
        Hsizes = np.array(f['Inputs/Hsizes'])
        Tvals = np.array(f['Inputs/Tvals'])
        svals = np.array(f['Inputs/svals'])
        discs = np.array(f['Inputs/discs'])
        # these are outputs i.e. actual data
        f_probs = np.array(f['Outputs/f_probs'])
        r_probs = np.array(f['Outputs/r_probs'])
        bfrem_probs = np.array(f['Outputs/bfrem_probs'])
        bfrem_psizes = np.array(f['Outputs/bfrem_psizes'])
        avgfrem_probs = np.array(f['Outputs/avgfrem_probs'])
        p_btf = np.array(f['Outputs/p_btf'])
        p_btr = np.array(f['Outputs/p_btr'])

    df = pd.DataFrame({'Hsizes': Hsizes, 'Tvals': Tvals, 'svals': svals,
                       'discs': discs, 'f_probs': f_probs,
                       'r_probs': r_probs, 'bfrem_probs': bfrem_probs,
                       'bfrem_psizes': bfrem_psizes,
                       'avgfrem_probs': avgfrem_probs,
                       'p_btf': p_btf, 'p_btr': p_btr})

    df.inputstr = inputstr

    return df

def all_graph_plot(df, indvar='Hsizes', figname=None):
    """
    Invokes prob_comp_plot, bt_comp_plot, and best_HR_plot with x-axis
    over indvar assuming that other variables are kept constant.
    """
    fig, axes = plt.subplots(1, 3, figsize = (16, 16))
    prob_comp_plot(df, indvar, None, ax=axes[0])
    bt_comp_plot(df, indvar, None, ax=axes[1])
    best_HR_plot(df, indvar, None, ax=axes[2])

    if figname is not None:
        fig.savefig(figname, dpi=500, transparent=False,
                    bbox_inches='tight')

    return fig
    

def prob_comp_plot(df, indvar='Hsizes', figname=None, ax=None):
    """
    Compares probability of sampling the ground-state of forward,
    reverse, and FREM annealing.
    """
    # extract and clean up the data
    complist = ['f_probs', 'r_probs', 'bfrem_probs', 'avgfrem_probs']
    data = df.melt(id_vars=[indvar], value_vars=complist,
                    var_name='Anneal Type', value_name='Probability')
    
    # plot the data
    if ax is None:
        fig, ax = plt.subplots()
    g = sns.catplot(x=indvar, y='Probability', hue='Anneal Type',
                data=data, kind='bar', legend_out=True, ax=ax)
    ax.set(ylim=(0, 1))
    
    # change the labels
    new_labels = ['F', 'R', 'Best FREM', 'Avg FREM']
    for t, l in zip(ax.legend().get_texts(), new_labels): t.set_text(l)
    # add probability label above bars
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        ax.annotate('{:.0}'.format(height), (p.get_x()+.5*width,
                    p.get_y() + height + 0.01), ha = 'center')

    # save figure if figname provided
    if figname is not None:
        fig.savefig(figname, dpi=500, transparent=False,
                    bbox_inches='tight')

def bt_comp_plot(df, indvar='Hsizes', figname=None, ax=None):
    """
    Plots the fraction of FREM runs over partitions in a sim that are
    better than (bt) forward and reverse annealing.
    """
    # extract and clean up data
    data = df.melt(id_vars=[indvar], value_vars=['p_btf', 'p_btr'],
                    var_name='Anneal Type',
                   value_name='Probability FREM Better GS')

    # create the plot
    if ax is None:
        fig, ax = plt.subplots()
    sns.catplot(x=indvar, y='Probability FREM Better GS',
                   hue='Anneal Type', data=data, kind='bar',
                   legend_out=True, ax=ax)
    ax.set(ylim=(0, 1))
    
    # relabel legend
    new_labels = ['F', 'R']
    for t, l in zip(ax.legend().get_texts(), new_labels): t.set_text(l)
    # add numbers above bars
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        ax.annotate('{:0}'.format(height), (p.get_x()+.5*width,
                    p.get_y() + height + 0.01), ha = 'center')

    if figname is not None:
        fig.savefig(figname, dpi=500, transparent=False,
                    bbox_inches='tight')

def best_HR_plot(df, indvar='Hsizes', figname=None, ax=None):
    """
    Of those partitions tried in a simulation, this plots the size of HR
    that produced the best result.
    """
    # extract and clean up the data
    data = df.melt(id_vars=[indvar], value_vars=['bfrem_psizes'],
                   var_name='Anneal Type', value_name='Best HR Size')

    # make the plot
    if ax is None:
        fig, ax = plt.subplots()
    sns.catplot(x=indvar, y='Best HR Size', hue='Anneal Type',
                   data=data, kind='bar', legend_out=True, ax=ax)
    ax.set(ylim=(0, max(df[indvar])))

    # change the labels
    new_labels = ['FREM']
    for t, l in zip(ax.legend().get_texts(), new_labels): t.set_text(l)
    # add numbers above bars
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        ax.annotate('{:d}'.format(int(height)), (p.get_x()+.5*width,
                    p.get_y() + height + 0.01), ha = 'center') 

    if figname is not None:
        fig.savefig(figname, dpi=500, transparent=False,
                    bbox_inches='tight')

def get_state_plot(data, figsize=(12, 8), filename=None, title='Distribution of Final States'):
    ncount = len(data)

    plt.figure(figsize=figsize)
    ax = sns.countplot(x='states', data=data)
    plt.title(title)
    plt.xlabel('State')

    # Make twin axis
    ax2 = ax.twinx()

    # Switch so count axis is on right, frequency on left
    ax2.yaxis.tick_left()
    ax.yaxis.tick_right()

    # Also switch the labels over
    ax.yaxis.set_label_position('right')
    ax2.yaxis.set_label_position('left')

    ax2.set_ylabel('Frequency [%]')

    for p in ax.patches:
        x = p.get_bbox().get_points()[:, 0]
        y = p.get_bbox().get_points()[1, 1]
        ax.annotate('{:.1f}%'.format(100. * y / ncount), (x.mean(), y),
                    ha='center', va='bottom')  # set the alignment of the text

    # Use a LinearLocator to ensure the correct number of ticks
    ax.yaxis.set_major_locator(ticker.LinearLocator(11))

    # Fix the frequency range to 0-100
    ax2.set_ylim(0, 100)
    ax.set_ylim(0, ncount)

    # And use a MultipleLocator to ensure a tick spacing of 10
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

    # Need to turn the grid on ax2 off, otherwise the gridlines end up on top of the bars
    # ax2.grid(None)

    if filename:
        plt.savefig(filename, dpi=300)

    return plt
