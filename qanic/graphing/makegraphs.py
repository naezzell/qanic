# imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

def get_frem_data(directory, xaxis, yaxis):
    """
    Looks through all files in 'directory' and creates two lists containing
    the 'xaxis' and 'yaxis' data.

    Inputs
    --------------------------------------------------
    directory: string -- directory containg frem_sim files
    cqubit: Bool -- True if contains critical qubit; false otherwise
    """
    # get list of all files in directory
    files = next(os.walk(directory))[2]
    filepaths = [os.path.join(directory, fn) for fn in files]

    # open each file and extract the data
    for f in files:
        with open(f) as summfile:
            break

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
