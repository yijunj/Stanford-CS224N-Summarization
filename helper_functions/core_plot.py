import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import string

def apply_sublabels(axs, invert_color_inds=[], location = [5,-5], size=10, extra_titles = [], weight='bold'):
   if(len(extra_titles)==0):
       for n, ax in enumerate(axs):
           if np.in1d(n, invert_color_inds):
              color='w'
           else:
              color='k'
           ax.annotate(' '+string.ascii_lowercase[n]+')', bbox=dict(boxstyle="round", fc="w", edgecolor = 'white'), \
                       xy=(0,1), xytext=(location[0], location[1]), xycoords='axes fraction', textcoords='offset points', color=color, size=size,\
                       weight=weight, horizontalalignment='left', verticalalignment='top')
   else:
       for n, ax in enumerate(axs):
           if np.in1d(n, invert_color_inds):
              color='w'
           else:
              color='k'
           ax.annotate(' '+string.ascii_lowercase[n]+') '+extra_titles[n], bbox=dict(boxstyle="round", fc="w", edgecolor = 'white'), \
                       xy=(0,1), xytext=(location[0], location[1]), xycoords='axes fraction', textcoords='offset points', color=color, size=size,\
                       weight=weight, horizontalalignment='left', verticalalignment='top')


def core_formatter(ax, label_dict, spec_dict, color_dict, tick_param_dict, sty = 'default'):
    '''
    to account for the fact that the actual plot submitted by a person could be anything, we
    have this, which only performs formatting on an existing plot

    :param ax: axis object from subplot
    :return:
    '''

    ax.set_xlabel(label_dict['xlabel'], fontsize = spec_dict['label_size'])
    ax.set_ylabel(label_dict['ylabel'], fontsize = spec_dict['label_size']);
    ax.set_title(label_dict['title'], fontsize = spec_dict['title_size'])
    #post-active font size formatting

    ##mpl update only works in cython or the console...
    mpl.rcParams.update({'font.size': spec_dict['font']});
    mpl.rcParams.update({'axes.titlesize': spec_dict['title_size']})
    mpl.rcParams.update({'axes.labelsize': spec_dict['label_size']})
    mpl.rcParams.update({'xtick.labelsize': spec_dict['xtick']})
    mpl.rcParams.update({'ytick.labelsize': spec_dict['ytick']})
    mpl.rcParams.update({'legend.fontsize': spec_dict['legend']})
    mpl.rcParams.update({'figure.titlesize': spec_dict['title']})

    ax.tick_params(direction='out', length=tick_param_dict['tick_length'], width=tick_param_dict['tick_width'], \
                   colors=color_dict['border'])

    # color formatting
    ax.spines['bottom'].set_color(color_dict['border'])
    ax.spines['bottom'].set_linewidth(tick_param_dict['line_border']);

    ax.spines['top'].set_color(color_dict['border'])
    ax.spines['top'].set_linewidth(tick_param_dict['line_border']);

    ax.spines['left'].set_color(color_dict['border'])
    ax.spines['left'].set_linewidth(tick_param_dict['line_border']);

    ax.spines['right'].set_color(color_dict['border'])
    ax.spines['right'].set_linewidth(tick_param_dict['line_border']);

    #sets the ticks to the same color as the line border but keeps the labels black
    ax.tick_params(color=color_dict['border'], colors='black')

    # geometry formatting
    mpl.rcParams.update({'axes.linewidth': tick_param_dict['line_border']});

    ax.tick_params(direction='out', length=tick_param_dict['tick_length'], width=tick_param_dict['tick_length'],\
                    colors=color_dict['border'])


    ## miscellaneous formatting

def core_formatter_list(ax_list, label_dict, spec_dict, color_dict, tick_param_dict, sty = 'default'):
    '''
        if you want to format multiple plots the same way without the hassle of using core_formatter

    :param ax_list: list of  axis object from subplot
    :return:
    '''
    for ax in ax_list:
        ax.set_xlabel(label_dict['xlabel'], fontsize = spec_dict['label_size'])
        ax.set_ylabel(label_dict['ylabel'], fontsize = spec_dict['label_size']);
        ax.set_title(label_dict['title'], fontsize = spec_dict['title_size'])
        #post-active font size formatting

        ##mpl update only works in cython or the console...
        mpl.rcParams.update({'font.size': spec_dict['font']});
        mpl.rcParams.update({'axes.titlesize': spec_dict['title_size']})
        mpl.rcParams.update({'axes.labelsize': spec_dict['label_size']})
        mpl.rcParams.update({'xtick.labelsize': spec_dict['xtick']})
        mpl.rcParams.update({'ytick.labelsize': spec_dict['ytick']})
        mpl.rcParams.update({'legend.fontsize': spec_dict['legend']})
        mpl.rcParams.update({'figure.titlesize': spec_dict['title']})

        ax.tick_params(direction='out', length=tick_param_dict['tick_length'], width=tick_param_dict['tick_width'], \
                       colors=color_dict['border'])

        # color formatting
        ax.spines['bottom'].set_color(color_dict['border'])
        ax.spines['bottom'].set_linewidth(tick_param_dict['line_border']);

        ax.spines['top'].set_color(color_dict['border'])
        ax.spines['top'].set_linewidth(tick_param_dict['line_border']);

        ax.spines['left'].set_color(color_dict['border'])
        ax.spines['left'].set_linewidth(tick_param_dict['line_border']);

        ax.spines['right'].set_color(color_dict['border'])
        ax.spines['right'].set_linewidth(tick_param_dict['line_border']);

        #sets the ticks to the same color as the line border but keeps the labels black
        ax.tick_params(color=color_dict['border'], colors='black')

        # geometry formatting
        mpl.rcParams.update({'axes.linewidth': tick_param_dict['line_border']});

        ax.tick_params(direction='out', length=tick_param_dict['tick_length'], width=tick_param_dict['tick_length'],\
                        colors=color_dict['border'])


        ## miscellaneous formatting
