from itertools import zip_longest

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

SMALL_SIZE = 13
MEDIUM_SIZE = 16
BIGGER_SIZE = 30

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize

plt.rcParams['figure.figsize'] = [9, 7]




def plot_with_error(
    x, y, y_error, label=None, color=None, alpha_fill=0.3, ax=None, base_kwargs=None
):
    """
    Plot with transparent error.
    :param x: List of abscissas.
    :param y: List of ordinates.
    :param y_error: Amplitude of the error.
    :param label: Label of the curve (legend).
    :param color: Color of the plot.
    :param alpha_fill: Transparency of the error (default 0.3).
    :param ax: Axe where to plot the curve. If None (default), creates a new one.
    """
    if base_kwargs is None:
        base_kwargs = {}
    x, y = np.array(x), np.array(y)
    ax = ax if ax is not None else plt.gca()
    if np.isscalar(y_error) or len(y_error) == len(y):
        y_min = y - y_error
        y_max = y + y_error
    elif len(y_error) == 2:
        y_min, y_max = y_error
    else:
        raise ValueError(
            f"y_error must be either a scalar, a list of the same length as y, "
            f"or a tuple containing the min and the max errors. Found {y_error}"
        )
    (base_line,) = ax.plot(x, y, label=label, color=color, **base_kwargs)
    if color is None:
        color = base_line.get_color()
    ax.fill_between(x, y_max, y_min, color=color, alpha=alpha_fill)


def _find_axes_to_move(ys, n1, n2):
    shape = np.array(np.shape(ys))
    desired = np.array([n1, n2])

    shape = np.broadcast_to(shape, (len(desired), len(shape)))

    return np.where((shape.T == desired).T)[1]


def plot_curves(
    x,
    ys,
    labels=None,
    ax=None,
    alpha_fill: float = 0.3,
    move_axes=None,
    only_mean: bool = False,
):
    if ax is None:
        ax = plt.gca()

    if labels is None:
        labels = []

    if move_axes is None:
        move_axes = _find_axes_to_move(ys, len(labels), len(x))
    ys = np.moveaxis(ys, move_axes, (i for i in range(len(move_axes))))

    for y, label in zip_longest(ys, labels):
        if np.ndim(y) == 2:
            if only_mean:
                ax.plot(x, np.mean(y, axis=-1), label=label)
            else:
                plot_with_error(
                    x,
                    np.mean(y, axis=-1),
                    np.std(y, axis=-1),
                    label=label,
                    alpha_fill=alpha_fill,
                    ax=ax,
                )
        else:
            ax.plot(x, y, label=label)

    return ax


def full_extent(ax, pad=0.0):
    """
    Get the full extent of an axes, including axes labels, tick labels, and titles.
    :param ax:
    :param pad:
    :return:
    """
    # For text objects, we need to draw the figure first, otherwise the extents are undefined.
    ax.figure.canvas.draw()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    items += [ax, ax.title]
    bbox = Bbox.union([item.get_window_extent() for item in items])
    return bbox.expanded(pad + 1, pad + 1)


def set_size(w, h, ax=None):
    """
    :param w:
    :param h:
    :param ax:
    :return:
    """
    if not ax:
        ax = plt.gca()
    left = ax.figure.subplotpars.left
    right = ax.figure.subplotpars.right
    top = ax.figure.subplotpars.top
    bottom = ax.figure.subplotpars.bottom
    fig_w = float(w) / (right - left)
    fig_h = float(h) / (top - bottom)
    ax.figure.set_size_inches(fig_w, fig_h)
