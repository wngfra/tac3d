import base64

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns
import nengo_spa as spa

from ssp.pointers import normalize


class Plotter(object):
    """Maintains and interactive plot of a heat map produced from an SSP."""

    def __init__(self, figsize=(7, 7)):

        # initialize plotting for updates during runtime
        plt.figure(figsize=figsize)
        self.ax = plt.subplot(111)

        # ensure plot is interactive so that it can update
        plt.ion()
        plt.show()

        # counter to determine whether to create or update plot
        self.count = 0

    def show(self, ssp_map):
        """Plot or update heatmap using current SSP representation."""
        cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

        if self.count == 0:
            # build initial plots based on first map, utility values
            self.im = self.ax.imshow(
                ssp_map.heatmap_scores,
                interpolation="none",
                extent=(0, ssp_map.x_len, 0, ssp_map.y_len),
                cmap=cmap,
            )
        else:
            # update plots with new data if they are already built
            self.im.set_data(ssp_map.heatmap_scores)

        plt.draw()
        plt.pause(0.0001)
        self.count += 1


def lineplot_animation(lines, figsize, titles=None, interval=100):
    """Animate transitions between a series of lines on a plot"""
    n_plots = len(lines)  # number of linesets plot
    n_steps = len(lines[0])  # length of each lineset array

    fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=figsize)

    images = []
    for step in range(n_steps):
        frame = []
        for i, ax in enumerate(axes.flatten()):
            line = ax.plot(lines[i][step], animated=True)[0]
            frame.append(line)

        images.append(frame)

    for ax in axes:
        ax.set_xticks([])
        ax.set_xlabel("Axis Position")
        ax.set_ylabel("Effective Encoding Width")

    if titles is not None:
        for i, ax in enumerate(axes):
            ax.set_title(titles[i])

    ani = animation.ArtistAnimation(fig, images, interval=interval, blit=True)
    plt.close()

    return ani


def heatmap_animation(
    sims,
    figsize,
    titles=None,
    quiver=None,
    ticks=False,
    text=False,
    interval=100,
    cmap=sns.diverging_palette(220, 20, sep=20, as_cmap=True),
    vmin=-1,
    vmax=1,
):
    """Create heatmaps of SSP decodings over time with optional quiver field"""
    n_plots = len(sims)  # number of heatmap arrays to plot
    n_steps = len(sims[0])  # length of each heatmap array

    fig, axes = plt.subplots(nrows=1, ncols=n_plots, figsize=figsize)
    if n_plots == 1:
        axes = np.asarray([axes])

    # build the quiver artists once to avoid recomputing things over and over
    if quiver:
        X, Y = quiver[0], quiver[1]
        U = np.flip(quiver[2], axis=0)
        V = np.flip(quiver[3], axis=0)

        qs = []
        for i, ax in enumerate(axes.flatten()):
            q = ax.quiver(X, Y, U, V)
            qs.append(q)

    images = []
    for step in range(n_steps):
        frame = []
        for i, ax in enumerate(axes.flatten()):
            heatmap = ax.imshow(
                sims[i][step], vmin=vmin, vmax=vmax, cmap=cmap, animated=True
            )
            frame.append(heatmap)
            if quiver:
                frame.append(qs[i])
            if text:
                frame.append(plt.text(1, 1, "Frame %d" % step))

        images.append(frame)

    if not ticks:
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

    if titles is not None:
        for i, ax in enumerate(axes):
            ax.set_title(titles[i])

    ani = animation.ArtistAnimation(fig, images, interval=interval, blit=True)
    plt.close()

    return ani


def eigenvector_animation(eigvals, coeffs, sims, interval=None):
    """Create animation of SSP as linear combination of eigenvectors"""
    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 8))
    ax1, ax2, ax3 = axes

    images = []
    for step in range(len(sims)):
        frame = []

        eigval_line = ax1.plot(
            np.arange(len(eigvals)), np.abs(eigvals), color="b", animated=True
        )

        coeff_bars = ax2.bar(
            np.arange(len(eigvals)), np.abs(coeffs[step]), color="b", animated=True
        )

        sim = ax3.imshow(sims[step], vmin=-1, vmax=1, cmap=cmap, animated=True)
        frame = eigval_line + coeff_bars.patches + [sim]

        images.append(frame)

    ax1.set_xlabel("Eigenvector")
    ax1.set_ylabel("Eigenvalue")
    ax2.set_xlabel("Eigenvector")
    ax2.set_ylabel("Coefficienct")
    ax3.set_xticks([])
    ax3.set_yticks([])

    ani = animation.ArtistAnimation(fig, images, interval=interval, blit=True)
    plt.close()

    return ani


def create_quiver(model, ssp_map, obj_name):
    """Create data for quiver plot to overlay with heatmap animation"""
    X, Y = np.meshgrid(np.arange(len(ssp_map.xs)), np.arange(len(ssp_map.ys)))
    U = np.zeros_like(X, dtype=np.float64) + 0.1
    V = np.zeros_like(Y, dtype=np.float64) + 0.1

    for x_ind, x in enumerate(ssp_map.xs):
        for y_ind, y in enumerate(ssp_map.ys):
            inp = ssp_map.encode_point(x, y, name=obj_name).v[np.newaxis, :]
            out = spa.SemanticPointer(np.squeeze(model(inp)))

            coords = ssp_map.decode_ssp_coords(
                out, obj_name, top_only=False, average=True
            )

            if coords is None:
                arrow = np.array([0, 0])
            else:
                arrow = normalize(np.array([coords[0] - x, coords[1] - y]))

            U[y_ind, x_ind] = arrow[0]
            V[y_ind, x_ind] = arrow[1]

    # size the arrows to look nice on plots
    U = U / np.sqrt(U ** 2 + V ** 2) / 2
    V = V / np.sqrt(U ** 2 + V ** 2) / 2

    return [X, Y, U, V]


def create_gif(ani, fname=".temp.gif"):
    """Create a gif file to render in HTML in a Jupyter notebook"""
    ani.save(fname, writer="imagemagick")
    gif = open(fname, "rb").read()
    gif_base64 = base64.b64encode(gif).decode()

    return gif_base64


def plot_error(log, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    data = log.to_dataframe()

    xs = data.loc[(data["axis"] == "X")]
    ys = data.loc[(data["axis"] == "Y")]

    x = sns.lineplot(
        x="t", y="val", hue="Object", style="Prediction", data=xs, ax=axes[0]
    )
    x.set_title("X Coordinates")
    y = sns.lineplot(
        x="t", y="val", hue="Object", style="Prediction", data=ys, ax=axes[1]
    )
    y.set_title("Y Coordinates")

    fig.suptitle(title)
    plt.show()
