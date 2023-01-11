import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt


def display_video(frames, framerate=30, figsize=None, dpi=100):
    try:
        height, width, _ = frames[0].shape
    except ValueError as e:
        height, width = frames[0].shape
    orig_backend = matplotlib.get_backend()
    # Switch to headless 'Agg' to inhibit figure rendering.
    matplotlib.use("Agg")
    if figsize is None:
        figsize = (width / dpi, height / dpi)
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.set_axis_off()
    ax.set_aspect("equal")
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    matplotlib.use(orig_backend)

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(
        fig=fig, func=update, frames=frames, interval=interval, blit=True, repeat=False
    )
    return anim.to_html5_video()
