"""
utils.py  –  Shared helper functions for the forgery-detection project.

Functions
---------
load_image   : Read an image from disk (BGR, RGB, or grayscale).
save_image   : Write an image to disk (creates directories automatically).
show_images  : Display or save a side-by-side comparison figure.
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend (no GUI pop-ups)
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────
# 1.  load_image
# ──────────────────────────────────────────────────────────────────────
def load_image(path: str, color: str = "bgr") -> np.ndarray:
    """
    Read an image from *path* and return it as a NumPy array.

    Parameters
    ----------
    path  : str
        Absolute or relative path to the image file.
    color : str, optional
        Colour space of the returned array.
        'bgr'  → OpenCV default (Blue-Green-Red)
        'rgb'  → converted to Red-Green-Blue
        'gray' → converted to single-channel grayscale

    Returns
    -------
    np.ndarray
        The loaded image.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    ValueError
        If *color* is not one of the accepted values.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_COLOR)     # always load as colour first
    if img is None:
        raise ValueError(f"OpenCV could not decode: {path}")

    color = color.lower()
    if color == "rgb":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif color != "bgr":
        raise ValueError(f"Unknown color mode '{color}'. Use 'bgr', 'rgb', or 'gray'.")

    return img


# ──────────────────────────────────────────────────────────────────────
# 2.  save_image
# ──────────────────────────────────────────────────────────────────────
def save_image(image: np.ndarray, path: str) -> None:
    """
    Save *image* to *path*, creating parent directories if needed.

    If the image has 3 channels and is in RGB order, it is automatically
    converted to BGR before writing (OpenCV expects BGR).

    Parameters
    ----------
    image : np.ndarray
        Image array (grayscale or 3-channel).
    path  : str
        Destination file path (e.g. 'results/output.png').
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cv2.imwrite(path, image)


# ──────────────────────────────────────────────────────────────────────
# 3.  show_images
# ──────────────────────────────────────────────────────────────────────
def show_images(
    images: list,
    titles: list[str],
    save_path: str | None = None,
    figsize_per_img: tuple = (5, 4),
) -> None:
    """
    Display (or save) several images side-by-side in one Matplotlib figure.

    Parameters
    ----------
    images         : list of np.ndarray
        Images to show (grayscale or RGB).
    titles         : list of str
        Title for each subplot.
    save_path      : str or None
        If given, the figure is saved here instead of being shown.
    figsize_per_img: tuple
        (width, height) in inches per subplot.
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(figsize_per_img[0] * n,
                                            figsize_per_img[1]))
    if n == 1:
        axes = [axes]                         # make iterable for a single image

    for ax, img, title in zip(axes, images, titles):
        # Decide colour map: gray for single-channel, none for colour
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)                    # expects RGB
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[utils] Saved figure -> {save_path}")
    else:
        plt.show()

    plt.close(fig)                            # free memory


# ──────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[utils] Module loaded OK.  Functions: load_image, save_image, show_images")
