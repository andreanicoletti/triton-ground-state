import numpy as np
import matplotlib.pyplot as plt

def plot_matrix(matrices, titles=None, shape=None, clip=None, figsize=None, cmap=None):
    if shape is None:
        # automatically set a square grid
        ncols = int(np.ceil(np.sqrt(n)))
        nrows = int(np.ceil(n / ncols))
    else:
        nrows, ncols = shape

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    
    vmin, vmax = clip if clip is not None else (None, None)
    cmap = cmap if cmap is not None else 'viridis'
    if titles is None:
        titles = [f"Matrix {i}" for i in range(n)]
    
    for i, mat in enumerate(matrices):
        im = axes[i].imshow(np.abs(mat), vmin=vmin, vmax=vmax, cmap=cmap, origin='upper')
        axes[i].set_title(titles[i])
        fig.colorbar(im, ax=axes[i])

    # we hide axis since they are not very indicative of the basis the matrix is in
    for j in range(len(axes)):
        axes[j].set_axis_off()

    fig.tight_layout()
    return fig