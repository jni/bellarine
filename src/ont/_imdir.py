"""Read nested directories of images into an n-dimensional array."""
import os
import sys
from pathlib import Path
from glob import glob
from math import prod
import imageio.v3 as iio
import dask
import dask.array as da
import numpy as np
import napari
import toolz as tz


@tz.curry
def _load_block(files_array, block_id=None,
        *,
        n_leading_dim,
        load_func=iio.imread):
    image = np.asarray(load_func(files_array[block_id[:n_leading_dim]]))
    return image[(np.newaxis,) * n_leading_dim]


def _find_shape(file_sequence):
    n_total = len(file_sequence)
    parents = {p.parent for p in file_sequence}
    n_parents = len(parents)
    if n_parents == 1:
        return (n_total,)
    else:
        return _find_shape(parents) + (n_total // n_parents,)


def imreads(root, pattern='*.tif', load_func=iio.imread):
    """Read images from root (heh) folder.

    Parameters
    ----------
    root : str | pathlib.Path
        The root folder containing the hierarchy of image files.
    pattern : str
        A glob pattern with zero or more levels of subdirectories. Each level
        will be counted as a dimension in the output array. Directories *must*
        be specified with a forward slash ("/").
    load_func : Callable[Path | str, np.ndarray]
        The function to load individual arrays from files.

    Returns
    -------
    stacked : dask.array.Array
        The stacked dask array. The array will have the number of dimensions of
        each image plus one per directory level.
    """
    root = Path(root)
    files = sorted(root.glob(pattern))
    if len(files) == 0:
        raise ValueError(
                f'no files found at path {root} with pattern {pattern}.'
                )
    leading_shape = _find_shape(files)
    n_leading_dim = len(leading_shape)
    file_props = iio.improps(files[0])
    lagging_shape = file_props.shape
    files_array = np.array(list(files)).reshape(leading_shape)
    stacked = da.map_blocks(
        _load_block(n_leading_dim=n_leading_dim, load_func=load_func),
        files_array,
        chunks=tuple([(1,) * shp for shp in leading_shape]) + lagging_shape,
        dtype=file_props.dtype,
    )
    return stacked

