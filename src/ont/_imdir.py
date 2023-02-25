"""Read nested directories of images into an n-dimensional array."""
import os
import sys
from dataclasses import dataclass
from typing import Protocol, Tuple
from pathlib import Path
from glob import glob
from math import prod
import imageio.v3 as iio
import dask
import dask.array as da
import numpy as np
import napari
import toolz as tz


class ImagePropertiesProto(Protocol):
    shape : Tuple[int]
    dtype : type


@dataclass
class ImageProperties:
    shape : Tuple[int]
    dtype : type


def image_properties(filename) -> ImagePropertiesProto:
    """Return the shape and dtype of the image data in filename.

    This function uses iio.v3.improps."""
    return iio.improps(filename)


@tz.curry
def image_properties_loaded(filename, load_func=iio.imread) -> ImageProperties:
    loaded = load_func(filename)
    return ImageProperties(shape=loaded.shape, dtype=loaded.dtype)


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


def imreads(root, pattern='*.tif', load_func=iio.imread, props_func=None):
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
    props_func : Callable[Path | str, ImageProperties]
        A function to get the array shape from a file. If omitted, `load_func`
        is called on the first file to get the shape. In some cases,
        `image_properties` is the most efficient function here and may avoid
        loading a large image into memory.

    Returns
    -------
    stacked : dask.array.Array
        The stacked dask array. The array will have the number of dimensions of
        each image plus one per directory level.
    """
    if props_func is None:
        if load_func is not iio.imread:
            props_func = image_properties_loaded(load_func=load_func)
        else:
            props_func = image_properties
    root = Path(root)
    files = sorted(root.glob(pattern))
    if len(files) == 0:
        raise ValueError(
                f'no files found at path {root} with pattern {pattern}.'
                )
    leading_shape = _find_shape(files)
    n_leading_dim = len(leading_shape)
    props = props_func(files[0])
    lagging_shape = props.shape
    files_array = np.array(list(files)).reshape(leading_shape)
    chunks = tuple((1,) * shp for shp in leading_shape) + lagging_shape
    stacked = da.map_blocks(
            _load_block(n_leading_dim=n_leading_dim, load_func=load_func),
            files_array,
            chunks=chunks,
            dtype=props.dtype,
            )
    return stacked

