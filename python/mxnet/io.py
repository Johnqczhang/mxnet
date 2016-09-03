# coding: utf-8
# pylint: disable=invalid-name, protected-access, fixme, too-many-arguments, W0221, W0201, no-self-use

"""NDArray interface of mxnet"""
from __future__ import absolute_import
from collections import OrderedDict

import ctypes
import cv2
import os
import sys
import numpy as np
import logging
import threading
from .base import _LIB
from .base import c_array, c_str, mx_uint, py_str
from .base import DataIterHandle, NDArrayHandle
from .base import check_call, ctypes2docstring
from .ndarray import NDArray
from .ndarray import array
from .ndarray import load


class DataBatch(object):
    """Default object for holding a mini-batch of data and related information."""
    def __init__(self, data, label, pad=None, index=None,
                 bucket_key=None, provide_data=None, provide_label=None):
        self.data = data
        self.label = label
        self.pad = pad
        self.index = index

        # the following properties are only used when bucketing is used
        self.bucket_key = bucket_key
        self.provide_data = provide_data
        self.provide_label = provide_label

class DataIter(object):
    """DataIter object in mxnet. """

    def __init__(self):
        self.batch_size = 0

    def __iter__(self):
        return self

    def reset(self):
        """Reset the iterator. """
        pass

    def next(self):
        """Get next data batch from iterator. Equivalent to
        self.iter_next()
        DataBatch(self.getdata(), self.getlabel(), self.getpad(), None)

        Returns
        -------
        data : DataBatch
            The data of next batch.
        """
        if self.iter_next():
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=self.getindex())
        else:
            raise StopIteration

    def __next__(self):
        return self.next()

    def iter_next(self):
        """Iterate to next batch.

        Returns
        -------
        has_next : boolean
            Whether the move is successful.
        """
        pass

    def getdata(self):
        """Get data of current batch.

        Returns
        -------
        data : NDArray
            The data of current batch.
        """
        pass

    def getlabel(self):
        """Get label of current batch.

        Returns
        -------
        label : NDArray
            The label of current batch.
        """
        pass

    def getindex(self):
        """Get index of the current batch.

        Returns
        -------
        index : numpy.array
            The index of current batch
        """
        return None

    def getpad(self):
        """Get the number of padding examples in current batch.

        Returns
        -------
        pad : int
            Number of padding examples in current batch
        """
        pass

class ResizeIter(DataIter):
    """Resize a DataIter to given number of batches per epoch.
    May produce incomplete batch in the middle of an epoch due
    to padding from internal iterator.

    Parameters
    ----------
    data_iter : DataIter
        Internal data iterator.
    size : number of batches per epoch to resize to.
    reset_internal : whether to reset internal iterator on ResizeIter.reset
    """

    def __init__(self, data_iter, size, reset_internal=True):
        super(ResizeIter, self).__init__()
        self.data_iter = data_iter
        self.size = size
        self.reset_internal = reset_internal
        self.cur = 0
        self.current_batch = None

        self.provide_data = data_iter.provide_data
        self.provide_label = data_iter.provide_label
        self.batch_size = data_iter.batch_size

    def reset(self):
        self.cur = 0
        if self.reset_internal:
            self.data_iter.reset()

    def iter_next(self):
        if self.cur == self.size:
            return False
        try:
            self.current_batch = self.data_iter.next()
        except StopIteration:
            self.data_iter.reset()
            self.current_batch = self.data_iter.next()

        self.cur += 1
        return True

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad

class PrefetchingIter(DataIter):
    """Base class for prefetching iterators. Takes one or more DataIters (
    or any class with "reset" and "read" methods) and combine them with
    prefetching. For example:

    Parameters
    ----------
    iters : DataIter or list of DataIter
        one or more DataIters (or any class with "reset" and "read" methods)
    rename_data : None or list of dict
        i-th element is a renaming map for i-th iter, in the form of
        {'original_name' : 'new_name'}. Should have one entry for each entry
        in iter[i].provide_data
    rename_label : None or list of dict
        Similar to rename_data

    Examples
    --------
    iter = PrefetchingIter([NDArrayIter({'data': X1}), NDArrayIter({'data': X2})],
                           rename_data=[{'data': 'data1'}, {'data': 'data2'}])
    """
    def __init__(self, iters, rename_data=None, rename_label=None):
        super(PrefetchingIter, self).__init__()
        if not isinstance(iters, list):
            iters = [iters]
        self.n_iter = len(iters)
        assert self.n_iter > 0
        self.iters = iters
        if rename_data is None:
            self.provide_data = sum([i.provide_data for i in iters], [])
            print self.provide_data
        else:
            self.provide_data = sum([[(r[n], s) for n, s in i.provide_data] \
                                    for r, i in zip(rename_data, iters)], [])
        if rename_label is None:
            self.provide_label = sum([i.provide_label for i in iters], [])
        else:
            self.provide_label = sum([[(r[n], s) for n, s in i.provide_label] \
                                    for r, i in zip(rename_label, iters)], [])
        self.batch_size = self.provide_data[0][1][0]
        self.data_ready = [threading.Event() for i in range(self.n_iter)]
        self.data_taken = [threading.Event() for i in range(self.n_iter)]
        for e in self.data_taken:
            e.set()
        self.started = True
        self.current_batch = [None for i in range(self.n_iter)]
        self.next_batch = [None for i in range(self.n_iter)]
        def prefetch_func(self, i):
            """Thread entry"""
            while True:
                self.data_taken[i].wait()
                if not self.started:
                    break
                try:
                    self.next_batch[i] = self.iters[i].next()
                except StopIteration:
                    self.next_batch[i] = None
                self.data_taken[i].clear()
                self.data_ready[i].set()
        self.prefetch_threads = [threading.Thread(target=prefetch_func, args=[self, i]) \
                                 for i in range(self.n_iter)]
        for thread in self.prefetch_threads:
            thread.setDaemon(True)
            thread.start()

    def __del__(self):
        self.started = False
        for e in self.data_taken:
            e.set()
        for thread in self.prefetch_threads:
            thread.join()

    def reset(self):
        for e in self.data_ready:
            e.wait()
        for i in self.iters:
            i.reset()
        for e in self.data_ready:
            e.clear()
        for e in self.data_taken:
            e.set()

    def iter_next(self):
        for e in self.data_ready:
            e.wait()
        if self.next_batch[0] is None:
            for i in self.next_batch:
                assert i is None, "Number of entry mismatches between iterators"
            return False
        else:
            for batch in self.next_batch:
                assert batch.pad == self.next_batch[0].pad, \
                    "Number of entry mismatches between iterators"
            self.current_batch = DataBatch(sum([batch.data for batch in self.next_batch], []),
                                           sum([batch.label for batch in self.next_batch], []),
                                           self.next_batch[0].pad,
                                           self.next_batch[0].index)
            for e in self.data_ready:
                e.clear()
            for e in self.data_taken:
                e.set()
            return True

    def next(self):
        if self.iter_next():
            return self.current_batch
        else:
            raise StopIteration

    def getdata(self):
        return self.current_batch.data

    def getlabel(self):
        return self.current_batch.label

    def getindex(self):
        return self.current_batch.index

    def getpad(self):
        return self.current_batch.pad

def _init_data(data, allow_empty, default_name):
    """Convert data into canonical form."""
    assert (data is not None) or allow_empty
    if data is None:
        data = []

    if isinstance(data, (np.ndarray, NDArray)):
        data = [data]
    if isinstance(data, list):
        if not allow_empty:
            assert(len(data) > 0)
        if len(data) == 1:
            data = OrderedDict([(default_name, data[0])])
        else:
            data = OrderedDict([('_%d_%s' % (i, default_name), d) for i, d in enumerate(data)])
    if not isinstance(data, dict):
        raise TypeError("Input must be NDArray, numpy.ndarray, " + \
                "a list of them or dict with them as values")
    for k, v in data.items():
        if isinstance(v, NDArray):
            data[k] = v.asnumpy()
    for k, v in data.items():
        if not isinstance(v, np.ndarray):
            raise TypeError(("Invalid type '%s' for %s, "  % (type(v), k)) + \
                    "should be NDArray or numpy.ndarray")

    return list(data.items())

class NDArrayIter(DataIter):
    """NDArrayIter object in mxnet. Taking NDArray or numpy array to get dataiter.
    Parameters
    ----------
    data: NDArray or numpy.ndarray, a list of them, or a dict of string to them.
        NDArrayIter supports single or multiple data and label.
    label: NDArray or numpy.ndarray, a list of them, or a dict of them.
        Same as data, but is not fed to the model during testing.
    batch_size: int
        Batch Size
    shuffle: bool
        Whether to shuffle the data
    last_batch_handle: 'pad', 'discard' or 'roll_over'
        How to handle the last batch
    Note
    ----
    This iterator will pad, discard or roll over the last batch if
    the size of data does not match batch_size. Roll over is intended
    for training and can cause problems if used for prediction.
    """
    def __init__(self, data, label=None, batch_size=1, shuffle=False, last_batch_handle='pad'):
        # pylint: disable=W0201

        super(NDArrayIter, self).__init__()

        self.data = _init_data(data, allow_empty=False, default_name='data')
        self.label = _init_data(label, allow_empty=True, default_name='softmax_label')

        # shuffle data
        if shuffle:
            idx = np.arange(self.data[0][1].shape[0])
            np.random.shuffle(idx)
            self.data = [(k, v[idx]) for k, v in self.data]
            self.label = [(k, v[idx]) for k, v in self.label]

        self.data_list = [x[1] for x in self.data] + [x[1] for x in self.label]
        self.num_source = len(self.data_list)

        # batching
        if last_batch_handle == 'discard':
            new_n = self.data_list[0].shape[0] - self.data_list[0].shape[0] % batch_size
            data_dict = OrderedDict(self.data)
            label_dict = OrderedDict(self.label)
            for k, _ in self.data:
                data_dict[k] = data_dict[k][:new_n]
            for k, _ in self.label:
                label_dict[k] = label_dict[k][:new_n]
            self.data = data_dict.items()
            self.label = label_dict.items()
        self.num_data = self.data_list[0].shape[0]
        assert self.num_data >= batch_size, \
            "batch_size need to be smaller than data size."
        self.cursor = -batch_size
        self.batch_size = batch_size
        self.last_batch_handle = last_batch_handle

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self.label]


    def hard_reset(self):
        """Igore roll over data and set to start"""
        self.cursor = -self.batch_size

    def reset(self):
        if self.last_batch_handle == 'roll_over' and self.cursor > self.num_data:
            self.cursor = -self.batch_size + (self.cursor%self.num_data)%self.batch_size
        else:
            self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            return DataBatch(data=self.getdata(), label=self.getlabel(), \
                    pad=self.getpad(), index=None)
        else:
            raise StopIteration

    def _getdata(self, data_source):
        """Load data from underlying arrays, internal use only"""
        assert(self.cursor < self.num_data), "DataIter needs reset."
        if self.cursor + self.batch_size <= self.num_data:
            return [array(x[1][self.cursor:self.cursor+self.batch_size]) for x in data_source]
        else:
            pad = self.batch_size - self.num_data + self.cursor
            return [array(np.concatenate((x[1][self.cursor:], x[1][:pad]),
                                         axis=0)) for x in data_source]

    def getdata(self):
        return self._getdata(self.data)

    def getlabel(self):
        return self._getdata(self.label)

    def getpad(self):
        if self.last_batch_handle == 'pad' and \
           self.cursor + self.batch_size > self.num_data:
            return self.cursor + self.batch_size - self.num_data
        else:
            return 0


class MXDataIter(DataIter):
    """DataIter built in MXNet. List all the needed functions here.
    Parameters
    ----------
    handle : DataIterHandle
        the handle to the underlying C++ Data Iterator
    """
    def __init__(self, handle, data_name='data', label_name='softmax_label', **_):
        super(MXDataIter, self).__init__()
        self.handle = handle
        # debug option, used to test the speed with io effect eliminated
        self._debug_skip_load = False


        # load the first batch to get shape information
        self.first_batch = None
        self.first_batch = self.next()
        data = self.first_batch.data[0]
        label = self.first_batch.label[0]

        # properties
        self.provide_data = [(data_name, data.shape)]
        self.provide_label = [(label_name, label.shape)]
        self.batch_size = data.shape[0]


    def __del__(self):
        check_call(_LIB.MXDataIterFree(self.handle))

    def debug_skip_load(self):
        """Set the iterator to simply return always first batch.
        Notes
        -----
        This can be used to test the speed of network without taking
        the loading delay into account.
        """
        self._debug_skip_load = True
        logging.info('Set debug_skip_load to be true, will simply return first batch')

    def reset(self):
        self._debug_at_begin = True
        self.first_batch = None
        check_call(_LIB.MXDataIterBeforeFirst(self.handle))

    def next(self):
        if self._debug_skip_load and not self._debug_at_begin:
            return  DataBatch(data=[self.getdata()], label=[self.getlabel()], pad=self.getpad(),
                              index=self.getindex())
        if self.first_batch is not None:
            batch = self.first_batch
            self.first_batch = None
            return batch
        self._debug_at_begin = False
        next_res = ctypes.c_int(0)
        check_call(_LIB.MXDataIterNext(self.handle, ctypes.byref(next_res)))
        if next_res.value:
            return DataBatch(data=[self.getdata()], label=[self.getlabel()], pad=self.getpad(),
                             index=self.getindex())
        else:
            raise StopIteration

    def iter_next(self):
        if self.first_batch is not None:
            return True
        next_res = ctypes.c_int(0)
        check_call(_LIB.MXDataIterNext(self.handle, ctypes.byref(next_res)))
        return next_res.value

    def getdata(self):
        hdl = NDArrayHandle()
        check_call(_LIB.MXDataIterGetData(self.handle, ctypes.byref(hdl)))
        return NDArray(hdl, False)

    def getlabel(self):
        hdl = NDArrayHandle()
        check_call(_LIB.MXDataIterGetLabel(self.handle, ctypes.byref(hdl)))
        return NDArray(hdl, False)

    def getindex(self):
        index_size = ctypes.c_uint64(0)
        index_data = ctypes.POINTER(ctypes.c_uint64)()
        check_call(_LIB.MXDataIterGetIndex(self.handle,
                                           ctypes.byref(index_data),
                                           ctypes.byref(index_size)))
        address = ctypes.addressof(index_data.contents)
        dbuffer = (ctypes.c_uint64* index_size.value).from_address(address)
        np_index = np.frombuffer(dbuffer, dtype=np.uint64)
        return np_index.copy()

    def getpad(self):
        pad = ctypes.c_int(0)
        check_call(_LIB.MXDataIterGetPadNum(self.handle, ctypes.byref(pad)))
        return pad.value

def _make_io_iterator(handle):
    """Create an io iterator by handle."""
    name = ctypes.c_char_p()
    desc = ctypes.c_char_p()
    num_args = mx_uint()
    arg_names = ctypes.POINTER(ctypes.c_char_p)()
    arg_types = ctypes.POINTER(ctypes.c_char_p)()
    arg_descs = ctypes.POINTER(ctypes.c_char_p)()

    check_call(_LIB.MXDataIterGetIterInfo( \
            handle, ctypes.byref(name), ctypes.byref(desc), \
            ctypes.byref(num_args), \
            ctypes.byref(arg_names), \
            ctypes.byref(arg_types), \
            ctypes.byref(arg_descs)))
    iter_name = py_str(name.value)
    param_str = ctypes2docstring(num_args, arg_names, arg_types, arg_descs)

    doc_str = ('%s\n\n' +
               '%s\n' +
               'name : string, required.\n' +
               '    Name of the resulting data iterator.\n\n' +
               'Returns\n' +
               '-------\n' +
               'iterator: DataIter\n'+
               '    The result iterator.')
    doc_str = doc_str % (desc.value, param_str)

    def creator(*args, **kwargs):
        """Create an iterator.
        The parameters listed below can be passed in as keyword arguments.
        Parameters
        ----------
        name : string, required.
            Name of the resulting data iterator.
        Returns
        -------
        dataiter: Dataiter
            the resulting data iterator
        """
        param_keys = []
        param_vals = []

        for k, val in kwargs.items():
            param_keys.append(c_str(k))
            param_vals.append(c_str(str(val)))
        # create atomic symbol
        param_keys = c_array(ctypes.c_char_p, param_keys)
        param_vals = c_array(ctypes.c_char_p, param_vals)
        iter_handle = DataIterHandle()
        check_call(_LIB.MXDataIterCreateIter(
            handle,
            mx_uint(len(param_keys)),
            param_keys, param_vals,
            ctypes.byref(iter_handle)))

        if len(args):
            raise TypeError('%s can only accept keyword arguments' % iter_name)

        return MXDataIter(iter_handle, **kwargs)

    creator.__name__ = iter_name
    creator.__doc__ = doc_str
    return creator

def _init_io_module():
    """List and add all the data iterators to current module."""
    plist = ctypes.POINTER(ctypes.c_void_p)()
    size = ctypes.c_uint()
    check_call(_LIB.MXListDataIters(ctypes.byref(size), ctypes.byref(plist)))
    module_obj = sys.modules[__name__]
    for i in range(size.value):
        hdl = ctypes.c_void_p(plist[i])
        dataiter = _make_io_iterator(hdl)
        setattr(module_obj, dataiter.__name__, dataiter)

# Initialize the io in startups
_init_io_module()


def load_image(path_img, color=1):
    """
    Load an image converting from grayscale or alpha as needed.

    Parameters
    ----------
    path_img: string
    color: integer
        flag for color format. 1 (default) loads as RGB while 0
        loads as intensity (if image is already grayscale).

    Returns
    -------
    image: an image with type np.uint8 in range [0, 255] of size (H x W x 3) in RGB.
    """
    try:
        img = cv2.imread(path_img, color)  # Height x Width x Channel
    except Exception, e:
        print 'cv2.imread error:', path_img, e
        return
    return img


def resize_image(img, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.

    Parameters
    ----------
    img: (H x W x C) ndarray
    new_dims: (height, width) tuple of new dimensions.
    interp_order: interpolation order, 1 (default) is bilinear interpolation. 

    Returns
    -------
    resized_img: resized ndarray with shape (new_dims[0], new_dims[1], C)
    """    
    resized_img = cv2.resize(img, (new_dims[1], new_dims[0]), interpolation=interp_order)

    return resized_img


def subtract_mean(img, img_mean):
    """
    Subtract the mean for centering the data.

    Parameters
    ----------
    img: (C x H x W) ndarray
    img_mean: numpy.ndarray or tuple, 
        for numpy.ndarray, subtract mean per pixel
        for tuple, subtract mean per channel, mean_img should be either 3-dim tuple (mean_r, mean_g, mean_b)
        or 4-dim tuple (mean_r, mean_g, mean_b, mean_a) 

    Returns
    -------
    mean_img: subtracted mean ndarray with shape (C x H x W)
    """ 
    # Subtract mean per channel
    if isinstance(img_mean, tuple): 
        mean_img = np.empty(img.shape)
        mean_img[0] = img[0] - img_mean[0]
        if img.shape[0] >= 3:
            mean_img[1] = img[1] - img_mean[1]
            mean_img[2] = img[2] - img_mean[2]
        if img.shape[0] == 4:
            mean_img[3] = img[3] - img_mean[3]
    # Subtract mean per pixel
    else:
        assert img.shape == img_mean.shape, 'subtract mean per pixel error:, mean_img shape dismatch.'
        mean_img = img - img_mean
    
    return mean_img


def image_preprocess(path_img, img_mean, resize_dims=None, 
                     rand_crop=False, crop_size=None, 
                     rand_mirror=False, mirror=False):
    """Preprocess image

    Parameters
    ----------
    path_img: string, the path of image.
    img_mean: numpy.ndarray or tuple.
    resize_dims: list.
        If resize_dims has exactly one element, it should be either an integer or a tuple, for an integer, it 
        indicates the new dimension of the shorter side of resized image; for a tuple, it indicates a/an 
        range/interval from which the new dimension of the shorter side of resized image will be chosen randomly.
        Otherwise resize_dims should have at least two distinct integers from which the new dimension of the 
        shorter side of resized image will be chosen randomly. 
    crop_size: (cropped_height, cropped_width) tuple of new cropped dimensions.

    Returns
    -------
    img: preprocessed image ndarray with shape (C x H x W)
    """
    # Load image from path_img
    img = load_image(path_img)
    if img is None:
        print 'load image error:', path_img, 'is None.'
        return

    # Resize image isotropically
    if resize_dims is not None:
        # Get the new dimension of the shorter side of the resized image
        # pick a resize dim randomly if resize_dims has more than one element
        if len(resize_dims) > 1:
            resize_dim = int(resize_dims[np.random.randint(len(resize_dims))])
        # pick a resize dim randomly from the interval indicated by the tuple
        elif isinstance(resize_dims[0], tuple):
            resize_dim = np.random.randint(resize_dims[0][0], resize_dims[0][1]+1)
        # otherwise, resize_dim(s) only has one integer
        else:
            resize_dim = int(resize_dims[0])

        # Get the shorter dim and longer dim of the original image
        shorter = 0 if img.shape[0] < img.shape[1] else 1
        short_dim = img.shape[shorter]

        if short_dim != resize_dim:
            long_dim = img.shape[1-shorter]
            new_long_dim = int(round(resize_dim*long_dim*1.0 / short_dim))
            new_shape = (resize_dim, new_long_dim) if shorter == 0 else (new_long_dim, resize_dim)
            # resize the image
            img = resize_image(img, new_shape)

    # convert image data type from int to float
    img = img.astype(np.float32)
    # Swap the order of dims from 'h x w x c' to 'c x h x w'
    img = img.transpose((2, 0, 1))
    # Swap the order of channels from BGR to RGB
    img = img[(2, 1, 0), :, :]

    # Crop the image
    if crop_size is not None and img.shape[1:] != crop_size:
        h_max = img.shape[1] - crop_size[0]
        w_max = img.shape[2] - crop_size[1]
        try:
            assert h_max >= 0 and w_max >= 0
        except:
            print 'crop error:crop_size', crop_size, 'is larger than image_size', img.shape[1:]
            return
        if rand_crop:
            h = np.random.randint(0, h_max+1)
            w = np.random.randint(0, w_max+1)
        else:  # center crop
            h = h_max / 2
            w = w_max / 2
        img = img[:, h:h+crop_size[0], w:w+crop_size[1]]

    # Color jittering
    # TODO(johnqczhang): will implement color space augmentation later

    # Subtract mean
    img = subtract_mean(img, img_mean)

    # Mirror
    if (rand_mirror and np.random.randint(2)) or mirror:
        img = img[:, :, ::-1]

    return img


def oversample_10(img, crop_dims):
    """
    Standard 10 crops: Crop image into the four corners, center, and their mirrored versions.

    Parameters
    ----------
    img: (H x W x C) ndarray
    crop_dims: (height, width) tuple for the crops.
    
    Returns
    -------
    crops: (10 x H x W x C) ndarray of crops.
    """
    # Dimensions and center.
    im_shape = np.array(img.shape)
    crop_dims = np.array(crop_dims)
    im_center = im_shape[:2] / 2.0

    # Make crop coordinates
    h_indices = (0, im_shape[0] - crop_dims[0])
    w_indices = (0, im_shape[1] - crop_dims[1])
    crops_ix = np.empty((5, 4), dtype=int)
    curr = 0
    for i in h_indices:
        for j in w_indices:
            crops_ix[curr] = (i, j, i + crop_dims[0], j + crop_dims[1])
            curr += 1
    crops_ix[4] = np.tile(im_center, (1, 2)) + np.concatenate([
        -crop_dims / 2.0,
         crop_dims / 2.0
    ])
    # crops_ix = np.tile(crops_ix, (2, 1))

    # Extract crops
    crops = np.empty((10, crop_dims[0], crop_dims[1], im_shape[-1]), 
                     dtype=np.float32)
    ix = 0

    for crop in crops_ix:
        crops[ix] = img[crop[0]:crop[2], crop[1]:crop[3], :]
        ix += 2
    # flip for mirrors
    crops[1:10:2] = crops[0:9:2, :, ::-1, :]
    
    return crops


def oversample_12(img, crop_dims):
    """
    Crop image into 12 crops: 10 standard crops plus 
        resize the square image into crop_dims with mirrored version.
    
    Parameters
    ----------
    img: (H x W x C) ndarray
    crop_dims: (height, width) tuple for the crops.
    
    Returns
    -------
    crops: (12 x H x W x C) ndarray of crops.
    """
    crops = np.empty((12, crop_dims[0], crop_dims[1], img.shape[-1]), 
                     dtype=np.float32)
    # get 10 standard crops
    crops[0:10] = oversample_10(img, crop_dims)

    # Extract the resized crops and its mirrored version
    crops[10] = resize_image(img, crop_dims)
    crops[11] = crops[10, :, ::-1, :]

    return crops


def oversample_square(img, crop_dims, resize_dims=None):
    """
    Aggressive cropping (from GoogleNet): Given a square image, first rescale the image such that 
        each side is in resize_dims list. Then, for each square, take the 4 corners, center as 
        well as the square resized to crop_dims, and their mirrored versions.

    Parameters
    ----------
    img: (H x W x C) ndarray
    crop_dims: (height, width) tuple for the crops.
    resize_dims: list of the shorter dims for resize before cropping.

    Returns
    -------
    crops: (N x H x W x C) ndarray of crops that N = 12 * len(resize_dims).
    """ 
    # Do not resize if resized dimensions not specified
    if not resize_dims:
        resize_dims = [img.shape[0]]
    
    N = len(resize_dims) * 12
    crops = np.empty((N, crop_dims[0], crop_dims[1], img.shape[-1]), 
                     dtype=np.float32)
    crop_id = 0

    for resize_dim in resize_dims:
        # Resize the image
        if resize_dim != img.shape[0]:
            img = resize_image(img, (resize_dim, resize_dim))
        # Extract 12 crops for each resized image.
        crops[crop_id:crop_id+12] = oversample_12(img, crop_dims)
        crop_id += 12
 
    return crops


def oversample_rect(img, crop_dims, resize_dims=None):
    """
    Aggressive cropping (from GoogleNet): Given a rectangular image, first rescale the image such that 
        the shorter side is in resize_dims list, then take the left, center and right square of these 
        resized images (in the case of portrait images, take the top, center and bottom squares).
        For each square, take the 4 corners, center as well as the square resized to crop_dims, and 
        their mirrored versions.
    
    Parameters
    ----------
    img: (H x W x K) ndarray
    crop_dims: (height, width) tuple for the crops.
    resize_dims: list of the shorter dims for resize before cropping.
    
    Returns
    -------
    crops: (N x H x W x C) ndarray of crops that N = 36 * len(resize_dims).
    """
    # Get the shorter side of original image
    shorter = 0 if img.shape[0] < img.shape[1] else 1
    
    # Dimensions of original image
    short_dim = img.shape[shorter]
    long_dim = img.shape[1-shorter]
    
    # Do not resize if resized dimensions not specified
    if not resize_dims:
        resize_dims = [short_dim]
    
    N = len(resize_dims) * 36
    crops = np.empty((N, crop_dims[0], crop_dims[1], img.shape[-1]), 
                     dtype=np.float32)
    crop_id = 0

    for resize_dim in resize_dims:
        # Resize the image isotropically
        if resize_dim != short_dim:
            new_long_dim = int(round(resize_dim*long_dim*1.0 / short_dim))
            new_dims = (resize_dim, new_long_dim) if shorter == 0 else (new_long_dim, resize_dim)
            img = resize_image(img, new_dims)
        else:
            new_long_dim = long_dim
        
        # Take the left (or top), center and right (or bottom) square of resized image
        for left in (0, (new_long_dim-resize_dim)/2, new_long_dim-resize_dim):
            im_square = img[:, left:left+resize_dim, :] if shorter == 0 else img[left:left+resize_dim, :, :]
            # Take 12 crops for each square image
            # raise ValueError
            crops[crop_id:crop_id+12] = oversample_12(im_square, crop_dims)
            crop_id += 12

    return crops


class ImageDataIter(DataIter):
    """DataIter that loads images directly from the disk

    Parameters
    ----------
    img_lst: string, the path of the image list file (.lst).
    data_shape: tuple, (batch_size, channel, height, width), here the height and width should be
        the cropped size that will be input to the network.
    mean_img: string, the path of mean file.
    mean_rgb: tuple, (mean_r, mean_g, mean_b).
        Note: Either mean_img or mean_rgb should be given.
    mean_a: float, the mean value of alpha channel.
    batch_size: integer, the number of images in a mini-batch.
    root: string, the full path of image data directory.
    rand_crop: boolean, whether crop a patch randomly from the original image.
    resize_dims: list, tuple or integer.
    rand_mirror: boolean, whether take the mirrored version of the image randomly.
    mirror: boolean, whether take the mirrored version of the image.
    shuffle: boolean, whether shuffle the image list.
    """
    def __init__(self, data_shape, img_lst=None, mean_img=None, mean_rgb=None, 
                 mean_a=0.0, batch_size=None, root=None, resize_dims=None, 
                 rand_crop=False, rand_mirror=False, mirror=False, shuffle=False):
        assert (mean_img is None) != (mean_rgb is None), 'mean error: either mean_img or mean_rgb should be given.'

        super(ImageDataIter, self).__init__()
        
        # reading image list file
        with open(img_lst) as f:
            lines = f.readlines()
        # shuffle data
        if shuffle:
            np.random.shuffle(lines)
        self.img = lines
        # the total number of all images
        self.num_data = len(lines)
        # the data shape that will be input to the network
        self.data_shape = data_shape
        self.crop_size = (data_shape[1], data_shape[2])

        # the mean file over pixels or mean value over channels
        if mean_img is not None:
            print 'Load mean image from %s' % mean_img
            self.mean_img = load(mean_img)['mean_img'].asnumpy()
        else:
            self.mean_img = tuple(list(mean_rgb)+[mean_a])

        self.batch_size = batch_size
        self.root = root
        # convert resize_dims into a list
        if resize_dims is not None and not isinstance(resize_dims, list):
            resize_dims = [resize_dims]
        self.resize_dims = resize_dims
        self.rand_crop = rand_crop
        self.rand_mirror = rand_mirror
        self.mirror = mirror
        self.shuffle = shuffle

        # the cursor that points to the current batch (iteration)
        self.cursor = -batch_size
        
    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [("data", tuple([self.batch_size] + list(self.data_shape)))]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [("softmax_label", tuple([self.batch_size]))]

    def hard_reset(self):
        """Ignore roll over data and set to start"""
        self.cursor = -self.batch_size

    def reset(self):
        self.cursor = -self.batch_size
        # shuffle the order of image
        if self.shuffle:
            np.random.shuffle(self.img)

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            data, label = self.getdata_label()
            return DataBatch(data=data, label=label, pad=self.getpad(), index=None)
        else:
            raise StopIteration

    def getdata_label(self):
        data = np.empty([self.batch_size] + list(self.data_shape), np.float32)
        label = np.empty(self.batch_size, np.uint32)
        for i in range(self.batch_size):
            # Read image
            cur_ = (self.cursor + i) % self.num_data
            img_info = self.img[cur_].split()
            filename = img_info[-1].strip('/')
            label[i] = int(img_info[1])
            data[i] = image_preprocess(os.path.join(self.root, filename), self.mean_img, 
                                       self.resize_dims, self.rand_crop, self.crop_size, 
                                       self.rand_mirror, self.mirror)
            assert data[i] is not None, 'image preprocess error: %s is None.' % filename

        return [array(data)], [array(label)]
    
    def getpad(self):
        return 0


class ImageSampleIter(DataIter):
    """DataIter that loads images directly and samples images in a balanced way
    
    Parameters
    ----------
    img_lists: list, each element is a string that indicates the path of per-class image list file
    data_shape: tuple, (batch_size, channel, height, width), here the height and width should be
        the cropped size that will be input to the network.
    mean_img: string, the path of mean file.
    mean_rgb: tuple, (mean_r, mean_g, mean_b).
        Note: Either mean_img or mean_rgb should be given.
    mean_a: float, the mean value of alpha channel.
    batch_size: integer, the number of images in a mini-batch.
    root: string, the full path of image data directory.
    rand_crop: boolean, whether crop a patch randomly from the original image.
    resize_dims: list, tuple or integer.
    rand_mirror: boolean, whether take the mirrored version of the image randomly.
    mirror: boolean, whether take the mirrored version of the image.
    shuffle: boolean, whether shuffle the image list.
    """
    def __init__(self, img_lists, data_shape, mean_img=None, mean_rgb=None, 
                 mean_a=0.0, batch_size=None, root=None, resize_dims=None, 
                 rand_crop=False, rand_mirror=False, mirror=False, shuffle=False):
        # reading per-class image lists
        self.img_lists = []
        for img_lst in img_lists:
            with open(img_lst) as fin:
                # logging.info('Loading %s', img_lst)
                print 'Loading %s', img_lst
                lines = fin.readlines()
            # shuffle data to sample an image from per-class image list randomly
            if shuffle:
                np.random.shuffle(lines)
            
            self.img_lists.append(lines)

        if img_lists[0].endswith('lst'):
            self.is_lst = True
        else:
            self.is_lst = False

        # get the number of class and generate the class_id list
        self.num_class = len(self.img_lists)
        self.class_id_list = np.arange(self.num_class)
        # For training, first shuffle the class_id list to sample a class randomly
        if shuffle:
            np.random.shuffle(self.class_id_list)
            
        # get the number of images in each per-class image list
        self.num_per_class = [len(img_list) for img_list in self.img_lists]
        # get the total number of images over all classes
        self.num_data = sum(self.num_per_class)

        if mean_img is not None:
            print 'Load mean image from %s' % mean_img
            self.mean_img = load(mean_img)['mean_img'].asnumpy()
        else:
            self.mean_img = tuple(list(mean_rgb)+[mean_a])
        
        self.data_shape = data_shape
        self.crop_size = (data_shape[1], data_shape[2])
        self.batch_size = batch_size
        # the image root directory
        self.root = root 
        # resize dims (could be multiple scales, an interval, or just one dim)
        if resize_dims  is not None and not isinstance(resize_dims, list):
            resize_dims = [resize_dims]
        self.resize_dims = resize_dims
        
        # whether randomly crop a patch over the original image
        self.rand_crop = rand_crop
        # horizontal flip
        self.rand_mirror = rand_mirror
        self.mirror = mirror
        self.shuffle = shuffle

        # the cursor that points to the current batch (iteration)
        self.cursor = -batch_size
        # the cursor that points to the current class_id
        self.class_cursor = 0
        # the cursor that points to the current image in each per-class image list
        self.per_class_cursor = [0] * self.num_class

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [("data", tuple([self.batch_size] + list(self.data_shape)))]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [("softmax_label", tuple([self.batch_size]))]

    def hard_reset(self):
        """Ignore roll over data and set to start"""
        self.cursor = -self.batch_size

    def reset(self):
        self.cursor = -self.batch_size

    def iter_next(self):
        self.cursor += self.batch_size
        if self.cursor < self.num_data:
            return True
        else:
            return False

    def next(self):
        if self.iter_next():
            data, label = self.getdata_label()
            return DataBatch(data=data, label=label, pad=self.getpad(), index=None)
        else:
            raise StopIteration

    def update_cursor(self, class_id):
        # Update class cursor
        class_cursor = self.class_cursor
        if class_cursor < self.num_class - 1:
            self.class_cursor = class_cursor + 1
        else:   # reach the end of class list
            # shuffle class list to reorder the classes
            np.random.shuffle(self.class_id_list)
            # reset class_cursor
            self.class_cursor = 0
        
        # Update image cursor in per-class image list
        per_class_cursor = self.per_class_cursor[class_id]
        if per_class_cursor < self.num_per_class[class_id] - 1:
            self.per_class_cursor[class_id] = per_class_cursor + 1
        else:   # reach the end of per-class image list of class class_id
            # shuffle per-class image list to reorder the images of class_id
            np.random.shuffle(self.img_lists[class_id])
            self.per_class_cursor[class_id] = 0

    def getdata_label(self):
        # data: N (batch_size) x Channel x Height x Width
        data = np.empty([self.batch_size] + list(self.data_shape), np.float32)
        # labels: N (batch_size), one dim, range: 0 ~ 2^32-1
        label = np.empty(self.batch_size, np.uint32)
        for i in range(self.batch_size):
            # get the current class index
            class_id = self.class_id_list[self.class_cursor]
            # get the per-class images list of current class
            img_list = self.img_lists[class_id]
            # get the current image cursor of current class and get image info
            img_info = img_list[self.per_class_cursor[class_id]].split()
            # Note: there are two kinds of format about img_info depending on whether using
            # .lst file or .txt file
            if self.is_lst:
                filename = img_info[-1].strip('/')
            else:
                filename = img_info[0].strip('/')
            label[i] = class_id
            data[i] = image_preprocess(os.path.join(self.root, filename), self.mean_img, 
                                       self.resize_dims, self.rand_crop, self.crop_size, 
                                       self.rand_mirror, self.mirror)
            assert data[i] is not None, 'image preprocess error: %s is None.' % filename
            
            # update cursors
            self.update_cursor(class_id=class_id)
        return [array(data)], [array(label)]  
    
    def getpad(self):
        return 0
