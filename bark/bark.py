# -*- coding: utf-8 -*-
# -*- mode: python -*-
from __future__ import division, print_function, absolute_import, \
        unicode_literals
from datetime import datetime, timedelta
import sys
import os.path
from os import listdir
from glob import glob
from collections import namedtuple
from uuid import uuid4
import yaml
import numpy as np
import pandas as pd

BUFFER_SIZE = 10000

spec_version = "0.1"
__version__ = version = "0.1"

__doc__ = """
This is BARK, a python library for storing and accessing audio and ephys data in
directories and simple file formats.

Library versions:
 bark: %s
""" % (version)

# heirarchical classes
Root = namedtuple('Root', ['entries', 'path', 'attrs'])
Entry = namedtuple('Entry', ['datasets', 'path', 'attrs'])
SampledData = namedtuple('SampledData', ['data', 'path', 'attrs'])
EventData = namedtuple('EventData', ['data', 'path', 'attrs'])


def write_sampled(datfile, data, sampling_rate, units, **params):
    if len(data.shape) == 1:
        params["n_channels"] = 1 
    else: 
        params["n_channels"] = data.shape[1]
    params["dtype"] = data.dtype.name
    shape = data.shape
    mdata = np.memmap(datfile,
                     dtype=params["dtype"],
                     mode="w+", shape=shape)
    mdata[:] = data[:]
    params["filetype"] = "rawbinary"
    write_metadata(datfile + ".meta", sampling_rate=sampling_rate,
            units=units, **params)
    return SampledData(mdata, datfile, params)


def read_sampled(datfile, mode="r"):
    """ loads raw binary file

    mode may be "r" or "r+", use "r+" for modifiying the data (not
    recommended).
    """
    path = os.path.abspath(datfile)
    params = read_metadata(datfile + ".meta")
    data = np.memmap(datfile, dtype=params["dtype"], mode=mode)
    data = data.reshape(-1, params["n_channels"])
    return SampledData(data, path, params) 


def write_events(eventsfile, data, **params):
    assert "units" in params and params["units"] in ["s" or "samples"]
    data.to_csv(eventsfile, index=False)
    params["filetype"] = "csv"
    write_metadata(eventsfile+".meta", **params)
    return read_events(eventsfile)


def read_events(eventsfile):
    data = pd.read_csv(eventsfile)
    params = read_metadata(eventsfile + ".meta")
    return EventData(data, eventsfile, params)


def read_dataset(fname):
    "determines is file is sampled or events and reads accordingly"
    params = read_metadata(fname + ".meta")
    if "units" in params and params["units"] in ("s", "seconds"):
        return read_events(fname)
    else:
        return read_sampled(fname)


def read_metadata(metafile):
    try:
        with open(metafile, 'r') as fp:
            params = yaml.load(fp)
        return params
    except IOError as err:
        fname = os.path.splitext(metafile)[0]
        if fname == "meta":
            return {}
        print("""
{dat} is missing an associated .meta file, which should be named {dat}.meta

The .meta is plaintext YAML file of the following format:

---
dtype: int16
n_channels: 4
sampling_rate: 30000.0

(you may include any other metadata you like, such as experimenter, date etc.)

to create a .meta file interactively, type:

$ dat-meta {dat}
        """.format(dat=metafile))
        sys.exit(0)


def write_metadata(filename, **params):
    for k, v in params.items():
        if isinstance(v, (np.ndarray, np.generic)):
            params[k] = v.tolist()
    with open(filename, 'w') as yaml_file:
        header = """# metadata using YAML syntax\n---\n"""
        yaml_file.write(header)
        yaml_file.write(yaml.dump(params, default_flow_style=False))

def create_root(name, **attrs):
    """creates a new BARK top level directory"""
    path = os.path.abspath(name)
    os.makedirs(name)
    write_metadata(os.path.join(path, "meta"), **attrs)
    return read_root(name)


def read_root(name):
    path = os.path.abspath(name)
    attrs = read_metadata(os.path.join(path, "meta"))
    all_sub = [os.path.join(name, x) for x in  listdir(path)]
    print(all_sub)
    subdirs = [x for x in all_sub if os.path.isdir(x) and x[-1] != '.']
    entries = {os.path.split(x)[-1]: read_entry(x) for x in subdirs}
    return Root(entries, path, attrs)


def create_entry(name, timestamp, **attributes):
    """Creates a new BARK entry under group, setting required attributes.

    An entry is an abstract collection of data which all refer to the same time
    frame. Data can include physiological recordings, sound recordings, and
    derived data such as spike times and labels. See add_data() for information
    on how data are stored.

    name -- the name of the new entry. any valid python string.

    timestamp -- timestamp of entry (datetime object, or seconds since
               January 1, 1970). Can be an integer, a float, or a tuple
               of integers (seconds, microsceconds)

    Additional keyword arguments are set as attributes on created entry.

    Returns: newly created entry object
    """
    path = os.path.abspath(name)
    os.makedirs(path)
    if "uuid" not in attributes:
        attributes["uuid"] = str(uuid4())
    attributes["timestamp"] = convert_timestamp(timestamp)
    write_metadata(os.path.join(name, "meta"), **attributes)
    return read_entry(name)

def read_entry(name):
    path = os.path.abspath(name)
    dsets = {}
    attrs = read_metadata(os.path.join(path, "meta"))
    # load only files with associated metadata files
    dset_metas = glob(os.path.join(path, "*.meta"))
    dset_names = [x[:-5] for x in dset_metas]
    datasets = {name: read_dataset(name) for name in dset_names}
    return Entry(datasets, path, attrs)

def convert_timestamp(obj):
    """Makes a BARK timestamp from an object.

    Argument can be a datetime.datetime object, a time.struct_time, an integer,
    a float, or a tuple of integers. The returned value is a numpy array with
    the integer number of seconds since the Epoch and any additional
    microseconds.

    Note that because floating point values are approximate, the conversion
    between float and integer tuple may not be reversible.

    """
    import numbers
    from datetime import datetime
    from time import mktime, struct_time

    out = np.zeros(2, dtype='int64')
    if isinstance(obj, datetime):
        out[0] = mktime(obj.timetuple())
        out[1] = obj.microsecond
    elif isinstance(obj, struct_time):
        out[0] = mktime(obj)
    elif isinstance(obj, numbers.Integral):
        out[0] = obj
    elif isinstance(obj, numbers.Real):
        out[0] = obj
        out[1] = (obj - out[0]) * 1e6
    else:
        try:
            out[:2] = obj[:2]
        except:
            raise TypeError("unable to convert %s to timestamp" % obj)
    return out


def timestamp_to_datetime(timestamp):
    """Converts an BARK timestamp a datetime.datetime object (naive local time)"""
    obj = datetime.fromtimestamp(timestamp[0])
    return obj + timedelta(microseconds=int(timestamp[1]))


def timestamp_to_float(timestamp):
    """Converts an BARK timestamp to a floating point (sec since epoch) """
    return np.dot(timestamp, (1.0, 1e-6))


def parse_timestamp_string(string):
    if len(string) == len("YYYY-MM-DD"):
        timestamp = datetime.strptime(string, "%Y-%m-%d")
    else:
        timestamp = datetime.strptime(string, "%Y-%m-%d_%H-%M-%S.%f")
    return timestamp


def get_uuid(obj):
    """Returns the uuid for obj, or None if none is set"""
    return obj.attrs.get('uuid', None)


def count_children(obj):
    """Returns the number of children of obj"""
    if isinstance(obj, Root):
        return len(obj.entries)
    else:
        return len(obj.datasets)


def is_sampled(dset):
    """Returns True if dset is a sampled time series (units are not time)"""
    return isinstance(dset, np.memmap)


def is_events(dset):
    """Returns True if dset is a marked point process (a complex dtype with 'start' field)"""
    return isinstance(dset.data, pd.DataFrame) and 'start' in dset.data.columns

