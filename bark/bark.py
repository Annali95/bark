# -*- coding: utf-8 -*-
# -*- mode: python -*-
from __future__ import division, print_function, absolute_import, \
        unicode_literals
from datetime import datetime, timedelta
import sys
import os.path
from os import listdir
from glob import glob
from uuid import uuid4
import codecs
from collections import namedtuple
import arrow
import yaml
import numpy as np
from bark import stream

BUFFER_SIZE = 10000

spec_version = "0.2"
__version__ = version = "0.2"

__doc__ = """
This is BARK, a python library for storing and accessing audio and ephys data
in directories and simple file formats.

Library versions:
 bark: %s
""" % (version)

_Units = namedtuple('_Units', ['TIME_UNITS'])
UNITS = _Units(TIME_UNITS=('s', 'samples'))

_pairs = ((None, None), ('UNDEFINED', 0), ('ACOUSTIC', 1), ('EXTRAC_HP', 2),
          ('EXTRAC_LF', 3), ('EXTRAC_EEG', 4), ('INTRAC_CC', 5),
          ('INTRAC_VC', 6), ('EVENT', 1000), ('SPIKET', 1001),
          ('BEHAVET', 1002), ('INTERVAL', 2000), ('STIMI', 2001),
          ('COMPONENTL', 2002))
_dt = namedtuple('_dt', ['name_to_code', 'code_to_name'])
DATATYPES = _dt(name_to_code={name: code
                              for name, code in _pairs},
                code_to_name={code: name
                              for name, code in _pairs})


# hierarchical classes
class Root():
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.name = os.path.split(self.path)[-1]
        all_sub = [os.path.join(path, x) for x in listdir(self.path)]
        subdirs = [x for x in all_sub if os.path.isdir(x) and x[-1] != '.']
        self.entries = {os.path.split(x)[-1]: read_entry(x) for x in subdirs}

    def __getitem__(self, item):
        return self.entries[item]

    def __len__(self):
        return self.entries.__len__()

    def __contains__(self, item):
        return self.entries.__contains__(item)


class Entry():
    def __init__(self, datasets, path, attrs):
        self.datasets = datasets
        self.path = path
        self.name = os.path.split(path)[-1]
        self.attrs = attrs
        self.timestamp = timestamp_to_datetime(attrs["timestamp"])
        self.uuid = attrs["uuid"]

    def __getitem__(self, item):
        return self.datasets[item]

    def __len__(self):
        return self.datasets.__len__()

    def __contains__(self, item):
        return self.datasets.__contains__(item)

    def __lt__(self, other):
        return self.timestamp < other.timestamp


class Data():
    def __init__(self, data, path, attrs):
        self.data = data
        self.path = path
        self.attrs = attrs
        self.name = os.path.split(path)[-1]

    def __getitem__(self, item):
        return self.data[item]

    @property
    def datatype_name(self):
        """Returns the dataset's datatype name, or None if unspecified."""
        return DATATYPES.name_to_code[self.attrs.get('datatype',
                                                     self.default_datatype())]

    def default_datatype(self):
        if isinstance(self, EventData):
            return 1000
        return 0


class SampledData(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_rate = self.attrs['sampling_rate']

    def toStream(self):
        return stream.read(self.path)

    def write(self, path=None):
        if path is None:
            path = self.path
        write_sampled(path, self.data, **self.attrs)


class EventData(Data):
    def write(self, path=None):
        "Saves data to file"
        if path is None:
            path = self.path
        write_events(path, self.data, **self.attrs)


def template_columns(fields):
    return {f: {'units': None} for f in fields}


def event_columns(dataframe, columns=None):
    if columns is None:
        return template_columns(dataframe.columns)
    for fieldkey in columns:
        if fieldkey not in dataframe.columns:
            del columns[fieldkey]
    for col in dataframe.columns:
        if col not in columns:
            columns[col] = {'units': None}
        if 'units' not in columns[col]:
            columns[col]['units'] = None


def sampled_columns(data, columns=None):
    'If columns=None, create new columns attribute, otherwise verify columns.'
    if len(data.shape) == 1:
        n_channels = 1
    else:
        n_channels = data.shape[1]
    if columns is None:
        return template_columns(range(n_channels))
    if len(columns) != n_channels:
        raise ValueError(
            'the columns attribute does not match the number of columns')
    for i in range(n_channels):
        if i not in columns:
            raise ValueError(
                'the columns attribute is missing column {}'.format(i))
        if 'units' not in columns[i]:
            columns[i]['units'] = None


def write_sampled(datfile, data, sampling_rate, **params):
    if 'columns' not in params:
        params['columns'] = sampled_columns(data)
    params["dtype"] = data.dtype.str
    shape = data.shape
    mdata = np.memmap(datfile, dtype=params["dtype"], mode="w+", shape=shape)
    mdata[:] = data[:]
    write_metadata(datfile, sampling_rate=sampling_rate, **params)
    params['sampling_rate'] = sampling_rate
    return SampledData(mdata, datfile, params)


def read_sampled(datfile, mode="r"):
    """ loads raw binary file

    mode may be "r" or "r+"; use "r+" for modifiying the data (not
    recommended).
    """
    path = os.path.abspath(datfile)
    params = read_metadata(datfile)
    data = np.memmap(datfile, dtype=params["dtype"], mode=mode)
    data = data.reshape(-1, len(params['columns']))
    return SampledData(data, path, params)


def write_events(eventsfile, data, **params):
    if 'columns' not in params:
        params['columns'] = event_columns(data)
    data.to_csv(eventsfile, index=False)
    write_metadata(eventsfile, **params)
    return read_events(eventsfile)


def read_events(eventsfile):
    import pandas as pd
    data = pd.read_csv(eventsfile).fillna('')
    params = read_metadata(eventsfile)
    return EventData(data, eventsfile, params)


def read_dataset(fname):
    "determines if file is sampled or event data and reads accordingly"
    params = read_metadata(fname)
    if 'dtype' in params:
        dset = read_sampled(fname)
    else:
        dset = read_events(fname)
    return dset


def read_metadata(path, meta='.meta.yaml'):
    if os.path.isdir(path):
        metafile = os.path.join(path, meta[1:])
        return yaml.safe_load(open(metafile, 'r'))
    if os.path.isfile(path):
        metafile = path + meta
        if os.path.isfile(metafile):
            return yaml.safe_load(open(metafile, 'r'))
        elif os.path.splitext(path)[-1] == meta:
            print("Tried to open metadata file instead of data file.")
        if os.path.exists(path):
            print("{} is missing an associated meta file, should named {}"
                  .format(path, meta))
    else:
        print("{} does not exist".format(path))
    sys.exit(1)


def write_metadata(path, meta='.meta.yaml', **params):
    if 'n_channels' in params:
        del params['n_channels']
    if 'n_samples' in params:
        del params['n_samples']
    if os.path.isdir(path):
        metafile = os.path.join(path, meta[1:])
    else:
        metafile = path + meta
    for k, v in params.items():
        if isinstance(v, (np.ndarray, np.generic)):
            params[k] = v.tolist()
    with codecs.open(metafile, 'w', encoding='utf-8') as yaml_file:
        yaml_file.write(yaml.safe_dump(params, default_flow_style=False))


def read_root(name):
    return Root(name)


def create_entry(name, timestamp, parents=False, **attributes):
    """Creates a new BARK entry under group, setting required attributes.

    An entry is an abstract collection of data which all refer to the same time
    frame. Data can include physiological recordings, sound recordings, and
    derived data such as spike times and labels. See add_data() for information
    on how data are stored.

    name -- the name of the new entry. any valid python string.

    timestamp -- timestamp of entry (datetime object, or seconds since
               January 1, 1970). Can be an integer, a float, or a tuple
               of integers (seconds, microsceconds)

    parents -- if True, no error is raised if file already exists. Metadata
                is overwritten

    Additional keyword arguments are set as attributes on created entry.

    Returns: newly created entry object
    """
    path = os.path.abspath(name)
    if os.path.isdir(path):
        if not parents:
            raise IOError("{} already exists".format(path))
    else:
        os.makedirs(path)

    if "uuid" not in attributes:
        attributes["uuid"] = str(uuid4())
    attributes["timestamp"] = convert_timestamp(timestamp)
    write_metadata(os.path.join(name), **attributes)
    return read_entry(name)


def read_entry(name, meta=".meta.yaml"):
    path = os.path.abspath(name)
    attrs = read_metadata(path, meta)
    # load only files with associated metadata files
    dset_metas = glob(os.path.join(path, "*" + meta))
    dset_full_names = [x[:-len(meta)] for x in dset_metas]
    dset_names = [os.path.split(x)[-1] for x in dset_full_names]
    datasets = {name: read_dataset(full_name)
                for name, full_name in zip(dset_names, dset_full_names)}
    return Entry(datasets, path, attrs)


def convert_timestamp(obj, default_tz='America/Chicago'):
    """Makes a BARK timestamp from an object.
    If the object is not UTC aware, the timezone is set by default_tz."""
    dt = timestamp_to_datetime(obj)
    if dt.tzinfo:
        return arrow.get(dt).isoformat()
    else:
        return arrow.get(dt, default_tz).isoformat()


def timestamp_to_datetime(obj):
    """Converts a BARK timestamp to a datetime.datetime object (aware local time)

    Argument can be an ISO 8601 string, a datetime.datetime object,
    a time.struct_time, an integer,
    a float, or a tuple of integers. The returned value is a string in
    ISO 8601 format.

    Note that because floating point values are approximate, the conversion
    between float and integer tuple may not be reversible.
    """
    import numbers
    from time import mktime, struct_time
    if isinstance(obj, datetime):
        dt = obj
    elif isinstance(obj, arrow.Arrow):
        dt = obj.datetime
    elif isinstance(obj, str):
        dt = arrow.get(obj).datetime
    elif isinstance(obj, struct_time):
        dt = mktime(obj)
    elif isinstance(obj, numbers.Number):
        dt = datetime.fromtimestamp(obj)
    elif hasattr(obj, '__getitem__'):
        dt = datetime.fromtimestamp(obj[0])
        dt += timedelta(microseconds=int(obj[1]))
    else:
        raise TypeError("unable to convert %s to timestamp" % obj)
    return dt


def timestamp_to_float(timestamp):
    """Converts a BARK timestamp to a floating point value (sec since epoch)"""
    return timestamp_to_datetime(timestamp).timestamp()
