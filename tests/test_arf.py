# -*- coding: utf-8 -*-
# -*- mode: python -*-

# test harness for arf interface. assumes the underlying hdf5 and h5py libraries
# are working.

from nose.tools import *
from nose.plugins.skip import SkipTest

import numpy as nx
import h5py
import arf
import time
from numpy.random import randn, randint, rand

test_file = 'tests/test.arf'
fp = arf.open_file(test_file,'w')
entry_base = "entry_%03d"
tstamp = time.mktime(time.localtime())
entry_attributes = { 'intattr' : 1,
                     'vecattr' : [1, 2, 3],
                     'arrattr' : randn(5),
                     'strattr' : "an attribute",
                     }
datasets = [ dict(name="acoustic",
                  data=randn(100000),
                  sampling_rate=20000,
                  datatype=arf.DataTypes.ACOUSTIC,
                  maxshape=(None,),
                  microphone="DK-1234",
                  compression=0),
             dict(name="neural",
                  data=(randn(100000)*2**16).astype('h'),
                  sampling_rate=20000,
                  datatype=arf.DataTypes.EXTRAC_HP,
                  compression=9),
             dict(name="spikes",
                  data=randint(0,100000,100),
                  datatype=arf.DataTypes.SPIKET,
                  units="samples",
                  sampling_rate=20000,  # required
                  ),
             dict(name="empty-spikes",
                  data=nx.array([],dtype='f'),
                  datatype=arf.DataTypes.SPIKET,
                  method="broken",
                  units="s",
                  ),
             dict(name="events",
                  data=nx.rec.fromrecords([(1.0, 1, "stimulus"),(5.0,0,"stimulus")],
                                          names=("start","state","name")), # 'start' required
                  datatype=arf.DataTypes.EVENT,
                  units="s")
             ]

bad_datasets = [ dict(name="string datatype",
                      data="a string"),
                 dict(name="object datatype",
                      data=basestring),
                 dict(name="missing samplerate/units",
                      data=randn(1000)),
                 dict(name="missing samplerate for units=samples",
                      data=randn(1000),
                      units="samples"),
                 dict(name="missing start field",
                      data=nx.rec.fromrecords([(1.0,1),(2.0,2)],
                                              names=("time","state")),
                      units="s"),
                 dict(name="missing units for complex dtype",
                      data=nx.rec.fromrecords([(1.0, 1, "stimulus"),(5.0,0,"stimulus")],
                                              names=("start","state","name"))),
                 dict(name="wrong length units for complex dtype",
                      data=nx.rec.fromrecords([(1.0, 1, "stimulus"),(5.0,0,"stimulus")],
                                              names=("start","state","name")),
                      units=("seconds",)),
                 ]


def create_entry(name):
    g = arf.create_entry(fp, name, tstamp, **entry_attributes)
    assert_true(name in fp)
    assert_greater(arf.timestamp_to_float(g.attrs['timestamp']), 0)
    for k in entry_attributes:
        assert_true(k in g.attrs)


def create_dataset(g, dset):
    d = arf.create_dataset(g, **dset)
    assert_equal(d.shape, dset['data'].shape)


def test00_create_entries():
    for i in range(25):
        yield create_entry, entry_base % i
    assert_equal(len(fp), 25)


@raises(ValueError)
def test01_create_existing_entry():
    arf.create_entry(fp, entry_base % 0,tstamp,**entry_attributes)


def test02_create_datasets():
    for name in arf.keys_by_creation(fp):
        entry = fp[name]
        for dset in datasets:
            yield create_dataset, entry, dset
        assert_equal(len(entry), len(datasets))
        assert_items_equal(entry.keys(), (dset['name'] for dset in datasets))


def test03_set_attributes():
    # tests the set_attributes convenience method
    arf.set_attributes(fp["entry_001/spikes"], mystr="myvalue", myint=5000)
    assert_equal(arf.get_attributes(fp["entry_001/spikes"], "myint"), 5000)
    assert_true("myint" in arf.get_attributes(fp["entry_001/spikes"]))


def test04_create_bad_dataset():
    f = raises(ValueError)(create_dataset)
    e = fp['entry_001']
    for dset in bad_datasets:
        yield f, e, dset


def test05_null_uuid():
    # nulls in a uuid can make various things barf
    from uuid import UUID
    uuid = UUID(bytes=''.rjust(16,'\0'))
    e = fp['entry_001']
    arf.set_uuid(e, uuid)

    assert_equal(arf.get_uuid(e), uuid)

def test98_creation_iter():
    fp = arf.open_file("test_mem", mode="a", driver="core", backing_store=False)
    entry_names = ('z','y','a','q','zzyfij')
    for name in entry_names:
        g = arf.create_entry(fp, name, 0)
        arf.create_dataset(g, "dset", (), sampling_rate=1)

    assert_sequence_equal(arf.keys_by_creation(fp), entry_names)


def test99_various():
    # test some functions difficult to cover otherwise
    arf.DataTypes._doc()
    arf.DataTypes._todict()


# Variables:
# End:
