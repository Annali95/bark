from setuptools import setup

setup(name='bark',
      version='0.2',
      description='tools for reading and writing Bark formatted data',
      url='http://github.com/kylerbrown/bark',
      author='Kyler Brown',
      author_email='kylerjbrown@gmail.com',
      license='GPL',
      packages=['bark',
                'bark.tools',
                'bark.io',
                'bark.io.rhd',
                'bark.io.openephys', ],
      zip_safe=False,
      entry_points={
          'console_scripts': [
              'bark-entry=bark.tools.barkutils:mk_entry',
              'bark-attribute=bark.tools.barkutils:meta_attr',
              'bark-column-attribute=bark.tools.barkutils:meta_column_attr',
              'bark-clean-orphan-metas=bark.tools.barkutils:clean_metafiles',
              'bark-scope=bark.tools.barkscope:main',
              'csv-from-waveclus=bark.io.waveclus:_waveclus2csv',
              'csv-from-textgrid=bark.io.textgrid:textgrid2csv',
              'csv-from-lbl=bark.io.lbl:_lbl_csv',
              'csv-from-plexon-csv=bark.io.plexon:_plexon_csv_to_bark_csv',
              'bark-convert-rhd=bark.io.rhd.rhd2bark:bark_rhd_to_entry',
              'bark-convert-openephys=bark.io.openephys.kwik2dat:kwd_to_entry',
              'bark-convert-arf=bark.io.arf2bark:_main',
              'bark-db=bark.io.db:_run',
              'dat-decimate=bark.tools.barkutils:rb_decimate',
              'dat-resample=bark.tools.barkutils:rb_resample',
              'dat-select=bark.tools.barkutils:rb_select',
              'dat-cat=bark.tools.barkutils:rb_concat',
              'dat-join=bark.tools.barkutils:rb_join',
              'dat-segment=bark.tools.datsegment:_run',
              'dat-filter=bark.tools.barkutils:rb_filter',
              'dat-diff=bark.tools.barkutils:rb_diff',
              'dat-ref=bark.tools.datref:main',
              'dat-artifact=bark.tools.datartifact:main',
              'dat-enrich=bark.tools.datenrich:main',
              'dat-spike-detect=bark.tools.datspike:_run',
              'dat-envelope-classify=bark.tools.datenvclassify:_run',
              'dat-split=bark.tools.barkutils:_datchunk',
              'dat-to-audio=bark.tools.barkutils:rb_to_audio',
              'dat-to-wave-clus=bark.tools.barkutils:rb_to_wave_clus',
              'bark-label-view=bark.tools.labelview:_run',
              'bark-for-each=bark.tools.barkforeach:_main',
          ]
      })
