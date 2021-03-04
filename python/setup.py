from distutils.core import setup
from distutils.extension import Extension
import os
import sys
import platform

openmm_dir = '@OPENMM_DIR@'
constvplugin_header_dir = '@CONSTVPLUGIN_HEADER_DIR@'
constvplugin_library_dir = '@CONSTVPLUGIN_LIBRARY_DIR@'

# setup extra compile and link arguments on Mac
extra_compile_args = []
extra_link_args = []

if platform.system() == 'Darwin':
    extra_compile_args += ['-std=c++11', '-mmacosx-version-min=10.7']
    extra_link_args += ['-std=c++11', '-mmacosx-version-min=10.7', '-Wl', '-rpath', openmm_dir+'/lib']

extension = Extension(name='_constvplugin',
                      sources=['ConstVPluginWrapper.cpp'],
                      libraries=['OpenMM', 'OpenMMConstV'],
                      include_dirs=[os.path.join(openmm_dir, 'include'), constvplugin_header_dir],
                      library_dirs=[os.path.join(openmm_dir, 'lib'), constvplugin_library_dir],
                      extra_compile_args=extra_compile_args,
                      extra_link_args=extra_link_args
                     )

setup(name='constvplugin',
      version='1.0',
      py_modules=['constvplugin'],
      ext_modules=[extension],
      packages=["mdtools"],
      data_files= [],
      package_data={"mdtools" : []}
     )
