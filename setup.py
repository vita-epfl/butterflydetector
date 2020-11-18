from setuptools import setup
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ImportError:
    cythonize = None
try:
    import numpy
except ImportError:
    numpy = None


# extract version from __init__.py
with open('butterflydetector/__init__.py', 'r') as f:
    VERSION_LINE = [l for l in f if l.startswith('__version__')][0]
    VERSION = VERSION_LINE.split('=')[1].strip()[1:-1]


class NumpyIncludePath(object):
    """Lazy import of numpy to get include path."""
    @staticmethod
    def __str__():
        import numpy
        return numpy.get_include()


if cythonize is not None and numpy is not None:
    EXTENSIONS = cythonize([Extension('butterflydetector.functional',
                                      ['butterflydetector/functional.pyx'],
                                      include_dirs=[numpy.get_include()]),
                           ],
                           annotate=True,
                           compiler_directives={'language_level': 3})
else:
    EXTENSIONS = [Extension('butterflydetector.functional',
                            ['butterflydetector/functional.pyx'],
                            include_dirs=[NumpyIncludePath()])]


setup(
    name='butterflydetector',
    version=VERSION,
    packages=[
        'butterflydetector',
        'butterflydetector.decoder',
        'butterflydetector.decoder.generator',
        'butterflydetector.encoder',
        'butterflydetector.network',
        'butterflydetector.transforms',
        'butterflydetector.data_manager',
    ],
    license='GNU AGPLv3',
    description='Butterfly Detector for Aerial Images',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='George Adaimi',
    author_email='georgeadaimi@gmail.com',
    url='https://github.com/vita-epfl/butterflydetector',
    ext_modules=EXTENSIONS,
    zip_safe=False,

    install_requires=[
        'numpy>=1.16',
        'pysparkling',  # for log analysis
        'python-json-logger',
        'scipy',
        'pandas',
        'torch>=1.1.0',
        'torchvision>=0.3',
        'yacs',
        'matplotlib',
        'pillow<7',  # temporary compat requirement for torchvision
    ],
    extras_require={
        'onnx': [
            'onnx',
            'onnx-simplifier',
        ],
        'test': [
            'pylint',
            'pytest',
            'opencv-python',
        ],
        'train': [
            'matplotlib',
            'torch>=1.3.0',
            'torchvision>=0.4',
        ],
    },
)
