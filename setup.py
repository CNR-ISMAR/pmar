import os
from setuptools import setup


def read(*rnames):
    """Read and return the contents of a file located relative to this file's directory."""
    return open(os.path.join(os.path.dirname(__file__), *rnames)).read()

setup(
    name="pmar",
    version="0.15.0",
    author="Sofia Bosi",
    author_email="sofiabosi@cnr.it",
    description="Pressure models for marine activities",
    long_description=(read('README.md')),
    # Full list of classifiers can be found at:
    # http://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
    ],
    license="GPL3",
    keywords="Lagrangian dispersion, Pressure models for marine activities",
    url='https://github.com/CNR-ISMAR/pmar',
    packages=['pmar',],
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.11',  # the real floor across this dependency set is networkx's >=3.11
    install_requires=[
        'cartopy', 'cmap', 'geocube', 'geopandas', 'matplotlib', 'networkx',
        'numpy', 'opendrift', 'owslib', 'pandas', 'pydap', 'rasterio',
        'requests', 'rioxarray', 'scipy', 'seaborn', 'shapely', 'xarray',
        'xhistogram',
    ],
)
