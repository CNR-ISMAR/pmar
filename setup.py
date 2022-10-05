import os
from distutils.core import setup


def read(*rnames):
    return open(os.path.join(os.path.dirname(__file__), *rnames)).read()

setup(
    name="pmar",
    version="0.0.1",
    author="Sofia Bosi",
    author_email="sofia.bosi@ismar.cnr.it",
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
    install_requires=[]
)
