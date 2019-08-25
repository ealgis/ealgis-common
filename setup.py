from setuptools import setup, find_packages

install_requires = ['SQLAlchemy==1.3.7', 'GeoAlchemy2==0.6.3', 'SQLAlchemy-Utils==0.34.2']

setup(
    author="Grahame Bowland",
    author_email="grahame@oreamnos.com.au",
    description="EAlGIS common functionality (loaders and backend)",
    long_description="EAlGIS data schema",
    license="GPL3",
    keywords="ealgis",
    url="https://github.com/ealgis/ealgis-common",
    name="ealgis_common",
    version="0.8.4",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=install_requires,
)
