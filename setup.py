from setuptools import setup, find_packages

install_requires = ['SQLAlchemy==1.1.9', 'GeoAlchemy2==0.4.0', 'SQLAlchemy==1.1.9', 'SQLAlchemy-Utils==0.32.14']

setup(
    author="Grahame Bowland",
    author_email="grahame@angrygoats.net",
    description="EAlGIS common functionality (loaders and backend)",
    long_description="EAlGIS data schema",
    license="GPL3",
    keywords="ealgis",
    url="https://github.com/ealgis/ealgis-data-schema",
    name="ealgis_common",
    version="0.6.0",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=install_requires,
)
