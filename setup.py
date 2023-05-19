from setuptools import setup, find_packages

def readme():
    with open('README.md') as f:
        return f.read()

setup(name = "photcalib",
    version = 1.0,
    description = "calibration tool for photometric data",
    long_description = readme(),
    author = "Zhen Yuan",
    author_email = "",
    url = "https://github.com/zyuan-astro/PhotCalib",
    packages = find_packages(),
    package_data = {},
    include_package_data = True,
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Science/Research',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      'Topic :: Scientific/Engineering :: Astronomy'
      ],
    zip_safe=False
)