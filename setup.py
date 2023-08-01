from setuptools import setup, find_packages

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name = "photcalib",
    version = 0.2,
    description = "calibration tool for photometric data",
    long_description = readme(),
    long_description_content_type="text/x-rst",
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