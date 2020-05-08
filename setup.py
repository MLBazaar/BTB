#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup


try:
    with open('README.md') as readme_file:
        readme = readme_file.read()

except IOError:
    readme = ''


try:
    with open('HISTORY.md') as history_file:
        history = history_file.read()

except IOError:
    history = ''


install_requires = [
    'numpy>=1.14.0,<1.18.0',
    'scikit-learn>0.20.0,<0.23',
    'scipy>=1.0.1,<1.4.0',
    'pandas>=0.21.0,<0.26.0',
    'tqdm>=4.36.1,<4.50.0',
]


tests_require = [
    'pytest>=3.4.2',
    'pytest-cov>=2.6.0',
]


setup_requires = [
    'pytest-runner>=2.11.1',
]


development_requires = [
    # general
    'pip>=9.0.1',
    'bumpversion>=0.5.3,<0.6',
    'watchdog>=0.8.3,<0.11',

    # docs
    'm2r>=0.2.0,<0.3',
    'nbsphinx>=0.5.0,<0.7',
    'Sphinx>=1.7.1,<3',
    'sphinx_rtd_theme>=0.2.4,<0.5',

    # style check
    'flake8>=3.7.7,<4',
    'isort>=4.3.4,<5',

    # fix style issues
    'autoflake>=1.1,<2',
    'autopep8>=1.4.3,<2',

    # distribute on PyPI
    'twine>=1.10.0,<4',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1,<6',
    'tox>=2.9.1,<4',

    # others
    'jupyter>=1.0.0',
]


setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    description='Bayesian Tuning and Bandits',
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='machine learning hyperparameters tuning classification',
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    name='baytune',
    packages=find_packages(include=['btb', 'btb.*']),
    python_requires='>=3.5',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/HDI-Project/BTB',
    version='0.3.8',
    zip_safe=False,
)
