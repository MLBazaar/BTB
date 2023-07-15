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


def github_dependency(user, name, commit):
    return f'{name} @ git+https://github.com/{user}/{name}@{commit}#egg={name}'

github_req =  github_dependency(
    'csala', 'dask-kubernetes', 'issue-170-ssl-error-when-cleaning-up-pods')

install_requires = [
    'dask>=2.15.0,<3',
    'kubernetes>=11.0.0,<11.1',
    'distributed>=2.15.2,<2.16',
    'hyperopt>=0.2.3,<3',
    'tabulate>=0.8.3,<0.9',
    'xgboost>=1.0.2,<1.1.0',
    'docutils>=0.10,<0.16',
    'boto3>=1.9.18,<1.10',
    'urllib3<1.26,>=1.20',
    'numpy>=1.14.0',
    'scikit-learn>=0.20.0',
    'pandas>=1,<2',
    'XlsxWriter>=1.2.8,<1.3',
    github_req,
    'ax-platform>=0.1.9,<0.1.13',
    'configspace==0.4.12',
    'smac>=0.12.1,<0.13',
    'scikit-optimize>=0.7.4,<0.9',
    'emcee>=2.1.0,<3',
    'pyDOE>=0.3.8<0.4',
]


examples_require = [
    'jupyter>=1.0.0',
    'matplotlib>=3.1.1',
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
    'bumpversion>=0.5.3',
    'pip>=9.0.1',
    'watchdog>=0.8.3',

    # docs
    'autodocsumm>=0.1.10',
    'ipython>=6.5.0',
    'm2r>=0.2.0',
    'Sphinx>=1.7.1,<2.4',
    'sphinx_rtd_theme>=0.2.4',

    # style check
    'flake8>=3.7.7',
    'isort>=4.3.4',

    # fix style issues
    'autoflake>=1.2',
    'autopep8>=1.4.3',

    # distribute on PyPI
    'twine>=1.10.0',
    'wheel>=0.30.0',

    # Advanced testing
    'coverage>=4.5.1',
    'tox>=2.9.1,<4',
    'importlib-metadata<2.0.0,>=0.12'
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
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description='Bayesian Tuning and Bandits',
    entry_points={
        'console_scripts': [
            'btb_benchmark=btb_benchmark.__main__:main'
        ],
    },
    extras_require={
        'examples': examples_require,
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords='machine learning hyperparameters tuning classification',
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    name='btb_benchmark',
    packages=find_packages(include=['btb_benchmark', 'btb_benchmark.*']),
    python_requires='>=3.6,<3.9',
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/HDI-Project/BTB',
    version='0.4.1.dev0',
    zip_safe=False,
)
