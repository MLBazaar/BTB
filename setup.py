#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

requirements = [
    'enum34>=1.1.6; python_version=="2.7"',
    'numpy>=1.14.2',
    'scikit-learn>=0.19.1',
    'scipy>=1.0.1'
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author='MIT Data To AI Lab',
    author_email='dailabmit@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description='Bayesian Tuning and Bandits',
    install_requires=requirements,
    license='MIT license',
    long_description=readme + '\n\n' + history,
    long_description_content_type='text/markdown',
    include_package_data=True,
    keywords='machine learning hyperparameters tuning classification',
    name='baytune',
    packages=find_packages(include=['btb']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/HDI-Project/BTB',
    version='0.1.1',
    zip_safe=False,
)
