.. image:: https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow
    :target: https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha
    :alt: Development Status Shield

.. image:: https://img.shields.io/pypi/v/baytune.svg
    :target: https://pypi.python.org/pypi/baytune
    :alt: PyPI Shield

.. image:: https://travis-ci.org/HDI-Project/BTB.svg?branch=master
    :target: https://travis-ci.org/HDI-Project/BTB
    :alt: Travis CI Shield

.. image:: https://codecov.io/gh/HDI-Project/BTB/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/HDI-Project/BTB
    :alt: Codecov Shield

.. image:: https://pepy.tech/badge/baytune
    :target: https://pepy.tech/project/baytune
    :alt: Downloads Shield

|

.. image:: images/BTB-Icon-small.png
   :alt: BTB Logo
   :align: center
   :target: https://github.com/HDI-Project/BTB

.. centered:: A simple, extensible backend for developing auto-tuning systems

Overview
========

* License: `MIT <https://github.com/HDI-Project/BTB/blob/master/LICENSE>`_
* Development Status: `Pre-Alpha <https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha>`_
* Documentation: https://HDI-Project.github.io/BTB
* Homepage: https://github.com/HDI-Project/BTB

BTB ("Bayesian Tuning and Bandits") is a simple, extensible backend for developing auto-tuning
systems such as AutoML systems. It provides an easy-to-use interface for *tuning* and *selection*.

It is currently being used in several AutoML systems:
* `ATM <https://github.com/HDI-Project/ATM>`_, distributed, multi-tenant AutoML system for
classifier tuning
* `mit-d3m-ta2 <https://github.com/HDI-Project/mit-d3m-ta2/>`_, MIT's system for the DARPA
`Data-driven discovery of models <https://www.darpa.mil/program/data-driven-discovery-of-models>`_ (D3M) program
* `AutoBazaar <https://github.com/HDI-Project/AutoBazaar>`_, a flexible, general-purpose
AutoML system

History
-------

In its first iteration, in 2018, BTB was designed as an open source library that handles
the problems of tuning the hyperparameters of a machine learning pipeline, selecting
between multiple pipelines and recommending a pipeline. A good reference to see our design
rationale at that time is Laura Gustafson’s thesis, written under the supervision of
Kalyan Veeramachaneni:

* `Bayesian Tuning and Bandits`_.
  Laura Gustafson. Masters thesis, MIT EECS, 2018.

Later in 2018, Carles Sala joined the project to make it grow as a reliable open-source library
that would become part of a bigger software ecosystem designed to facilitate the development of
robust end-to-end solutions based on Machine Learning tools. This second iteration of our work
was presented in 2019 as part of the Machine Learning Bazaar:

* `The Machine Learning Bazaar: Harnessing the ML Ecosystem for Effective System Development`_.
  Micah J. Smith, Carles Sala, James Max Kanter, and Kalyan Veeramachaneni. Sigmod 2020.

.. toctree::
   :hidden:
   :maxdepth: 1

   Overview<self>
   install
   tutorials/00_Quickstart

.. toctree::
   :caption: User Guides
   :maxdepth: 1

   tutorials/01_Tuning
   tutorials/02_Selection
   tutorials/03_Session
   benchmark
   kubernetes

.. toctree::
   :caption: Resources
   :maxdepth: 1
   :titlesonly:

   API Reference <api/btb>
   contributing
   history
   authors

Indices and tables
==================
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Bayesian Tuning and Bandits: https://dai.lids.mit.edu/wp-content/uploads/2018/05/Laura_MEng_Final.pdf

.. _The Machine Learning Bazaar\: Harnessing the ML Ecosystem for Effective System Development: https://arxiv.org/abs/1905.08942
