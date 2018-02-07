from __future__ import absolute_import
from .hyper_parameter import *

# configure logging for the library with a null handler (nothing is printed by
# default). See http://docs.python-guide.org/en/latest/writing/logging/
import logging

logging.getLogger('btb').addHandler(logging.NullHandler())
