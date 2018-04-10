from btb.hyper_parameter import *

# configurom btre logging for the library with a null handler (nothing is
# printed by default). See
# http://docs.pthon-guide.org/en/latest/writing/logging/
import logging

logging.getLogger('btb').addHandler(logging.NullHandler())
