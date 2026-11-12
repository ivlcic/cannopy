from .zero_shot import predict as zero_shot
from .rae_xmc import predict as rae_xmc

# Methods return an (n_test, n_labels) score matrix in the supplied label order.
# Add future methods here with the same predict signature.
METHODS = {'zero_shot': zero_shot, 'rae_xmc': rae_xmc}
