import logging
reload(logging)
import scipy

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())


def call_logger():
    print_messages(verbose=50, logger=logger)


def print_messages(verbose=0, logger=None):
    logger = logger or logging.getLogger(__name__)
    logger.setLevel(50 - verbose)
    logger.debug("debugging %4i, %2.25f", 42, scipy.pi)
    logger.warn("informative comment")
    logger.debug("debug information")
    logger.exception("Auch, error!")
