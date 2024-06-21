from utils import *
# used to track results
def initiate_log(PATH):
    logging.basicConfig(filename=PATH + "20240529_finalrun.log",
                        format='%(asctime)s %(message)s',
                        level=logging.INFO)

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)

    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)

    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    logger = logging.getLogger(__name__)
    return logger