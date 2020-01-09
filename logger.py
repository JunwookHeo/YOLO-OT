import logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s.%(msecs)03d %(levelname)s \t%(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

