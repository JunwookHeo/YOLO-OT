import logging
import datetime
import time

class ElapsedFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
        self.start_time = time.time()
    
    def formatTime(self, record, datefmt=None):
        ct = time.gmtime(time.time() - self.start_time)
        if datefmt:
            s = time.strftime(datefmt, ct)
            s = '{:03d}:{:s}'.format((ct.tm_yday - 1)*24+ct.tm_hour, s)
        else:
            t = time.strftime(self.default_time_format, ct)
            s = self.default_msec_format % (t, record.msecs)
        return s

__handler = logging.StreamHandler()
__handler.setFormatter(ElapsedFormatter('%(asctime)s.%(msecs)03d %(levelname)s \t%(message)s', '%M:%S'))

logger = logging.getLogger(__name__)
logger.addHandler(__handler)
logger.setLevel(logging.DEBUG)