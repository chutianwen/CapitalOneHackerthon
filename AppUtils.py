import logging
from logging.config import dictConfig

logging_config = dict(
    version=1,
    formatters={
        'f': {'format': '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'},
        'f1': {'format': '%(asctime)s %(levelname)-8s %(message)s'}
    },
    handlers={
        'sh': {'class': 'logging.StreamHandler',
               'formatter': 'f',
               'level': logging.FATAL
               },
        'fh': {'class': 'logging.FileHandler',
               'filename': './log/DsLog',
               'formatter': 'f1',
               'level': logging.DEBUG
               }
    },
    root={
        'handlers': ['sh', 'fh'],
        'level': logging.DEBUG,
    },
)
dictConfig(logging_config)
logger = logging.getLogger()

class TaskReporter(object):
    def __init__(self, task):
        self.task = task

    def __call__(self, original_func):
        decorator_self = self

        def wrapper(*args, **kwargs):
            logger.info("Processing task: {}...".format(decorator_self.task))
            res = original_func(*args, **kwargs)
            logger.info("Task: {} is done".format(decorator_self.task))
            return res
        return wrapper