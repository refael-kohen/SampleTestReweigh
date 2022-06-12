import logging.handlers
import os



class Logger:
    def __init__(self, rep_counter='', v=None):
        cntstr = ''
        if rep_counter:  # in main log or in non-parallel no print number
            cntstr = str(rep_counter).zfill(5) + '_' if v.RunV.multiproc_num else ''
        log_file_name = os.path.join(v.PathV.log_path, cntstr + v.PathV.date + '_' + str(v.SampCompV.sample_size_s) + '.log')
        self.formatter = logging.Formatter(
            "%(asctime)s [%(filename)-20.20s] [%(lineno)-3d] [%(levelname)-5.5s]  %(message)s")
        self._g_logger = self.setup_logger(logger_name=cntstr + 'general_logger', log_file=log_file_name,
                                           level=v.RunV.log_level, stdout=True,
                                           level_stdout=v.RunV.log_level_stdout)

        log_file_name = os.path.join(v.PathV.log_path, cntstr + v.PathV.date + '_' + str(v.SampCompV.sample_size_s) + '_weights.log')
        self._weights_logger = self.setup_logger(logger_name=cntstr + 'weights_logger', log_file=log_file_name,
                                                 level=v.RunV.log_level, stdout=False)

    def info(self, msg):
        self._g_logger.info(msg)

    def debug(self, msg):
        self._g_logger.debug(msg)

    def gwinfo(self, msg):
        self._g_logger.info(msg)
        self._weights_logger.info(msg)

    def gwdebug(self, msg):
        self._g_logger.debug(msg)
        self._weights_logger.debug(msg)

    def winfo(self, msg):
        self._weights_logger.info(msg)

    def wdebug(self, msg):
        self._weights_logger.debug(msg)

    def setup_logger(self, logger_name, log_file, level, stdout=False, level_stdout=None):
        """To setup as many loggers as you want"""

        handler_file = logging.FileHandler(log_file)
        handler_file.setFormatter(self.formatter)
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.addHandler(handler_file)
        if stdout:
            handler_stream = logging.StreamHandler()
            handler_stream.level = level_stdout
            handler_stream.setFormatter(self.formatter)
            logger.addHandler(handler_stream)
        return logger

    @staticmethod
    def listener_configurer():
        f = logging.Formatter(
            "%(asctime)s [%(filename)-20.20s] [%(lineno)-3d] [%(levelname)-5.5s]  %(message)s")
        root = logging.getLogger()
        # h = logging.handlers.RotatingFileHandler('mptest.log', 'a', 300, 10)
        h = logging.handlers.RotatingFileHandler('mptest.log')
        h.setFormatter(f)
        root.addHandler(h)

    @staticmethod
    def listener_process(queue, configurer):
        configurer()
        while True:
            try:
                record = queue.get()
                if record is None:  # We send this as a sentinel to tell the listener to quit.
                    break
                logger = logging.getLogger(record.name)
                logger.handle(record)  # No level or filter logic applied - just do it!
            except Exception:
                import sys, traceback
                print('Whoops! Problem:', file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

    @staticmethod
    def worker_configurer(queue):
        h = logging.handlers.QueueHandler(queue)  # Just the one handler needed
        root = logging.getLogger()
        root.addHandler(h)
        # send all messages, for demo; no other level or filter logic applied.
        root.setLevel(logging.DEBUG)
