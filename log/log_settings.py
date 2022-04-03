#!/root/anaconda3/bin/python

import logging
import logging.handlers


class Logger():
    level_relations = {
        'debug':logging.DEBUG,
        'info':logging.INFO,
        'warning':logging.WARNING,
        'error':logging.ERROR,
        'critical':logging.CRITICAL
    }
    
    def __init__(self,filename,level = 'info',when = 'D',backupCount = 60,fmt = '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        formatter = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations[level])
        
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        
        th = logging.handlers.TimedRotatingFileHandler(filename = filename,when = when,backupCount = backupCount,encoding = 'utf-8')
        th.setFormatter(formatter)
        
        self.logger.addHandler(sh)
        self.logger.addHandler(th)
 
 
if __name__ == "__main__":
    
    pass