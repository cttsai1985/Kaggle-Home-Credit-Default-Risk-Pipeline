import logging
formatter = '%(asctime)s %(filename)s:%(lineno)d: %(message)s'
logging.basicConfig(format=formatter, level='INFO')
logger = logging.getLogger(__name__)

import sys, os
sys.path.insert(0, './lib')
sys.path.insert(0, './external')


logger.info('working directory: {}'.format(os.getcwd()))
for p in sys.path[:sys.path.index(os.getcwd())]:
    logger.info('expend to {}'.format(p))

from lib.Utility import MkDirSafe
from lib.LibConfigs import file_dir_path

for path in file_dir_path:
    MkDirSafe(path)
