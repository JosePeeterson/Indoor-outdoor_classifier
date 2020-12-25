import socket
import os
import sys
import logging
import time
import shutil

def init_logger(debug_flag=False,
                log_name='output',
                print_screen=True,
                print_file=True,
                log_dir='./output/'):

    hostname = socket.gethostname()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    output_folder_path = './output/' + hostname + '/' + timestr

    if not os.path.exists(output_folder_path + '/'):
        os.makedirs(output_folder_path + '/')  
    
    if print_file:
        output_file = '%s%s/%s/%s.log' % (log_dir, hostname, timestr, log_name)
    else:
        output_file = '%s%s/%s/%s.log' % (log_dir, hostname, 'temp', log_name)

    src_fold = './configs/main_config.yaml'
    dest_fold = output_folder_path
    shutil.copy(src_fold, dest_fold)

    # config log
    logger = logging.getLogger('logger_%s' % log_name)
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(output_file, mode='a')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    stdout_handler = logging.StreamHandler(sys.stdout)

    if debug_flag:
        stdout_handler.setLevel(logging.DEBUG)
    else:
        stdout_handler.setLevel(logging.INFO)
        if print_file:
            logger.addHandler(file_handler)

    if print_screen:
        logger.addHandler(stdout_handler)

    return logger, timestr