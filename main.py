import shutil
import signal
import socket
import sys
from datetime import datetime

from omegaconf import OmegaConf

from agent import Agent
from model.ppo import PPO
from parallelAgent import ParallelAgent
from utils.logger import init_logger

launch_time = None
time_str = None


def signal_handler(signal, frame):
    clear_temp(launch_time, time_str)
    sys.exit()


def clear_temp(launch_time, time_str):
    curr_time = datetime.now()
    hostname = socket.gethostname()
    exe_time = (curr_time - launch_time).total_seconds()
    output_folder = './output/' + hostname + '/' + time_str
    if exe_time < 60 * 5:
        print '\n\nExecution time(mins): %f \nDeleted output folder: %s \n' % (exe_time/60, output_folder)
        shutil.rmtree(output_folder)


def main(args):
    global launch_time
    global time_str
    launch_time = datetime.now()
    logger, time_str = init_logger(debug_flag=args.debug)
    agent_class = ParallelAgent if args.use_parallel_agent else Agent
    agent_class.time_str = time_str
    agent = agent_class(main_config=OmegaConf.load(args.config),
                        world_type=args.world,
                        eval_flag=args.eval,
                        logger=logger,
                        write_outputs=args.save)
    
    policy = PPO(config=agent.config,
                 eval_flag=args.eval,
                 logger=logger,
                 debug_flag=args.debug) if agent.rank == 0 else None

    try:
        agent.run(policy=policy)
    except KeyboardInterrupt:
        pass
    clear_temp(launch_time, time_str)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--world',
                        required=True,
                        help='Enter world type (same as .world file)\
                        e.g. python main.py --w circle')
    parser.add_argument('--config',
                        default='./configs/main_config.yaml',
                        help='Specify relative path to config folder')
    parser.add_argument('--debug',
                        action='store_true',
                        help='Enable debug mode for verbosity')
    parser.add_argument('--eval',
                        action='store_true',
                        help='Enable eval mode, off exploration \
                        & model update')
    parser.add_argument('--save',
                        action='store_true',
                        help='Enable saving of logs, results & data')
    parser.add_argument('--use_parallel_agent',
                        action='store_true',
                        help='Use parallel agent for scene_open')
    args = parser.parse_args()
    if args.use_parallel_agent and args.world != 'scene_open':
        raise NotImplementedError
    
    signal.signal(signal.SIGQUIT, signal_handler)
    
    main(args)
