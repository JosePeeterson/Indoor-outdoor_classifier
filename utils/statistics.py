import datetime
import os
import socket
from collections import OrderedDict

import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


class Statistics:
    def __init__(self, config, logger, write_outputs, time_str, no_episodes):
        self.config = config
        self.logger = logger
        self.env = None
        self.write_outputs = write_outputs
        hostname = socket.gethostname()
        self.output_folder_path = './output/' + hostname + '/' + time_str

        if not os.path.exists('./output/' + hostname):
            os.makedirs('./output/' + hostname)
        self.stats_file = self.output_folder_path + '/stats.csv'
        self.stats = OrderedDict()
        self.initialize_stat()
        self.max_episodes = no_episodes
        self.total_episodes = 0
        self.total_success = 0
        self.total_failure = 0
        self.total_timeout = 0

    def initialize_stat(self):
        for stat in [
            'status', 'time', 'speed', 'avgspd', 'n_minima', 'tdist',
            'gdist', 'avg_angle'
        ]:
            self.stats[stat] = []

    @staticmethod
    def find_local_minima(ep_speeds, theta1):
        if len(ep_speeds) < 2: return 0
        count = result = 0
        for v, _ in ep_speeds:
            if np.abs(v) < theta1:  # Implies stuck
                count += 1
                result = max(result, count)
            else:
                count = 0
        return result

    def get_stats_array(self):
        stats = self.stats
        stats_all = []
        for i in zip(*stats.values()):
            stats_all.append(list(i))
        stats_array = np.asarray(stats_all)
        return stats_array

    def store_results(self, no_step, result, ep):
        stats = self.stats
        env = self.env

        if result == 'Reach Goal':
            status = 1
        elif result == 'Crashed':
            status = -1
        else:
            status = 0
        stats['status'].append(status)
        stats['time'].append(no_step * 0.1)
        stats['speed'].append(env.total_dist / (no_step * 0.1))
        v = np.mean(np.abs(env.speed_list), axis=0)
        stats['avgspd'].append(v[0])
        stats['avg_angle'].append(v[1] * no_step * 0.1)
        count_local_minima = self.find_local_minima(env.speed_list, 0.02)
        stats['n_minima'].append(count_local_minima)
        stats['tdist'].append(env.total_dist)
        stats['gdist'].append(
            np.sqrt((env.goal_pose[0] - env.init_pose[0])**2 +
                    (env.goal_pose[1] - env.init_pose[1])**2))

        self.logger.debug(
            "Env:%d, Ep:%d, %s, NSteps:%d, Speed:%.2f, AvgSpd: %.2f, Minima: %d, Dist: %.2f, GoalDist: %.2f, AvgAngle: %.2f" % (
                env.index, ep, result, no_step, stats['speed'][-1], stats['avgspd'][-1], stats['n_minima'][-1],
                stats['tdist'][-1], stats['gdist'][-1], stats['avg_angle'][-1]))

    def print_stats(self, rank):
        stats = self.stats
        stats_array = self.get_stats_array()
        self.logger.debug('\n------------ALL STATS------------')
        self.logger.debug(stats.keys())
        self.logger.debug(stats_array)

        stats_suc = stats_array[stats_array[:, 0] == 1, 1:]
        stats_fail = stats_array[stats_array[:, 0] == 0, 1:]
        stats_crash = stats_array[stats_array[:, 0] == -1, 1:]
        n = len(stats['status'])
        
        self.logger.info(
            '-------Cummulative Env %02d---> Total: %d, Success: %d, Timeout: %d, Collision: %d'
            % (rank, self.total_episodes + n, self.total_success + len(stats_suc), self.total_timeout + len(stats_fail), self.total_failure + len(stats_crash)))
        
        if n == self.max_episodes:
            self.total_episodes += n
            self.total_success += len(stats_suc)    
            self.total_timeout += len(stats_fail)
            self.total_failure += len(stats_crash)
        
        self.logger.debug(
            "Success rate: %.2f, Timeout rate: %.3f, Collision rate: %.3f" %
            (float(len(stats_suc)) / n, float(len(stats_fail)) / n,
             float(len(stats_crash)) / n))

        if len(stats_suc):
            self.logger.debug('----Success Stats----')
            for stat, mu, sd in zip(stats.keys()[1:], np.mean(stats_suc, 0),
                                    np.std(stats_suc, 0)):
                self.logger.debug('%s: %.3f +- %.3f' % (stat, mu, sd))
        else:
            self.logger.debug('----No Success----')

        if len(stats_fail):
            self.logger.debug('----Non Success Stats----')
            for stat, mu, sd in zip(stats.keys()[1:], np.mean(stats_fail, 0),
                                    np.std(stats_fail, 0)):
                self.logger.debug('%s: %.3f +- %.3f' % (stat, mu, sd))
        else:
            self.logger.debug('----No Timeouts----')

        if len(stats_crash):
            self.logger.debug('----Collision Stats----')
            for stat, mu, sd in zip(stats.keys()[1:], np.mean(stats_crash, 0),
                                    np.std(stats_crash, 0)):
                self.logger.debug('%s: %.3f +- %.3f' % (stat, mu, sd))
        else:
            self.logger.debug('----No Collisions----')

    def write_all(self, policy, scene_desc, stageros):
        if self.write_outputs:
            self.write_csv(policy, scene_desc, stageros)
            self.write_data(self.get_stats_array(), scene_desc)

    def write_csv(self, policy, scene_desc, stageros):
        if self.write_outputs:
            stats = self.stats
            if not os.path.exists(self.stats_file):
                stats_wtf = open(self.stats_file, 'w')
                stats_wtf.write(
                    "Date and Time,Policy,Stage,Scene,Success-rate,Timeout-rate,Collision-rate,Status,Time-taken,"
                    "Speed,Avg-Speed,N-minima,Travel-dist,Goal-dist,Avg-Angle\n"
                )
            else:
                stats_wtf = open(self.stats_file, 'a')

            stats_array = self.get_stats_array()
            date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            stats_wtf.write("%s, %s, %s, %s ," %
                            (date, policy, stageros, scene_desc))

            stats_suc = stats_array[stats_array[:, 0] == 1, 1:]
            stats_fail = stats_array[stats_array[:, 0] == 0, 1:]
            stats_crash = stats_array[stats_array[:, 0] == -1, 1:]
            n = len(stats['status'])

            succces_rate = float(len(stats_suc)) / n
            timeout_rate = float(len(stats_fail)) / n
            collision_rate = float(len(stats_crash)) / n
            stats_wtf.write("%.2f, %.3f, %.3f," %
                            (succces_rate, timeout_rate, collision_rate))

            if len(stats_suc):
                stats_wtf.write("Success,")
                for stat, mu, sd in zip(stats.keys()[1:],
                                        np.mean(stats_suc, 0),
                                        np.std(stats_suc, 0)):
                    stats_wtf.write('%.3f +- %.3f' % (mu, sd) + ",")
            else:
                stats_wtf.write("Success,,,,,,")

            if len(stats_fail):
                stats_wtf.write("\n,,,,,,,Timeout,")
                for stat, mu, sd in zip(stats.keys()[1:],
                                        np.mean(stats_fail, 0),
                                        np.std(stats_fail, 0)):
                    stats_wtf.write('%.3f +- %.3f' % (mu, sd) + ",")
            else:
                stats_wtf.write("\n,,,,,,,Timeout,,,,,,")

            if len(stats_crash):
                stats_wtf.write("\n,,,,,,,Collision,")
                for stat, mu, sd in zip(stats.keys()[1:],
                                        np.mean(stats_crash, 0),
                                        np.std(stats_crash, 0)):
                    stats_wtf.write('%.3f +- %.3f' % (mu, sd) + ",")
            else:
                stats_wtf.write("\n,,,,,,,Collision,,,,,,")
            stats_wtf.write("\n")
            stats_wtf.close()

    def write_data(self, data, scene_desc):
        save_folder = '/data/'
        filename = '%s' % scene_desc
        self.logger.debug('Data File Logged: %s' % filename)

        folder_path = self.output_folder_path + save_folder
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        results_file = folder_path + filename + '.pkl'

        try:
            import cPickle as pickle
        except ImportError:  # python 3.x
            import pickle

        with open(results_file, 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)