import numpy as np
from geometry_msgs.msg import Pose
from sensor_msgs.msg import LaserScan
import time
import rospy
import tf
import time
import pickle

class dataset():

    def __init__(self):
        rospy.init_node('dataset', anonymous=None)
        self.cmd_pose = rospy.Publisher('robot_0/cmd_pose', Pose, queue_size=10)
        self.scan = None
        self.full_data = []
        rospy.Subscriber('robot_0/base_scan', LaserScan, self.scan_callback)

    def scan_callback(self,data):
        self.scan = np.array(data.ranges)

    def control_pose(self,func, pose):
        pose_cmd = Pose()
        assert len(pose) == 3
        pose_cmd.position.x = pose[0]
        pose_cmd.position.y = pose[1]
        pose_cmd.position.z = 0

        qtn = tf.transformations.quaternion_from_euler(0, 0, pose[2], 'rxyz')
        pose_cmd.orientation.x = qtn[0]
        pose_cmd.orientation.y = qtn[1]
        pose_cmd.orientation.z = qtn[2]
        pose_cmd.orientation.w = qtn[3]
        func.publish(pose_cmd)      
    
    def save_scans_plus(self):
        Angles = [float(2*n*(np.pi))/10 for n in range(1,11)]  # /10 (1,11)
        env = 1 # 1 indoor, 0 outdoor
        while not rospy.is_shutdown():
            for x in range (-2,3): #-2,3
                y = 0
                z = x                   
                for a in Angles:    
                    if (z >= 0):       
                        y = -z          
                        x = 0           
                    self.control_pose(self.cmd_pose, [4*x, 4*y, a])
                    time.sleep(0.1)
                    #print(self.scan)
                    self.scan_env = np.append(self.scan,env)
                    #print(len(self.scan_env))
                    self.full_data.append(self.scan_env)
            for x in range (-2,3): #-2,3
                y = 0
                z = x                  
                for a in Angles:    
                    if (z <= 0):       
                        y = -z          
                        x = 0          
                    self.control_pose(self.cmd_pose, [4*x, 4*y, a])
                    time.sleep(0.1)
                    #print(self.scan)
                    self.scan_env = np.append(self.scan,env)
                    #print(len(self.scan_env))
                    self.full_data.append(self.scan_env)
            break
        #print(len(self.full_data[8]))
        results_file = 'plus_corridor' # saved pickle file.
        with open(results_file,'wb') as fp:
            pickle.dump(self.full_data,fp)
        file.close(fp)

    def classify(self):
        tot = 0
        for s in self.full_data:
            s = np.array(s) 
            s = s[:len(s)-1] # remove the label
            tot = tot + len(s[s<5])
        if float(tot)/(len(self.full_data)*360) < 0.5:
            print('open env')
        else:
            print('closed env')
'''
    def save_scans_others(self):
        Angles = [float(2*n*(np.pi))/10 for n in range(1,11)]  # /10 (1,11)
        env = 0 # 1 indoor, 0 outdoor
        while not rospy.is_shutdown():
            for x in range (-4,5): #-2,3
                y = 0
                for a in Angles:    
                    self.control_pose(self.cmd_pose, [2*x, y, a])
                    time.sleep(0.1)
                    #print(self.scan)
                    self.scan_env = np.append(self.scan,env)
                    #print(len(self.scan_env))
                    self.full_data.append(self.scan_env)
            break
        #print(len(self.full_data[8]))
        results_file = 'corridor' # saved pickle file.
        with open(results_file,'wb') as fp:
            pickle.dump(self.full_data,fp)
        file.close(fp)

    def save_scans_l(self):
        Angles = [float(2*n*(np.pi))/10 for n in range(1,11)]  # /10 (1,11)
        env = 1 # 1 indoor, 0 outdoor
        while not rospy.is_shutdown():
            for x in range (-4,5): #-2,3
                y = 0
                z = x                   # only for l-corridor
                for a in Angles:    
                    if (z >= 0):        # only for l-corridor
                        y = z           # only for l-corridor
                        x = 0           # only for l-corridor
                    self.control_pose(self.cmd_pose, [2*x, 2*y, a])
                    time.sleep(0.1)
                    #print(self.scan)
                    self.scan_env = np.append(self.scan,env)
                    #print(len(self.scan_env))
                    self.full_data.append(self.scan_env)
            break
        #print(len(self.full_data[8]))
        results_file = 'l_corridor' # saved pickle file.
        with open(results_file,'wb') as fp:
            pickle.dump(self.full_data,fp)
        file.close(fp)
'''

if __name__ == '__main__':
    d = dataset()
    d.save_scans_plus()
    d.classify()








