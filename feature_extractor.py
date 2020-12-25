
import rospy
import time
from itertools import combinations
import numpy as np
from collections import deque  
import queue
import matplotlib.pyplot as plt
import pickle

from geometry_msgs.msg import Point
from std_msgs.msg import String
from obstacle_detector.msg import Obstacles
from sensor_msgs.msg import LaserScan

class  Listener():

    def __init__(self):
        rospy.init_node('listener', anonymous = True)
        rospy.Subscriber('/obstacles', Obstacles, self.obstacle_callback)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.scan = None
        self.feat_vec = [0]*8
        self.full_data = []
    

    def obstacle_callback(self,Obstacles):
        self.feat_vec = [0]*8   
        self.all_segments = Obstacles.segments	
        self.all_circles = Obstacles.circles
        #self.feat_vec[0] = 1  if len(self.all_segments) != 0 else 0             # line segment present
        self.feat_vec[0] = len(self.all_segments)                               # Number of line segments
        self.all_segments = Obstacles.segments
        #print(len(self.all_segments))
        self.corner_detection()
        if (len(self.all_circles) != 12):
            self.obstacle_detection()
        
    def scan_callback(self,data):
        self.scan = np.array(data.ranges)
        self.percentage_continuity()
        self.out_of_range()
        self.feat_vec[7] = 0 # 1 indoor 0 outdoor
        self.full_data.append(self.feat_vec)
        #print(self.feat_vec)

    def corner_detection(self):
        seg_size = len(self.all_segments)
        coords = {'s_x':[],'s_y':[]}
        for s in self.all_segments:
            coords['s_x'].append(s.first_point.x)
            coords['s_y'].append(s.first_point.y)
            coords['s_x'].append(s.last_point.x)
            coords['s_y'].append(s.last_point.y)
        comb = combinations(range(0,seg_size*2), 2)  
        comb = list(comb)
        comb2 = [x for x in comb if (x[0]//2 != x[1]//2) ]
        #print(len(comb))
        cor_cnt = 0
        for i in range(0,len(comb2)): 
            p1 = np.array((coords['s_x'][comb2[i][0]],coords['s_y'][comb2[i][0]])) 
            p2 = np.array((coords['s_x'][comb2[i][1]],coords['s_y'][comb2[i][1]]))
            #print(p1,p2)
            dist = np.linalg.norm(p1-p2)
            #print(dist)
            if dist <= 0.9: # need to modify this
                # check that the dot product is ~0 to ensure vectors are perpendicular
                # Note: even indies of coords[''] are first_points and odd indies are last_points   
                vertex_array = np.array([comb2[i][0], comb2[i][1]])
                rem_array = np.array([comb2[i][0], comb2[i][1]]) % 2
                other_p1 = comb2[i][0] + 1 if rem_array[0] == 0 else comb2[i][0] - 1
                other_p2 = comb2[i][1] + 1 if rem_array[1] == 0 else comb2[i][1] - 1

                p3 = np.array((coords['s_x'][other_p1],coords['s_y'][other_p1])) 
                p4 = np.array((coords['s_x'][other_p2],coords['s_y'][other_p2]))
                
                vec1 = p3 - p1
                vec2 = p4 - p2
                #print(p1)
                #print(np.dot(vec1,vec2))

                if (abs(np.dot(vec1,vec2)) <= 0.9):   #need to adjust this.   
                    #print('angle = ', np.arccos(float(np.dot(vec1,vec2))/(np.linalg.norm(vec1)*np.linalg.norm(vec2) )))          
                    #self.feat_vec[2] = 1                    # corner present
                    cor_cnt = cor_cnt + 1
        self.feat_vec[1] = cor_cnt                      # count of number of corners
        #print('total corners = ',cor_cnt)

    def obstacle_detection(self): 
        coords = {'c_x':[],'c_y':[]}
        for c in self.all_circles:
            self.feat_vec[2] = 1  if c.true_radius > 0.7  else 0          # is obstacle bigger than 0.7 meters? set using "max_circle_radius"
            coords['c_x'].append(c.center.x)
            coords['c_y'].append(c.center.y)
        c_x = np.array(coords['c_x'])
        c_y = np.array(coords['c_y'])
        if np.size(c_x) != 0:
            x_bins = int(float(( np.max(c_x) - np.min(c_x) ))/2)         # should it be int(float (x_bins)/2) to ensure each bin is 2m wide as obstacles are 0.5m??
            y_bins = int(float(( np.max(c_y) - np.min(c_y) ))/2) 
            H = np.histogram2d(c_x, c_y,(x_bins+1,y_bins+1))
            #print(H[0])
            filled_bins = np.count_nonzero(H[0])
            #print(filled_bins)
            #print(len(self.all_circles))
            self.feat_vec[3] = float(len(self.all_circles))/filled_bins     # 2d histogram density, larger the value closer the obstacles are together
            #print(self.feat_vec[5])
        self.feat_vec[4] = len(self.all_circles)                            # number of obstacles
    
    def percentage_continuity(self):
        valid_scan_size = len(self.scan[self.scan<6])
        valid_scans = self.scan[self.scan<6]
        if(len(valid_scans) != 0):
            pts1 = deque(valid_scans)
            pts2 = deque(valid_scans)
            pts2.popleft()
            pts2.append(valid_scans[0])
            diff_array = np.array(pts1) - np.array(pts2)
            num_cont_pts = len(diff_array[diff_array < 0.05])
            self.feat_vec[5] = float(num_cont_pts)/len(diff_array)          # percentage of valid scans that have immediate neighbours < 5 cm
        #print(self.feat_vec[7])
        #print(self.scan[self.scan<6][0])
        #print(len(valid_scans))
        #print(len(q2.popleft()))

    def out_of_range(self):
        invalid_scan_size = len(self.scan[self.scan>5])
        self.feat_vec[6] = float(invalid_scan_size)/len(self.scan)

    #def line_detector(self):
        
    def save_scans(self):
        results_file = 'one_wall_fv'
        with open(results_file,'wb') as fp:
            pickle.dump(self.full_data,fp)
        file.close(fp)    

        
if __name__ == '__main__':
    
    l = Listener()
    init_tim = time.time()
    init_num = len(l.full_data)
    while not rospy.is_shutdown():
        time.sleep(.1)
        if(time.time() - init_tim > 2):
            if (len(l.full_data) - init_num > 0):
                init_num = len(l.full_data)
            else:
                break
            init_tim = time.time()
    l.save_scans()
    #rospy.sleep(1)
   
    



