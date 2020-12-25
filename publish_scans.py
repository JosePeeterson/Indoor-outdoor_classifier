
import time
import pickle
import numpy as np
import rospy
from sensor_msgs.msg import LaserScan

def collate_data():
    results_file = ['one_wall']#['one_wall','out_corner','open',#,'corridor','l_corridor','plus_corridor']

    all_data = []
    for r in results_file:
        data = []
        with (open(r,"rb")) as fp:
            while True:
                try:
                    data.append(pickle.load(fp))
                except EOFError:
                    break
        all_data.append(data)
    return all_data

def publish_scans(all_data):
    rospy.init_node('laser_scan_publisher')
    scan_pub = rospy.Publisher('/scan', LaserScan, queue_size=50)
    
    num_readings = len(all_data)*len(all_data[0][0])
    laser_frequency = 40

    count = 0
    r = rospy.Rate(1.0)
    while not rospy.is_shutdown():
        current_time = rospy.Time.now()

        scan = LaserScan()

        scan.header.stamp = current_time
        scan.header.frame_id = 'laser_frame'
        scan.angle_min = 0
        scan.angle_max = 6.2657
        scan.angle_increment = 3.14 / 180
        scan.time_increment = (1.0 / laser_frequency) / (360)
        scan.range_min = 0.0
        scan.range_max = 6.0

        

        for i in range(0,len(all_data)):  #
            full_data = []
            full_data = all_data[i]
            tot1 = 0
            for a in full_data[0]:  # a = 90
                a = np.array(a) 
                a = a[:len(a)-1] # remove the label
                scan.ranges = []
                scan.intensities = []
                for m in range(0,len(a)):
                    scan.ranges.append(a[m])
                    scan.intensities.append(1)
                scan_pub.publish(scan)
                r.sleep()
        break

        

if __name__ == '__main__':

    all_data = collate_data()
    publish_scans(all_data)

