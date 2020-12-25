"""
The codes are designed to read-in costmaps and pre-process.
Can be used for both off-line and on-line costmap conversion.
The costmap must be square (width == height)

Created by Shen Ren, 2020.09.02
"""
import subprocess
import yaml

import numpy as np
import rosbag
import tf
from scipy import ndimage
import matplotlib.pyplot as plt


def ros_format(data_frame):
    """
    input: data_frame -> OccupancyGrid (probability of obstacles,
    row-major order starting from (0, 0))
    output: np.array -> [height*width, 1], value in [0, 100]
    """
    costmap = np.array(data_frame.data)
    return costmap


def matrix_format(data_frame, angle):
    """
    input: data_frame -> OccupancyGrid (probability of obstacles,
    row-major order starting from (0, 0))
    angle -> [0, 360]
    output: np.array -> [height, width], value in [0, 100]
    """
    costmap = np.array(data_frame.data)

    height = data_frame.info.height
    width = data_frame.info.width

    costmap_matrix = np.reshape(costmap, (height, width))
    costmap_rotated = ndimage.rotate(costmap_matrix, angle, reshape=False)
    return costmap_rotated


def laser_scan_format(data_frame, angle, prob_thres=50):
    """
    From costmap to laser scan, compensated for heading directions.

    input: data_frame -> OccupancyGrid (probability of obstacles,
    row-major order starting from (0, 0))
    angle -> [0, 360]
    output: np.array -> [360, 1], value in [-0.5, 0.5], representing the
    scaled distance to obstacles from degree 0 to 360.
    """
    width = data_frame.info.width
    height = data_frame.info.height
    assert width == height, "The width and the height of the given costmap" \
                            " is not equal"

    costmap_scan = np.full(360, 0.5)
    costmap_matrix = np.reshape(np.array(data_frame.data), (height, width))
    # plt.imshow(costmap_matrix)

    # rotate the costmap according to the most recent heading direction of
    # the given vehicle
    costmap_rotated = ndimage.rotate(costmap_matrix,
                                     angle, reshape=False)
    # plt.imshow(costmap_rotated)
    # plt.show()
    costmap_rotated[costmap_rotated < prob_thres] = 0

    def get_one_direction(m, current_angle):
        """
        Get one beam reading from the center.
        """
        for index in range(half_w):

            # Set the correct signs for x
            if 0 < current_angle < np.pi/2 or np.pi*3/2 < current_angle < 2*np.pi:
                x = index
            else:
                x = -index

            y = np.round(np.tan(current_angle) * x)
            current_pos = (int(y + center[0]), int(x + center[1]))
            if 0 <= current_pos[0] < height and 0 <= current_pos[1] < width:
                readings = m[current_pos[0], current_pos[1]]
                if readings > 0:
                    return np.linalg.norm(
                        np.array(current_pos) - np.array(center))
            else:
                continue
        # no obstacles at this direction
        return -1

    def special_case(m, deg):
        array = None
        if deg == 0:
            array = m[center[0], center[1]:] > 0
        if deg == 90:
            array = m[center[0]:, center[1]] > 0
        if deg == 180:
            array = m[center[0], :center[1]] > 0
        if deg == 270:
            array = m[:center[0], center[1]] > 0
        if np.any(array):
            return np.argmax(array)
        else:
            return half_h

    # Get 360 degree
    half_h = height/2
    half_w = width/2
    center = (half_h, half_w)
    for degree in range(360):
        if degree in [0, 90, 180, 270]:
            dist = special_case(costmap_rotated, degree)
            rescale = float(dist)/half_h - 0.5
        else:
            moving_angle = degree/360.0 * (2 * np.pi)
            dist = get_one_direction(costmap_rotated, moving_angle)
            # scale the final dist into [-0.5, 0.5]
            if dist == -1:
                rescale = 0.5
            else:
                rescale = float(dist)/(np.sqrt(half_h**2+half_w**2)) - 0.5
        costmap_scan[degree] = rescale
    return costmap_scan


def crop_to_180(costmap_scan):
    """
    The function is used to crop the 360 degree costmap_scan to 180
    and scanning from left to right as in original readings.
    """
    half_costmap = costmap_scan[:180]
    half_costmap = half_costmap[::-1]
    return half_costmap


def get_odometry_angle(odometry):
    """
    Get euler angular position in the range of [0, 360] degree
    """
    quaternious = odometry.pose.pose.orientation
    euler = tf.transformations.euler_from_quaternion(
        [quaternious.x, quaternious.y, quaternious.z, quaternious.w])
    euler_angle = euler[2]
    angle = (euler_angle + np.pi) * 360 / (2 * np.pi)
    return angle


def read_in_rosbag(path):
    """
    Read-in recorded rosbag.
    """
    bag = rosbag.Bag(path)
    costmaps = []
    prev_angle = 0
    for topic, msg, t in bag.read_messages(
            topics=['/odometry/filtered', '/move_base/local_costmap/costmap']):
        # read odometry
        if topic == '/odometry/filtered':
            prev_angle = get_odometry_angle(msg)
        elif topic == '/move_base/local_costmap/costmap':
            costmap = ros_format(msg)
            costmap_matrix = matrix_format(msg, prev_angle)
            costmap_scan = laser_scan_format(msg, prev_angle)
            costmap_half = crop_to_180(costmap_scan)
            costmaps.append(costmap_scan)
        else:
            print("This topic is not processed.")
            continue

    return costmaps


def rosbag_info(path):
    bag = rosbag.Bag(path)
    info_dict = yaml.load(
        subprocess.Popen(['rosbag', 'info', '--yaml', path],
                         stdout=subprocess.PIPE).communicate()[0])
    print(info_dict)
    topics = bag.get_type_and_topic_info()[1].keys()
    types = []
    for i in range(0, len(bag.get_type_and_topic_info()[1].values())):
        types.append(bag.get_type_and_topic_info()[1].values()[i][0])

    print(topics)
    print(types)


if __name__ == '__main__':
    # rosbag_info('../data/subset.bag')
    read_in_rosbag('../data/subset.bag')
