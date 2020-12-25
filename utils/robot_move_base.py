from move_base_msgs.msg import *
from std_srvs.srv import Empty
import rospy
from tf.transformations import quaternion_from_euler
import actionlib
from actionlib_msgs.msg import GoalStatusArray, GoalID
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal, MoveBaseFeedback


class WaypointNavigator:
    """
    Navigates the robot to a global position and orientation using move_base.
    ># pose     Pose2D      Target waypoint for navigation.
    """

    def __init__(self, reference="map"):
        """Constructor"""
        self._action_topic = "/move_base"
        self._cancel_topic = "/move_base/cancel"
        self._clear_costmaps = "/move_base/clear_costmaps"

        self._client = actionlib.SimpleActionClient(
            self._action_topic, MoveBaseAction)
        self.move_base_cancel_goal_pub = rospy.Publisher(
            self._cancel_topic, GoalID, queue_size=1)

        self._arrived = False
        self._failed = False
        self.reference = reference

        rospy.loginfo('Waiting for move_base action server to be up')
        self._client.wait_for_server()

    def set_goal(self, pose, ep):
        rospy.wait_for_service(self._clear_costmaps)
        serv = rospy.ServiceProxy(self._clear_costmaps, Empty)
        serv()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = self.reference

        goal.target_pose.header.stamp = rospy.Time.now()
        goal.target_pose.pose.position.x = pose[0]
        goal.target_pose.pose.position.y = pose[1]
        goal.target_pose.pose.position.z = 0
        q = quaternion_from_euler(0, 0, 0)
        goal.target_pose.pose.orientation.x = q[0]
        goal.target_pose.pose.orientation.y = q[1]
        goal.target_pose.pose.orientation.z = q[2]
        goal.target_pose.pose.orientation.w = q[3]
        rospy.loginfo('Sending goal id %d to move_base', ep)

        # Send the action goal for execution
        try:
            self._client.send_goal(goal)
        except Exception as e:
            rospy.logwarn("Unable to send navigation action goal:\n%s" % str(e))
            self._failed = True

    def _wait_for_navigation(self):
        """Wait for navigation system to be up
        """
        services = [
            'move_base/local_costmap/set_parameters',
            'move_base/global_costmap/set_parameters',
            'move_base/set_parameters'
        ]
        for service in services:
            while True:
                try:
                    rospy.loginfo('Waiting for service %s to be up', service)
                    rospy.wait_for_service(service)
                    break
                except rospy.ROSException:
                    rospy.sleep(0.5)

    def cancel_goal(self):
        self.move_base_cancel_goal_pub.publish(GoalID())
        rospy.loginfo('Cancelled active action goal.')
        rospy.sleep(0.2)
