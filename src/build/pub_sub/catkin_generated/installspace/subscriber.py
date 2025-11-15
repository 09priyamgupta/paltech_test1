#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def callback(data):
    '''
        Callback function to process received messages.
    '''
    rospy.loginfo(f"Received message: {data.data}")


def subscriber():
    # Initialize ROS node
    rospy.init_node('subscriber_node', anonymous=True)

    # Creating the subscriber
    rospy.Subscriber('/publisher', String, callback)

    rospy.loginfo("Subscriber node started, listening to '/publisher' topic.")

    # Keep the node running
    rospy.spin()


if __name__ == '__main__':
    try:
        subscriber()
    except rospy.ROSInterruptException:
        print("ROS Node Interrupted")
        pass