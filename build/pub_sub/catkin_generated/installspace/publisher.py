#!/usr/bin/env python3

import rospy
from std_msgs.msg import String

def publisher():
    # Initialize ROS node
    rospy.init_node('publisher_node', anonymous=True)

    # Creating the publisher
    pub = rospy.Publisher('/publisher', String, queue_size=10)

    # Setting the publishing rate
    rate = rospy.Rate(1)            # 1 Hz

    counter = 0

    rospy.loginfo("Publisher node started, publishing messages to '/publisher' topic.")

    while not rospy.is_shutdown():
        # Create the message
        message = f"Hello PALTECH team! Message {counter}"
        
        # Log the message to console
        rospy.loginfo(f"Publishing: {message}")
        
        # Publish the message
        pub.publish(message)
        
        # Increment the counter
        counter += 1
        
        # Sleep to maintain the loop rate
        rate.sleep()


if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        print("ROS Node Interrupted")
        pass