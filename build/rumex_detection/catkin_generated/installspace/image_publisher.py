#!/usr/bin/env python3.10

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from pathlib import Path
import os
import rospkg


class ImagePublisher:
    def __init__(self):
        rospy.init_node('image_publisher', anonymous=True)
        self.publisher = rospy.Publisher('camera/image_raw', Image, queue_size=10)
        self.bridge = CvBridge()
        
        rospack = rospkg.RosPack()
        pkg_path = rospack.get_path('rumex_detection')
        self.image_dir = Path(pkg_path) / 'paltech_test_AI_2025_2' / 'front_camera'
        self.image_files = list(self.image_dir.glob('*.jpg'))
        self.current_index = 0
        
        if not self.image_files:
            rospy.logerr('No images found in front_camera directory')
            return
            
        rospy.loginfo(f'Found {len(self.image_files)} images to publish')
        for img in self.image_files:
            rospy.loginfo(f'  - {img.name}')
        
        # Publish images every 5 seconds
        self.timer = rospy.Timer(rospy.Duration(5), self.publish_image)
    
    def publish_image(self, event):
        if not self.image_files:
            return
            
        # Get current image
        image_path = self.image_files[self.current_index]
        cv_image = cv2.imread(str(image_path))
        
        if cv_image is not None:
            # Convert to ROS message and publish
            try:
                ros_image = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                ros_image.header.stamp = rospy.Time.now()
                ros_image.header.frame_id = "camera"
                
                self.publisher.publish(ros_image)
                rospy.loginfo(f'Published image: {image_path.name}')
            except Exception as e:
                rospy.logerr(f'Error converting image: {e}')
        else:
            rospy.logerr(f'Failed to load image: {image_path}')
        
        # Move to next image (loop)
        self.current_index = (self.current_index + 1) % len(self.image_files)
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        publisher = ImagePublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass
