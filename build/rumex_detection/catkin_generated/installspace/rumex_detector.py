#!/usr/bin/env python3.10

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
import time
from pathlib import Path
import rospkg

class RumexDetector:
    def __init__(self):
        rospy.init_node('rumex_detector', anonymous=True)
        
        # Subscriber for images
        self.subscriber = rospy.Subscriber('camera/image_raw', Image, self.image_callback, queue_size=1)
        
        # Publisher for plant centers
        self.detection_pub = rospy.Publisher('plant_centers', Float32MultiArray, queue_size=10)
        self.result_pub = rospy.Publisher('detection_result', Image, queue_size=10)
        
        self.bridge = CvBridge()
        
        # Load YOLO model
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('rumex_detection')
        model_path = Path(self.pkg_path) / 'paltech_test_AI_2025_2' / 'yolo11m_finetuned.pt'
        
        self.model = YOLO(str(model_path))
        self.conf_threshold = 0.4

        self.pkg_path = rospack.get_path('rumex_detection')
        self.results_path = Path(self.pkg_path) / 'results'
        self.results_path.mkdir(exist_ok=True)
        
        rospy.loginfo('Rumex detector node initialized')
        rospy.loginfo(f'Using model: {model_path}')
    
    def image_callback(self, msg):
        start_time = time.time()
        
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            orig_height, orig_width = cv_image_rgb.shape[:2]
            
            # Run YOLO inference
            results = self.model(cv_image_rgb, conf=self.conf_threshold, verbose=False)
            
            plant_centers = []
            result_image = cv_image_rgb.copy()
            
            if len(results) > 0 and results[0].masks is not None:
                # Get and resize masks
                masks = results[0].masks.data.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                
                # Resize masks to original dimensions
                resized_masks = []
                for mask in masks:
                    if mask.shape != (orig_height, orig_width):
                        mask_resized = cv2.resize(mask, (orig_width, orig_height), 
                                                interpolation=cv2.INTER_NEAREST)
                        resized_masks.append(mask_resized)
                    else:
                        resized_masks.append(mask)
                
                masks = np.array(resized_masks)
                
                # Draw leaf masks and collect centers for clustering
                leaf_centers = []
                for mask in masks:
                    # Draw leaf mask
                    color = np.random.randint(0, 255, 3)
                    mask_bool = mask.astype(bool)
                    
                    # Create colored overlay
                    overlay = result_image.copy()
                    overlay[mask_bool] = color
                    result_image = cv2.addWeighted(result_image, 0.7, overlay, 0.3, 0)
                    
                    # Draw contour
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        cv2.drawContours(result_image, contours, -1, color.tolist(), 2)
                    
                    # Calculate leaf center for clustering
                    y_coords, x_coords = np.where(mask_bool)
                    if len(x_coords) > 0 and len(y_coords) > 0:
                        center_x = np.mean(x_coords)
                        center_y = np.mean(y_coords)
                        leaf_centers.append([center_x, center_y])
                
                # Cluster leaves into plants
                if len(leaf_centers) > 0:
                    leaf_centers = np.array(leaf_centers)
                    diag = np.sqrt(orig_width**2 + orig_height**2)
                    eps = max(20, int(diag * 0.05))
                    clustering = DBSCAN(eps=eps, min_samples=1).fit(leaf_centers)
                    
                    unique_labels = set(clustering.labels_)
                    
                    for label in unique_labels:
                        if label == -1:
                            continue
                            
                        plant_indices = np.where(clustering.labels_ == label)[0]
                        plant_leaves_centers = leaf_centers[plant_indices]
                        
                        if len(plant_leaves_centers) > 0:
                            min_x, max_x = np.min(plant_leaves_centers[:, 0]), np.max(plant_leaves_centers[:, 0])
                            min_y, max_y = np.min(plant_leaves_centers[:, 1]), np.max(plant_leaves_centers[:, 1])
                            
                            plant_center_x = (min_x + max_x) / 2
                            plant_center_y = (min_y + max_y) / 2
                            
                            plant_centers.extend([plant_center_x, plant_center_y])
                            
                            # Draw plant bounding box and center
                            cv2.rectangle(result_image, 
                                         (int(min_x), int(min_y)), 
                                         (int(max_x), int(max_y)), 
                                         (255, 0, 0), 3)
                            cv2.circle(result_image, 
                                      (int(plant_center_x), int(plant_center_y)), 
                                      8, (0, 255, 0), -1)
                            
                            plant_num = len(plant_centers) // 2
                            cv2.putText(result_image, f'Plant {plant_num}', 
                                       (int(min_x), int(min_y) - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            # Publish plant centers
            centers_msg = Float32MultiArray()
            centers_msg.data = plant_centers
            self.detection_pub.publish(centers_msg)
            
            # Publish result image
            result_ros = self.bridge.cv2_to_imgmsg(
                cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR), "bgr8")
            result_ros.header = msg.header  # Keep the same header
            self.result_pub.publish(result_ros)
            
            processing_time = time.time() - start_time
            num_plants = len(plant_centers) // 2
            rospy.loginfo(f'Detected {num_plants} plants in {processing_time:.3f}s')
                
        except Exception as e:
            rospy.logerr(f'Error processing image: {str(e)}')
    
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        detector = RumexDetector()
        detector.run()
    except rospy.ROSInterruptException:
        pass