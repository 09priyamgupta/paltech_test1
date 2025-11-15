#!/usr/bin/env python3

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
import json
from datetime import datetime

class RumexDetector:
    def __init__(self):
        rospy.init_node('rumex_detector', anonymous=True)
        
        # Subscriber for images
        self.subscriber = rospy.Subscriber('camera/image_raw', Image, self.image_callback, queue_size=1)
        
        # Publisher for plant centers - ROS1 syntax
        self.detection_pub = rospy.Publisher('plant_centers', Float32MultiArray, queue_size=10)
        self.result_pub = rospy.Publisher('detection_result', Image, queue_size=10)
        
        self.bridge = CvBridge()
        
        # Load YOLO model
        rospack = rospkg.RosPack()
        self.pkg_path = rospack.get_path('rumex_detection')
        model_path = Path(self.pkg_path) / 'paltech_test_AI_2025_2' / 'yolo11m_finetuned.pt'
        
        self.model = YOLO(str(model_path))
        self.conf_threshold = 0.4

        # Create results directory
        self.results_path = Path(self.pkg_path) / 'results'
        self.results_path.mkdir(exist_ok=True)
        
        # Counter for sequential image naming
        self.image_counter = 0
        
        rospy.loginfo('Rumex detector node initialized')
        rospy.loginfo(f'Using model: {model_path}')
        rospy.loginfo(f'Results will be saved to: {self.results_path}')
    
    def save_detection_results(self, original_image, result_image, masks, plants, processing_time, image_name=None):
        """
        Save detection results to the results folder
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if image_name is None:
            self.image_counter += 1
            image_name = f"detection_{self.image_counter:04d}"
        else:
            image_name = Path(image_name).stem
        
        # Save original image
        original_filename = self.results_path / f"{image_name}_original.jpg"
        cv2.imwrite(str(original_filename), cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
        
        # Save result image with detections
        result_filename = self.results_path / f"{image_name}_result.jpg"
        cv2.imwrite(str(result_filename), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        # Only save leaf image if there are leaves detected
        if len(masks) > 0:
            leaf_image = original_image.copy()
            for mask in masks:
                color = np.random.randint(0, 255, 3)
                mask_bool = mask.astype(bool)
                overlay = leaf_image.copy()
                overlay[mask_bool] = color
                leaf_image = cv2.addWeighted(leaf_image, 0.7, overlay, 0.3, 0)
                
                mask_uint8 = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(leaf_image, contours, -1, color.tolist(), 2)
            
            leaf_filename = self.results_path / f"{image_name}_leaves.jpg"
            cv2.imwrite(str(leaf_filename), cv2.cvtColor(leaf_image, cv2.COLOR_RGB2BGR))
        else:
            leaf_filename = None
        
        # Save detection data as JSON
        detection_data = {
            'timestamp': timestamp,
            'image_name': image_name,
            'processing_time': processing_time,
            'num_leaves': len(masks),
            'num_plants': len(plants),
            'plants': [],
            'detection_parameters': {
                'confidence_threshold': self.conf_threshold,
                'dbscan_eps': 'adaptive'
            }
        }
        
        for i, plant in enumerate(plants):
            plant_data = {
                'plant_id': i + 1,
                'bounding_box': {
                    'x_min': float(plant['bbox'][0]),
                    'y_min': float(plant['bbox'][1]),
                    'x_max': float(plant['bbox'][2]),
                    'y_max': float(plant['bbox'][3])
                },
                'center': {
                    'x': float(plant['center'][0]),
                    'y': float(plant['center'][1])
                },
                'num_leaves': plant['num_leaves']
            }
            detection_data['plants'].append(plant_data)
        
        json_filename = self.results_path / f"{image_name}_data.json"
        with open(json_filename, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        # Save plant centers as CSV for easy analysis
        csv_filename = self.results_path / f"{image_name}_centers.csv"
        with open(csv_filename, 'w') as f:
            f.write("plant_id,center_x,center_y,num_leaves\n")
            for i, plant in enumerate(plants):
                f.write(f"{i+1},{plant['center'][0]:.2f},{plant['center'][1]:.2f},{plant['num_leaves']}\n")
        
        # Log what was saved
        log_message = f"Saved detection results for {image_name}:"
        log_message += f"\n  - Images: {original_filename.name}, {result_filename.name}"
        if leaf_filename:
            log_message += f", {leaf_filename.name}"
        log_message += f"\n  - Data: {json_filename.name}, {csv_filename.name}"
        log_message += f"\n  - Stats: {len(masks)} leaves, {len(plants)} plants, {processing_time:.2f}s processing time"
        
        rospy.loginfo(log_message)
    
    def image_callback(self, msg):
        start_time = time.time()
        
        try:
            # Convert ROS Image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            orig_height, orig_width = cv_image_rgb.shape[:2]
            
            # Extract image name from header if available
            image_name = None
            if hasattr(msg, 'header') and hasattr(msg.header, 'frame_id'):
                image_name = msg.header.frame_id
            
            # Run YOLO inference
            results = self.model(cv_image_rgb, conf=self.conf_threshold, verbose=False)
            
            plant_centers = []
            result_image = cv_image_rgb.copy()
            plants = []  # Store plant data for saving
            masks = []   # Initialize masks as empty list
            
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
                            
                            # Store plant data for saving
                            plants.append({
                                'bbox': [min_x, min_y, max_x, max_y],
                                'center': [plant_center_x, plant_center_y],
                                'num_leaves': len(plant_indices)
                            })
                            
                            # Draw plant bounding box and center
                            cv2.rectangle(result_image, 
                                         (int(min_x), int(min_y)), 
                                         (int(max_x), int(max_y)), 
                                         (255, 0, 0), 3)
                            cv2.circle(result_image, 
                                      (int(plant_center_x), int(plant_center_y)), 
                                      8, (0, 255, 0), -1)
                            
                            plant_num = len(plants)
                            cv2.putText(result_image, f'Plant {plant_num}', 
                                       (int(min_x), int(min_y) - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                # No detections found
                rospy.loginfo(f"No leaves detected in image {image_name}")
            
            # Save detection results
            processing_time = time.time() - start_time
            self.save_detection_results(cv_image_rgb, result_image, masks, plants, processing_time, image_name)
            
            # Publish plant centers
            centers_msg = Float32MultiArray()
            centers_msg.data = plant_centers
            self.detection_pub.publish(centers_msg)
            
            # Publish result image
            result_ros = self.bridge.cv2_to_imgmsg(
                cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR), "bgr8")
            result_ros.header = msg.header  # Keep the same header
            self.result_pub.publish(result_ros)
            
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