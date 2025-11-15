#!/usr/bin/env python3.10

import cv2
import torch
import numpy as np
import time
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from ultralytics import YOLO
import json

class RumexSegmentor:
    def __init__(self, model_path, conf_threshold=0.4):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
    def segment_image(self, image_path):
        """
            Perform instance segmentation on a single image
        """
        start_time = time.time()
        
        # Run YOLO inference
        results = self.model(image_path, conf=self.conf_threshold, verbose=False)
        
        # Extract masks and boxes
        if len(results) == 0 or results[0].masks is None:
            return None, None, time.time() - start_time
            
        # Get original image dimensions
        orig_image = cv2.imread(image_path)
        if orig_image is None:
            print(f"Failed to load image: {image_path}")
            return None, None, time.time() - start_time
            
        orig_height, orig_width = orig_image.shape[:2]
        
        # Get masks and convert to original image size
        masks = results[0].masks.data.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        
        # Resize masks to original image dimensions if needed
        resized_masks = []
        for mask in masks:
            # Check if mask needs resizing
            if mask.shape != (orig_height, orig_width):
                mask_resized = cv2.resize(mask, (orig_width, orig_height), interpolation=cv2.INTER_NEAREST)
                resized_masks.append(mask_resized)
            else:
                resized_masks.append(mask)
        
        masks = np.array(resized_masks)
        
        # Convert image for visualization
        image_rgb = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        
        # Draw leaf masks
        leaf_image = image_rgb.copy()
        for mask in masks:
            # Create colored mask with same dimensions as image
            color = np.random.randint(0, 255, 3)
            colored_mask = np.zeros_like(leaf_image)
            
            # Ensure mask is boolean and has correct dimensions
            mask_bool = mask.astype(bool)
            colored_mask[mask_bool] = color
            
            # Blend mask with image
            leaf_image = cv2.addWeighted(leaf_image, 0.7, colored_mask, 0.3, 0)
            
            # Draw mask contour
            mask_uint8 = (mask * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:  # Only draw if contours are found
                cv2.drawContours(leaf_image, contours, -1, color.tolist(), 2)
        
        inference_time = time.time() - start_time
        return leaf_image, (masks, boxes), inference_time
    
    def cluster_leaves_to_plants(self, masks, boxes, image_shape):
        """
            Cluster individual leaves into plants using DBSCAN
        """
        if masks is None or len(masks) == 0:
            return []
            
        # Calculate centers of each leaf mask
        leaf_centers = []
        for mask in masks:
            # Find non-zero coordinates in mask
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) > 0 and len(y_coords) > 0:
                center_x = np.mean(x_coords)
                center_y = np.mean(y_coords)
                leaf_centers.append([center_x, center_y])
        
        if len(leaf_centers) == 0:
            return []
            
        leaf_centers = np.array(leaf_centers)
        
        # Use DBSCAN to cluster leaves into plants
        # eps: maximum distance between leaves in the same plant
        # min_samples: minimum number of leaves to form a plant
        clustering = DBSCAN(eps=150, min_samples=1).fit(leaf_centers)
        
        plants = []
        unique_labels = set(clustering.labels_)
        
        for label in unique_labels:
            if label == -1:  # Noise points (shouldn't happen with min_samples=1)
                continue
                
            # Get leaves belonging to this plant
            plant_indices = np.where(clustering.labels_ == label)[0]
            plant_leaves_centers = leaf_centers[plant_indices]
            
            if len(plant_leaves_centers) > 0:
                # Calculate plant bounding box from all leaf centers
                min_x = np.min(plant_leaves_centers[:, 0])
                max_x = np.max(plant_leaves_centers[:, 0])
                min_y = np.min(plant_leaves_centers[:, 1])
                max_y = np.max(plant_leaves_centers[:, 1])
                
                # Calculate plant center
                plant_center_x = (min_x + max_x) / 2
                plant_center_y = (min_y + max_y) / 2
                
                plants.append({
                    'bbox': [min_x, min_y, max_x, max_y],
                    'center': [plant_center_x, plant_center_y],
                    'num_leaves': len(plant_indices)
                })
        
        return plants
    
    def process_image(self, image_path, output_dir):
        """
            Complete processing pipeline for one image
        """
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nProcessing image: {image_path}")
        
        # Perform segmentation
        leaf_image, segmentation_data, inference_time = self.segment_image(image_path)
        
        if leaf_image is None:
            print(f"No detections in {image_path}")
            return None, inference_time
            
        masks, boxes = segmentation_data
        
        # Cluster leaves into plants
        clustering_start = time.time()
        plants = self.cluster_leaves_to_plants(masks, boxes, leaf_image.shape)
        clustering_time = time.time() - clustering_start
        
        total_time = inference_time + clustering_time
        
        # Create plant visualization
        plant_image = leaf_image.copy()
        
        # Draw plant bounding boxes and centers
        for i, plant in enumerate(plants):
            bbox = plant['bbox']
            center = plant['center']
            
            # Draw plant bounding box
            cv2.rectangle(plant_image, 
                         (int(bbox[0]), int(bbox[1])), 
                         (int(bbox[2]), int(bbox[3])), 
                         (255, 0, 0), 3)
            
            # Draw plant center
            cv2.circle(plant_image, 
                      (int(center[0]), int(center[1])), 
                      8, (0, 255, 0), -1)
            
            # Add plant label
            cv2.putText(plant_image, f'Plant {i+1}', 
                       (int(bbox[0]), int(bbox[1]) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Save results
        image_name = Path(image_path).stem
        cv2.imwrite(str(output_dir / f'{image_name}_leaves.jpg'), 
                   cv2.cvtColor(leaf_image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / f'{image_name}_plants.jpg'), 
                   cv2.cvtColor(plant_image, cv2.COLOR_RGB2BGR))
        
        print(f"Processed {image_path}:")
        print(f"    Found {len(plants)} plants")
        print(f"    Total leaves: {len(masks)}")
        print(f"    Inference time: {inference_time:.3f}s")
        print(f"    Clustering time: {clustering_time:.3f}s")
        print(f"    Total time: {total_time:.3f}s")
        
        return plants, total_time

def main():
    # Initialize segmentor with correct path
    model_path = "paltech_test_AI_2025_2/yolo11m_finetuned.pt"
    
    try:
        segmentor = RumexSegmentor(model_path, conf_threshold=0.4)
    except Exception as e:
        print(f"Error initializing segmentor: {e}")
        return
    
    # Process test images from front_camera directory
    image_dir = Path("paltech_test_AI_2025_2/front_camera")
    test_images = list(image_dir.glob("*.jpg"))
    
    print(f"Found {len(test_images)} test images:")
    for img in test_images:
        print(f"  - {img}")
    
    all_times = []
    for image_path in test_images:
        try:
            plants, proc_time = segmentor.process_image(str(image_path), 'outputs')
            if proc_time:
                all_times.append(proc_time)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    if all_times:
        print(f"\nAverage processing time per image: {np.mean(all_times):.3f}s")

if __name__ == "__main__":
    main()