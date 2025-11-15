# Rumex Obtusifolius Detection System

## Project Overview
A complete ROS-based system for real-time detection and segmentation of Rumex Obtusifolius plants using YOLO11m instance segmentation and DBSCAN clustering. The system detects individual leaves, clusters them into plants, and estimates plant centers for agricultural monitoring applications.

## Deliverables

### 1. ROS Package Structure
```
rumex_detection/
├── package.xml
├── CMakeLists.txt
├── launch/
│   └── rumex_detection.launch
├── scripts/
│   ├── image_publisher.py
│   ├── rumex_detector.py
│   └── yolo.py (standalone DL/AI version)
├── results/ (auto-created)
│   ├── detection_0001_original.jpg
│   ├── detection_0001_result.jpg
│   ├── detection_0001_leaves.jpg
│   ├── detection_0001_data.json
│   └── detection_0001_centers.csv
└── paltech_test_AI_2025_2/
    ├── yolo11m_finetuned.pt
    └── front_camera/
        ├── 1.jpg
        ├── 2.jpg
        ├── 3.jpg
        └── 4.jpg
```

### 2. ROS Nodes
**Node 1: Image Publisher (`image_publisher.py`)**
- Publishes test images from the front_camera directory to ROS topics
- Loops through images continuously for testing
- Publishes to topic: `/camera/image_raw`

**Node 2: Rumex Detector (`rumex_detector.py`)**
- Subscribes to `/camera/image_raw`
- Performs YOLO11m instance segmentation with confidence threshold 0.4
- Clusters individual leaves into plants using DBSCAN
- Estimates plant centers and bounding boxes
- Publishes plant centers to `/plant_centers` as `Float32MultiArray`
- Publishes annotated images to `/detection_result`
- Saves comprehensive results to the `results` folder

**Standalone DL/AI Component (`yolo.py`)**
- Complete Python implementation without ROS dependencies
- Processes images and saves results to `outputs` folder
- Useful for testing and development without ROS

## Installation and Setup

### Prerequisites
- Ubuntu 20.04 with ROS Noetic
- Python 3.10 for AI components
- Python 3.8 for ROS core

### Step 1: Install Dependencies
```bash
pip3 install ultralytics opencv-python scikit-learn numpy
```

### Step 2: Build ROS Workspace
```bash
cd ~/Desktop/paltech_internship/ros_ws
catkin_make
source devel/setup.bash
```

## Running the System

### Method 1: Using ROS Launch File
```bash
roslaunch rumex_detection rumex_detection.launch
```

### Method 2: Running Nodes Manually
```bash
# Terminal 1 - ROS Core
roscore

# Terminal 2 - Image Publisher
rosrun rumex_detection image_publisher.py

# Terminal 3 - Rumex Detector
rosrun rumex_detection rumex_detector.py

# Terminal 4 - Monitor Output
rostopic echo /plant_centers
```

### Method 3: Direct Python Execution (Bypassing ROS Wrapper)
```bash
# Terminal 1
roscore

# Terminal 2
python3.10 src/rumex_detection/scripts/image_publisher.py

# Terminal 3
python3.10 src/rumex_detection/scripts/rumex_detector.py
```

### Method 4: Standalone DL/AI Testing
```bash
cd src/rumex_detection/scripts
python3.10 yolo.py
```

## Plant Center Estimation Algorithm
1. **Leaf Detection**: YOLO11m performs instance segmentation to detect individual Rumex leaves.
2. **Leaf Center Calculation**: Compute centroids from mask pixels.
3. **Spatial Clustering**: DBSCAN clusters leaf centers into plants.
4. **Plant Bounding Box**: Minimum bounding rectangle around each cluster.
5. **Plant Center Estimation**: Centroid of bounding box.

## Technical Implementation Details
- **Libraries**: YOLO11m, OpenCV, scikit-learn, ROS Noetic, NumPy
- **Performance Metrics**: ~2.94 seconds per image, 4-8 plants detected per test image

## Challenges and Solutions
- **Python Version Conflicts**: ROS Noetic uses Python 3.8, AI packages need 3.10.
- **Dependency Management**: NumPy and PyTorch conflicts resolved by reinstalling required libraries.
- **Mask Dimension Mismatches**: YOLO output masks resized to match original image dimensions.
- **Clustering Parameter Tuning**: Adaptive DBSCAN parameters.

## Project Achievements
- Successful YOLO11m + ROS integration for plant detection.
- Robust DBSCAN clustering of leaves into plants.
- Results saved as images, JSON, and CSV.
- Supports both ROS and standalone execution.
- Adaptive parameters for different image sizes.

## Future Improvements
- Performance optimization: quantization, batch processing, GPU acceleration.
- Advanced features: plant health, growth monitoring, multi-camera.
- Robustness enhancements: overlapping plants, confidence scoring.
- User interface: web dashboard, real-time monitoring.

## Limitations and Workarounds
- Fixed confidence threshold of 0.4.
- DBSCAN parameters are adaptive but could be improved.
- Currently static image processing.
- Python 3.10 + ROS Noetic conflict handled by reinstalling PyTorch and YOLO libraries manually.

## What I'm Proud Of, Challenges, and Future Work
**What I am proud of:**
* I managed to implement a complete pipeline from raw images to plant center coordinates in just 2 hours.
* The system successfully detects individual Rumex leaves, clusters them into plants, and estimates plant centers.
* Results are saved in multiple formats (images, JSON, CSV), making it easy to analyze and visualize outputs.
* I integrated YOLO11m with ROS, and handled adaptive mask resizing and DBSCAN clustering dynamically.

**Challenges:**
* I had to work with two versions of Python (3.8 for ROS Noetic and 3.10 for AI libraries), which caused a lot of dependency conflicts with PyTorch and YOLO.
* Mask resizing and DBSCAN clustering parameters required quick experimentation due to limited time.
* Due to the tight time limit, I could not implement advanced features like real-time video processing or a web dashboard.

**What I could achieve with a full week:**
* Resolve all Python and library conflicts properly and unify the environment.
* Optimize performance with GPU acceleration, batch processing, and model quantization.
* Implement multi-camera synchronization, overlapping plant handling, and confidence scoring.
* Add a real-time dashboard with ROS web_video_server and rosbridge for monitoring results.
* Integrate plant size, health estimation, and growth tracking over time.


## Conclusion
Demonstrates a complete pipeline for agricultural plant detection, from leaf detection to plant center estimation, with ROS integration and standalone operation.

*Note: Due to time constraints (2 hours max), some improvements could not be implemented, including advanced real-time features and Python version harmonization.*

