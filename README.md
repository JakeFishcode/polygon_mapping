

## Install

### Requirements
- Tested on `Ubuntu 20.04 / ROS Noetic`

### Dependencies
This package depends on
- [pyrealsense2]
- [cupy]

### Installation procedure
It is assumed that ROS is installed.

1. Clone to your catkin_ws
```bash
mkdir -p catkin_ws/src
cd catkin_ws/src
git clone https://github.com/BTFrontier/polygon_mapping.git

```

2. Install dependent packages.
```bash
......
```

3. Build a package.
```bash
cd catkin_ws
catkin build polygon_mapping
```

## Datasets
You can download the test dataset [here](https://1drv.ms/f/c/1e83680b5fbc1ae4/Et2MgY6eCHRMpczAZAwRXBUBvlHg70gRJopAoxf9fdi9vg?e=DQmDKZ).
Once downloaded, extract the `dataset_39` and `dataset_71` folders into the following directory:
```bash
catkin_ws/src/polygon_mapping/data
```
Now, you can run the test case with the following command:
```bash
rosrun polygon_mapping read_dataset.py
```
You can modify the dataset directory in `read_dataset.py` to use the dataset of your choice. During the testing process, intermediate images will be saved in the `processed` subdirectory within the dataset directory, allowing you to review the results later.
