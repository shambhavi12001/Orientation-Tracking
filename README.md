# Orientation Tracking and Panorama Reconstruction

## Overview

This project implements 3D orientation tracking using inertial measurement unit (IMU) data and uses the estimated orientations to reconstruct panoramic images from camera frames.

The main goal is to estimate the orientation of a rotating body over time using gyroscope and accelerometer measurements. The estimated orientation trajectory is then used to align camera images into a common world frame and generate panoramic views.

The project is divided into three main parts:

1. IMU calibration
2. Orientation estimation using projected gradient descent
3. Panorama reconstruction using estimated orientations

## Problem Description

Raw IMU measurements contain noise, bias, and scaling effects. Directly integrating gyroscope measurements can estimate orientation, but this often leads to drift over time. To improve accuracy, this project combines gyroscope-based motion prediction with accelerometer-based gravity alignment.

The orientation of the body is represented using unit quaternions. A projected gradient descent optimization is used to estimate the full orientation trajectory while enforcing the unit-norm quaternion constraint.

After orientation estimation, camera images are projected into a common global frame to create panoramic images.

## Method

### 1. IMU Calibration

The raw accelerometer and gyroscope measurements are first calibrated.

During the initial static portion of each dataset, the body is assumed to be stationary. This allows estimation of:

- Accelerometer bias
- Gyroscope bias
- Sensor scaling based on datasheet sensitivity values

The calibrated IMU measurements are then used for orientation estimation.

### 2. Baseline Gyroscope Integration

As an initial estimate, the calibrated gyroscope measurements are numerically integrated over time.

At each time step:

- Angular velocity is converted into a small rotation
- The incremental rotation is represented as a quaternion
- The current orientation is updated by quaternion multiplication
- The quaternion is normalized to maintain unit length

This provides an initial orientation trajectory, but it can drift because gyroscope integration accumulates error.

### 3. Quaternion-Based Orientation Estimation

The final orientation trajectory is estimated using projected gradient descent.

The cost function includes two terms:

#### Motion Model Error

The motion model predicts the next orientation using the current orientation and gyroscope measurement. The error measures the difference between the predicted next quaternion and the estimated next quaternion.

#### Observation Model Error

The accelerometer is used as a gravity reference. Under the pure rotation assumption, the measured acceleration should align with gravity. The observation error measures the difference between the calibrated accelerometer reading and the predicted gravity direction from the estimated quaternion.

The total cost combines both terms and is minimized over the full orientation trajectory.

After each gradient descent update, every quaternion is projected back onto the unit sphere to ensure it remains a valid rotation.

### 4. Orientation Evaluation

For training datasets, the estimated orientations are compared against VICON ground truth.

The quaternion trajectories are converted to roll, pitch, and yaw angles. The estimated and ground-truth trajectories are time-aligned using nearest-neighbor timestamp matching.

The results show that projected gradient descent improves roll and pitch estimation compared to simple gyroscope integration, especially because the accelerometer provides a gravity constraint. Yaw remains more difficult because gravity does not provide absolute heading information.

### 5. Panorama Reconstruction

Using the estimated orientation trajectory, camera images are stitched into panoramas.

For each camera frame:

- The closest previous IMU orientation estimate is selected
- Pixel rays are projected into 3D using a pinhole camera model
- Rays are rotated into a global reference frame using the estimated orientation
- The rotated rays are mapped onto an equirectangular panorama canvas
- Pixel colors are written to the panorama image

A simple overwrite strategy is used for overlapping pixels.
