import numpy as np
import cv2
import matplotlib.pyplot as plt

# Initialize the Kalman filter
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1e-4

# Generate odom1 and odom2 as sine functions with some noise added
np.random.seed(0)
t = np.linspace(0, 4*np.pi, 100)  # 100 points between 0 and 4*pi
odom1 = np.column_stack((t, np.sin(t))) + np.random.randn(100, 2) * 0.1  # noise standard deviation is 0.1
odom2 = np.column_stack((t, np.sin(t))) + np.random.randn(100, 2) * 0.1  # noise standard deviation is 0.1

# Initialize lists to store the x and y coordinates of the combined odometry
combined_odom_x = []
combined_odom_y = []

# Combine the measurements from odom1 and odom2
for i in range(len(odom1)):
    # Predict the next state
    predicted = kalman.predict()

    # Get the measurements from odom1 and odom2
    measurement1 = np.array([[np.float32(odom1[i][0])], [np.float32(odom1[i][1])]])
    measurement2 = np.array([[np.float32(odom2[i][0])], [np.float32(odom2[i][1])]])

    # Update the Kalman filter with the measurements
    kalman.correct(measurement1)
    kalman.correct(measurement2)

    # The combined odometry is the state of the Kalman filter
    combined_odom = kalman.statePost

    # Store the x and y coordinates of the combined odometry
    combined_odom_x.append(combined_odom[0])
    combined_odom_y.append(combined_odom[1])

# Plot the odometry data
plt.figure()
plt.plot(odom1[:, 0], odom1[:, 1], label='odom1')
plt.plot(odom2[:, 0], odom2[:, 1], label='odom2')
plt.plot(combined_odom_x, combined_odom_y, label='combined_odom')
plt.legend()
plt.show()