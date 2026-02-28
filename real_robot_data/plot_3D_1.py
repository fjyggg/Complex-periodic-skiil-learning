import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Z
data = np.loadtxt("/home/fjy/fjy_ws/src/franka_ros/2025RAL/real_robot_data/T_shape_3D_exp2_data.txt")[1:,:]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(data[:,0],data[:,1],data[:,2],'r')


plt.show()


