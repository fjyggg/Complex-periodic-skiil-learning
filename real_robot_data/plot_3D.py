import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Z
data = np.loadtxt("/home/fjy/fjy_ws/src/franka_ros/2025RAL/real_robot_data/J_shape_3D_exp1_data.txt")[1:,:]

print(data[1,0],data[1,1])
fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot(data[:,1],data[:,2],data[:,3],'r')

plt.plot(data[:,0],data[:,1])

plt.show()


