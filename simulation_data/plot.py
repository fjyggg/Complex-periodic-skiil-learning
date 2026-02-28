import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Type = 'U_shape'
data = np.loadtxt('D:\computer document\pythonProject_phd\Complex_periodicity_DS\simulation_data/' + 'sim_data_' + Type + '.txt')
print(data.shape)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(data[:,0],data[:,1],data[:,2])
plt.show()


