import numpy as np
from DMP import DMP
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
np.random.seed(5)

def compute_Sea(x_set1, x_set2):
    def compute_area(A, B, C, D):
        comp1 = np.linalg.det(np.vstack((A, B)).T)
        comp2 = np.linalg.det(np.vstack((B, C)).T)
        comp3 = np.linalg.det(np.vstack((C, D)).T)
        comp4 = np.linalg.det(np.vstack((D, A)).T)
        S = np.abs((comp1 + comp2 + comp3 + comp4) / 2)
        return S

    S = 0.0
    size = np.shape(x_set1)[0]
    for k in range(size - 1):
        area = compute_area(x_set1[k, :], x_set1[k + 1, :], x_set2[k + 1, :], x_set2[k, :])
        S = S + area
    return S


# type = 'U_shape'
type = 'T_shape'

x_set = np.loadtxt("D:\computer document\pythonProject_phd\Complex_periodicity_DS\dataset\ZJUT_dataset/"+ type + '.txt')[::5,:]
x_set1 = np.loadtxt("D:\computer document\pythonProject_phd\Complex_periodicity_DS\dataset\ZJUT_dataset/"+ type + '.txt')[::5,1:4]
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(x_set[80:,1],x_set[80:,2],x_set[80:,3])
# plt.show()


x_set = x_set[80:,1:4]
set_size = np.shape(x_set)[0]
t_set = 0.02 * np.arange(set_size)
T_length = None
tau = None
Period = None
step_T = 0.02
if type == 'U_shape':
    T_length = 390
else:
    T_length = 355

Period = T_length * step_T
dphi = 2 * np.pi / Period
tau = 1 / dphi
test_T_set = np.arange(0.0, 2 * Period, 0.02)
test_phi_set = test_T_set * (1 / tau)

NN = 50
rr = 1

dmp1 = DMP(x_set=x_set[:, 0], tau=tau, N=NN, r=rr)
savepath1 = 'D:\computer document\pythonProject_phd\Complex_periodicity_DS\Comparison method\para\DMP1_parameters_' + type + '.txt'
omega1 = dmp1.train(savepath=savepath1)

dmp2 = DMP(x_set=x_set[:, 1], tau=tau, N=NN, r=rr)
savepath2 = 'D:\computer document\pythonProject_phd\Complex_periodicity_DS\Comparison method\para\DMP2_parameters_' + type + '.txt'
omega2 = dmp2.train(savepath=savepath2)

dmp3 = DMP(x_set=x_set[:, 2], tau=tau, N=NN, r=rr)
savepath3 = 'D:\computer document\pythonProject_phd\Complex_periodicity_DS\Comparison method\para\DMP2_parameters_' + type + '.txt'
omega3 = dmp3.train(savepath=savepath3)



off_x = np.random.normal(0.0, 5 * 1e-2, 3)
x = x_set1[0, :]  # + off_x
dx = np.zeros(3)
# Max_steps = int(Period / step_T)
# Max_steps = x_set.shape[0]
Max_steps = 3000
k = 60.0
d = 2 * np.sqrt(k)

################################################### 这个是新加的 ########################################################
# 添加目标点参数
target_point = np.array([0.0, 0.0, 0.0])  # 示例目标点
# target_point = np.array([0.4, 0.5, 0.0])  # 示例目标点
tolerance = 0.02  # 到达容差
reached_target = False
########################################################################################################################

x_tra = [x]
dx_tra = [dx]
target_reached_time = []

for i in range(Max_steps):
    time = i * step_T
    phi = time * (1 / tau)
    g1 = dmp1.f(phi, omega1)
    g2 = dmp2.f(phi, omega2)
    g3 = dmp3.f(phi, omega3)
    g = np.array([g1, g2, g3])

    # 检查是否到达目标点
    if not reached_target and np.linalg.norm(x - target_point) < tolerance:
        reached_target = True
        target_reached_time.append(time)
        print(f"Reached target point at time {time}")

    if 1.0 <= time <= 0.1:
        dx = 0.0
        ddx = 0.0
        x = x
    else:
        ddx = -d * dx - k * (x - g)
        x = x + dx * step_T + 0.5 * ddx * step_T**2
        dx = dx + ddx * step_T

    x_tra.append(x)
    dx_tra.append(dx)

x_tra = np.array(x_tra)
dx_tra = np.array(dx_tra)

np.savetxt('D:\computer document\pythonProject_phd\Complex_periodicity_DS\Comparison method\sim_data/sim_'+type+'.txt',x_tra)
# plt.figure()
# plt.subplots_adjust(left=0.12, right=0.99, wspace=0.15, hspace=0.2, bottom=0.1, top=0.99)
# plt.plot(x_tra[:, 0], x_tra[:, 1], c='black',label='Dmp')
#
# plt.scatter(x_set[::2, 0], x_set[::2, 1], marker='x', color='blue',label='Demo')
# plt.xlabel(r'$\mathit{x_1}$/m')
# plt.ylabel(r'$\mathit{x_2}$/m')
#
# plt.legend(loc='upper left')
# plt.show()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.plot(x_tra[:, 0], x_tra[:, 1], x_tra[:, 2], c='black',label='Dmp')
ax.scatter(x_set1[::3, 0], x_set1[::3, 1], x_set1[::3, 2], marker='x', color='blue',label='Demo')
ax.scatter(x_set1[0, 0], x_set1[0, 1], x_set1[0, 2], s=60, marker='o',color='black',label='Repro. start')


# # Ushape trajectory
# Mean = np.array([0.4362341, 0.02314081, 0.447768602215292])
# data = np.loadtxt('D:\computer document\pythonProject_phd\Complex_periodicity_DS\simulation_data/' + 'sim_data_' + type + '.txt') + Mean
# ax.plot(data[:,0],data[:,1],data[:,2], label='PA exp. tra.')


ax.set_xlabel(r'$\mathit{x_1}$/m', fontsize=12, labelpad=10)
ax.set_ylabel(r'$\mathit{x_2}$/m', fontsize=12, labelpad=10)
ax.set_zlabel(r'$\mathit{x_3}$/m', fontsize=12, labelpad=10)
plt.legend(loc='upper left')
plt.show()



