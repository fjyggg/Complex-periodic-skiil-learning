import numpy as np
from DMP import DMP
import matplotlib.pyplot as plt
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


# type = '2D_unregular'
type = 'RShape'
# type = 'dataset_2D_unregular'
# type = 'dataset_2D_circle'
x_set = np.loadtxt('D:\computer document\pythonProject_phd\Complex_periodicity_DS\dataset/'+ type +'.txt')
# x_set = np.loadtxt('data_set/filter_set/filtered_x_' + type + '.txt')

print(x_set.shape)
# plt.figure()
# plt.plot(x_set[:295,0],x_set[:295,1])
# plt.show()

set_size = np.shape(x_set)[0]
t_set = 0.02 * np.arange(set_size)

T_length = None
tau = None
Period = None
step_T = 0.02
if type == '2D_circle':
    T_length = 313
elif type == '2D_rectangle':
    T_length = 378
elif type == '2D_star':
    T_length = 770
elif type == 'RShape':
    T_length = 262
else:
    T_length = 278

Period = T_length * step_T
dphi = 2 * np.pi / Period
tau = 1 / dphi
test_T_set = np.arange(0.0, 2 * Period, 0.02)
test_phi_set = test_T_set * (1 / tau)

NN = 50
rr = 1

dmp1 = DMP(x_set=x_set[:, 0], tau=tau, N=NN, r=rr)
# savepath1 = 'parameters/DMP1_parameters_' + type + '.txt'
savepath1 = 'D:\computer document\pythonProject_phd\Complex_periodicity_DS\Comparison method\para\DMP1_parameters_' + type + '.txt'
# omega1 = dmp1.train(savepath=savepath1)
omega1 = np.loadtxt(savepath1)

dmp2 = DMP(x_set=x_set[:, 1], tau=tau, N=NN, r=rr)
# savepath2 = 'parameters/DMP2_parameters_' + type + '.txt'
savepath2 = 'D:\computer document\pythonProject_phd\Complex_periodicity_DS\Comparison method\para\DMP2_parameters_' + type + '.txt'
# omega2 = dmp2.train(savepath=savepath2)
omega2 = np.loadtxt(savepath2)

off_x = np.random.normal(0.0, 5 * 1e-2, 2)
x = x_set[0, :]  # + off_x
dx = np.zeros(2)
# Max_steps = int(Period / step_T)
Max_steps = x_set.shape[0]
k = 60.0
d = 2 * np.sqrt(k)

################################################### 这个是新加的 ########################################################
# 添加目标点参数
target_point = np.array([0.4, 0.5])  # 示例目标点
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
    g = np.array([g1, g2])

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



def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.square(y_true - y_pred)))


ads_sim_data = np.loadtxt('D:\computer document\pythonProject_phd\Complex_periodicity_DS11\Complex_periodicity_DS\sim\sim_data_'+ type + '.txt')
ads_sim_data = ads_sim_data +np.array([-0.06103107, -0.06123884])  #
# ads_sim_data = ads_sim_data + np.array([ 0.53424137, -0.00482774])
E_ADS_data = np.loadtxt('D:\computer document\pythonProject_phd\Complex_periodicity_DS11\Complex_periodicity_DS\sim\sim_data_2D_unregular.txt')+np.array([ 0.53424137, -0.00482774])


S = compute_Sea(x_set, x_tra)
S1 = compute_Sea(x_set, ads_sim_data)
# S2 = compute_Sea(x_set, E_ADS_data)

Rmse_dmp = rmse(x_set, x_tra[:-1,:])
Rmse_ds =rmse(x_set, ads_sim_data[:-1,:])
# Rmse_eds = rmse(x_set, E_ADS_data[:-1,:])
plt.figure()
plt.plot(ads_sim_data[:,0])
# plt.plot(E_ADS_data[:,0],c='b')
plt.plot(x_tra[:,0],c='g')
# plt.show()


print(Rmse_dmp)
print(Rmse_ds)
# print(Rmse_eds)

print('dmp is:', S)
print('PA is:', S1)
# print('DS is:', S2)
plt.figure()
plt.subplots_adjust(left=0.12, right=0.99, wspace=0.15, hspace=0.2, bottom=0.1, top=0.99)
plt.plot(ads_sim_data[:,0], ads_sim_data[:,1], c='r',label='PA')
plt.plot(x_tra[:, 0], x_tra[:, 1], c='black',label='Dmp')
# plt.plot(E_ADS_data[:, 0], E_ADS_data[:, 1], c='#2ca02c',label='GP')

plt.scatter(x_set[::2, 0], x_set[::2, 1], marker='x', color='blue',label='Demo')
plt.scatter(x_set[0, 0], x_set[0, 1], s=60, marker='o',color='black',label='Repro. start')

plt.scatter(target_point[0],target_point[1])

plt.xlabel(r'$\mathit{x_1}$/m')
plt.ylabel(r'$\mathit{x_2}$/m')

plt.legend(loc='upper left')
plt.show()
# '''
# T = 10.0
# Max_steps = int(T / Period)
# for i in range(Max_steps):
# '''

