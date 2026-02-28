from algorithm.RE_NEUM import ParameterNetworks, Lyapunov_REF, Data_train, Lyapunov_function_Loss
from algorithm.Learn_GPR_ODS import LearnOds, SingleGpr
from algorithm.Learn_SDS import LearnSds
from algorithm.ADS_learn import ADS_learn
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyLasaDataset as lasa
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(40)  # 设置随机种子保证可重复性
# Z_shape、J_shape和T_shape的beta选择为2.5，U_shape的beta选择为
# "Z_shape", "J_shape", "U_shape", "T_shape"
# 新增两个速度快速项"T_shape_velocity"
Type = 'U_shape'


data_set = np.loadtxt("D:\computer document\pythonProject_phd\Complex_periodicity_DS\dataset\ZJUT_dataset/"+ Type + '.txt')

data = data_set[500:,(1,2,4,5)]
data = data[::5,:]

period = 0.01
X = data[:,:2]

Mean_X = np.average(X, 0)  # 按行求均值，主要是使这样的x点位于中心，方便后面求范数
x_set = X - Mean_X
print(Mean_X)

# dx = np.zeros_like(x_set)
# dx[:-1, :] = (x_set[1:,:] - x_set[:-1,:])/period
# dot_x_set = dx

dot_x_set = data[:,2:]

# 转换为pytorch张量
x_set_tensor = torch.from_numpy(x_set).float()
dot_x_set_tensor = torch.from_numpy(dot_x_set).float()

input_num = x_set_tensor.shape[1]  # 输入维度
output_num = dot_x_set_tensor.shape[1]  # 输出维度

# ------------------- model training --------------------
Learning_Lyapunov_model = Lyapunov_REF(x_set_tensor, dot_x_set_tensor, input_num, output_num, hidden_dim=10)

model_train = Data_train()
# model_train.train_model(Learning_Lyapunov_model, x_set_tensor, dot_x_set_tensor, Type, epochs=8000)

# set parameters, Z_shape是epoch4000，J_shape是epoch7500，T_shape是epoch2500, U_shape是epoch4000, T_shape_velocity是epoch6000
Learning_Lyapunov_model.load_state_dict(torch.load('parameters/NNs_parameter_for_' + Type + '_epoch4000' + '.pth'))
# Learning_Lyapunov_model.load_state_dict(torch.load('parameters/NNs_parameter_for_' + Type + '_epoch2500' + '.pth'))
# Learning_Lyapunov_model.load_state_dict(torch.load('parameters/NNs_parameter_for_' + Type + '_epoch6000' + '.pth'))
# Learning_Lyapunov_model.eval()  # 设置为评估模式

# # ------------------- plot V_result, checking--------------------
Learning_Lyapunov_model.plot_result(Type, epoch=2000)
plt.show()


# ------------------- Learning (Loading) original ADS --------------------
observation_noise = None
gamma_oads = 0.5
ods_learner = LearnOds(x=x_set,y=dot_x_set, observation_noise=observation_noise, gamma=gamma_oads)
print('--- Start original ads training (loading) ---')
save_path = 'parameters/Oads_parameter_for_' + Type + '.txt'
# ods_learner.train(save_path)
oads_parameters = np.loadtxt(save_path)
ods_learner.set_param(oads_parameters)
print('--- Training (Loading) completed ---')
# need plot to look the ods's result
print('---plotting stable ODS results ...')
# ods_learner.show_learning_result()
print('---plotting ODS finished')
# plt.clf()


# # ------------------- Formulate the stable ADS ----------------------
b = 3
max_v = 0.1
likelihood_noise = 0.05
training_options = {'feastol': 1e-9, 'abstol': 1e-9, 'reltol': 1e-9, 'maxiters': 50, 'show_progress': False}
ads = ADS_learn(ods_learner, model_train,Learning_Lyapunov_model, training_options=training_options, X=x_set, Y=dot_x_set, b=b, max_v=max_v, likelihood_noise=likelihood_noise)   # 将ads实例化
# print('---plotting stable period ADS results ...')
# ads.plot_vector_field(handle=None, gap=3)
# plt.gcf().savefig('figure/'+Type+'_ADS_result.png', dpi=300, bbox_inches='tight')
# plt.show()

# --------------------- learning the third dimension-------------------------
observation_noise = None
gamma_oads = 0.5
Z = data_set[500::5, 3]
Mean_Z = np.average(Z)
Z = Z - Mean_Z

z_predicter = SingleGpr(X=x_set, y=Z, observation_noise=observation_noise, gamma=gamma_oads)
save_path1 = 'parameters/Oads_parameter_for_3d_' + Type + '.txt'
# z_predicter.train(save_path1)
z_predicter_param = np.loadtxt(save_path1)
z_predicter.set_param(z_predicter_param)
#
Maxsteps = 3000
Period = 0.02
collect_data = []
training_options = {'feastol': 1e-9, 'abstol': 1e-9, 'reltol': 1e-9, 'maxiters': 50, 'show_progress': False}
x = data_set[0, 1:3] - Mean_X
x1 = torch.from_numpy(x).float()
z = data_set[0, 3] - Mean_Z
print(Mean_Z)
x_list = [x]
z_list = [z]
V_list = [Learning_Lyapunov_model(x1).detach().numpy()]
v_list = []
gain_z = 1.0
for step in range(3000):
    o_x_dot, u = ads.ads_evolution(x=x.reshape(1, -1))
    z_pre = z_predicter.predict_determined_input(x.reshape(1, -1))
    z_pre = z_pre.reshape(-1)
    z_dot = -gain_z * (z - z_pre) + z_predicter.gradient2input(x).dot(o_x_dot + u)
    z = z + z_dot * Period
    x = x + (o_x_dot + u) * Period
    x_list.append(x)
    z_list.append(z)
    x2 = torch.from_numpy(x).float()
    V_list.append(Learning_Lyapunov_model(x2).detach().numpy())
    v_list.append(np.hstack((o_x_dot + u, z_dot)))

x_list = np.array(x_list)
z_list = np.array(z_list)
V_list = np.array(V_list)
v_list = np.array(v_list)
#
np.savetxt('simulation_data/' + 'sim_data_' + Type + '.txt', np.hstack((x_list, z_list.reshape(-1, 1))))
# np.savetxt('simulation_data/' + 'sim_data_' + Type + '_V.txt', V_list)
# np.savetxt('simulation_data/' + 'sim_data_' + Type + '_velocity.txt', v_list)
#
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111,projection='3d')


ax.plot(x_list[:, 0]+Mean_X[0], x_list[:, 1]+Mean_X[1], z_list, c='red')
ax.scatter3D(X[0::3, 0], X[0::3, 1], Z[0::3], c='blue', alpha=1.0, s=10, marker='x')
plt.show()
