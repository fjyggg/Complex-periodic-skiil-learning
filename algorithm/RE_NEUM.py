import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time  # 需要在文件开头导入

# 定义模型
class Lyapunov_REF(nn.Module):
    def __init__(self, x_set, dot_x_set, input_dim=2, output_dim=2, hidden_dim=10,margin=0.01):
        super().__init__()
        self.x_set = x_set
        self.dot_x_set = dot_x_set
        self.d_x = np.shape(self.x_set)[1]

        row_num = x_set.shape[0]
        self.V_target = 1

        self.x_set = x_set  # 输入轨迹点 [N, 2]
        # 计算 x_set 的最小和最大值（按列）
        # x_set_tensor = torch.tensor(x_set) if not isinstance(x_set, torch.Tensor) else x_set
        # self.x_min = torch.min(x_set_tensor, dim=0, keepdim=True)[0]  # shape: [1, 2]
        # self.x_max = torch.max(x_set_tensor, dim=0, keepdim=True)[0]  # shape: [1, 2]
        # 计算轨迹的边界（带安全裕度）
        self.margin = margin
        self.x_min = torch.min(self.x_set, dim=0)[0] * (1 + margin)  # 扩大边界
        self.x_max = torch.max(self.x_set, dim=0)[0] * (1 + margin)


        # self.center = nn.Parameter(torch.mean(x_set, dim=0, keepdim=True))
        # self.center = torch.tensor([[0.04, -0.04]])  # 形状 (1, 3)
        # self.center = torch.tensor([[0., -0.]])  # 形状 (1, 3)

        # self.center = nn.Parameter(torch.mean(x_set, dim=0, keepdim=True))
        self.center = torch.mean(x_set, dim=0, keepdim=True) - torch.tensor([[0.0, 0.06]])  # 这个是Ushape的
        # print(self.center.shape)
        # self.center = torch.mean(x_set, dim=0, keepdim=True)
        # self.center = torch.tensor([[-0., -0.033]])  # 形状 (1, 3)

        # self.center.register_hook(lambda grad: grad * (self.x_min <= self.center) * (self.center <= self.x_max))

        # 参数网络 (ω, α, β ∈ R^m) 与旋转网络
        self.para_net = ParameterNetworks(input_dim, output_dim, hidden_dim)
        self.dir_net = DirectionNetworks(input_dim, hidden_dim=10, recursive=True)

            # 非线性变换ρ(s) = tanh(s) = (e^s - e^-s)/(e^s + e^-s)
        self.rho = nn.Tanh()

    def forward(self, xi):
        '''
        :param r_y: 暂时未加入旋转向量的最原始数据
        :return: V
        '''
        xi_centered = xi - self.center
        y = self.dir_net(xi_centered)
        # y = self.dir_net(xi)

        r_y_normalized = F.normalize(y + 1e-8, p=2, dim=-1)  # 添加小常数防止除零
        r_y = torch.norm(y, p=2, dim=1, keepdim=True)

        # 获取方向相关参数
        omega, alpha, beta = self.para_net(r_y_normalized)  # ω,α,β ∈ R^m #omega_shape: torch.Size([98, 2]), alpha_shape: torch.Size([98, 2]), beta_shape: torch.Size([98, 2])

        # 计算q(y) ∈ R^m
        s = alpha * r_y + beta  # α_i*r_y + β_i
        q = self.rho(s)        # ρ(s) = tanh(s)

        # 计算λ(y) = ω^T q
        lambda_y = (omega*q).sum(dim=-1, keepdim=True)  # 点积


        z = (lambda_y*y)

        V = torch.sum(torch.pow(z, 2), dim=1)
        # 最终输出
        return V  # fλ(y) = λ(y)·y


    def periodic_constraint(self):
        """计算周期轨迹上的Lyapunov值误差"""
        V_periodic = self(self.x_set)
        return torch.mean((V_periodic - self.V_target)**2)  # MSE损失

    def plot_result(self, Type, epoch):
    # def plot_result(self):
        '''plot'''
        x_set = self.x_set.detach().numpy()
        d_x = self.dot_x_set.detach().numpy()
        x_1_min = np.min(x_set[:, 0])
        x_1_max = np.max(x_set[:, 0])
        x_2_min = np.min(x_set[:, 1])
        x_2_max = np.max(x_set[:, 1])

        delta_x1 = x_1_max - x_1_min
        x_1_min = x_1_min - 0.4 * delta_x1
        x_1_max = x_1_max + 0.3 * delta_x1
        delta_x2 = x_2_max - x_2_min
        x_2_min = x_2_min - 0.3 * delta_x2
        x_2_max = x_2_max + 0.3 * delta_x2

        num = 100
        num_levels = 10
        step = np.min(np.array([(x_1_max - x_1_min) / num, (x_2_max - x_2_min) / num]))
        area_Cartesian = {'x_1_min': x_1_min, 'x_1_max': x_1_max, 'x_2_min': x_2_min, 'x_2_max': x_2_max,
                          'step': step}
        area = area_Cartesian
        x1 = np.arange(area['x_1_min'], area['x_1_max'], step)
        x2 = np.arange(area['x_2_min'], area['x_2_max'], step)
        length_x1 = np.shape(x1)[0]
        length_x2 = np.shape(x2)[0]
        X1, X2 = np.meshgrid(x1, x2)
        V = np.zeros((length_x2, length_x1))
        V_list = np.zeros(length_x2 * length_x1)

        for i in range(length_x2):
            for j in range(length_x1):
                x = np.array([x1[j], x2[i]])
                x_tensor = torch.from_numpy(x).float()
                V11 = self(x_tensor.reshape(1, 2))
                V_array = V11.detach().numpy()
                V[i, j] = V_array
                V_list[i * length_x1 + j] = V[i, j]
        levels = np.sort(V_list, axis=0)[0::int(length_x2 * length_x1 / num_levels)]
        levels_ = []
        for i in range(np.shape(levels)[0]):
            if i == 0:
                levels_.append(levels[i])
            else:
                if levels[i] != levels[i - 1]:
                    levels_.append(levels[i])
        levels_ = np.array(levels_)
        contour = plt.contour(X1, X2, V, levels=levels_, alpha=0.8, linewidths=1.0)
        plt.clabel(contour, inline=True, fontsize=8, fmt='%.2f')  # 在等高线上标注函数值

        center = self.center.detach().numpy()
        # plt.scatter(center[:,0], center[:,0], marker='D', s=50, label = 'Center point')

        plt.scatter(x_set[:, 0], x_set[:, 1], s=10, marker='o')

        # center = self.center.detach().numpy()
        # plt.scatter(center[:,0], center[:,1], s=10, marker='o')

        # # plt.show()
        # plt.gcf().savefig(f'figure/{Type}_{epoch}_lyapunov_result.png', dpi=300, bbox_inches='tight')
        # # plt.show()
        # plt.clf()  # 清除当前图形（Clear Figure）



class Exp(nn.Module):
    """自定义指数激活模块"""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.exp(x)

# 定义了三个参数矩阵
class ParameterNetworks(nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_dim=200):
        super().__init__()

        # fλ1网络：双隐藏层MLP → 输出ω ∈ R^5 (exp激活)
        self.f_lambda1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            Exp()

        )

        # fλ2网络：双隐藏层MLP → 输出α ∈ R^5 (exp激活)
        self.f_lambda2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            Exp()
        )

        # fλ3网络：双隐藏层MLP → 输出β ∈ R^5 (无激活)
        self.f_lambda3 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, r_y_normalized):
        # 满足径向增加的要求，y本身是带有旋转的
        omega = self.f_lambda1(r_y_normalized)  # ω ∈ R^5 (ω_i > 0)
        alpha = self.f_lambda2(r_y_normalized)  # α ∈ R^5 (α_i > 0)
        beta = self.f_lambda3(r_y_normalized)  # β ∈ R^5 (无约束)
        return omega, alpha, beta


# 这个先不加
class DirectionNetworks(nn.Module):
    def __init__(self, input_dim, hidden_dim=10, recursive=True):
        super().__init__()
        self.input_dim = input_dim
        self.recursive = recursive

        # 角度预测网络 fθ(rs)
        self.angle_net = nn.Sequential(
            nn.Linear(3, hidden_dim),  # 输入rs是标量
            nn.SiLU(),  # 更平滑的激活函数
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加归一化
            nn.SiLU(),  # 更平滑的激活函数
            nn.Linear(hidden_dim, 1)  # 输出旋转角度θ
        )

    def forward(self, s):
        # 基础二维旋转情况
        rs = s.norm(p=2, dim=-1, keepdim=True).pow(2)  # rs = s1² + s2²
        combined = torch.cat([s, rs], dim=-1)  # 输出形状 [171, 3]
        theta = self.angle_net(combined)* 3.1416*0.5 # θ = fθ(rs),限制在[-pi,pi],乘上系数0.5，避免过度非线性
        # theta = self.angle_net(rs)* 3.1416 *0.5  # θ = fθ(rs),限制在[-pi,pi],乘上系数0.5，避免过度非线性

        # 构造旋转矩阵
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)  #torch.Size([588, 1])
        # 二维旋转, 这里的乘法有问题
        rotation_matrix = torch.stack([
            torch.cat([cos_theta, sin_theta], dim=-1),
            torch.cat([-sin_theta, cos_theta], dim=-1)
        ], dim=1)

        # 批量矩阵乘法 [batch, 2, 2] @ [batch, 2] -> [batch, 2]

        s_rotated = torch.bmm(rotation_matrix, s.unsqueeze(-1)).squeeze(-1)
        return s_rotated


# 损失函数设置
class Lyapunov_function_Loss(nn.Module):
    def __init__(self, alpha = 0.1, beta = 5.0):
        super().__init__()
        self.alpha = alpha  # 正则化系数
        self.beta = beta

        self.rho1 = nn.Tanh()

    def dVdx(self,model , xi):
        # 确保y需要梯度
        xi = xi.clone().requires_grad_(True)
        # 计算V(y)
        V = model(xi)
        # 计算dV/dx
        dV_dx = torch.autograd.grad(
            outputs=V.sum(),  # 对batch求和以获得每个样本的梯度
            inputs=xi,
            create_graph=True,  # 保持计算图以用于二阶导数
            retain_graph=True,
            only_inputs=True
        )[0]
        return dV_dx

    def forward(self, model, xi, dxi):
        # 确保y需要梯度
        xi = xi.clone().requires_grad_(True)

        V = model(xi)

        dV_dxi = torch.autograd.grad(outputs=V.sum(),  # 对batch求和以获得每个样本的梯度
            inputs=xi, create_graph=True,  # 保持计算图以用于二阶导数
            retain_graph=True, only_inputs=True)[0]

        dxi_dt = dxi  # 需要从你的数据中获取 [batch_size, input_dim]
        J_sum = torch.sum(dV_dxi * dxi_dt, dim=1, keepdim=True)
        dV_dxi_norm = torch.norm(dV_dxi, p=2, dim=1, keepdim=True)
        dxi_dt_norm = torch.norm(dxi_dt, p=2, dim=1, keepdim=True)
        J = J_sum/(dV_dxi_norm*dxi_dt_norm + 1e-10)

        J_squared = J**2
        J_zeta = self.rho1(self.beta*J_squared)
        #
        main_loss = J_zeta.mean()

        # 添加参数正则化
        l2_loss = 0.0
        for param in model.parameters():
            l2_loss += param.pow(2).sum()  # L2正则化

        # 添加周期轨迹约束
        periodic_loss = model.periodic_constraint()

        # 新增趋势匹配损失（强制V与轨迹半径同步变化）
        r_traj = xi.norm(dim=1)
        V_values = model(xi)
        trend_loss = F.mse_loss(V_values.diff(), r_traj.diff())
        # trend_loss = V_values.diff().pow(2).mean()

        return main_loss + self.alpha*l2_loss + 0.5*periodic_loss + 0.5*trend_loss
        # return main_loss + self.alpha*l2_loss + 0.5*periodic_loss

class ProjectedAdam(torch.optim.Adam):
    def __init__(self, params, lr, model):
        super().__init__(params, lr)
        self.model = model

    def step(self):
        super().step()
        # 投影到可行域
        with torch.no_grad():
            self.model.center.data = torch.clamp(
                self.model.center.data,
                min=self.model.x_min,
                max=self.model.x_min
            )



class Data_train:
    def train_model(self, model, x_set, dot_x_set, Type, epochs=1000):

        start_time = time.time()
        # optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
        loss_fn = Lyapunov_function_Loss(alpha=1e-4 , beta=2.5)

        # 使用时：
        # optimizer = ProjectedAdam(model.parameters(), lr=1e-2, model=model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-2, steps_per_epoch=1, epochs=epochs
        )
        for epoch in range(epochs):
            # 混合训练
            optimizer.zero_grad()
            loss = loss_fn(model, x_set, dot_x_set)
            # 梯度处理
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # 参数更新
            optimizer.step()
            scheduler.step()

            if (epoch+1) % 200 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            if (epoch + 1) > 1999:
                if (epoch + 1) % 500 == 0:
                    model.plot_result(Type, epochs)
                    # plt.show()
                    save_path = f'parameters/NNs_parameter_for_{Type}_epoch{epoch + 1}.pth'
                    # 训练完成后保存模型
                    torch.save(model.state_dict(), save_path)
                    print(f"训练完成，模型参数已保存到 {save_path}")

                    plt.gcf().savefig(f'figure/{Type}_{epoch+1}_lyapunov_result.png', dpi=300, bbox_inches='tight')

                    plt.clf()  # 清除当前图形（Clear Figure）

        # 计算训练总时间
        end_time = time.time()
        training_time = end_time - start_time

        # 格式化输出训练时间
        hours = int(training_time // 3600)
        minutes = int((training_time % 3600) // 60)
        seconds = training_time % 60

        print(f"训练完成!")
        print(f"总训练时间: {training_time:.2f} 秒")
        print(f"格式化时间: {hours:02d}:{minutes:02d}:{seconds:05.2f}")
        print(f"平均每轮时间: {training_time / epochs:.4f} 秒")



    def get_dvdx(self, model, x_set):
        loss_fn = Lyapunov_function_Loss(alpha=1e-4 , beta=1.5)  #如果曲线太弯，调大这个值beta值
        dVdx = loss_fn.dVdx(model, x_set)
        dVdx = dVdx.detach().numpy()
        return dVdx.reshape(1,2)
