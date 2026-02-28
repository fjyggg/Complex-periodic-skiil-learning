import numpy as np
import matplotlib.pyplot as plt
from cvxopt import *
import torch

class ADS_learn():
    def __init__(self, ori_model, model_trai, Learning_Lyapunov_model, training_options, X,Y,b,max_v,likelihood_noise):
        self.X = X
        self.Y = Y
        self.b = b
        self.max_v = max_v
        # 导入初始模型
        self.ads_orginal = ori_model
        # 梯度获取
        self.grad_get = model_trai
        # lyapunov model
        self.lf_learner = Learning_Lyapunov_model
        self.training_options = training_options


    # 这里是与LF有关的约束优化
    def ads_evolution(self,x):
        o_x_dot = self.ads_orginal.predict(x.reshape(1, -1))   # (1,2)
        o_x_dot = matrix(o_x_dot)
        # max_v = np.max(np.array((self.max_v, self.b * lf.paper21(x.reshape(1,-1))),dtype=object)) # 速度的那一项拿出来了,速度给的最大是0.1

        def obj(u=None, z=None):
            if u is None:
                return 1, matrix(0.0, (2, 1))
            f_value = matrix(0.0, (2, 1))  # 这里还是有点点问题
            f_gradient = matrix(0.0, (2, 2))
            f_value[0, 0] = sum(u.T * u)
            # f_value[1, 0] = sum(u.T * u)
            # f_value[1, 0] = sum((u + o_x_dot).T * (u + o_x_dot) - max_v ** 2)  # 这步和下面的那步
            f_gradient[0, :] = 2 * u.T
            # f_gradient[1, :] = 2 * u.T
            # f_gradient[1, :] = 2 * (u + o_x_dot).T
            if z is None:
                return f_value, f_gradient
            I = spmatrix(1.0, range(2), range(2))
            return f_value, f_gradient, 2 * (z[0] + z[1]) * I

        # dv_dx = lf.dVdx(x)  #(1, 2)
        x_set_tensor = torch.from_numpy(x).float()
        dv_dx = self.grad_get.get_dvdx(self.lf_learner, x_set_tensor)
        dv_dx = dv_dx.astype(np.float64)

        Vx = self.lf_learner(x_set_tensor)
        Vx = Vx.detach().numpy()

        A = matrix(dv_dx, (1, 2))

        b = -self.b / (self.lf_learner.V_target-0.00) * Vx + self.b - dv_dx.dot(np.array(o_x_dot).reshape(-1))   # 这边就相当于是把（x+u)移位了，这里的优化参数是u啊
        # b = -self.b / 1.0 * Vx + self.b - dv_dx.dot(np.array(o_x_dot).reshape(-1))   # 这边就相当于是把（x+u)移位了，这里的优化参数是u啊
        b = matrix(b, (1, 1))

        solvers.options['feastol'] = self.training_options['feastol']
        solvers.options['abstol'] = self.training_options['abstol']
        solvers.options['reltol'] = self.training_options['reltol']
        solvers.options['maxiters'] = self.training_options['maxiters']
        solvers.options['show_progress'] = self.training_options['show_progress']  # 应该是求解要的参数吧，不懂
        u = solvers.cp(obj, A=A, b=b)['x']   # cp求解的是一个非线性目标
        # print(np.array(u))
        return np.array(o_x_dot).reshape(-1), np.array(u).reshape(-1)

    def plot_vector_field(self, Type, handle=None, gap=1):
        plot_flag = False
        if handle is None:
            handle = plt
            plot_flag = True

        x_set = self.X

        x_1_min = np.min(x_set[:, 0])
        x_1_max = np.max(x_set[:, 0])
        x_2_min = np.min(x_set[:, 1])
        x_2_max = np.max(x_set[:, 1])

        delta_x1 = x_1_max - x_1_min
        x_1_min = x_1_min - 0.3 * delta_x1
        x_1_max = x_1_max + 0.3 * delta_x1
        delta_x2 = x_2_max - x_2_min
        x_2_min = x_2_min - 0.3 * delta_x2
        x_2_max = x_2_max + 0.3 * delta_x2

        num = 100
        step = np.min(np.array([(x_1_max - x_1_min) / num, (x_2_max - x_2_min) / num]))
        area_Cartesian = {'x_1_min': x_1_min, 'x_1_max': x_1_max, 'x_2_min': x_2_min, 'x_2_max': x_2_max, 'step': step}

        area = area_Cartesian

        step = area['step']
        x = np.arange(area['x_1_min'], area['x_1_max'], step)
        y = np.arange(area['x_2_min'], area['x_2_max'], step)
        X, Y = np.meshgrid(x, y)  #  建立网格表
        length_x = np.shape(x)[0]
        length_y = np.shape(y)[0]
        Dot_x = np.zeros((length_y, length_x))
        Dot_y = np.zeros((length_y, length_x))   # 这一步为什么y是放在前面的
        for i in range(length_y):
            for j in range(length_x):
                pose = np.array([x[j], y[i]])
                o_x_dot, u = self.ads_evolution(pose)
                Dot_x[i, j], Dot_y[i, j] = o_x_dot + u
                # print(t2 - t1)
        # fig, ax = handle.subplots()
        # 这个是画流图用的
        handle.streamplot(X, Y, Dot_x, Dot_y, density=1.0, color='red', linewidth=0.3, maxlength=0.2, minlength=0.1, arrowstyle='simple', arrowsize=0.5)
        handle.scatter(self.X[0::4, 0], self.X[0::4, 1], c='blue', alpha=1.0, s=10, marker='x', label='Demonstration')

        # Rshape
        # x0s = x_set[0, :] + np.array([0.2,1.0])
        # x0s1 = x_set[0, :] + np.array([3,0.3])
        # x0s2 = x_set[0, :] + np.array([1.5,2.1])
        # x0s3 = x_set[0, :] + np.array([-0.7,3])
        # x0s4 = np.array([-0.186,0.97])

        # # dataset_2d_rectangle
        # x0s = np.array([0.0021, 0.0282])
        # x0s1 = np.array([0.0148, 0.1280])
        # x0s2 = np.array([-0.0271, -0.0481])
        # x0s3 = np.array([0.0663, -0.0946])
        # x0s4 = np.array([-0.0700, 0.1184])

        # # # dataset_2d_circle
        # x0s = np.array([-0.012, 0.0472])
        # x0s1 = np.array([-0.0162, -0.0203])
        # x0s2 = np.array([-0.0403, -0.1061])
        # x0s3 = np.array([-0.0443, 0.1314])
        # x0s4 = np.array([0.0654, -0.0666])

        # # # dataset_2d_circle
        # x0s = np.array([-0.0291, 0.0052])
        # x0s1 = np.array([-0.0016, 0.0642])
        # x0s2 = np.array([-0.0824, -0.0621])
        # x0s3 = np.array([0.0634, -0.0945])
        # x0s4 = np.array([-0.0287, 0.1298])

        # # # Ishape
        # x0s = np.array([-0.081, 0.063])
        # x0s1 = np.array([0.074, -0.079])
        # x0s2 = np.array([-0.121, 0.630])
        # x0s3 = np.array([0.191, 0.192])
        # x0s4 = np.array([-0.024, -0.252])

        # Oshape
        # x0s = np.array([-0.454, -0.499])
        # x0s1 = np.array([0.496, -0.313])
        # x0s2 = np.array([-0.0780, 0.304])
        # x0s3 = np.array([-0.518, 0.474])
        # x0s4 = np.array([0.267, 0.349])

        # Oshape
        # x0s = np.array([-0.0986, -0.0729])
        # x0s1 = np.array([-0.0698, -0.1198])
        # x0s2 = np.array([-0.0173, 0.0010])
        # x0s3 = np.array([0.0206, 0.0454])
        # x0s4 = np.array([0.0746, 0.0599])

        # Zshape_real_robot_experiment
        # x0s = np.array([-0.15, 0.0])
        # x0s = np.array([-0.14979077,  0.00319746 ])
        # x0s = np.array([-0.14346477,  0.0092514 ])
        # x0s = np.array([-0.14977508,  0.00310145])
        # x0s1 = np.array([-0.0698, -0.1198])
        # x0s2 = np.array([-0.0173, 0.0010])
        # x0s3 = np.array([0.0206, 0.0454])
        # x0s4 = np.array([0.0746, 0.0599])

        # Jshape real_robot_experiment
        x0s = np.array([-0.1304, 0.0743])
        # x0s1 = np.array([-0.0588, -0.1214])
        # x0s2 = np.array([0.1104, 0.1546])
        # x0s3 = np.array([0.0528, 0.0305])
        # x0s4 = np.array([-0.0230, -0.0430])

        # # # #
        # # # # #
        handle.scatter(x0s[0], x0s[1], c='black', alpha=1.0, s=50, marker='*', label='Start point')
        # handle.scatter(x0s1[0], x0s1[1], c='black', alpha=1.0, s=50, marker='*')
        # handle.scatter(x0s2[0], x0s2[1], c='black', alpha=1.0, s=50, marker='*')
        # handle.scatter(x0s3[0], x0s3[1], c='black', alpha=1.0, s=50, marker='*')
        # handle.scatter(x0s4[0], x0s4[1], c='black', alpha=1.0, s=50, marker='*')

        self.aa = True
        self.plot_repro(x0s, plot_handle=handle)
        # self.plot_repro1(x0s, Type, plot_handle=handle)
        # self.plot_repro(x0s1, plot_handle=handle)
        # self.plot_repro(x0s2, plot_handle=handle)
        # self.plot_repro(x0s3, plot_handle=handle)
        # self.plot_repro(x0s4, plot_handle=handle)
        if plot_flag is True:
            handle.show()

    def plot_repro1(self, x0, Type, plot_handle=None):
        x = x0
        period = 1e-2
        steps = int(30 / period)
        x_tra = [x]
        for i in range(steps):
            desired_v, u = self.ads_evolution(x)
            x = x + (desired_v+u) * period
            x_tra.append(x)
        x_tra = np.array(x_tra)
        np.savetxt('simulation_data/' + 'sim_data_' + Type + '.txt', x_tra)
        show_flag = False
        if plot_handle is None:
            import matplotlib.pyplot as plt
            plot_handle = plt
            show_flag = True
        x_tra = np.array(x_tra)
        plot_handle.plot(x_tra[:, 0], x_tra[:, 1], c='black', linewidth=2, alpha=1.0)
        if show_flag is True:
            plot_handle.show()

    def plot_repro(self, x0, plot_handle=None):
        x = x0
        period = 1e-2
        steps = int(30 / period)
        x_tra = [x]
        for i in range(steps):
            desired_v, u = self.ads_evolution(x)
            x = x + (desired_v+u) * period
            x_tra.append(x)

        show_flag = False
        if plot_handle is None:
            import matplotlib.pyplot as plt
            plot_handle = plt
            show_flag = True
        x_tra = np.array(x_tra)

        if self.aa ==True:
            plot_handle.plot(x_tra[:, 0], x_tra[:, 1], c='black', linewidth=2, alpha=1.0, label='Reproduction traj.')
            self.aa = False
        else:
            plot_handle.plot(x_tra[:, 0], x_tra[:, 1], c='black', linewidth=2, alpha=1.0)
        # 显示图例
        plot_handle.xlabel('X/m', fontsize=12, style='italic')  # X轴标签斜体
        plot_handle.ylabel('Y/m', fontsize=12, style='italic')  # Y轴标签斜体
        plot_handle.legend(fontsize=10, loc='upper left')
        if show_flag is True:
            plot_handle.show()