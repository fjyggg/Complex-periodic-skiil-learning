import autograd.numpy as np
from autograd import value_and_grad
from scipy.optimize import minimize
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.numpy.linalg import solve
import time
np.random.seed(5)

class SingleGpr:
    def __init__(self, X, y, observation_noise=None, gamma=0.5):
        '''
        Initializing the single GPR
        :param X: Input set, (input_num * input_dim)
        :param y: Output set for one dim, (n_size,)
        :param observation_noise: the standard deviation for observation_noise
        :param gamma: a scalar which will be used if observation_noise is None
        '''
        self.input_dim = np.shape(X)[1]
        # add the equilibrium-point (0, 0) to the set
        self.X = np.vstack((X, np.zeros(self.input_dim).reshape(1, -1)))
        self.y = np.hstack((y, np.array([0.0])))
        self.input_num = np.shape(self.X)[0]
        if observation_noise is None:
            self.observation_noise = gamma * np.sqrt(np.average(self.y**2))
        else:
            self.observation_noise = observation_noise
        self.param = self.init_random_param()
        # Used to tune the degree of the equilibrium-point confidence
        self.determined_point_degree = 0.0
        self.solve_cov_y = None

    def init_random_param(self):
        '''
        Initializing the hyper-parameters
        '''
        sqrt_kernel_length_scale = np.sqrt(np.diag(np.cov(self.X.T)))
        kernel_noise = np.sqrt(np.average(self.y**2))
        param = np.hstack((kernel_noise, sqrt_kernel_length_scale))
        return param

    def set_param(self, param):
        '''
        Manually set the hyper-parameters
        '''
        self.param = param.copy()
        '''
        pre-computations for prediction
        '''
        self.cov_y_y = self.rbf(self.X, self.X, self.param)
        temp = self.observation_noise ** 2 * np.eye(self.input_num)
        # observation noises for determined set should be zero
        temp[-1, -1] = self.determined_point_degree
        self.cov_y_y = self.cov_y_y + temp
        self.beta = solve(self.cov_y_y, self.y)  # The constant vector of the mean prediction function
        self.inv_cov_y_y = solve(self.cov_y_y, np.eye(self.input_num))

        self.solve_cov_y = solve(self.cov_y_y, self.y)

    def build_objective(self, param):
        '''
        The obj of Single GPR
        '''
        cov_y_y = self.rbf(self.X, self.X, param)
        temp = self.observation_noise**2 * np.eye(self.input_num)
        # observation noises for determined set should be zero
        temp[-1, -1] = self.determined_point_degree
        cov_y_y = cov_y_y + temp
        out = - mvn.logpdf(self.y, np.zeros(self.input_num), cov_y_y)
        return out

    def train(self, save_path=None):
        '''
        Training Single GPR
        '''
        start_time = time.time()
        result = minimize(value_and_grad(self.build_objective), self.param, jac=True, method='L-BFGS-B', tol=1e-8,
                          options={'maxiter': 50, 'disp': False})
        self.param = result.x
        # pre-computation for prediction
        self.cov_y_y = self.rbf(self.X, self.X, self.param)
        temp = self.observation_noise ** 2 * np.eye(self.input_num)
        temp[-1, -1] = self.determined_point_degree
        self.cov_y_y = self.cov_y_y + temp
        self.beta = solve(self.cov_y_y, self.y)
        self.inv_cov_y_y = solve(self.cov_y_y, np.eye(self.input_num))

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

        if save_path is not None:
            np.savetxt(save_path, self.param)

    def rbf(self, x, x_, param):
        '''
        Construct the kernel matrix (or scalar, vector)
        '''
        kn = param[0]  # abbreviation for kernel_noise
        sqrt_kls = param[1:]  # abbreviation for sqrt_kernel_length_scale
        '''
        Using the broadcast technique to accelerate computation
        '''
        diffs = np.expand_dims(x / sqrt_kls, 1) - np.expand_dims(x_ / sqrt_kls, 0)
        return kn ** 2 * np.exp(-0.5 * np.sum(diffs ** 2, axis=2))

    def predict_determined_input(self, inputs):
        '''
        Prediction of Single GPR
        '''
        cov_y_f = self.rbf(self.X, inputs, self.param)
        means = np.dot(cov_y_f.T, self.beta)  # (m,)
        return means

    def gradient2input(self, input):
        sqrt_kern_length_scale = self.param[1:]
        temp1 = np.dot(self.X - input, np.diag(1 / (sqrt_kern_length_scale**2)))
        cov_y_f = self.rbf(self.X, input.reshape(1, -1), self.param)  # 300,1
        temp2 = (temp1*cov_y_f).T
        gradient = temp2@self.solve_cov_y.reshape(-1,1)
        return gradient.reshape(-1)


class MultiGpr:
    def __init__(self, X, Y, observation_noise=None, gamma=0.5):
        '''
        MultiGPR is a stack of multiple single GPRs;
        :param X: Input set, (input_num * input_dim)
        :param Y: Output set, (input_num * output_dim)
        :param observation_noise: the standard deviation for observation_noise
        '''
        self.X = X
        self.Y = Y
        self.input_dim = np.shape(X)[1]
        self.input_num = np.shape(X)[0]
        self.output_dim = np.shape(Y)[1]
        self.observation_noise = observation_noise
        self.gamma = gamma
        self.models = self.create_models()

    def set_param(self, param):
        '''
        Manually set the parameter
        '''
        for i in range(self.output_dim):
            self.models[i].set_param(param[i])

    def create_models(self):
        '''
        Creating a stack of single GPR
        '''
        models = []
        for i in range(self.output_dim):
            if self.observation_noise is not None:
                models.append(SingleGpr(self.X, self.Y[:, i], observation_noise=self.observation_noise[i], gamma=self.gamma))
            else:
                models.append(SingleGpr(self.X, self.Y[:, i], observation_noise=None, gamma=self.gamma))
        return models

    def train(self, save_path=None):
        '''
        Training multi-PGR
        '''
        for i in range(self.output_dim):
            print('training model ', i, '...')
            self.models[i].train()
            if i == 0:
                param = self.models[i].param.copy()
            else:
                param = np.vstack((param, self.models[i].param.copy()))
        if save_path is not None:
            np.savetxt(save_path, param)

    def predict_determined_input(self, inputs):
        '''
        Prediction of Multi-GPR
        '''
        for i in range(self.output_dim):
            if i == 0:
                means = self.models[0].predict_determined_input(inputs).copy()
            else:
                means = np.vstack((means, self.models[i].predict_determined_input(inputs).copy()))
        means = means.T
        return means  # (m, output_dimension)


class LearnOds:
    def __init__(self, x, y, observation_noise=None, gamma=0.5):
        '''
        Initializing the original ADS
        :param manually_design_set: (x_set, dot_x_set, t_set)
        :param observation_noise: the standard deviation for observation_noise
        :param gamma: a scalar which will be used if observation_noise is None
        '''
        self.x_set = x
        self.dot_x_set = y

        self.d_x = np.shape(self.x_set)[1]
        if np.isscalar(observation_noise):
            self.MultiGpr = MultiGpr(self.x_set.reshape(-1, self.d_x), self.dot_x_set.reshape(-1, self.d_x), np.ones(self.d_x) * observation_noise, gamma=gamma)
        else:
            self.MultiGpr = MultiGpr(self.x_set.reshape(-1, self.d_x), self.dot_x_set.reshape(-1, self.d_x), observation_noise, gamma=gamma)

    def set_param(self, param):
        '''
        Set parameters of the original ADS
        '''
        self.MultiGpr.set_param(param)

    def train(self, save_path=None):
        '''
        Training the original ADS
        '''
        self.MultiGpr.train(save_path=save_path)

    def predict(self, inputs):
        '''
        Prediction of the orginal ADS
        '''
        outputs = self.MultiGpr.predict_determined_input(inputs)
        return outputs


    def show_learning_result(self, plot_handle=None, stream_flag=True, energy_flag=False, area_Cartesian=None):

        if area_Cartesian is None:
            x_1_min = np.min(self.x_set.reshape(-1, self.d_x)[:, 0])
            x_1_max = np.max(self.x_set.reshape(-1, self.d_x)[:, 0])
            x_2_min = np.min(self.x_set.reshape(-1, self.d_x)[:, 1])
            x_2_max = np.max(self.x_set.reshape(-1, self.d_x)[:, 1])

            delta_x1 = x_1_max - x_1_min
            x_1_min = x_1_min - 0.2 * delta_x1
            x_1_max = x_1_max + 0.2 * delta_x1
            delta_x2 = x_2_max - x_2_min
            x_2_min = x_2_min - 0.2 * delta_x2
            x_2_max = x_2_max + 0.2 * delta_x2

            num = 100
            step = np.min(np.array([(x_1_max - x_1_min) / num, (x_2_max - x_2_min) / num]))
            area_Cartesian = {'x_1_min': x_1_min, 'x_1_max':x_1_max, 'x_2_min': x_2_min, 'x_2_max': x_2_max, 'step': step}
            area = area_Cartesian
            step = area['step']
            x1 = np.arange(area['x_1_min'], area['x_1_max'], step)
            x2 = np.arange(area['x_2_min'], area['x_2_max'], step)
            length_x1 = np.shape(x1)[0]
            length_x2 = np.shape(x2)[0]
            X1, X2 = np.meshgrid(x1, x2)
            Dot_x1 = np.zeros((length_x2, length_x1))
            Dot_x2 = np.zeros((length_x2, length_x1))

            if stream_flag is True:
                for i in range(length_x2):
                    for j in range(length_x1):
                        x = np.array([x1[j], x2[i]])
                        desired_v = self.predict(x)
                        desired_v = desired_v.reshape(1,2)
                        Dot_x1[i, j], Dot_x2[i, j] = desired_v[0,0],desired_v[0,1]
            show_flag = False
            if plot_handle is None:
                import matplotlib.pyplot as plt
                plot_handle = plt
                show_flag = True
            if stream_flag is True:
                plot_handle.streamplot(X1, X2, Dot_x1, Dot_x2, density=1.0, linewidth=0.5, maxlength=1.0, minlength=0.1,
                                       arrowstyle='simple', arrowsize=0.8)

            mark_size = 3
            plot_handle.scatter(0, 0, c='black', alpha=1.0, s=50, marker='X',label='Goal')
            plot_handle.scatter(self.x_set.reshape(-1, self.d_x)[::1, 0],
                                self.x_set.reshape(-1, self.d_x)[::1, 1], c='red', alpha=1.0, s=mark_size,
                                marker='o',label='Demos')

            plot_handle.legend()

            if show_flag is True:
                plot_handle.show()

    def plot_repro(self, x0, plot_handle=None):
        x = x0
        period = 1e-2
        steps = int(40 / period)
        x_tra = x
        for i in range(steps):
            desired_v= self.predict(x)
            x = x + desired_v * period
            x = x.reshape(1,2)
            x_tra = np.vstack((x_tra,x))
        show_flag = False
        if plot_handle is None:
            import matplotlib.pyplot as plt
            plot_handle = plt
            show_flag = True
        x_tra = np.array(x_tra)
        plot_handle.plot(x_tra[:, 0], x_tra[:, 1], c='blue', linewidth=2, alpha=1.0,label='Repeo. trajectory')
        if show_flag is True:
            plot_handle.show()

    def plot_repro1(self, x0, plot_handle=None):
        x = x0
        period = 1e-2
        steps = 250
        x_tra = x
        for i in range(steps):
            desired_v= self.predict(x)
            x = x + desired_v * period
            x = x.reshape(1,2)
            x_tra = np.vstack((x_tra,x))
        show_flag = False
        if plot_handle is None:
            import matplotlib.pyplot as plt
            plot_handle = plt
            show_flag = True
        x_tra = np.array(x_tra)
        plot_handle.plot(x_tra[:, 0], x_tra[:, 1], c='blue', linewidth=2, alpha=1.0)
        if show_flag is True:
            plot_handle.show()

    def plot_repro2(self, x0, plot_handle=None):
        x = x0
        period = 1e-2
        steps = int(80 / period)
        x_tra = x
        for i in range(steps):
            desired_v= self.predict(x)
            x = x + desired_v * period
            x = x.reshape(1,2)
            x_tra = np.vstack((x_tra,x))
        show_flag = False
        if plot_handle is None:
            import matplotlib.pyplot as plt
            plot_handle = plt
            show_flag = True
        x_tra = np.array(x_tra)
        plot_handle.plot(x_tra[:, 0], x_tra[:, 1], c='blue', linewidth=2, alpha=1.0)
        plot_handle.scatter(x_tra[-1, 0], x_tra[-1, 1], c='black', alpha=1.0, s=20, marker='D',label='Spurious attractor')
        if show_flag is True:
            plot_handle.show()

