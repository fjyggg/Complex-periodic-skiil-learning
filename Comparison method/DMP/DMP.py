import numpy as np
from scipy.optimize import minimize
np.random.seed(5)


class DMP:
    def __init__(self, x_set, tau=1.0, N=10, r=1.0):
        '''
        :param x_set:
        :param tau: need to be computed according to the period of the objective function
        :param N:
        :param r:
        '''
        self.tau = tau
        self.N = N
        self.r = r
        self.c = np.linspace(0.0, 2 * np.pi, N)
        self.x_set = x_set
        self.d_phi = 1.0 / tau
        self.data_size = np.shape(x_set)[0]
        phi = 0.0
        phi_set = [phi]
        Period = 0.02
        for i in range(self.data_size - 1):
            phi = phi + self.d_phi * Period
            phi_set.append(phi)
        self.phi_set = np.array(phi_set)

    def Psi(self, phi):
        c = self.c
        temp1 = np.cos(phi - c) - 1
        Psi = np.exp(temp1)
        return Psi

    def Psi_set(self, phi_set):
        c = self.c
        temp1 = np.cos(np.expand_dims(phi_set, axis=1) - np.expand_dims(c, axis=0)) - 1
        Psi_set = np.exp(temp1)
        return Psi_set  # (set_size, N)

    def f(self, phi, omega):
        r = self.r
        Psi = self.Psi(phi)
        dem = np.sum(Psi)
        f = Psi.dot(omega) * r / dem
        return f

    def f_set(self, phi_set, omega):
        Psi_set = self.Psi_set(phi_set)
        Psi_sum_set = np.sum(Psi_set, axis=1)
        weighted_Psi_set = Psi_set / np.expand_dims(Psi_sum_set, axis=1)  # (set_size, N)
        f_set = weighted_Psi_set.dot(omega) * self.r
        return f_set, weighted_Psi_set * self.r   # (set_size,), (set_size, N)

    def obj(self, omega):
        predicted_x_set, d_predicted_x_set_d_omega = self.f_set(self.phi_set, omega)
        gap = 1.0 / (2 * self.data_size) * (predicted_x_set - self.x_set).dot((predicted_x_set - self.x_set))
        gradient = 1.0 / self.data_size * d_predicted_x_set_d_omega.T.dot(predicted_x_set - self.x_set)
        return gap, gradient

    def train(self, savepath):
        omega = np.random.uniform(-0.5, 0.5, self.N)
        result = minimize(self.obj, omega, jac=True, method='L-BFGS-B', tol=1e-20)
        omega = result.x
        np.savetxt(savepath, omega)
        return omega














