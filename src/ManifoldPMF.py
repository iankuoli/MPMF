import numpy as np
from scipy import *
from scipy.sparse import *
from scipy.special import *
#import matplotlib.pyplot as plt
import pickle
import random

import similarity as sim
import distribution as dist
import Measure
import time


class ManifoldPMF:
    """
    Manifold Poisson Matrix Factorization
    The idea is inspired from "Content-based recommendation with Poisson factorization", in NIPS, 2014.
    """

    def __init__(self, val_k, m, n, params, ini_scale, ini):

        self.K = val_k
        self.M = m
        self.N = n
        self.delta = 0
        self.epsilon = 0
        self.mu = 0
        self.r_u = 0
        self.r_i = 0
        self.ini_scale = ini_scale
        [self.a, self.b, self.c, self.d, self.e, self.f, self.p, self.q, self.r, self.s] = params

        """ vvv---------- Start: Initialize the matrices ----------vvv """
        if ini == 1:
            self.mat_epsilon_shp = ini_scale * np.random.rand(self.M)
            self.mat_epsilon_rte = ini_scale * np.random.rand(self.M)
            self.matEpsilon = self.mat_epsilon_shp / self.mat_epsilon_rte

            self.mat_eta_shp = ini_scale * np.random.rand(self.N)
            self.mat_eta_rte = ini_scale * np.random.rand(self.N)
            self.mat_eta = self.mat_eta_shp / self.mat_eta_rte

            self.mat_pi_shp = ini_scale * np.random.rand(self.M)
            self.mat_pi_rte = ini_scale * np.random.rand(self.M)
            self.mat_pi = self.mat_pi_shp / self.mat_pi_rte

            self.mat_gamma_shp = ini_scale * np.random.rand(self.N)
            self.mat_gamma_rte = ini_scale * np.random.rand(self.N)
            self.mat_gamma = self.mat_gamma_shp / self.mat_gamma_rte

            self.mat_theta_shp = ini_scale * np.random.rand(self.M, self.K)
            self.mat_theta_rte = ini_scale * np.random.rand(self.M, self.K)
            self.mat_theta = self.mat_theta_shp / self.mat_theta_rte

            self.mat_beta_shp = ini_scale * np.random.rand(self.N, self.K)
            self.mat_beta_rte = ini_scale * np.random.rand(self.N, self.K)
            self.mat_beta = self.mat_beta_shp / self.mat_beta_rte

            pickle.dump(self.mat_epsilon_shp, open("../model/ini/mat_epsilon_shp.p", 'wb'))
            pickle.dump(self.mat_epsilon_rte, open("../model/ini/mat_epsilon_rte.p", 'wb'))
            pickle.dump(self.mat_eta_shp, open("../model/ini/mat_eta_shp.p", 'wb'))
            pickle.dump(self.mat_eta_rte, open("../model/ini/mat_eta_rte.p", 'wb'))
            pickle.dump(self.mat_pi_shp, open("../model/ini/mat_pi_shp.p", 'wb'))
            pickle.dump(self.mat_pi_rte, open("../model/ini/mat_pi_rte.p", 'wb'))
            pickle.dump(self.mat_gamma_shp, open("../model/ini/mat_gamma_shp.p", 'wb'))
            pickle.dump(self.mat_gamma_rte, open("../model/ini/mat_gamma_rte.p", 'wb'))
            pickle.dump(self.mat_theta_shp, open("../model/ini/mat_theta_shp.p", 'wb'))
            pickle.dump(self.mat_theta_rte, open("../model/ini/mat_theta_rte.p", 'wb'))
            pickle.dump(self.mat_beta_shp, open("../model/ini/mat_beta_shp.p", 'wb'))
            pickle.dump(self.mat_beta_rte, open("../model/ini/mat_beta_rte.p", 'wb'))
        else:
            self.mat_epsilon_shp = pickle.load(open("../model/ini/mat_epsilon_shp.p", "rb"))
            self.mat_epsilon_rte = pickle.load(open("../model/ini/mat_epsilon_rte.p", "rb"))
            self.matEpsilon = self.mat_epsilon_shp / self.mat_epsilon_rte

            self.mat_eta_shp = pickle.load(open("../model/ini/mat_eta_shp.p", "rb"))
            self.mat_eta_rte = pickle.load(open("../model/ini/mat_eta_rte.p", "rb"))
            self.mat_eta = self.mat_eta_shp / self.mat_eta_rte

            self.mat_pi_shp = pickle.load(open("../model/ini/mat_pi_shp.p", "rb"))
            self.mat_pi_rte = pickle.load(open("../model/ini/mat_pi_rte.p", "rb"))
            self.mat_pi = self.mat_pi_shp / self.mat_pi_rte

            self.mat_gamma_shp = pickle.load(open("../model/ini/mat_gamma_shp.p", "rb"))
            self.mat_gamma_rte = pickle.load(open("../model/ini/mat_gamma_rte.p", "rb"))
            self.mat_gamma = self.mat_gamma_shp / self.mat_gamma_rte

            self.mat_theta_shp = pickle.load(open("../model/ini/mat_theta_shp.p", "rb"))
            self.mat_theta_rte = pickle.load(open("../model/ini/mat_theta_rte.p", "rb"))
            self.mat_theta = self.mat_theta_shp / self.mat_theta_rte

            self.mat_beta_shp = pickle.load(open("../model/ini/mat_beta_shp.p", "rb"))
            self.mat_beta_rte = pickle.load(open("../model/ini/mat_beta_rte.p", "rb"))
            self.mat_beta = self.mat_beta_shp / self.mat_beta_rte

        self.tensor_phi = []
        self.tensor_rho = []
        self.tensor_sigma = []
        for kk in range(self.K):
            self.tensor_phi.append(csr_matrix((self.M, self.N)))
            self.tensor_rho.append(csr_matrix((self.M, self.M)))
            self.tensor_sigma.append(csr_matrix((self.N, self.N)))

        """ ^^^---------- Finish: Initialize the matrices ----------^^^ """

    def coordinate_ascent(self, mat_x, delta, epsilon, mu, r_u, r_i, alpha, max_itr, ini=False):

        if ini is True:
            self.initialize()

        self.delta = delta
        self.epsilon = epsilon
        self.mu = mu
        self.r_u = r_u
        self.r_i = r_i

        """ vvv---------- Start: Build the kernel matrix ----------vvv """
        """ Build mat_w """
        if self.delta > 0:
            mat_w = sim.similarity(mat_x.todense(), mat_x.todense(), 'gamma')
            mat_w -= eye(self.M)

        """ Build mat_s """
        if self.mu > 0:
            mat_s = sim.similarity(mat_x.todense().T, mat_x.todense().T, 'gamma')
            mat_s -= eye(self.N)
        """ ^^^---------- Finish: Build the kernel matrix ----------^^^ """

        is_converge = False
        i = 0
        l = 0

        if self.delta > 0 and self.M < 1000:
            mat_ww = np.linalg.pinv(np.eye(self.M) - (1 - alpha) * mat_w)

        if self.mu > 0 and self.N < 1000:
            mat_ss = np.linalg.pinv(np.eye(self.N) - (1 - alpha) * mat_s)

        while is_converge is False and i < max_itr:
            i += 1

            print("\n Index: ", i, "  ---------------------------------- \n")

            """ vvv---------- Start: Diffusion process of %mat_theta% ----------vvv """
            if self.delta > 0:
                if alpha < 1:
                    if self.M < 1000:
                        mat_theta_shp_diff = np.dot(mat_ww, self.mat_theta_shp)
                        mat_theta_diff = np.dot(mat_ww, self.mat_theta)
                    else:
                        mat_theta_shp_diff = self.mat_theta_shp
                        mat_theta_diff = self.mat_theta

                        for t in range(100):
                            mat_theta_shp_diff = \
                                (1 - alpha) * np.dot(mat_x, mat_theta_shp_diff) + alpha * self.mat_theta_shp

                            mat_theta_diff = (1 - alpha) * np.dot(mat_x, mat_theta_diff) + alpha * self.mat_theta

                        mat_theta_shp_diff = mat_theta_shp_diff / alpha
                        mat_theta_diff = mat_theta_diff / alpha

                    mat_theta_shp_diff = mat_theta_shp_diff * (sum(self.mat_theta_shp) / sum(mat_theta_shp_diff))
                    mat_theta_diff = mat_theta_diff * (sum(self.mat_theta) / sum(mat_theta_diff))
                else:
                    mat_theta_shp_diff = self.mat_theta_shp
                    mat_theta_diff = self.mat_theta

                mat_theta_shp_psi_diff = psi(mat_theta_shp_diff)

            mat_theta_shp_psi = psi(self.mat_theta_shp)
            mat_theta_rte_log = log(self.mat_theta_rte)
            """ ^^^---------- Finish: Diffusion process of %mat_theta% ----------^^^ """

            """ vvv---------- Start: Diffusion process of %mat_beta% ----------vvv """
            if self.mu > 0:
                if alpha < 1:
                    if self.N < 1000:
                        mat_beta_shp_diff = np.dot(mat_ss, self.mat_beta_shp)
                        mat_beta_diff = np.dot(mat_ss, self.mat_beta)
                    else:
                        mat_beta_shp_diff = self.mat_beta_shp
                        mat_beta_diff = self.mat_beta

                        for t in range(100):
                            mat_beta_shp_diff = \
                                (1 - alpha) * np.dot(self.matZ, mat_beta_shp_diff) + alpha * self.mat_beta_shp

                            mat_beta_diff = (1 - alpha) * np.dot(self.matZ, mat_beta_diff) + alpha * self.mat_beta

                        mat_beta_shp_diff = mat_beta_shp_diff / alpha
                        mat_beta_diff = mat_beta_diff / alpha

                    mat_beta_shp_diff = mat_beta_shp_diff * (sum(self.mat_beta_shp) / sum(mat_beta_shp_diff))
                    mat_beta_diff = mat_beta_diff * (sum(self.mat_beta) / sum(mat_beta_diff))
                else:
                    mat_beta_shp_diff = self.mat_beta_shp
                    mat_beta_diff = self.mat_beta

                mat_beta_shp_psi_diff = psi(mat_beta_shp_diff)

            mat_beta_shp_psi = psi(self.mat_beta_shp)
            mat_beta_rte_log = log(self.mat_beta_rte)
            """ ^^^---------- Finish: Diffusion process of %mat_beta% ----------^^^ """

            mat_pi_shp_psi = psi(self.mat_pi_shp)
            mat_pi_rte_log = log(self.mat_pi_rte)
            mat_gamma_shp_psi = psi(self.mat_gamma_shp)
            mat_gamma_rte_log = log(self.mat_gamma_rte)

            """
             Update tensor_phi
            """
            if self.epsilon > 0:
                print('Update tensor_phi ...  k = ')
                mat_phi_sum = csr_matrix((self.M, self.N))
                # mat_x_One = mat_x > 0
                for kk in range(self.K):
                    print(kk, ", ", end="")

                    self.tensor_phi[kk] = diags(mat_theta_shp_psi[:, kk] - mat_theta_rte_log[:, kk]) * (mat_x > 0)
                    self.tensor_phi[kk] += np.dot((mat_x > 0), diags(mat_beta_shp_psi[:, kk] -
                                                                          mat_beta_rte_log[:, kk]))

                    [ii, jj, ss] = find(self.tensor_phi[kk])
                    self.tensor_phi[kk] = csr_matrix((exp(ss), (ii, jj)), shape=(self.M, self.N))
                    mat_phi_sum += self.tensor_phi[kk]

                for kk in range(self.K):
                    [x_phi, y_phi, v_phi] = find(self.tensor_phi[kk])
                    v_phi_sum = find(mat_phi_sum.multiply(self.tensor_phi[kk] > 0))[2]
                    self.tensor_phi[kk] = csr_matrix((v_phi / v_phi_sum, (x_phi, y_phi)), shape=(self.M, self.N))

            """
             Update tensor_rho
            """
            if self.delta > 0:
                print("\nUpdate tensor_rho ...  k = ")
                self.manifold_tensor_update(tensor_manifold=self.tensor_rho,
                                            mat_manifold=mat_w,
                                            manifold_size=self.M,
                                            mat_data_shp_psi=mat_theta_shp_psi,
                                            mat_data_rte_log=mat_theta_rte_log,
                                            mat_norm_shp_psi=mat_pi_shp_psi,
                                            mat_norm_rte_log=mat_pi_rte_log)

            """
             Update tensor_sigma
            """
            if self.mu > 0:
                print("\nUpdate tensor_sigma ...  k = ")
                self.manifold_tensor_update(tensor_manifold=self.tensor_sigma,
                                            mat_manifold=mat_s,
                                            manifold_size=self.N,
                                            mat_data_shp_psi=mat_beta_shp_psi,
                                            mat_data_rte_log=mat_beta_rte_log,
                                            mat_norm_shp_psi=mat_gamma_shp_psi,
                                            mat_norm_rte_log=mat_gamma_rte_log)

            """
            Update Latent Matrix Variables
            """

            """
             Update mat_theta_shp, mat_theta_rte, mat_thetaD
            """
            if self.delta > 0:
                self.mat_pi_shp = np.squeeze(np.asarray(self.p + mat_w.sum(1)))
                self.mat_pi_rte = self.q + np.dot(self.mat_theta, sum(self.mat_theta * self.mat_pi[:, None], 0).T)
                self.mat_pi = self.mat_pi_shp / self.mat_pi_rte
            else:
                self.mat_pi = np.zeros(self.M, 1)

            if self.epsilon > 0 or self.delta > 0:
                for kk in range(self.K):
                    self.mat_theta_shp[:, kk] = self.epsilon * np.squeeze(
                        np.asarray(mat_x.multiply(self.tensor_phi[kk]).sum(1)))

                    self.mat_theta_shp[:, kk] += self.delta * np.squeeze(
                        np.asarray(mat_w.multiply(self.tensor_rho[kk]).sum(1)))

                self.mat_theta_shp += self.a

                self.mat_theta_rte = self.delta * (np.sum(self.mat_theta * self.mat_pi[:, None], 0) *
                                                   self.mat_pi[:, None] -
                                                   self.mat_theta * (self.mat_pi * self.mat_pi)[:, None])
                self.mat_theta_rte += self.epsilon * np.sum(self.mat_beta, 0)[None, :]
                self.mat_theta_rte += self.matEpsilon[:, None]

                self.mat_theta = self.mat_theta_shp / self.mat_theta_rte

            """
             Update mat_beta_shp, mat_beta_rte, mat_betaD
            """
            if self.mu > 0:
                self.mat_gamma_shp = np.squeeze(np.asarray(self.r + mat_s.sum(1)))
                self.mat_gamma_rte = self.s + np.dot(self.mat_beta, sum(self.mat_beta * self.mat_gamma[:, None], 0).T)
                self.mat_gamma = self.mat_gamma_shp / self.mat_gamma_rte
            else:
                self.mat_gamma = zeros(self.N, 1)

            if self.epsilon > 0 or self.mu > 0:
                for kk in range(self.K):
                    self.mat_beta_shp[:, kk] = self.epsilon * np.squeeze(
                        np.asarray(mat_x.multiply(self.tensor_phi[kk]).sum(0)))

                    self.mat_beta_shp[:, kk] += self.mu * np.squeeze(
                        np.asarray(mat_s.multiply(self.tensor_sigma[kk]).sum(1)))

                self.mat_beta_shp += self.d

                self.mat_beta_rte = self.mu * (np.sum(self.mat_beta * self.mat_gamma[:, None], 0) *
                                               self.mat_gamma[:, None] -
                                               self.mat_beta * (self.mat_gamma * self.mat_gamma)[:, None])

                self.mat_beta_rte += self.epsilon * np.sum(self.mat_theta, 0)[None, :]
                self.mat_beta_rte += self.mat_eta[:, None]

                self.mat_beta = self.mat_beta_shp / self.mat_beta_rte

            """
             Update mat_epsilon_shp, mat_epsilon_rte
            """
            self.mat_epsilon_shp = self.b + self.K * self.a
            self.mat_epsilon_rte = self.c + np.sum(self.mat_theta, 1)
            self.matEpsilon = self.mat_epsilon_shp / self.mat_epsilon_rte

            """
             Update mat_eta_shp, mat_eta_rte
            """
            self.mat_eta_shp = self.e + self.K * self.d
            self.mat_eta_rte = self.f + np.sum(self.mat_beta, 1)
            self.mat_eta = self.mat_eta_shp / self.mat_eta_rte

            """
            Terminiation Checkout
            """

            """
             Calculate the likelihood to determine when to terminate.
            """
            if i > 0:

                new_1 = 0
                new_2 = 0
                new_3 = 0

                if self.epsilon > 0:
                    new_1 = self.epsilon / mat_x.nnz * dist.log_poisson(mat_x,
                                                                             self.mat_theta, self.mat_beta.T)

                if self.delta > 0:
                    new_2 = self.delta / mat_w.nnz * dist.log_poisson(mat_w,
                                                                           self.mat_theta * self.mat_pi[:, None],
                                                                           (self.mat_theta * self.mat_pi[:, None]).T)

                if self.mu > 0:
                    new_3 = self.mu / mat_s.nnz * dist.log_poisson(mat_s,
                                                                        self.mat_beta * self.mat_gamma[:, None],
                                                                        (self.mat_beta * self.mat_gamma[:, None]).T)

                new_l = new_1 + new_2 + new_3

                if abs(new_l - l) < 0.1:
                    is_converge = True

                l = new_l
                print("\nLikelihood: ", l, "  ( ", new_1, ", ", new_2, ", ", new_3, " )")

        pickle.dump(self.mat_theta, open("mat_theta.p", 'wb'))
        pickle.dump(self.mat_beta, open("mat_beta.p", 'wb'))
        pickle.dump(self.matEpsilon, open("matEpsilon.p", 'wb'))
        pickle.dump(self.mat_eta, open("mat_eta.p", 'wb'))
        pickle.dump(self.mat_pi, open("mat_pi.p", 'wb'))
        pickle.dump(self.mat_gamma, open("mat_gamma.p", 'wb'))

    def stochastic_coordinate_ascent(self, mat_x, mat_valid, batch_size, delta, epsilon, mu,
                                     kappa, max_itr, ini=False):

        if ini is True:
            self.initialize()

        self.delta = delta
        self.epsilon = epsilon
        self.mu = mu

        nnz = mat_x.nnz

        is_converge = False
        i = 0
        #l = 0

        list_valid_usrs = list(set(find(mat_valid)[0]))[0:100]

        while is_converge is False and i < max_itr:
            i += 1

            """ Set the learning rate """
            lr = i ** -kappa

            # Sample data
            # a1 = time.time()
            usr_idx = list()
            itm_idx = list()
            while len(usr_idx) < 2 or len(itm_idx) < 2:
                rand_index = random.sample(range(nnz), batch_size)
                usr_idx = list(set(find(mat_x)[0][rand_index]))
                itm_idx = list(set(find(mat_x)[1][rand_index]))
            # a2 = time.time()
            # print("sampling time: ", a2 - a1, " sec")

            print("\n Index: ", i, "  --------------------------- ", len(usr_idx), " / ", len(itm_idx), " - ", lr, "\n")

            """ vvv---------- Start: Build the kernel matrix ----------vvv """
            """ Build mat_w """
            if self.delta > 0:
                mat_w = sim.similarity(mat_x[usr_idx, :],
                                       mat_x[usr_idx, :], 'gamma2')

                mat_w -= eye(len(usr_idx))
                # mat_w[np.ix_(usr_idx, usr_idx)] = mat_w

            """ Build mat_s """
            if self.mu > 0:
                mat_s = sim.similarity(mat_x[:, itm_idx].T,
                                       mat_x[:, itm_idx].T, 'gamma2')

                mat_s -= eye(len(itm_idx))
                # mat_s[np.ix_(itm_idx, itm_idx)] = mat_s
            """ ^^^---------- Finish: Build the kernel matrix ----------^^^ """

            mat_theta_shp_psi = psi(self.mat_theta_shp)
            mat_theta_rte_log = log(self.mat_theta_rte)
            mat_beta_shp_psi = psi(self.mat_beta_shp)
            mat_beta_rte_log = log(self.mat_beta_rte)

            mat_pi_shp_psi = psi(self.mat_pi_shp)
            mat_pi_rte_log = log(self.mat_pi_rte)
            mat_gamma_shp_psi = psi(self.mat_gamma_shp)
            mat_gamma_rte_log = log(self.mat_gamma_rte)

            """
             Update tensor_phi
            """
            if self.epsilon > 0:
                print('Update tensor_phi ...  k = ')
                self.data_tensor_partial_update(mat_x=mat_x,
                                                usr_idx=usr_idx,
                                                itm_idx=itm_idx,
                                                usr_size=len(usr_idx),
                                                itm_size=len(itm_idx),
                                                mat_theta_shp_psi=mat_theta_shp_psi,
                                                mat_theta_rte_log=mat_theta_rte_log,
                                                mat_beta_shp_psi=mat_beta_shp_psi,
                                                mat_beta_rte_log=mat_beta_rte_log)

            """
             Update tensor_rho
            """
            if self.delta > 0:
                print("\nUpdate tensor_rho ...  k = ")
                self.manifold_tensor_partial_update(big_tensor_manifold=self.tensor_rho,
                                                    mat_manifold=mat_w,
                                                    data_idx=usr_idx,
                                                    mat_data_shp_psi=mat_theta_shp_psi,
                                                    mat_data_rte_log=mat_theta_rte_log,
                                                    mat_norm_shp_psi=mat_pi_shp_psi,
                                                    mat_norm_rte_log=mat_pi_rte_log)

            """
             Update tensor_sigma
            """
            if self.mu > 0:
                print("\nUpdate tensor_sigma ...  k = ")
                self.manifold_tensor_partial_update(big_tensor_manifold=self.tensor_sigma,
                                                    mat_manifold=mat_s,
                                                    data_idx=itm_idx,
                                                    mat_data_shp_psi=mat_beta_shp_psi,
                                                    mat_data_rte_log=mat_beta_rte_log,
                                                    mat_norm_shp_psi=mat_gamma_shp_psi,
                                                    mat_norm_rte_log=mat_gamma_rte_log)

            """
            Update Latent Matrix Variables
            """

            """
             Update mat_theta_shp, mat_theta_rte, mat_thetaD
            """
            mat_theta_old = self.mat_theta[usr_idx, :]
            scale = (self.M - 1) / (len(usr_idx) - 1)
            if self.delta > 0:
                self.mat_pi_shp[usr_idx] = (1 - lr) * self.mat_pi_shp[usr_idx] + lr * (
                    np.squeeze(np.asarray(self.p + scale * mat_w.sum(1))))

                self.mat_pi_rte[usr_idx] = (1 - lr) * self.mat_pi_rte[usr_idx] + lr * (
                    self.q + scale * np.dot(self.mat_theta[usr_idx, :],
                                            np.sum(self.mat_theta[usr_idx, :] * self.mat_pi[usr_idx, None], 0).T))

                self.mat_pi[usr_idx] = self.mat_pi_shp[usr_idx] / self.mat_pi_rte[usr_idx]

            if self.epsilon > 0 or self.delta > 0:
                scale1 = np.squeeze(np.asarray((mat_x[usr_idx, :] > 0).sum(1) /
                                               (mat_x[np.ix_(usr_idx, itm_idx)] > 0).sum(1)))
                for kk in range(self.K):
                    self.mat_theta_shp[usr_idx, kk] = (1 - lr) * self.mat_theta_shp[usr_idx, kk] + lr * (
                        self.epsilon * scale1 * np.squeeze(
                            np.asarray(mat_x[np.ix_(usr_idx, itm_idx)].multiply(
                                self.tensor_phi[kk][np.ix_(usr_idx, itm_idx)]).sum(1))) +
                        self.delta * scale * np.squeeze(
                            np.asarray(mat_w.multiply(
                                self.tensor_rho[kk][np.ix_(usr_idx, usr_idx)]).sum(1))))
                self.mat_theta_shp[usr_idx, :] += lr * self.a

                self.mat_theta_rte[usr_idx, :] = (1 - lr) * self.mat_theta_rte[usr_idx, :] + lr * (
                    self.delta * scale * (np.sum(self.mat_theta[usr_idx, :] * self.mat_pi[usr_idx, None], 0) *
                                          self.mat_pi[usr_idx, None] -
                                          self.mat_theta[usr_idx, :] * (self.mat_pi[usr_idx] *
                                                                        self.mat_pi[usr_idx])[:, None]) +
                    self.epsilon * self.N / len(itm_idx) * np.sum(self.mat_beta[itm_idx, :], 0)[None, :] +
                    self.matEpsilon[usr_idx, None])

                self.mat_theta[usr_idx, :] = self.mat_theta_shp[usr_idx, :] / self.mat_theta_rte[usr_idx, :]

            """
             Update mat_beta_shp, mat_beta_rte, mat_betaD
            """
            mat_beta_old = self.mat_beta[itm_idx, :]
            scale = (self.N - 1) / (len(itm_idx) - 1)
            if self.mu > 0:
                self.mat_gamma_shp[itm_idx] = (1 - lr) * self.mat_gamma_shp[itm_idx] + lr * (
                    np.squeeze(np.asarray(self.r + scale * mat_s.sum(1))))

                self.mat_gamma_rte[itm_idx] = (1 - lr) * self.mat_gamma_rte[itm_idx] + lr *(
                    self.s + scale * np.dot(self.mat_beta[itm_idx, :],
                                            np.sum(self.mat_beta[itm_idx, :] * self.mat_gamma[itm_idx, None], 0).T))

                self.mat_gamma[itm_idx] = self.mat_gamma_shp[itm_idx] / self.mat_gamma_rte[itm_idx]

            if self.epsilon > 0 or self.mu > 0:
                scale2 = np.squeeze(np.asarray((mat_x[:, itm_idx] > 0).sum(0) /
                                               (mat_x[np.ix_(usr_idx, itm_idx)] > 0).sum(0)))
                for kk in range(self.K):
                    self.mat_beta_shp[itm_idx, kk] = (1 - lr) * self.mat_beta_shp[itm_idx, kk] + lr * (
                        self.epsilon * scale2 * np.squeeze(
                            np.asarray(mat_x[np.ix_(usr_idx, itm_idx)].multiply(
                                self.tensor_phi[kk][np.ix_(usr_idx, itm_idx)]).sum(0))) +
                        self.mu * scale * np.squeeze(
                            np.asarray(mat_s.multiply(
                                self.tensor_sigma[kk][np.ix_(itm_idx, itm_idx)]).sum(1))))

                self.mat_beta_shp[itm_idx, :] += lr * self.d

                self.mat_beta_rte[itm_idx, :] = (1 - lr) * self.mat_beta_rte[itm_idx, :] + lr * (
                    self.mu * scale * (np.sum(self.mat_beta[itm_idx, :] * self.mat_gamma[itm_idx, None], 0) *
                                       self.mat_gamma[itm_idx, None] -
                                       self.mat_beta[itm_idx, :] * (self.mat_gamma[itm_idx] *
                                                                    self.mat_gamma[itm_idx])[:, None]) +
                    self.epsilon * self.M / len(usr_idx) * np.sum(self.mat_theta[usr_idx, :], 0)[None, :] +
                    self.mat_eta[itm_idx, None])

                self.mat_beta[itm_idx, :] = self.mat_beta_shp[itm_idx, :] / self.mat_beta_rte[itm_idx, :]

            """
             Update mat_epsilon_shp, mat_epsilon_rte
            """
            self.mat_epsilon_shp = self.b + self.K * self.a
            self.mat_epsilon_rte = self.c + np.sum(self.mat_theta, 1)
            self.matEpsilon = self.mat_epsilon_shp / self.mat_epsilon_rte

            """
             Update mat_eta_shp, mat_eta_rte
            """
            self.mat_eta_shp = self.e + self.K * self.d
            self.mat_eta_rte = self.f + np.sum(self.mat_beta, 1)
            self.mat_eta = self.mat_eta_shp / self.mat_eta_rte

            """
            Termination Checkout
            """

            """
             Calculate the likelihood to determine when to terminate.
            """

            new_1 = 0
            new_2 = 0
            new_3 = 0

            # usr_idx = list(range(50))
            # itm_idx = list(range(40))

            if self.epsilon > 0:
                new_1 = self.epsilon / mat_x[np.ix_(usr_idx, itm_idx)].nnz * \
                        dist.log_poisson(mat_x[np.ix_(usr_idx, itm_idx)],
                                         self.mat_theta[usr_idx, :],
                                         self.mat_beta[itm_idx, :].T)

            if self.delta > 0:
                a = self.mat_theta[usr_idx, :] * self.mat_pi[usr_idx, None]
                new_2 = self.delta / mat_w.nnz * \
                        dist.log_poisson(mat_w,
                                         self.mat_theta[usr_idx, :] * self.mat_pi[usr_idx, None],
                                         (self.mat_theta[usr_idx, :] * self.mat_pi[usr_idx, None]).T)

            if self.mu > 0:
                new_3 = self.mu / mat_s.nnz * \
                        dist.log_poisson(mat_s,
                                         self.mat_beta[itm_idx, :] * self.mat_gamma[itm_idx, None],
                                         (self.mat_beta[itm_idx, :] * self.mat_gamma[itm_idx, None]).T)

            converge = dist.log_poisson(mat_theta_old.dot(mat_beta_old.T),mat_theta_old,mat_beta_old.T)
                                        #self.mat_theta[usr_idx, :],
                                        #self.mat_beta[itm_idx, :].T)

            # if abs(new_l - l) < 0.001:
            if converge > - 1:
                is_converge = True

            # l = new_l
            print("\nLikelihood: ", converge, "  ( ", new_1, ", ", new_2, ", ", new_3, " )")


            """
             Validate the precision of the recommendation in validation set
            """

            #list_valid_usrs = list(range(50))

            rand_idx = random.sample(range(len(list_valid_usrs)), len(list_valid_usrs))
            rand_users = [list_valid_usrs[i] for i in rand_idx]
            predict_matrix = self.mat_theta[rand_users, :].dot(self.mat_beta.T)
            predict_matrix -= mat_x[rand_users, :].multiply(predict_matrix > 0)
            avg_precision = 0
            avg_recall = 0
            for idx_t in range(len(rand_users)):
                uid = rand_users[idx_t]
                precision, recall = Measure.precision_recall_at_k(np.squeeze(mat_valid[uid, :].toarray()),
                                                                  np.squeeze(np.array(predict_matrix[idx_t, :])), 3)
                avg_precision += precision
                avg_recall += recall
            avg_precision /= len(list_valid_usrs)
            avg_recall /= len(list_valid_usrs)
            print("precision: ", avg_precision, " ,  recall: ", avg_recall)

    def dump_model(self, meta_info):
        pickle.dump(self, open("../model/MPMF_", meta_info, ".p", 'wb'))

    # def static load_model(meta_info):
    #    return pickle.load(open("../model/MPMF_", meta_info, ".p", 'rb'))
        # self.mat_theta = pickle.load(open(meta_info, "/mat_theta.p", 'rb'))
        # self.mat_beta = pickle.load(open(meta_info, "mat_beta.p", 'rb'))
        # self.matEpsilon = pickle.load(open(meta_info, "matEpsilon.p", 'rb'))
        # self.mat_eta = pickle.load(open(meta_info, "mat_eta.p", 'rb'))
        # self.mat_pi = pickle.load(open(meta_info, "mat_pi.p", 'rb'))
        # self.mat_gamma = pickle.load(open(meta_info, "mat_gamma.p", 'rb'))
        # [self.a, self.b, self.c, self.d, self.e, self.f, self.p, self.q, self.r, self.s,
        # self.delta, self.epsilon, self.mu] = pickle.load(open(meta_info, "mat_params.p", 'rb'))

    def manifold_tensor_update(self, tensor_manifold, mat_manifold, manifold_size,
                               mat_data_shp_psi, mat_data_rte_log, mat_norm_shp_psi, mat_norm_rte_log):

        mat_manifold_sum = csr_matrix((manifold_size, manifold_size))
        for kk in range(self.K):
            print(kk, ", ", end="")

            tensor_manifold[kk] = \
                diags(mat_data_shp_psi[:, kk] - mat_data_rte_log[:, kk] +
                      mat_norm_shp_psi - mat_norm_rte_log) * \
                (mat_manifold > 0) + \
                (mat_manifold > 0) * \
                diags(mat_data_shp_psi[:, kk] - mat_data_rte_log[:, kk] +
                      mat_norm_shp_psi - mat_norm_rte_log)

            [ii, jj, ss] = find(tensor_manifold[kk])
            tensor_manifold[kk] = csr_matrix((exp(ss), (ii, jj)), shape=(manifold_size, manifold_size))
            mat_manifold_sum += tensor_manifold[kk]

        for kk in range(self.K):
            [x_m, y_m, v_m] = find(tensor_manifold[kk])
            v_m_sum = find(mat_manifold_sum.multiply(tensor_manifold[kk] > 0))[2]
            tensor_manifold[kk] = csr_matrix((v_m / v_m_sum, (x_m, y_m)), shape=(manifold_size, manifold_size))

    def manifold_tensor_partial_update(self, big_tensor_manifold, mat_manifold, data_idx,
                                       mat_data_shp_psi, mat_data_rte_log, mat_norm_shp_psi, mat_norm_rte_log):

        manifold_size = len(data_idx)
        mat_manifold_sum = csr_matrix((manifold_size, manifold_size))
        tensor_manifold = []
        for kk in range(self.K):
            tensor_manifold.append(csr_matrix((manifold_size, manifold_size)))

        for kk in range(self.K):
            print(kk, ", ", end="")

            tensor_manifold[kk] = \
                diags(mat_data_shp_psi[data_idx, kk] - mat_data_rte_log[data_idx, kk] +
                      mat_norm_shp_psi[data_idx] - mat_norm_rte_log[data_idx]) * \
                (mat_manifold > 0) + \
                (mat_manifold > 0) * \
                diags(mat_data_shp_psi[data_idx, kk] - mat_data_rte_log[data_idx, kk] +
                      mat_norm_shp_psi[data_idx] - mat_norm_rte_log[data_idx])

            [ii, jj, ss] = find(tensor_manifold[kk])

            tensor_manifold[kk] = \
                csr_matrix((exp(ss), (ii, jj)), shape=(manifold_size, manifold_size))

            mat_manifold_sum += tensor_manifold[kk]

        for kk in range(self.K):
            [x_m, y_m, v_m] = find(tensor_manifold[kk])
            v_m_sum = find(mat_manifold_sum.multiply(tensor_manifold[kk] > 0))[2]
            big_tensor_manifold[kk][np.ix_(data_idx, data_idx)] = csr_matrix((v_m / v_m_sum, (x_m, y_m)),
                                                                             shape=(manifold_size, manifold_size))

    def data_tensor_partial_update(self, mat_x, usr_idx, itm_idx, usr_size, itm_size,
                                   mat_theta_shp_psi, mat_theta_rte_log, mat_beta_shp_psi, mat_beta_rte_log):

        mat_phi_sum = csr_matrix((usr_size, itm_size))

        tensor_phi = []
        for kk in range(self.K):
            tensor_phi.append(csr_matrix((usr_size, itm_size)))

        for kk in range(self.K):
            print(kk, ", ", end="")

            tensor_phi[kk] = diags(mat_theta_shp_psi[usr_idx, kk] - mat_theta_rte_log[usr_idx, kk]) * \
                             (mat_x[np.ix_(usr_idx, itm_idx)] > 0) + \
                             (mat_x[np.ix_(usr_idx, itm_idx)] > 0) * \
                             diags(mat_beta_shp_psi[itm_idx, kk] - mat_beta_rte_log[itm_idx, kk])

            [ii, jj, ss] = find(tensor_phi[kk])
            tensor_phi[kk] = csr_matrix((exp(ss), (ii, jj)), shape=(usr_size, itm_size))
            mat_phi_sum += tensor_phi[kk]

        for kk in range(self.K):
            [x_phi, y_phi, v_phi] = find(tensor_phi[kk])
            v_phi_sum = find(mat_phi_sum.multiply(tensor_phi[kk] > 0))[2]
            self.tensor_phi[kk][np.ix_(usr_idx, itm_idx)] = csr_matrix((v_phi / v_phi_sum, (x_phi, y_phi)),
                                                                       shape=(usr_size, itm_size))

    def recommend_for_users(self, vec_query_user_index):
        print(vec_query_user_index)
        return np.dot(self.mat_theta[vec_query_user_index, :], self.mat_beta)

    def recommend_for_items(self, vec_query_item_index):
        return np.dot(self.mat_beta[vec_query_item_index, :], self.mat_theta)

    def initialize(self):
        self.mat_epsilon_shp = pickle.load(open("ini/mat_epsilon_shp.p", "rb"))
        self.mat_epsilon_rte = pickle.load(open("ini/mat_epsilon_rte.p", "rb"))
        self.matEpsilon = self.mat_epsilon_shp / self.mat_epsilon_rte

        self.mat_eta_shp = pickle.load(open("ini/mat_eta_shp.p", "rb"))
        self.mat_eta_rte = pickle.load(open("ini/mat_eta_rte.p", "rb"))
        self.mat_eta = self.mat_eta_shp / self.mat_eta_rte

        self.mat_pi_shp = pickle.load(open("ini/mat_pi_shp.p", "rb"))
        self.mat_pi_rte = pickle.load(open("ini/mat_pi_rte.p", "rb"))
        self.mat_pi = self.mat_pi_shp / self.mat_pi_rte

        self.mat_gamma_shp = pickle.load(open("ini/mat_gamma_shp.p", "rb"))
        self.mat_gamma_rte = pickle.load(open("ini/mat_gamma_rte.p", "rb"))
        self.mat_gamma = self.mat_gamma_shp / self.mat_gamma_rte

        self.mat_theta_shp = pickle.load(open("ini/mat_theta_shp.p", "rb"))
        self.mat_theta_rte = pickle.load(open("ini/mat_theta_rte.p", "rb"))
        self.mat_theta = self.mat_theta_shp / self.mat_theta_rte

        self.mat_beta_shp = pickle.load(open("ini/mat_beta_shp.p", "rb"))
        self.mat_beta_rte = pickle.load(open("ini/mat_beta_rte.p", "rb"))
        self.mat_beta = self.mat_beta_shp / self.mat_beta_rte
