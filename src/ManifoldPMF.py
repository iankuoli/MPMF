import numpy as np
from scipy import *
from scipy.sparse import *
from scipy.special import *

import similarity as sim
import distribution as dist


class ManifoldPMF:
    """
    Manifold Poisson Matrix Factorization
    The idea is inspired from "Content-based recommendation with Poisson factorization", in NIPS, 2014.
    """
    def __init__(self, val_k, data_x, params, delta, epsilon, mu, r_u, r_i, ini_scale):

        self.K = val_k
        self.matX = data_x
        self.M = data_x.shape[0]
        self.N = data_x.shape[1]
        self.delta = delta
        self.epsilon = epsilon
        self.mu = mu
        self.r_u = r_u
        self.r_i = r_i
        self.ini_scale = ini_scale
        [self.a, self.b, self.c, self.d, self.e, self.f, self.p, self.q, self.r, self.s] = params

        """ vvv---------- Start: Build the kernel matrix ----------vvv """
        """ Build matW """
        self.matW = sim.similarity(self.matX, self.matX, 'gamma')
        vec_inv_sqrt_rowsum_matX = 1 / np.sqrt(sum(self.matW))
        self.matW = (self.matW * vec_inv_sqrt_rowsum_matX).T * vec_inv_sqrt_rowsum_matX

        """ Build matS """
        self.matS = sim.similarity(self.matS, self.matS, 'gamma')
        vec_inv_sqrt_rowsum_matY = 1 / np.sqrt(sum(self.matS))
        self.matS = (self.matS * vec_inv_sqrt_rowsum_matY).T * vec_inv_sqrt_rowsum_matY
        """ ^^^---------- Finish: Build the kernel matrix ----------^^^ """

        """ vvv---------- Start: Initialize the matrices ----------vvv """
        self.matEpsilon_Shp = ini_scale * rand(self.M, 1)
        self.matEpsilon_Rte = ini_scale * rand(self.M, 1)
        self.matEpsilon = self.matEpsilon_Shp / self.matEpsilon_Rte

        self.matEta_Shp = ini_scale * rand(self.N, 1)
        self.matEta_Rte = ini_scale * rand(self.N, 1)
        self.matEta = self.matEta_Shp / self.matEta_Rte

        self.matPi_Shp = ini_scale * rand(self.M, 1)
        self.matPi_Rte = ini_scale * rand(self.M, 1)
        self.matPi = self.matPi_Shp / self.matPi_Rte

        self.matGamma_Shp = ini_scale * rand(self.N, 1)
        self.matGamma_Rte = ini_scale * rand(self.N, 1)
        self.matGamma = self.matGamma_Shp / self.matGamma_Rte

        self.matBeta_Shp = ini_scale * rand(self.N, self.K)
        self.matBeta_Rte = ini_scale * rand(self.N, self.K)
        self.matTheta = self.matTheta_Shp / self.matTheta_Rte

        self.matTheta_Shp = ini_scale * rand(self.M, self.K)
        self.matTheta_Rte = ini_scale * rand(self.M, self.K)
        self.matBeta = self.matBeta_Shp / self.matBeta_Rte
        """ ^^^---------- Finish: Initialize the matrices ----------^^^ """

    def coordinate_ascent(self, alpha, max_itr):

        is_converge = False
        i = 0
        l = 0

        if self.delta > 0 and self.M < 1000:
            matWW = np.linalg.pinv(np.eye(self.M) - (1 - alpha) * self.matW)

        if self.mu > 0 and self.N < 1000:
            matSS = np.linalg.pinv(np.eye(self.N) - (1 - alpha) * self.matS)

        while is_converge == False and i < max_itr:
            i += 1
            print("\n Index: ", i, "  ---------------------------------- \n")

            """ vvv---------- Start: Diffusion process of %matTheta% ----------vvv """
            if self.delta > 0:
                if alpha < 1:
                    if self.M < 1000:
                        matTheta_Shp_diff = np.dot(matWW,  self.matTheta_Shp)
                        matTheta_diff = np.dot(matWW, self.matTheta)
                    else:
                        matTheta_Shp_diff = self.matTheta_Shp
                        matTheta_diff = self.matTheta

                        for t in range(100):
                            matTheta_Shp_diff = (1 - alpha) * np.dot(self.matX, matTheta_Shp_diff) \
                                                + alpha * self.matTheta_Shp
                            matTheta_diff = (1 - alpha) * np.dot(self.matX, matTheta_diff) + alpha * self.matTheta

                        matTheta_Shp_diff = matTheta_Shp_diff / alpha
                        matTheta_diff = matTheta_diff / alpha

                    matTheta_Shp_diff = matTheta_Shp_diff * (sum(self.matTheta_Shp) / sum(matTheta_Shp_diff))
                    matTheta_diff = matTheta_diff * (sum(self.matTheta) / sum(matTheta_diff))
                else:
                    matTheta_Shp_diff = self.matTheta_Shp
                    matTheta_diff = self.matTheta

                matTheta_Shp_psi_diff = psi(matTheta_Shp_diff)

            matTheta_Shp_psi = psi(self.matTheta_Shp)
            matTheta_Rte_log = log(self.matTheta_Rte)
            """ ^^^---------- Finish: Diffusion process of %matTheta% ----------^^^ """

            """ vvv---------- Start: Diffusion process of %matBeta% ----------vvv """
            if self.mu > 0:
                if alpha < 1:
                    if self.N < 1000:
                        matBeta_Shp_diff = matSS * self.matBeta_Shp
                        matBeta_diff = matSS * self.matBeta
                    else:
                        matBeta_Shp_diff = self.matBeta_Shp
                        matBeta_diff = self.matBeta

                        for t in range(100):
                            matBeta_Shp_diff = (1 - alpha) * np.dot(self.matZ, matBeta_Shp_diff) \
                                               + alpha * self.matBeta_Shp
                            matBeta_diff = (1 - alpha) * np.dot(self.matZ, matBeta_diff) + alpha * self.matBeta

                        matBeta_Shp_diff = matBeta_Shp_diff / alpha
                        matBeta_diff = matBeta_diff / alpha

                    matBeta_Shp_diff = matBeta_Shp_diff * (sum(self.matBeta_Shp) / sum(matBeta_Shp_diff))
                    matBeta_diff = matBeta_diff * (sum(self.matBeta) / sum(matBeta_diff))
                else:
                    matBeta_Shp_diff = self.matBeta_Shp
                    matBeta_diff = self.matBeta

                matBeta_Shp_psi_diff = psi(matBeta_Shp_diff)

            matBeta_Shp_psi = psi(self.matBeta_Shp)
            matBeta_Rte_log = log(self.matBeta_Rte)
            """ ^^^---------- Finish: Diffusion process of %matBeta% ----------^^^ """

            matPi_Shp_psi = psi(self.matPi_Shp)
            matPi_Rte_log = log(self.matPi_Rte)
            matGamma_Shp_psi = psi(self.matGamma_Shp)
            matGamma_Rte_log = log(self.matGamma_Rte)

            """
             Update tensorPhi
            """
            if self.epsilon > 0:
                print('Update tensorPhi ...  k = ')
                matPhi_sum = csr_matrix((self.M, self.N))
                matX_One = self.matX > 0
                for k in range(self.K):
                    print(k, ", ")

                    self.tensorPhi[k] = diags(matTheta_Shp_psi[:, k] - matTheta_Rte_log[:, k]).dot(matX_One)
                    self.tensorPhi[k] += matX_One.dot(diags(matBeta_Shp_psi[:, k] - matBeta_Rte_log[:, k]))

                    [ii, jj, ss] = find(self.tensorPhi[k])
                    self.tensorPhi[k] = csr_matrix((exp(ss), (ii, jj)), shape=(self.M, self.N))
                    matPhi_sum = matPhi_sum + self.tensorPhi[k]
                
                for k in range(self.K):
                    [x_Phi, y_Phi, v_Phi] = find(self.tensorPhi[k])
                    v_PSum = find(matPhi_sum * (self.tensorPhi[k] > 0))[2]
                    self.tensorPhi[k] = csr_matrix((v_Phi / v_PSum, (x_Phi, y_Phi)), shape=(self.M, self.N))

            """
             Update tensorRho
            """
            if self.delta > 0:
                print("\nUpdate tensorRho ...  k = ")
                matRho_sum = csr_matrix((self.M, self.M))
                matX_One = self.matW > 0
                for k in range(self.K):
                    print(k, ", ")
        
                    self.tensorRho[k] = diags(self.matTheta_Shp_psi[:, k] - self.matTheta_Rte_log[:, k]
                                              + self.matPi_Shp_psi - self.matPi_Rte_log) * matX_One
                    self.tensorRho[k] += matX_One * diags(matTheta_Shp_psi[:, k] - matTheta_Rte_log[:, k]
                                                          + matPi_Shp_psi - matPi_Rte_log)
        
                    [ii, jj, ss] = find(self.tensorRho[k])
                    self.tensorRho[k] = csr_matrix((exp(ss), (ii, jj)), shape=(self.M, self.M))
                    matRho_sum = matRho_sum + self.tensorRho[k]
            
                for k in range(self.K):
                    [x_Phi, y_Phi, v_Phi] = find(self.tensorRho[k])
                    v_PSum = find(matRho_sum * (self.tensorRho[k] > 0))[2]
                    self.tensorRho[k] = csr_matrix((v_Phi / v_PSum, (x_Phi, y_Phi)), shape=(self.M, self.M))

            """
             Update tensorSigma
            """
            if self.mu > 0:
                print("\nUpdate tensorSigma ...  k = ")
                matSigma_sum = csr_matrix((self.N, self.N))
                matZ_One = self.matS > 0
                for k in range(self.K):
                    print(k, ", ")
                
                    self.tensorSigma[k] = diags(self.matBeta_Shp_psi[:, k] - self.matBeta_Rte_log[:, k] +
                                                self.matGamma_Shp_psi - self.matGamma_Rte_log) * matZ_One
                    self.tensorSigma[k] += matZ_One * diags(matBeta_Shp_psi[:, k] - matBeta_Rte_log[:, k]
                                                            + matGamma_Shp_psi - matGamma_Rte_log)
                
                    [ii, jj, ss] = find(self.tensorSigma[k])
                    self.tensorSigma[k] = csr_matrix((exp(ss), (ii, jj)), shape=(self.N, self.N))
                    matSigma_sum = matSigma_sum + self.tensorSigma[k]

                for k in range(self.K):
                    [x_Phi, y_Phi, v_Phi] = find(self.tensorSigma[k])
                    v_PSum = find(matSigma_sum * (self.tensorSigma[k] > 0))[2]
                    self.tensorSigma[k] = csr_matrix((v_Phi / v_PSum, (x_Phi, y_Phi)), shape=(self.N, self.N))

            """
            Update Latent Matrix Variables
            """

            """
             Update matTheta_Shp, matTheta_Rte, matThetaD
            """
            if self.delta > 0:
                self.matPi_Shp = self.p + sum(self.matW.T)
                tmp = sum(self.matTheta * self.matPi)
                self.matPi_Rte = self.q + self.matTheta * tmp.T
                self.matPi = self.matPi_Shp / self.matPi_Rte
            else:
                self.matPi = np.zeros(self.M, 1)

            if self.epsilon > 0 or self.delta > 0:
                for k in range(self.K):
                    self.matTheta_Shp[:, k] = self.epsilon * sum((self.matX * self.tensorPhi[k]).T) \
                                              + self.delta * sum(self.matW * self.tensorRho[k])

                self.matTheta_Shp = self.a + self.matTheta_Shp

                self.matTheta_Rte = np.tile(self.epsilon * sum(self.matBeta), (self.M, 1))
                self.matTheta_Rte += np.tile(self.delta * sum(self.matTheta * self.matPi), (self.M, 1)) * self.matPi
                self.matTheta_Rte += self.matEpsilon

                self.matTheta = self.matTheta_Shp / self.matTheta_Rte

            """
             Update matBeta_Shp, matBeta_Rte, matBetaD
            """
            if self.mu > 0:
                self.matGamma_Shp = self.r + sum(self.matS.T)
                tmp = sum(self.matBeta * self.matGamma)
                self.matGamma_Rte = self.s + np.dot(self.matBeta, tmp.T)
                self.matGamma = self.matGamma_Shp / self.matGamma_Rte
            else:
                self.matGamma = zeros(self.N, 1)

            if self.epsilon > 0 or self.mu > 0:
                for k in range(self.K):
                    self.matBeta_Shp[:, k] = self.epsilon * sum(self.matX * self.tensorPhi[k]).T \
                                             + self.mu * sum((self.matS * self.tensorSigma[k]).T)

                self.matBeta_Shp = self.d + self.matBeta_Shp

                self.matBeta_Rte = np.tile(self.epsilon * sum(self.matTheta), (self.N, 1))
                self.matBeta_Rte += np.tile(self.mu * sum(self.matBeta * self.matGamma), (self.N, 1)) * self.matGamma
                self.matBeta_Rte += self.matEta

                self.matBeta = self.matBeta_Shp / self.matBeta_Rte
        
            """
             Update matEpsilon_Shp, matEpsilon_Rte
            """
            print("\nUpdate matEpsilon_Shp , matEpsilon_Rte ...")
            self.matEpsilon_Shp = self.b + self.K * self.a
            self.matEpsilon_Rte = self.c + sum(self.matTheta.T)
            self.matEpsilon = self.matEpsilon_Shp / self.matEpsilon_Rte

            """
             Update matEta_Shp, matEta_Rte
            """
            print("\nUpdate matEta_Shp , matEta_Rte ...")
            self.matEta_Shp = self.e + self.K * self.d
            self.matEta_Rte = self.f + sum(self.matBeta.T)
            self.matEta = self.matEta_Shp / self.matEta_Rte

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

                if self.epsilon> 0:
                    new_1 = self.epsilon / self.matX.nnz * dist.log_Poisson(self.matX, self.matTheta, self.matBeta.T)

                if self.delta > 0:
                    norm_matTheta = self.matTheta * self.matPi
                    new_2 = self.delta / self.matW.nnz * dist.log_Poisson(self.matW, norm_matTheta, norm_matTheta.T)

                if self.mu > 0:
                    norm_matBeta = self.matBeta * self.matGamma
                    new_3 = self.mu / self.matS.nnz * dist.log_Poisson(self.matS, norm_matBeta, norm_matBeta.T)

                new_l = new_1 + new_2 + new_3

                if abs(new_l - l) < 0.0001:
                    is_converge = True

                l = new_l
                print("\nLikelihood: ", l, "  ( ", new_1, ", ", new_2, ", ", new_3, " )\n")
