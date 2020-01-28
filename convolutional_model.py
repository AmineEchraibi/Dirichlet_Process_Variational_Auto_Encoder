import torch
import torch.nn as nn
from torch.nn import BCELoss
import numpy as np
from utils import cumsum_ex, initialise_phi_with_kmeans, Flatten, UnFlatten

class DirichletProcessVariationalAutoEncoder(nn.Module):
    def __init__(self, X_km, T, eta, d, p, dataset_size, device, init="kmeans"):
        super(DirichletProcessVariationalAutoEncoder, self).__init__()

        if init == "kmeans":
            self.phi = initialise_phi_with_kmeans(X_km, T).to(device)

        else:
            self.phi = torch.rand(dataset_size,T, requires_grad=False,device=device)
            self.phi = self.phi / torch.sum(self.phi,1,True)
        del X_km

        self.m = torch.randn(T, p, requires_grad=False,device=device)

        self.v2 = torch.ones(T, p, requires_grad=False,device=device)

        self.gamma_1 = torch.zeros(T, requires_grad=False,device=device)
        self.gamma_2 = torch.zeros(T, requires_grad=False,device=device)
        self.pi = torch.zeros(T, requires_grad=False,device=device)

        self.decoders = torch.nn.ModuleList([])
        for k in range(T):
            self.decoders.append(nn.Sequential(nn.Linear( p , 256), UnFlatten(),  nn.ConvTranspose2d(256, 128, 5, 2),
                                               nn.Tanh(), nn.ConvTranspose2d(128, 64, 5, 2), nn.Tanh(),
                                             nn.ConvTranspose2d(64,1,4,2), nn.Sigmoid()))

        self.encoder_network_mu = nn.Sequential(nn.Conv2d(1, 32, 4, 2), nn.Tanh(), nn.Conv2d(32, 64, 4, 2),nn.Tanh(), Flatten(),  nn.Linear(1600, p * T))
        self.encoder_network_logsigma = nn.Sequential(nn.Conv2d(1, 32, 4, 2), nn.Tanh(), nn.Conv2d(32, 64, 4, 2),nn.Tanh(), Flatten(),  nn.Linear(1600, p * T))

        self.N = dataset_size
        self.eta = eta
        self.d = d
        self.p = p
        self.T = T
        self.device = device


    def update_phi_batch(self, hidden_representation, batch_size, mu, logsigma2, batch, writer, Y, i):
        """

        :param batch_idx:
        :param batch_size:
        :param mu:
        :param logsigma2:
        :return:
        """

        with torch.no_grad():
            x_reconstructed = self.reconstruct_observation(hidden_representation)
            batch = batch.reshape([batch_size, -1])
            sigma2 = logsigma2.exp()

            logphi_batch = - BCELoss(reduction="none")(x_reconstructed, batch.unsqueeze(1).repeat(1,self.T,1)).sum(2) \
                    - 0.5 * (torch.exp(logsigma2) / self.v2.unsqueeze(0)).sum(2) - \
                   0.5 * (torch.pow(mu - self.m.unsqueeze(0),2) /  self.v2.unsqueeze(0)).sum(2) - 0.5 *  torch.log(self.v2.unsqueeze(0) / sigma2).sum(2) \
                   + (torch.digamma(self.gamma_1) - torch.digamma(self.gamma_1 + self.gamma_2) + cumsum_ex(torch.digamma(self.gamma_2) - torch.digamma(self.gamma_1 + self.gamma_2))).unsqueeze(0) \

            # + torch.log(self.pi).unsqueeze(0) \
            phi_batch = (logphi_batch - torch.logsumexp(logphi_batch,1,True)).exp()
            phi_batch = phi_batch / torch.sum(phi_batch, 1, True)

            writer.add_histogram("histogram logphi0:",  - BCELoss(reduction="none")(x_reconstructed, batch.unsqueeze(1).repeat(1,self.T,1)).sum(2)[0,:], i)
            writer.add_histogram("histogram betas : ",(torch.digamma(self.gamma_1) - torch.digamma(self.gamma_1 + self.gamma_2)
                                                       + cumsum_ex(torch.digamma(self.gamma_2) - torch.digamma(self.gamma_1 + self.gamma_2))), i)

            writer.add_histogram("histogram KL : ", (- 0.5 *  torch.log(self.v2.unsqueeze(0) / sigma2).sum(2) - 0.5 * (torch.exp(logsigma2) / self.v2.unsqueeze(0)).sum(2) -
                   0.5 * (torch.pow(mu - self.m.unsqueeze(0),2) /  self.v2.unsqueeze(0)).sum(2))[0,:] ,i)
          # print("log pi ",  torch.log(self.pi))
           # print("entropy : ", 0.5 * (self.p * torch.log(torch.tensor(2 * np.pi * np.e,device=self.device)) + logsigma2.sum(2)))
            #print("digamma : ", torch.digamma(self.gamma_1) - torch.digamma(self.gamma_1 + self.gamma_2) \
                  #          + cumsum_ex(torch.digamma(self.gamma_2) - torch.digamma(self.gamma_1 + self.gamma_2)) )
           # print("log phi ele0 of batch : ",torch.mean(- BCELoss(reduction="none")(x_reconstructed, batch.unsqueeze(0).unsqueeze(2).
            #                                                                        repeat(100,1,self.T,1)).sum(3),0)[0,:])
           # print("logN : ",(- 0.5 * (torch.exp(logsigma2) / self.v2.unsqueeze(0)).sum(2) -
             #      0.5 * (torch.pow(mu - self.m.unsqueeze(0),2) /  self.v2.unsqueeze(0)).sum(2))[0,:] )
            return phi_batch

    def update_phi(self, train_loader, writer, Y, i):
        """

        :param train_loader:
        :return:
        """
        for batch_idx, (data, _) in enumerate(train_loader):
            with torch.no_grad():
                data = data.to(self.device).view(-1, 784)
                x_reconstructed, mu, logsigma2, hidden_representation = self.forward(data)
                batch_size = data.shape[0]
                self.phi[batch_idx *batch_size:(batch_idx + 1)*batch_size, : ] = self.update_phi_batch(hidden_representation,
                                                                                                       batch_size,
                                                                                                       mu,
                                                                                                       logsigma2,
                                                                                                       data, writer, Y, i)


    def update_pi(self):

        self.pi = torch.sum(self.phi, 0) / torch.sum(self.phi)

    def update_gammas(self):
        """

        :return:
        """
        with torch.no_grad():
            N = self.phi.sum(0)
            self.gamma_1 = 1 + N
            self.gamma_2 = self.eta + cumsum_ex(N.flip(0)).flip(0)

    def get_batch_sum_m(self, mu, batch_idx, batch_size):
        """

        :param mu:
        :param batch_idx:
        :param batch_size:
        :return:
        """
        with torch.no_grad():
            batch_m = torch.sum(self.phi[batch_idx*batch_size:batch_size*(batch_idx+1),:].unsqueeze(2) * mu, 0)

        return batch_m

    def get_batch_sum_v2(self, logsigma2, mu, batch_idx, batch_size):
        """

        :param mu:
        :param batch_idx:
        :param batch_size:
        :return:
        """
        with torch.no_grad():
            batch_v2 = torch.sum(self.phi[batch_idx * batch_size:batch_size * (batch_idx + 1), :].unsqueeze(2)
                                 * (torch.exp(logsigma2) + (mu - self.m.unsqueeze(0)).pow(2)), 0)
        return batch_v2

    def update_m(self, train_loader):
        """

        :param batch_idx:
        :param batch_size:
        :param mu:
        :return:
        """

        N = torch.sum(self.phi, 0).unsqueeze(-1)
        for batch_idx, (data, _) in enumerate(train_loader):
            with torch.no_grad():
                data = data.to(self.device).view(-1, 784)
                x_reconstructed, mu, logsigma2, hidden_representation = self.forward(data)
                if batch_idx == 0:
                    self.m = self.get_batch_sum_m(mu,batch_idx,data.shape[0])
                else:
                    self.m = self.m + self.get_batch_sum_m(mu,batch_idx,data.shape[0])

        self.m = self.m / N


    def update_v2(self, train_loader):
        """

        :param logsigma2:
        :param mu:
        :param batch_idx:
        :param batch_size:
        :return:
        """
        N = torch.sum(self.phi, 0).unsqueeze(-1)
        for batch_idx, (data, _) in enumerate(train_loader):
            with torch.no_grad():
                data = data.to(self.device).view(-1, 784)
                x_reconstructed, mu, logsigma2, hidden_representation = self.forward(data)
                if batch_idx == 0:
                    self.v2 = self.get_batch_sum_v2(logsigma2,mu, batch_idx, data.shape[0])
                else:
                    self.v2 = self.v2 + self.get_batch_sum_v2(logsigma2,mu, batch_idx, data.shape[0])

        self.v2 = self.v2 / N

    def compute_hidden_layer_statistics(self, x):
        """

        :param x:
        :return:
        """
        mu = self.encoder_network_mu(x).reshape(torch.Size([x.shape[0], self.T, self.p]))
        logsigma2 = self.encoder_network_logsigma(x).reshape(torch.Size([x.shape[0], self.T, self.p]))
        return mu, logsigma2

    def reparametrize(self, mu, logsigma2):
        """

        :param mu:
        :param logsigma2:
        :return:
        """

        epsilon = torch.randn(mu.size()).to(self.device)
        hidden_representation = mu + torch.exp(0.5 * logsigma2) * epsilon

        return hidden_representation

    def reconstruct_observation(self, hidden_representation):
        """

        :param hidden_representation:
        :return:
        """
        for k in range(self.T):
            if k==0:
                x_reconstructed = self.decoders[k](hidden_representation[:,k,:]).reshape([hidden_representation.shape[0], -1]).unsqueeze(1)
            else:
                x_reconstructed = torch.cat([x_reconstructed, self.decoders[k](hidden_representation[:,k,:]).reshape([hidden_representation.shape[0], -1]).unsqueeze(1)],1)

        assert x_reconstructed.shape == torch.Size([hidden_representation.shape[0], self.T, 784])

        return x_reconstructed

    def forward(self, x):
        """

        :param x:
        :return:
        """
        mu, logsigma2 = self.compute_hidden_layer_statistics(x)
        hidden_representation = self.reparametrize(mu, logsigma2)
        x_reconstructed = self.reconstruct_observation(hidden_representation)

        return x_reconstructed, mu, logsigma2, hidden_representation

    def sample_from_model(self, sample_size):

        """

        :param sample_size:
        :return:
        """
        epsilon = torch.randn(torch.Size([sample_size, self.T, self.p])).to(self.device)
        hidden_reprenstations = self.m.unsqueeze(0) + self.v2.unsqueeze(0).sqrt() * epsilon
        x_reconstuctions = self.reconstruct_observation(hidden_reprenstations)
        return x_reconstuctions



def compute_evidence_lower_bound(X, x_reconstructed, mu, logsigma2, hidden_representation, phi_batch, m, v2):
    """

    :param X:
    :param x_reconstructed:
    :param mu:
    :param logsigma2:
    :param hidden_representation:
    :param phi:
    :return:
    """



    sigma2 = torch.exp(logsigma2)
    #print("min max phi values ", phi_batch.cpu().detach().numpy().min(), phi_batch.cpu().detach().numpy().max())
    #print("min max explogs ",torch.exp(logsigma2).detach().cpu().numpy().min(), torch.exp(logsigma2).detach().cpu().numpy().max())
    assert (x_reconstructed.detach().cpu().numpy() >= 0).all()  and (x_reconstructed.detach().cpu().numpy() <= 1).all()
    assert (v2.cpu().detach().numpy() > 0).all()
    assert (phi_batch.detach().cpu().numpy() >= 0).all()  and (phi_batch.detach().cpu().numpy() <= 1).all()
    assert (torch.exp(logsigma2).detach().cpu().numpy() > 0).all()
    X = X.reshape([X.shape[0], -1])
    X = X.unsqueeze(1).repeat(1, mu.shape[1], 1)

    g_theta_func = - BCELoss(reduction="none")(x_reconstructed, X).sum(2)

    #print( 0.5 * (torch.exp(logsigma2) / v2.unsqueeze(0)).sum(2))

    likelihood_term = (phi_batch * g_theta_func).sum()
    #print("phi batch :", phi_batch[0,:])
    #print("g theta : ", g_theta_func[0,:])
    #print("term 1 : ", - BCELoss(reduction="none")(x_reconstructed, X).sum(2)[0,:])
    #print("term 2 :", - 0.5 * (torch.exp(logsigma2) / v2.unsqueeze(0)).sum(2)[0,:])
    #print("term 3 :", - 0.5 * (torch.pow(mu - m.unsqueeze(0),2) /  v2.unsqueeze(0)).sum(2)[0,:])


    kullback_term = - 0.5 * (sigma2 / v2.unsqueeze(0)).sum(2) - \
                   0.5 * (torch.pow(mu - m.unsqueeze(0),2) /  v2.unsqueeze(0)).sum(2) \
                    - 0.5 *  torch.log(v2.unsqueeze(0) / sigma2).sum(2)


    regularization_term =  torch.sum( phi_batch * kullback_term)
    #condition = (regularization_term.detach().cpu().numpy() < - 3000).all()
    #print(condition)

    evidence_lower_bound = - (likelihood_term + regularization_term)

    return evidence_lower_bound, likelihood_term, regularization_term




