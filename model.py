import torch
import torch.nn as nn
from torch.nn import BCELoss
import numpy as np
from utils import cumsum_ex, initialise_phi_with_kmeans

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

        self.decoder_network = nn.Sequential(nn.Linear(p, 400), nn.Tanh(), nn.Linear(400, 500), nn.Tanh(),
                                             nn.Linear(500, d), nn.Sigmoid())

        self.encoder_network_mu = nn.Sequential(nn.Linear(d, 500), nn.Tanh(), nn.Linear(500, 400),nn.Tanh(),  nn.Linear(400, p * T))
        self.encoder_network_logsigma = nn.Sequential(nn.Linear(d, 500), nn.Tanh(), nn.Linear(500, 400),nn.Tanh(),  nn.Linear(400, p * T))

        self.N = dataset_size
        self.eta = eta
        self.d = d
        self.p = p
        self.T = T
        self.device = device


    def update_phi_batch(self, batch_idx, batch_size, mu, logsigma2, batch):
        """

        :param batch_idx:
        :param batch_size:
        :param mu:
        :param logsigma2:
        :return:
        """

        with torch.no_grad():
            epsilon = torch.randn(torch.Size([batch_size,  self.T, self.p])).to(self.device)
            hidden_representation = mu + torch.exp(0.5 * logsigma2) * epsilon
            x_reconstructed = self.reconstruct_observation(hidden_representation)
            logphi_batch = - BCELoss(reduction="none")(x_reconstructed, batch.unsqueeze(1).repeat(1,self.T,1)).sum(2) \
                    - 0.5 * (torch.exp(logsigma2) / self.v2.unsqueeze(0)).sum(2) - \
                   0.5 * (torch.pow(mu - self.m.unsqueeze(0),2) /  self.v2.unsqueeze(0)).sum(2) \
                    + (torch.digamma(self.gamma_1) - torch.digamma(self.gamma_1 + self.gamma_2) \
                            + cumsum_ex(torch.digamma(self.gamma_2) - torch.digamma(self.gamma_1 + self.gamma_2))).unsqueeze(0) \
                                         + 0.5 * (self.p * torch.log(torch.tensor(2 * np.pi * np.e,device=self.device)) + logsigma2.sum(2))

            phi_batch = (logphi_batch - torch.logsumexp(logphi_batch,1,True)).exp()

            #print("entropy : ", 0.5 * (self.p * torch.log(torch.tensor(2 * np.pi * np.e,device=self.device)) + logsigma2.sum(2)))
            #print("digamma : ", torch.digamma(self.gamma_1) - torch.digamma(self.gamma_1 + self.gamma_2) \
            #                + cumsum_ex(torch.digamma(self.gamma_2) - torch.digamma(self.gamma_1 + self.gamma_2)) )
            #print("log phi ele0 of batch : ",logphi_batch[0,:])
            return phi_batch

    def update_phi(self, train_loader):
        """

        :param train_loader:
        :return:
        """
        for batch_idx, (data, _) in enumerate(train_loader):
            with torch.no_grad():
                data = data.to(self.device).view(-1, 784)
                x_reconstructed, mu, logsigma2, hidden_representation = self.forward(data)
                batch_size = data.shape[0]
                self.phi[batch_idx *batch_size:(batch_idx + 1)*batch_size, : ] = self.update_phi_batch(batch_idx,
                                                                                                       batch_size,
                                                                                                       mu,
                                                                                                       logsigma2,
                                                                                                       data)



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
        return self.decoder_network(hidden_representation)

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

    X = X.unsqueeze(1).repeat(1,mu.shape[1],1)
    p = mu.shape[1]
    c = torch.tensor(2 * np.pi, device="cuda")

    g_theta_func = - BCELoss(reduction="none")(x_reconstructed, X).sum(2)   - \
                   0.5 * (torch.exp(logsigma2) / v2.unsqueeze(0)).sum(2) - \
                   0.5 * (torch.pow(mu - m.unsqueeze(0),2) /  v2.unsqueeze(0)).sum(2)

    #print( 0.5 * (torch.exp(logsigma2) / v2.unsqueeze(0)).sum(2))

    likelihood_term = (phi_batch * g_theta_func).sum()
    #print("phi batch :", phi_batch[0,:])
    #print("g theta : ", g_theta_func[0,:])
    #print("term 1 : ", - BCELoss(reduction="none")(x_reconstructed, X).sum(2)[0,:])
    #print("term 2 :", - 0.5 * (torch.exp(logsigma2) / v2.unsqueeze(0)).sum(2)[0,:])
    #print("term 3 :", - 0.5 * (torch.pow(mu - m.unsqueeze(0),2) /  v2.unsqueeze(0)).sum(2)[0,:])



    logNormal = - 0.5 * (p * torch.log(c) + logsigma2.sum(2)) \
                - 0.5 * (torch.pow(hidden_representation - mu,2) /  torch.exp(logsigma2)).sum(2)

    #print("log normal : ", logNormal[0,:])

    logNormalNoGrad = logNormal.detach().clone()
    assert logNormalNoGrad.requires_grad == False

    entropy_term = torch.sum(- phi_batch * (logNormal * logNormalNoGrad))

    evidence_lower_bound = - (likelihood_term + entropy_term)

    return evidence_lower_bound, likelihood_term, entropy_term




