import torch
import torch.nn as nn
from torch.nn import BCELoss
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
        self.pi = torch.zeros(T, requires_grad=False,device=device)

        self.decoders = torch.nn.ModuleList([])
        for k in range(T):
            self.decoders.append(nn.Sequential(nn.Linear(p, 400), nn.Tanh(), nn.Linear(400, 500), nn.Tanh(),
                                             nn.Linear(500, d), nn.Sigmoid()))

        self.encoder_network_mu = nn.Sequential(nn.Linear(d, 500), nn.Tanh(), nn.Linear(500, 400),nn.Tanh(),  nn.Linear(400, p * T))
        self.encoder_network_logsigma = nn.Sequential(nn.Linear(d, 500), nn.Tanh(), nn.Linear(500, 400),nn.Tanh(),  nn.Linear(400, p * T))

        self.N = dataset_size
        self.eta = eta
        self.d = d
        self.p = p
        self.T = T
        self.device = device


    def update_phi_batch(self, hidden_representation, batch_size, mu, logsigma2, batch, writer, Y, i):
        """
        Updates for phi per batch
        """

        with torch.no_grad():
            x_reconstructed = self.reconstruct_observation(hidden_representation)
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

            return phi_batch

    def update_phi(self, train_loader, writer, Y, i):
        """
        Update rule for phi
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
        """
        Update pi
        """
        self.pi = torch.sum(self.phi, 0) / torch.sum(self.phi)

    def update_gammas(self):
        """
        Update gammas
        """
        with torch.no_grad():
            N = self.phi.sum(0)
            self.gamma_1 = 1 + N
            self.gamma_2 = self.eta + cumsum_ex(N.flip(0)).flip(0)

    def get_batch_sum_m(self, mu, batch_idx, batch_size):
        """
        Batched update mean
        """
        with torch.no_grad():
            batch_m = torch.sum(self.phi[batch_idx*batch_size:batch_size*(batch_idx+1),:].unsqueeze(2) * mu, 0)

        return batch_m

    def get_batch_sum_v2(self, logsigma2, mu, batch_idx, batch_size):
        """
        batched updates variance
        """
        with torch.no_grad():
            batch_v2 = torch.sum(self.phi[batch_idx * batch_size:batch_size * (batch_idx + 1), :].unsqueeze(2)
                                 * (torch.exp(logsigma2) + (mu - self.m.unsqueeze(0)).pow(2)), 0)
        return batch_v2

    def update_m(self, train_loader):
        """
        update mean
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
        update variance
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
        hidden layer mean and variance
        """
        mu = self.encoder_network_mu(x).reshape(torch.Size([x.shape[0], self.T, self.p]))
        logsigma2 = self.encoder_network_logsigma(x).reshape(torch.Size([x.shape[0], self.T, self.p]))
        return mu, logsigma2

    def reparametrize(self, mu, logsigma2):
        """
        reparameterization trick
        """

        epsilon = torch.randn(mu.size()).to(self.device)
        hidden_representation = mu + torch.exp(0.5 * logsigma2) * epsilon

        return hidden_representation

    def reconstruct_observation(self, hidden_representation):
        """
        reconstruct the observation from hidden layer
        """
        for k in range(self.T):
            if k==0:
                x_reconstructed = self.decoders[k](hidden_representation[:,k,:]).unsqueeze(1)
            else:
                x_reconstructed = torch.cat([x_reconstructed, self.decoders[k](hidden_representation[:,k,:]).unsqueeze(1)],1)

        assert x_reconstructed.shape == torch.Size([hidden_representation.shape[0], self.T, self.d])

        return x_reconstructed

    def forward(self, x):
        """
        put all together
        """
        mu, logsigma2 = self.compute_hidden_layer_statistics(x)
        hidden_representation = self.reparametrize(mu, logsigma2)
        x_reconstructed = self.reconstruct_observation(hidden_representation)

        return x_reconstructed, mu, logsigma2, hidden_representation

    def sample_from_model(self, sample_size):
        """
        sample observation from model per cluster
        """
        epsilon = torch.randn(torch.Size([sample_size, self.T, self.p])).to(self.device)
        hidden_reprenstations = self.m.unsqueeze(0) + self.v2.unsqueeze(0).sqrt() * epsilon
        x_reconstuctions = self.reconstruct_observation(hidden_reprenstations)
        return x_reconstuctions



def compute_evidence_lower_bound(X, x_reconstructed, mu, logsigma2, hidden_representation, phi_batch, m, v2):
    """
    Optimization loss for gradient descent
    """

    X = X.unsqueeze(1).repeat(1,mu.shape[1],1)

    sigma2 = torch.exp(logsigma2)
    assert (x_reconstructed.detach().cpu().numpy() >= 0).all()  and (x_reconstructed.detach().cpu().numpy() <= 1).all()
    assert (v2.cpu().detach().numpy() > 0).all()
    assert (phi_batch.detach().cpu().numpy() >= 0).all()  and (phi_batch.detach().cpu().numpy() <= 1).all()
    assert (torch.exp(logsigma2).detach().cpu().numpy() > 0).all()

    g_theta_func = - BCELoss(reduction="none")(x_reconstructed, X).sum(2)

    likelihood_term = (phi_batch * g_theta_func).sum()


    kullback_term = - 0.5 * (sigma2 / v2.unsqueeze(0)).sum(2) - \
                   0.5 * (torch.pow(mu - m.unsqueeze(0),2) /  v2.unsqueeze(0)).sum(2) \
                    - 0.5 *  torch.log(v2.unsqueeze(0) / sigma2).sum(2)


    regularization_term =  torch.sum( phi_batch * kullback_term)

    evidence_lower_bound = - (likelihood_term + regularization_term)

    return evidence_lower_bound, likelihood_term, regularization_term




