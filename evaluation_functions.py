import torch
from torch.nn import BCELoss
from utils import cumsum_ex

def evidence_lower_bound_complete_VAE(X, x_reconstructed, mu, logsigma2, phi_batch, m, v2, convolutioanal=False):
    """
    The complete evidence lower bound computed for getting a lower bound on the log-likelihood
    :param X:
    :param x_reconstructed:
    :param mu:
    :param logsigma2:
    :param hidden_representation:
    :param phi_batch:
    :param m:
    :param v2:
    :return:
    """

    if convolutioanal:
        X = X.reshape([X.shape[0], -1])
    X = X.unsqueeze(1).repeat(1, mu.shape[1], 1)

    sigma2 = torch.exp(logsigma2)

    g_theta_func = - BCELoss(reduction="none")(x_reconstructed, X).sum(2)

    likelihood_term = (phi_batch * g_theta_func).sum()

    kullback_term = - 0.5 * (sigma2 / v2.unsqueeze(0)).sum(2) - \
                    0.5 * (torch.pow(mu - m.unsqueeze(0), 2) / v2.unsqueeze(0)).sum(2) \
                    - 0.5 * torch.log(v2.unsqueeze(0) / sigma2).sum(2)

    regularization_term = torch.sum(phi_batch * kullback_term)

    evidence_lower_bound = likelihood_term + regularization_term - torch.sum(phi_batch * torch.log(phi_batch.clamp(min=1e-9)))

    return evidence_lower_bound


def predictive_distribution(x_new, x_reconstructed, mu, gamma_1, gamma_2, convolutioanal=False):
    """
    Computing the predictive distribution of the cluster hidden variable for a new sample
    :param X:
    :param x_reconstructed:
    :param mu:
    :param logsigma2:
    :param hidden_representation:
    :param phi_batch:
    :param m:
    :param v2:
    :param convolutioanal:
    :return:
    """
    if convolutioanal:
        x_new = x_new.reshape([x_new.shape[0], -1])
    x_new = x_new.unsqueeze(1).repeat(1, mu.shape[1], 1)

    ln_E_pX = - BCELoss(reduction="none")(x_reconstructed, x_new).sum(2)

    ln_E_pi =  (torch.log(gamma_1.clamp(min=1e-9)) - torch.log((gamma_1 + gamma_2).clamp(min=1e-9)) + cumsum_ex(
        torch.log(gamma_2.clamp(min=1e-9)) - torch.log((gamma_1 + gamma_2).clamp(min=1e-9)))).unsqueeze(0)

    ln_predictive = ln_E_pi + ln_E_pX

    predictive = (ln_predictive - torch.logsumexp(ln_predictive,1,True)).exp()
    predictive = predictive / torch.sum(predictive, 1, True) # numerical stability

    return predictive

