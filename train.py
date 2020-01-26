import warnings

import torch.utils.data
from torch import optim
from model import DirichletProcessVariationalAutoEncoder, compute_evidence_lower_bound
from torchvision import datasets, transforms, utils
from tensorboardX import SummaryWriter
from utils import np,cluster_acc

warnings.filterwarnings("ignore")

# Globals
batch_size=100
N=60000
writer = SummaryWriter()
n_epochs = 3000
device = torch.device("cuda")
d = 784
p = 20
T = 10
eta = 100

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/mr/Desktop/data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=N, shuffle=False)

for batch_idx, (data, Y) in enumerate(train_loader):
    X_km = data.numpy().reshape(data.shape[0], -1)
    if batch_idx == 0:
        break

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/mr/Desktop/data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=False)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('/home/mr/Desktop/data', train=False, transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=True)

# init model
model = DirichletProcessVariationalAutoEncoder(X_km, T, eta, d, p, N, device, "random").to(device)

optimizer = optim.SGD(model.parameters(), lr=1e-4, weight_decay=0.0005)

i = 0
for epoch in range(1, n_epochs + 1):

    model.train()
    train_loss = 0


    if epoch == 1 or epoch % 10 == 0:
        model.update_m(train_loader)
        model.update_v2(train_loader)
        #model.update_gammas()
        model.update_pi()
        model.update_phi(train_loader)

        writer.add_histogram("v2",model.v2,i)
        writer.add_histogram("m", model.m, i)

        writer.add_scalar("clustering accuracy ", cluster_acc(np.argmax(model.phi.cpu().numpy(), 1), Y.numpy())[0], i)


    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device).view(-1, 784)
        optimizer.zero_grad()

        # model forward
        x_reconstructed, mu, logsigma2, hidden_representation = model(data)
        #print(x_reconstructed[0,0,:])

        # mixture parameters
        phi_batch =model.phi[batch_size*batch_idx:batch_size*(1 + batch_idx), :]



        evidence_lower_bound, likelihood_term, entropy_term  = compute_evidence_lower_bound(data, x_reconstructed, mu,
                                                                                            logsigma2,
                                                                                            hidden_representation,
                                                                                            phi_batch,
                                                                                            model.m, model.v2)

        writer.add_scalar("loss/log predictive", likelihood_term, i)
        writer.add_scalar("loss/entropy", entropy_term, i)
        writer.add_scalar("loss/evidence lower bound - constants", - evidence_lower_bound, i)
        for j, (data, _) in enumerate(test_loader):
            data = data.to(device).view(-1, 784)
            x_reconstructed, mu, logsigma2, hidden_representation   = model(data)
            test_loss, _,_ =  compute_evidence_lower_bound(data, x_reconstructed, mu, logsigma2,
                                                           hidden_representation,phi_batch,
                                                           model.m, model.v2)
            writer.add_scalar("test_loss", test_loss, i)
            if j == 0:
                break
        i+=1



        evidence_lower_bound.backward()

        # Update steps


        train_loss += evidence_lower_bound.item()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                evidence_lower_bound.item() / len(data)))



    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    #scheduler.step()

    # Generative record

    x_rec = model.sample_from_model(batch_size)
    #print(x_rec.shape)

    for k in range(T):
        act = utils.make_grid(x_rec[:, k, :].view(batch_size, 1, 28, 28)[:64, :, :, :], normalize=True,
                              scale_each=True)
        writer.add_image("Sampling probs : " + str(k), act, i)



torch.save(model.state_dict(), "weights/DPVAE.pt")