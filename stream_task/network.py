import torch
import torch.nn as nn
import torchvision.models as models


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.hidden_size = 256

        self.encoder = nn.Sequential(nn.Linear(28, 64), nn.ReLU(),
                                     nn.Linear(64, 128), nn.ReLU(),
                                     nn.Linear(128, 256))
        self.lstm = LSTM_unit(self.hidden_size)

        self.mean_linear = nn.Linear(self.hidden_size, 10)
        self.var_linear = nn.Linear(self.hidden_size, 10)

        self.decoder_projection = nn.Linear(10, self.hidden_size)
        self.decoder = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                     nn.Linear(128, 64), nn.ReLU(),
                                     nn.Linear(64, 28))

        self.criterion = VAELoss()

    def forward(self, x):
        h = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        c = torch.zeros(x.size(0), self.hidden_size).to(x.device)
        x_new = torch.zeros_like(x).to(x.device)
        loss = 0

        for i in range(28 - 1):
            xt = x[:, i, :]
            y = x[:, i+1, :]

            xt = self.encoder(xt)  # batch_size * 128

            # Module that can perceive context
            h, c = self.lstm(xt, h, c)

            # Get distribution
            mean = self.mean_linear(h)
            logvar = self.var_linear(h)

            # Sample
            eps = torch.randn_like(logvar)
            std = torch.exp(logvar / 2)
            z = self.decoder_projection(eps * std + mean)

            y_hat = self.decoder(z)
            x_new[:, i+1, :] = y_hat

            # Calculate loss
            loss += self.criterion(y, y_hat, mean, logvar)

        return x_new, loss / x.size(0) / (28 - 1)

    def sample(self, h=0, c=0, device='cuda'):
        if isinstance(h, int) or isinstance(c, int):
            h = torch.zeros(1, self.hidden_size).to(device)
            c = torch.zeros(1, self.hidden_size).to(device)
            z = torch.randn(1, 10).to(device)
        else:
            mean = self.mean_linear(h)
            logvar = self.var_linear(h)
            eps = torch.randn_like(logvar)
            std = torch.exp(logvar / 2)
            z = eps * std + mean

        z = self.decoder_projection(z)
        y = self.decoder(z)

        x = self.encoder(y)
        h, c = self.lstm(x, h, c)

        return y, h, c


class LSTM_unit(nn.Module):
    def __init__(self, size):
        super(LSTM_unit, self).__init__()
        # Input gate
        self.xi = nn.Linear(size, size)
        self.hi = nn.Linear(size, size)

        # Forget gate
        self.xf = nn.Linear(size, size)
        self.hf = nn.Linear(size, size)

        # Output gate
        self.xo = nn.Linear(size, size)
        self.ho = nn.Linear(size, size)

        # Candidate memory
        self.xc = nn.Linear(size, size)
        self.hc = nn.Linear(size, size)

        # Activation function
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x, h, c):
        i = self.sigmoid(self.xi(x) + self.hi(h))
        f = self.sigmoid(self.xf(x) + self.hf(h))
        o = self.sigmoid(self.xo(x) + self.ho(h))
        c_ = self.tanh(self.xc(x) + self.hc(h))
        c_new = f * c + i * c_
        h_new = o * self.tanh(c_new)
        return h_new, c_new


class VAELoss(nn.Module):
    def __init__(self, kl_weight=1):
        super(VAELoss, self).__init__()
        self.kl_weight = kl_weight
        self.recon_func = nn.MSELoss(reduction='sum')

    def forward(self, y, y_hat, mean, logvar):
        recons_loss = self.recon_func(y_hat, y)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean ** 2 - torch.exp(logvar))
        loss = recons_loss + kl_loss * self.kl_weight
        return loss


if __name__ == "__main__":
    # model = VAE()
    # test = torch.randn(1, 3, 512, 512)
    # h = torch.zeros(1, 128)
    # c = torch.zeros(1, 128)
    # print(model.encoder(test).shape)
    # print(model.sample().shape)

    tensor1 = torch.randn(1, 3, 28)

    # 创建一个形状为 (1, 1, 28) 的张量
    tensor2 = torch.randn(1, 1, 28)

    # 在第二维上连接张量
    result_tensor = torch.cat((tensor1, tensor2), dim=1)

    print(result_tensor.shape)  # 输出为 torch.Size([1, 4, 28])

    t = torch.empty(0)
    r = torch.cat((t, tensor2), dim=1)
    print(r.shape)
