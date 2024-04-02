import torch
import torch.nn as nn
import torchvision.models as models


class VAE(nn.Module):
    def __init__(self, input_channels=3, image_size=512, latent_dim=128, bn_momentum=0.1):
        super(VAE, self).__init__()
        self.input_channels = input_channels
        self.latent_dim = latent_dim
        self.bn_momentum = bn_momentum

        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        encoder_dict = torch.hub.load_state_dict_from_url(
            "https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth",
            model_dir="D:/coding/weights"
        )
        self.encoder.load_state_dict(encoder_dict)
        self.encoder = torch.nn.Sequential(*self.encoder.children())[:-2]

        encoded_code = self.encoder(torch.randn(1, input_channels, image_size, image_size))
        encoded_size, encoded_channel = encoded_code.shape[-1], encoded_code.shape[1]
        self.mean_linear = nn.Linear(encoded_channel * encoded_size * encoded_size, latent_dim)
        self.var_linear = nn.Linear(encoded_channel * encoded_size * encoded_size, latent_dim)

        decoder_channels = [256, 128, 64, 32, 16]
        decoder_kernel = [4, 4, 4, 4, 4]
        self.decoded_shape = (encoded_channel, encoded_size, encoded_size)
        self.decoder_projection = nn.Linear(latent_dim, encoded_channel * encoded_size * encoded_size)
        self.decoder = self.make_decoder_layer(decoder_channels, decoder_kernel)

    def make_decoder_layer(self, num_channels, num_kernel):
        layers = []
        prev_channels = self.decoded_shape[0]
        for channels, kernel in zip(num_channels, num_kernel):
            up = nn.ConvTranspose2d(
                in_channels=prev_channels,
                out_channels=channels,
                kernel_size=kernel,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False)

            layers.append(up)
            layers.append(nn.BatchNorm2d(channels, momentum=self.bn_momentum))
            layers.append(nn.ReLU(inplace=True))
            prev_channels = channels

        layers.append(
            nn.Sequential(
                nn.Conv2d(prev_channels, prev_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(prev_channels, momentum=self.bn_momentum),
                nn.ReLU(inplace=True),
                nn.Conv2d(prev_channels, self.input_channels, kernel_size=1, stride=1, padding=0, bias=True),
            )
        )

        return nn.Sequential(*layers)

    def forward(self, x):
        encoded = torch.flatten(self.encoder(x), 1)

        mean = self.mean_linear(encoded)
        logvar = self.var_linear(encoded)

        eps = torch.randn_like(logvar)
        std = torch.exp(logvar / 2)
        z = self.decoder_projection(eps * std + mean)

        decoded = self.decoder(torch.reshape(z, (-1, *self.decoded_shape)))

        return decoded, mean, logvar

    def sample(self, device='cuda'):
        z = torch.randn(1, self.latent_dim).to(device)
        x = self.decoder_projection(z)
        x = torch.reshape(x, (-1, *self.decoded_shape))
        decoded = self.decoder(x)
        return decoded


class VAELoss(nn.Module):
    def __init__(self, kl_weight=1):
        super(VAELoss, self).__init__()
        self.kl_weight = kl_weight
        self.recon_func = nn.MSELoss(reduction='mean')

    def forward(self, y, y_hat, mean, logvar):
        recons_loss = self.recon_func(y_hat, y)
        kl_loss = -0.5 * torch.mean(1 + logvar - mean ** 2 - torch.exp(logvar))
        loss = recons_loss + kl_loss * self.kl_weight
        return loss


if __name__ == "__main__":
    model = VAE()
    test = torch.randn(1, 3, 512, 512)
    print(model.encoder(test).shape)
    print(model.sample().shape)
