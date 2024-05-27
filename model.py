import torch
import torch.nn as nn
from loss import graph_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        try:
            nn.init.normal_(m.weight.data, 0.0, 0.01)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)


class Classification(nn.Module):
    def __init__(self, input_dim, class_num):
        super(Classification, self).__init__()
        self.classification = nn.Sequential(
            nn.Linear(input_dim, class_num),
        )

    def forward(self, x):
        out = self.classification(x)
        return out


class VAE1Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(VAE1Layer, self).__init__()
        self.input_dim = input_dim
        self.encoder_mean = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )
        self.encoder_log_var = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim),
            nn.Tanh()
        )
        self.apply(weights_init)

    def forward(self, x):
        batchsize = x.size(0)
        mean = self.encoder_mean(x)
        log_var = self.encoder_log_var(x)
        std = torch.exp(log_var / 2)
        output = mean + std * torch.randn_like(std)
        input_hat = self.decoder(output)
        kld = - 0.5 * torch.sum(
            1
            + log_var
            - torch.pow(mean, 2)
            - log_var.exp()
        ) / (self.input_dim * batchsize)
        return output, input_hat, kld


class VAEModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VAEModel, self).__init__()
        self.VAE_firstlayer = VAE1Layer(input_dim, hidden_dim)
        self.VAE_secondlayer = VAE1Layer(hidden_dim, output_dim)

    def forward(self, x):
        hidden1, input_hat_1, kld_1 = self.VAE_firstlayer(x)
        hidden2, input_hat_2, kld_2 = self.VAE_secondlayer(hidden1)
        return hidden1, input_hat_1, kld_1, hidden2, input_hat_2, kld_2


class CDVAE(nn.Module):
    def __init__(
            self,
            source_input_dim,
            target_input_dim,
            hidden_dim,
            r,
            class_num):
        super(CDVAE, self).__init__()
        self.source_VAE = VAEModel(source_input_dim, hidden_dim, r)
        self.target_VAE = VAEModel(target_input_dim, hidden_dim, r)
        self.classification = Classification(r, class_num)

    def loss(self):
        criterion_mse = nn.MSELoss(reduction='mean')
        criterion_crossentropy = nn.CrossEntropyLoss()
        criterion_graphloss = graph_loss()
        return criterion_mse, criterion_crossentropy, criterion_graphloss

    def forward(self, source_data, target_data):
        source_hidden, source_input_hat_1, source_kld_1, source_hidden_2, source_input_hat_2, source_kld_2 = self.source_VAE(
            source_data)
        target_hidden, target_input_hat_1, target_kld_1, target_hidden_2, target_input_hat_2, target_kld_2 = self.target_VAE(
            target_data)
        source_out = self.classification(source_hidden_2)
        target_out = self.classification(target_hidden_2)
        return source_hidden, source_input_hat_1, source_kld_1, source_hidden_2, source_input_hat_2, source_kld_2, \
               target_hidden, target_input_hat_1, target_kld_1, target_hidden_2, target_input_hat_2, target_kld_2, \
               source_out, target_out
