import torch
from torch import nn

class Discriminator(nn.Module):

    def __init__(self,batch_size,number_Features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(number_Features, number_Features*2),
            # nn.BatchNorm1d(number_Features*2),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(number_Features*2, number_Features*3),
            # nn.BatchNorm1d(number_Features * 3),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(number_Features*3, number_Features*4),
            # nn.BatchNorm1d(number_Features * 4),
            nn.Tanh(),
            nn.Linear(number_Features*4, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class Generator(nn.Module):

    def __init__(self,batch_size,number_Features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(number_Features, number_Features*2),
            nn.Tanh(),
            # nn.Dropout(0.5),
            nn.Linear(number_Features*2, number_Features*3),
            nn.Tanh(),
            nn.Linear(number_Features * 3, number_Features * 2),
            nn.Tanh(),
            # nn.Dropout(0.5),
            nn.Linear(number_Features*2, number_Features),

        )

    def forward(self, x):
        output = self.model(x)
        return output



def generator_module(args):
    net = Generator(batch_size=args.batch_size,number_Features=args.number_features).to(device=torch.device(args.device))
    return net

def discriminator_module(args):
    net = Discriminator(batch_size=args.batch_size,number_Features=args.number_features).to(device=torch.device(args.device))
    return net

