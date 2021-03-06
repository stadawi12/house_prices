import torch
import torch.nn as nn
import torch.nn.functional as F

# A function containing a linear layer followed by an activation
# function (ReLU) in this case
def linear(in_c, out_c):
    conv = nn.Sequential(
            nn.Linear(in_c, out_c),
            nn.ReLU(inplace=True)
            )
    return conv

class Linear(nn.Module):

    def __init__(self, n_features):
        super(Linear, self).__init__()

        # Allow for variable number of features
        self.n_features = n_features

        self.linear1 = linear(n_features, 160)
        self.linear2 = linear(160, 160)
        self.linear3 = linear(160, 80)
        self.linear4 = linear(80, 1)

    def forward(self, sample):
        x = self.linear1(sample)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.linear4(x)
        return x


if __name__ == "__main__":
    n_features = 79
    L = Linear(n_features)
    inpt = torch.randn(2,n_features)
    out = L(inpt)
    print(out,"\n", out.shape)
