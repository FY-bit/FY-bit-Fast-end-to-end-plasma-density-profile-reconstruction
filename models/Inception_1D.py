import torch
import torch.nn as nn
import torch.nn.functional as F


class Permute(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)

class SimMLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation_layer=nn.ReLU, dropout=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, in_features)
        self.act = activation_layer(inplace=False)
        self.fc2 = nn.Linear(in_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, hidden_features)
        self.fc4 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc3(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc4(x)
        x = self.drop(x)
        return x

class stem(nn.Module):
    def __init__(self, in_channels, branch_channels=64):
        """
        branch_expansion: 分支通道数 = in_channels // branch_expansion
        """
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, 1, stride=2),
                Permute((0, 2, 1)),
                nn.LayerNorm(branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, 1),
                Permute((0, 2, 1)),
                nn.LayerNorm(branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True),
                nn.Conv1d(branch_channels, branch_channels, 3,
                          stride=2, padding=1),
                Permute((0, 2, 1)),
                nn.LayerNorm(branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, 1),
                Permute((0, 2, 1)),
                nn.LayerNorm(branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True),
                nn.Conv1d(branch_channels, branch_channels, 5,
                          stride=2, padding=2),
                Permute((0, 2, 1)),
                nn.LayerNorm(branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv1d(in_channels, branch_channels, 1),
                Permute((0, 2, 1)),
                nn.LayerNorm(branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True),
                nn.Conv1d(branch_channels, branch_channels, 7,
                          stride=2, padding=3),
                Permute((0, 2, 1)),
                nn.LayerNorm(branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True)
            )
        ])


    def forward(self, x):
        branch_outputs = [branch(x) for branch in self.branches]
        cat = torch.cat(branch_outputs, dim=1)
        return cat

class ResidualInceptionModule1D(nn.Module):
    def __init__(self, in_channels, branch_channels, downsample=False):
        super(ResidualInceptionModule1D, self).__init__()
        stride = 1
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, 1, stride),
            Permute((0, 2, 1)),
            nn.LayerNorm(branch_channels),
            Permute((0, 2, 1)),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, 1),
            Permute((0, 2, 1)),
            nn.LayerNorm(branch_channels),
            Permute((0, 2, 1)),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, 3, stride, 1),
            Permute((0, 2, 1)),
            nn.LayerNorm(branch_channels),
            Permute((0, 2, 1)),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, 1),
            Permute((0, 2, 1)),
            nn.LayerNorm(branch_channels),
            Permute((0, 2, 1)),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, 5, stride, 2),
            Permute((0, 2, 1)),
            nn.LayerNorm(branch_channels),
            Permute((0, 2, 1)),
            nn.ReLU()
        )

        self.branch4 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, 1),
            Permute((0, 2, 1)),
            nn.LayerNorm(branch_channels),
            Permute((0, 2, 1)),
            nn.ReLU(),
            nn.Conv1d(branch_channels, branch_channels, 7, stride, 3),
            Permute((0, 2, 1)),
            nn.LayerNorm(branch_channels),
            Permute((0, 2, 1)),
            nn.ReLU()
        )

        self.branch5 = nn.Sequential(
            nn.MaxPool1d(3, stride, 1),
            nn.Conv1d(in_channels, branch_channels, 1),
            Permute((0, 2, 1)),
            nn.LayerNorm(branch_channels),
            Permute((0, 2, 1)),
            nn.ReLU()
        )

        self.fusion = nn.Sequential(
            nn.Conv1d(5 * branch_channels, in_channels, 1),
            Permute((0, 2, 1)),
            nn.LayerNorm(in_channels),
            Permute((0, 2, 1))
        )

        if downsample:
            self.res_conv = nn.Sequential(
                nn.Conv1d(in_channels, in_channels, 1, stride),
                Permute((0, 2, 1)),
                nn.LayerNorm(in_channels),
                Permute((0, 2, 1))
            )
        else:
            self.res_conv = None

        self.act = nn.ReLU()

    def forward(self, x):
        identity = x

        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        b5 = self.branch5(x)

        out = torch.cat([b1, b2, b3, b4, b5], dim=1)
        fused = self.fusion(out)

        if self.res_conv is not None:
            identity = self.res_conv(identity)

        return self.act(fused + identity)


class DownsampleInceptionModule1D(nn.Module):

    def __init__(self, in_channels, branch_expansion=4):

        super().__init__()
        self.branch_channels = in_channels // branch_expansion
        assert self.branch_channels > 0, "branch_channels必须大于0"

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, self.branch_channels, 1, stride=2),
                Permute((0, 2, 1)),
                nn.LayerNorm(self.branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv1d(in_channels, self.branch_channels, 1),
                Permute((0, 2, 1)),
                nn.LayerNorm(self.branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.branch_channels, self.branch_channels, 3,
                          stride=2, padding=1),
                Permute((0, 2, 1)),
                nn.LayerNorm(self.branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv1d(in_channels, self.branch_channels, 1),
                Permute((0, 2, 1)),
                nn.LayerNorm(self.branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.branch_channels, self.branch_channels, 5,
                          stride=2, padding=2),
                Permute((0, 2, 1)),
                nn.LayerNorm(self.branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.Conv1d(in_channels, self.branch_channels, 1),
                Permute((0, 2, 1)),
                nn.LayerNorm(self.branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True),
                nn.Conv1d(self.branch_channels, self.branch_channels, 7,
                          stride=2, padding=3),
                Permute((0, 2, 1)),
                nn.LayerNorm(self.branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True)
            ),

            nn.Sequential(
                nn.MaxPool1d(3, stride=2, padding=1),
                nn.Conv1d(in_channels, self.branch_channels, 1),
                Permute((0, 2, 1)),
                nn.LayerNorm(self.branch_channels),
                Permute((0, 2, 1)),
                nn.ReLU(inplace=True)
            )
        ])


        self.fusion = nn.Sequential(
            nn.Conv1d(5 * self.branch_channels, 2 * in_channels, 1),
            Permute((0, 2, 1)),
            nn.LayerNorm(2 * in_channels),
            Permute((0, 2, 1)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        branch_outputs = [branch(x) for branch in self.branches]

        fused = torch.cat(branch_outputs, dim=1)
        return self.fusion(fused)

class Inception1DNet_v2(nn.Module):
    def __init__(self,
                 input_length=12500,
                 input_channels=6,
                 initial_channels=16,
                 branch_channels_A=32,
                 branch_channels_B=32,
                 num_modules_A=4,
                 num_modules_B=4,
                 target_length=80,
                 dropout=0.0):

        super(Inception1DNet_v2, self).__init__()
        self.initial_conv_1 = nn.Conv1d(input_channels, initial_channels, kernel_size=1, stride=2, padding=0)
        self.initial_ln_1 = nn.LayerNorm(initial_channels)
        self.act = nn.ReLU()
        self.initial_conv_2 = nn.Conv1d(initial_channels, initial_channels*2, kernel_size=1, stride=2, padding=0)
        self.initial_ln_2 = nn.LayerNorm(initial_channels*2)
        self.modules_list_A = nn.ModuleList()
        in_channels_A = initial_channels*2
        for i in range(num_modules_A):
            module_A = ResidualInceptionModule1D(in_channels_A, branch_channels_A)
            self.modules_list_A.append(module_A)
        self.downsample = DownsampleInceptionModule1D(in_channels_A)
        self.modules_list_B = nn.ModuleList()
        in_channels_B = in_channels_A * 2
        for i in range(num_modules_B):
            module_B = ResidualInceptionModule1D(in_channels_B, branch_channels_B)
            self.modules_list_B.append(module_B)
        self.final_conv = nn.Conv1d(in_channels_B, 1, kernel_size=1)  # input_length//8
        self.final_mlp = SimMLP(1563, hidden_features=input_length//16 ,out_features=target_length,dropout=dropout)

    def forward(self, x):

        x = x.transpose(1, 2)
        x = self.initial_conv_1(x)
        x = x.transpose(1, 2)
        x = self.initial_ln_1(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        x = self.initial_conv_2(x)
        x = x.transpose(1, 2)
        x = self.initial_ln_2(x)
        x = x.transpose(1, 2)
        x = self.act(x)
        for module_A in self.modules_list_A:
            x = module_A(x)
        x = self.downsample(x)
        for module_B in self.modules_list_B:
            x = module_B(x)
        x = self.final_conv(x)
        x = self.final_mlp(x)
        x = x.transpose(1, 2)
        return x


if __name__ == "__main__":
    batch_size = 1
    seq_length = 12500
    feature_dim = 6
    target_length = 80
    model = Inception1DNet_v2(input_length=12500,
                            input_channels=6,
                            initial_channels=32,
                            branch_channels_A=128,
                            branch_channels_B=256,
                            num_modules_A=6,
                            num_modules_B=4,
                            target_length=80,
                            dropout=0.0)

    x = torch.randn(batch_size, seq_length, feature_dim)
    out = model(x)
    print("Output shape:", out.shape)