import torch
from torch import nn
from torch.nn import functional as F


class OCR(nn.Module):
    def __init__(self, features, in_channels, out_features, num_classes, num_layers=2):
        super(OCR, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_features = out_features
        self.num_layers = num_layers
        self.conv = self.conv_layers(features)
        self.linear = self.linear_layers()
        self.lstm = nn.GRU(self.out_features*2, self.out_features, bidirectional=True, num_layers=self.num_layers, dropout=0.25, batch_first=True)
        self.cls = nn.Linear(self.out_features*2, self.num_classes)
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')

    def forward(self, x, targets=None):
        bs, _, _, _ = x.size()
        x = self.conv(x)
        x = x.permute(0, 3, 1, 2)
        x = x.view(bs, x.size(1), -1)
        x = self.linear(x)
        x, _ = self.lstm(x)
        x = self.cls(x)
        x = x.permute(1, 0, 2)

        if targets is not None:
            loss = self.calculate_loss(x, targets)
            return x, loss

        if not self.training:
            x = F.log_softmax(x, 2).argmax(2).squeeze(1)
            return x

        return x, None

    def calculate_loss(self, logits, texts):
        input_len, batch_size, vocab_size = logits.size()
        logits = logits.log_softmax(2)
        logits_lens = torch.full(size=(batch_size,), fill_value=input_len, dtype=torch.int32)
        target, target_lengths = texts[0], texts[1]
        loss = self.criterion(logits, target, logits_lens, target_lengths)
        return loss


    def linear_layers(self):
        x = nn.Sequential(
            nn.Linear(in_features=2048, out_features = self.out_features*2),
            nn.BatchNorm1d(8),
            nn.ReLU()
        )
        return x

    def conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels
        for arg in architecture:
            if type(arg) == int:
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=arg, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                           nn.BatchNorm2d(arg),
                           nn.ReLU(),
                           nn.Dropout2d(0.1)]
                in_channels = arg
            elif arg == "M":
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]
        return nn.Sequential(*layers)




if __name__ == '__main__':
    args = [64, "M", 128, "M", 256, "M", 512, "M", 512]
    num_classes = 35
    net = OCR(features=args, in_channels=1, num_classes = num_classes, out_features=32, num_layers=2)
    print(net(torch.rand(1, 1, 64, 128))[0].shape) # [8, 1, num_classes]