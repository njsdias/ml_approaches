import torch
from torch import nn
from torch.nn import functional as F
import config


class CaptchaModel(nn.Module):
    def __init__(self, num_chars):
        # num_chars: number of different characters
        super(CaptchaModel, self).__init__()
        self.conv_1 = nn.Conv2d(3, 128, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

        # input of second layer is always equal to output of the previous layer
        self.conv_2 = nn.Conv2d(128, 64, kernel_size=(3, 3), padding=(1, 1))
        self.max_pool = nn.MaxPool2d(kernel_size=(2, 2))

        # 1152 was obtained when we print the view: x.view
        # 64 is the number of features
        self.linear_1 = nn.Linear(1152, 64)
        self.drop_1 = nn.Dropout(0.2)

        # GRU model
        # 64 was obtained after drop_1
        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=2, dropout=0.25)
        # we need to do +1 becasue we have an UNKNOWN
        self.output = nn.Linear(64, num_chars=64 + 1)

    def forward(self, images, targets=None):
        # batch_size, channel, height, width
        # batch size is the number of samples that we have in one batch
        bs, c, h, w = images.size()
        print(bs, c, h, w)

        # activation function Relu
        x = F.relu(self.conv_1(images))
        print(x.size())
        x = self.max_pool_1(x)
        print(x.size())

        x = F.relu(self.conv_2(x))
        print(x.size())
        x = self.max_pool_2(x)  # 1, 64, 18, 75
        print(x.size())

        # we want to view the width of image when we apply rnn model
        # the width was in the 3rd position and we want it in the 1st position
        x = x.permute(0, 3, 1, 2)   # 1, 75, 64, 18
        print(x.size())

        x = x.view(bs, x.size(1), -1)
        print(x.size())   # 1152 number of features

        x = self.linear_1(x)
        x = self.drop_1(x)
        print(x.size())     # 64 features

        x, _ = self.gru(x)
        print(x.size())

        x = self.output(x)
        print(x.size())

        # timestamp, batch size, values
        # to use CTC loss
        x = x.permute(1, 0, 2)
        print(x.size())

        # for CTC LOss
        if targets is not None:
            log_softmax_values = F.log_softmax(x, 2)  # 2 is the last number that we have in classes
            input_lengths = torch.full(
                size=(bs, ),
                fill_value=log_softmax_values.size(0),
                dtype=torch.int32
            )
            print(input_lengths)
            target_lengths = torch.full(
                size=(bs,),
                fill_value=target.size(1),  # in our case it is 5 becasue we have 5 characters in each label
                dtype=torch.int32
            )
            print(target_lengths)

            loss = nn.CTCLoss(blank=0)(
                log_softmax_values, targets, input_lengths, target_lengths
            )

            return x, loss


if __name__ == "__main__":
    # number of unique chars
    num_chars = 19
    batch_size = 5
    cm = CaptchaModel(num_chars)
    img = torch.rand(batch_size, 3, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)

    # target is the name of file that has 5 characters
    # here we are randomize to have some targets
    target = torch.randint(1, 20, (batch_size, 5))

    # capture the image and loss
    x, loss = cm(img, target)










