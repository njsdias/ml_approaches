import torch.nn as nn
import pretrainedmodels


def get_model(pretrained):

    # list of possible models
    model_type = ["alexnet", "resnet18"]

    if pretrained:
        model = pretrainedmodels.__dict__[model_type[0]](
            pretrained='imagenet'
        )
    else:
        model = pretrainedmodels.__dict__[model_type[0]](
            pretrained=None
        )

    # print the model here to know what is going on
    model.last_linear = nn.Sequential(
        nn.BatchNorm1d(4095),  # 512 for resnet18
        nn.Dropout(p=0.25),
        nn.Linear(in_features=4096, out_features=2048),  # in_features=512 for resnet18
        nn.ReLU(),
        nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1),
        nn.Dropout(p=0.5),
        nn.Linear(in_features=2048, out_features=1)
    )

    return model
