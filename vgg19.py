import torch
from torch import nn
from torchvision.models import VGG19_Weights, vgg19


class VGG19:
    """
    Custom version of VGG19 with the maxpool layers replaced with avgpool as per the paper
    """
    def __init__(self, freeze_weights):
        """
        If True, the gradients for the VGG params are turned off
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = vgg19(weights=VGG19_Weights(VGG19_Weights.DEFAULT)).to(device)

        # note: added one extra maxpool (layer 36) from the vgg... worked well so kept it in
        self.output_layers = [0, 4, 9, 18, 27, 36]  # vgg19 layers [convlayer1, maxpool, ..., maxpool]
        for layer in self.output_layers[1:]:  # convert the maxpool layers to an avgpool
            self.model.features[layer] = nn.AvgPool2d(kernel_size=2, stride=2)

        self.feature_maps = []
        for param in self.model.parameters():
            if freeze_weights:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def __call__(self, x):
        """
        Take in image, pass it through the VGG, capture feature maps at each of the output layers of VGG
        """
        self.feature_maps = []
        for index, layer in enumerate(self.model.features):
            # print(layer)
            x = layer(x)  # pass the img through the layer to get feature maps of the img
            if index in self.output_layers:
                self.feature_maps.append(x)
            if index == self.output_layers[-1]:
                # stop VGG execution as we've captured the feature maps from all the important layers
                break

        return self

    def get_gram_matrices(self):
        """
        Convert the featuremaps captured by the call method into gram matrices
        """
        gram_matrices = []
        for fm in self.feature_maps:
            n, x, y = fm.size()  # num filters n and (filter dims x and y)
            F = fm.reshape(n, x * y)  # reshape filterbank into a 2D mat before doing auto correlation
            gram_mat = (F @ F.t()) / (4. * n * x * y)  # auto corr + normalize by layer output dims
            gram_matrices.append(gram_mat)

        return gram_matrices
