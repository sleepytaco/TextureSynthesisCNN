from torch import nn
from torchvision.models import VGG19_Weights, vgg19


class VGG19:
    """
    Custom version of VGG19 with the maxpool layers replaced with avgpool as per the paper
    VGG19(image_tensor) returns a list of featuremaps at the specified important layers
    """
    def __init__(self, freeze_weights, device):
        """
        :param freeze_weights: If True, the gradients for the VGG params are turned off
        :param device: Torch device - cuda or cpu
        """
        self.model = vgg19(weights=VGG19_Weights(VGG19_Weights.DEFAULT)).to(device)
        # self.important_layers = [0, 4, 9, 16, 23]  # vgg16 layers at which there is a MaxPool
        self.important_layers = [0, 4, 9, 18, 27, 36]  # vgg19 layers [convlayer1, maxpool, ..., maxpool]
        for layer in self.important_layers[1:]:  # convert the maxpool layers to an avgpool
            self.model.features[layer] = nn.AvgPool2d(kernel_size=2, stride=2)

        self.feature_maps = []
        for param in self.model.parameters():
            if freeze_weights:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def __call__(self, x):
        """
        Take in image, pass it through the VGG, and capture feature map outputs at each of the important layers of VGG
        Return a list of feature maps
        """
        self.feature_maps = []
        for index, layer in enumerate(self.model.features):
            # print(layer)
            x = layer(x)
            if index in self.important_layers:
                self.feature_maps.append(x)
            if index == self.important_layers[-1]:
                # stop VGG execution as we've captured the feature maps from all the important layers
                break
        return self.feature_maps
