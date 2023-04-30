import torch
from torchvision.models import vgg16, VGG16_Weights, vgg19


def calculate_gram_matrices(feature_maps):
    return list(map(lambda x: torch.mm(x, x.t()), feature_maps))


class VGG16:
    def __init__(self, freeze_weights):
        self.model = vgg16(weights=VGG16_Weights(VGG16_Weights.DEFAULT))
        self.important_layers = [4, 9, 16, 23, 30]  # layers at which there is a MaxPool/AvgPool
        self.feature_maps = []
        for param in self.model.parameters():
            if freeze_weights:
                param.requires_grad = False
            else:
                param.requires_grad = True

    def __call__(self, x):
        """
        Take in image and return Gram Matrices at each of the important layers of VGG
        """
        self.feature_maps = []
        for index, layer in enumerate(self.model.features):
            x = layer(x)
            if index in self.important_layers:
                print(layer)
                self.feature_maps.append(x)
            if index == self.important_layers[-1]:
                break
        return self.feature_maps


class TextureSynthesisCNN:
    def __init__(self, texture_exemplar_image):

        # calculate and save gram matrices for the texture exemplar once (as this does not change)
        self.texture_exemplar_image = texture_exemplar_image  # ideal texture image w.r.t which we are synthesizing our textures
        vgg_exemplar = VGG16(freeze_weights=True)
        feature_maps_ideal = vgg_exemplar(texture_exemplar_image)
        self.gram_matrices_ideal = calculate_gram_matrices(feature_maps_ideal)

        # vgg whose weights will be trained
        self.output_image = torch.randn_like(texture_exemplar_image)  # output image is initially a random noise image
        self.output_image.requires_grad = True  # set to True so that the rand noise image can be optimized
        self.vgg_synthesis = VGG16(freeze_weights=False)

        self.optimizer = torch.optim.LBFGS([self.output_image])
        self.layer_weights = [10 ** 9, 10 ** 9, 10 ** 9, 10 ** 9, 10 ** 9]

    def optimize(self, num_epochs=10):
        losses = []
        intermediate_synth_images = []

        def closure():
            self.optimizer.zero_grad()
            loss = self.get_loss()  # 1. passes output_img through vgg_synth, 2. returns loss
            loss.backward()
            return loss

        for epoch in range(num_epochs):
            self.optimizer.step(closure)

        return self.output_image.detach().cpu()

    def get_loss(self) -> torch.Tensor:
        loss = torch.Tensor(0)

        feature_maps_pred = self.vgg_synthesis(self.output_image)
        gram_matrices_pred = calculate_gram_matrices(feature_maps_pred)
        for i, gm_ideal, gm_pred in enumerate(zip(self.gram_matrices_ideal, gram_matrices_pred)):
            E = self.layer_weights[i] * ((gm_ideal - gm_pred) ** 2).sum()
            loss += E

        return loss



