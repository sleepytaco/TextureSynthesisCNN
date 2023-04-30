import torch
from torchvision.models import vgg16, VGG16_Weights, vgg19
import utils


class VGG16:
    def __init__(self, freeze_weights, device):
        """
        :param freeze_weights: If True, the gradients for the VGG params are turned off
        :param device: Torch device - cuda or cpu
        """
        self.model = vgg16(weights=VGG16_Weights(VGG16_Weights.DEFAULT)).to(device)
        self.important_layers = [4, 9, 16, 23, 30]  # layers at which there is a MaxPool/AvgPool
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
            x = layer(x)
            if index in self.important_layers:
                print(layer)
                self.feature_maps.append(x)
            if index == self.important_layers[-1]:
                # stop VGG execution as we've captured the feature maps from all the important layers
                break
        return self.feature_maps


class TextureSynthesisCNN:
    def __init__(self, texture_exemplar_image: torch.Tensor, device):
        """
        :param texture_exemplar_image: ideal texture image w.r.t which we are synthesizing our textures
        :param device: torch device - cuda or cpu
        """
        # calculate and save gram matrices for the texture exemplar once (as this does not change)
        vgg_exemplar = VGG16(freeze_weights=True, device=device)
        feature_maps_ideal = vgg_exemplar(texture_exemplar_image)
        self.gram_matrices_ideal = utils.calculate_gram_matrices(feature_maps_ideal)

        # vgg whose weights will be trained
        self.output_image = torch.randn_like(texture_exemplar_image).to(device)  # output image is initially a random noise image
        self.output_image.requires_grad = True  # set to True so that the rand noise image can be optimized
        self.vgg_synthesis = VGG16(freeze_weights=False, device=device)

        self.optimizer = torch.optim.LBFGS([self.output_image])
        self.layer_weights = [10 ** 9, 10 ** 9, 10 ** 9, 10 ** 9, 10 ** 9]

        self.losses = []
        self.intermediate_synth_images = []

        self.reparametrization_function = utils.reparametrize_image(texture_exemplar_image)

    def optimize(self, num_epochs=100):
        """
        Perform num_epochs steps of L-BFGS algorithm
        """
        self.losses = []
        # self.intermediate_synth_images = []

        def closure():
            self.optimizer.zero_grad()
            loss = self.get_loss()  # 1. passes output_img through vgg_synth, 2. returns loss
            loss.backward()
            return loss

        for epoch in range(num_epochs):
            epoch_loss = self.get_loss()
            self.optimizer.step(closure)
            self.losses.append(epoch_loss)
            # self.intermediate_synth_images.append(self.output_image.clone().detach().cpu())
            print(f"Epoch {epoch}: Loss - {epoch_loss}")

        return self.output_image.detach().cpu()

    def get_loss(self) -> torch.Tensor:
        """
        Generates the feature maps for the current output synth image, and uses the ideal feature maps to come up
        with the loss E at one layer. All the E's are added up to return the overall loss.
        """
        loss = torch.Tensor(0)

        feature_maps_pred = self.vgg_synthesis(self.output_image)
        gram_matrices_pred = utils.calculate_gram_matrices(feature_maps_pred)
        for i, gm_ideal, gm_pred in enumerate(zip(self.gram_matrices_ideal, gram_matrices_pred)):
            E = self.layer_weights[i] * ((gm_ideal - gm_pred) ** 2).sum()
            loss += E

        return loss
