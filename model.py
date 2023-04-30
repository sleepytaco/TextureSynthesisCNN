import torch
from vgg19 import VGG19
from tqdm import tqdm
import utils


class TextureSynthesisCNN:
    def __init__(self, texture_exemplar_image: torch.Tensor, texture_output_image: torch.Tensor,device):
        """
        :param texture_exemplar_image: ideal texture image w.r.t which we are synthesizing our textures
        :param texture_output_image: the initial random noise output which we will optimize
        :param device: torch device - cuda or cpu
        """
        # calculate and save gram matrices for the texture exemplar once (as this does not change)
        vgg_exemplar = VGG19(freeze_weights=True, device=device)
        feature_maps_ideal = vgg_exemplar(texture_exemplar_image)
        self.gram_matrices_ideal = utils.calculate_gram_matrices(feature_maps_ideal)

        # vgg whose weights will be trained
        self.vgg_synthesis = VGG19(freeze_weights=False, device=device)
        self.output_image = texture_output_image
        self.output_image.requires_grad = True  # set to True so that the rand noise image can be optimized

        self.optimizer = torch.optim.LBFGS([self.output_image])
        self.layer_weights = [10**9, 10**9, 10**9, 10**9, 10**9]  # layer weights as per paper

        self.losses = []

    def optimize(self, num_epochs=100, checkpoint=25):
        """
        Perform num_epochs steps of L-BFGS algorithm
        """
        progress_bar = tqdm(total=num_epochs, desc="Optimizing...")
        epoch_offset = len(self.losses)

        for epoch in range(num_epochs):
            epoch_loss = self.get_loss().item()
            # print(f"Epoch {epoch+1}: Loss - {epoch_loss}")
            progress_bar.update(1)
            progress_bar.set_description(f"Loss @ Epoch {epoch_offset + epoch + 1}  - {epoch_loss} ")

            def closure():
                self.optimizer.zero_grad()
                loss = self.get_loss()  # 1. passes output_img through vgg_synth, 2. returns loss
                loss.backward()
                return loss

            self.optimizer.step(closure)  # LBFGS optimizer expects loss in the form of closure function
            self.losses.append(epoch_loss)

            if (epoch + 1) % checkpoint == 0:
                utils.save_image_tensor(self.output_image.clone().detach().cpu(),
                                        output_dir="intermediate_outputs/",
                                        image_name=f"output_at_epoch_{epoch_offset + epoch + 1:0>6}.png")

        return self.output_image.detach().cpu()

    def get_loss(self) -> torch.Tensor:
        """
        Generates the feature maps for the current output synth image, and uses the ideal feature maps to come up
        with the loss E at one layer. All the E's are added up to return the overall loss.
        """
        loss = 0

        # calculate the gram mats for the output_image at the time the get_loss is called
        feature_maps_pred = self.vgg_synthesis(self.output_image)
        gram_matrices_pred = utils.calculate_gram_matrices(feature_maps_pred)

        for i in range(len(self.layer_weights)):
            # E_l = w_l * ||G_ideal_l - G_pred_l||^2
            E = self.layer_weights[i] * ((self.gram_matrices_ideal[i] - gram_matrices_pred[i]) ** 2).sum()
            loss += E

        # loss = w1*E1 + w2*E2 + ... + w5*E5
        return loss
