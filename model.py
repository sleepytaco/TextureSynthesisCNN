import torch
from torchvision import transforms
from vgg19 import VGG19
from tqdm import tqdm
import utils
import os


class TextureSynthesisCNN:
    def __init__(self, tex_exemplar_path, device):
        """
        :param tex_exemplar_path: ideal texture image w.r.t which we are synthesizing our textures
        :param device: torch device - cuda or cpu
        """
        self.device = device
        self.tex_exemplar_name = os.path.splitext(os.path.basename(tex_exemplar_path))[0]

        # init VGGs
        self.vgg_exemplar = VGG19(freeze_weights=True, device=device)  # vgg to generate ideal feature maps
        self.vgg_synthesis = VGG19(freeze_weights=False, device=device)  # vgg whose weights will be trained

        # calculate and save gram matrices for the texture exemplar once (as this does not change)
        self.tex_exemplar_image = utils.load_image_tensor(tex_exemplar_path).to(device)  # "path" -> Tensor
        feature_maps_ideal = self.vgg_exemplar(self.tex_exemplar_image)
        self.gram_matrices_ideal = utils.calculate_gram_matrices(feature_maps_ideal)

        # set up the initial random noise image output which the network will optimize
        self.output_image = torch.randn_like(self.tex_exemplar_image).to(device)
        self.output_image.requires_grad = True  # set to True so that the rand noise image can be optimized

        self.optimizer = torch.optim.LBFGS([self.output_image])
        self.layer_weights = [10**9, 10**9, 10**9, 10**9, 10**9]  # layer weights as per paper

        self.losses = []

    def optimize(self, num_epochs=100, checkpoint=5):
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
                                        output_dir=f"intermediate_outputs_{self.tex_exemplar_name}/",
                                        image_name=f"output_at_epoch_{epoch_offset + epoch + 1:0>6}.png")

        return self.output_image.detach().cpu()

    def get_loss(self) -> torch.Tensor:
        """
        Generates the feature maps for the current output synth image, and uses the ideal feature maps to come up
        with the loss E at one layer. All the E's are added up to return the overall loss.
        """
        loss = 0
        i_tilde = utils.get_i_tilde(i_hat=self.output_image, i=self.tex_exemplar_image)
        i_hat = utils.get_grayscale(self.output_image)
        delta_spe = i_hat - i_tilde


        # calculate the gram mats for the output_image at the time the get_loss is called
        feature_maps_pred = self.vgg_synthesis(self.output_image)
        gram_matrices_pred = utils.calculate_gram_matrices(feature_maps_pred)

        for i in range(len(self.layer_weights)):
            # E_l = w_l * ||G_ideal_l - G_pred_l||^2
            E = self.layer_weights[i] * ((self.gram_matrices_ideal[i] - gram_matrices_pred[i]) ** 2).sum()
            loss += E

        # loss = w1*E1 + w2*E2 + ... + w5*E5
        return loss

    def update_texture_exemplar(self, new_tex_exemplar_path):
        """
        Keeps the output_image the same, but updates the existing tex exemplar to the provided tex exemplar image path
        As such, the ideal gram matrices need to be re-calculated for the new tex exemplar image
        """
        # make the last texture synth output the output image
        self.output_image = self.output_image.clone().detach().to(self.device)
        self.output_image.requires_grad = True
        self.optimizer = torch.optim.LBFGS([self.output_image])

        # update old texture exemplar to new tex exemplar
        self.tex_exemplar_name = os.path.splitext(os.path.basename(new_tex_exemplar_path))[0]
        self.tex_exemplar_image = utils.load_image_tensor(new_tex_exemplar_path).to(self.device)  # "path" -> Tensor

        # calculate and save gram matrices for the new texture exemplar
        feature_maps_ideal = self.vgg_exemplar(self.tex_exemplar_image)
        self.gram_matrices_ideal = utils.calculate_gram_matrices(feature_maps_ideal)

    def synthesize_texture(self, num_epochs=100, checkpoint=5, display_when_done=True):
        """
        - Can be called multiple times to generate different textures
        - Resets the output_image to random noise and runs the optimizer again. Each time the optimizer starts off from a
        random noise image, the network optimizes/synthesizes the tex exemplar in a slightly different way.
        """
        self.output_image = torch.randn_like(self.tex_exemplar_image).to(self.device)  # make the last texture synth output the output image
        self.output_image.requires_grad = True
        self.optimizer = torch.optim.LBFGS([self.output_image])

        synthesized_texture = self.optimize(num_epochs=num_epochs, checkpoint=checkpoint)
        if display_when_done: utils.display_image_tensor(synthesized_texture)

        return synthesized_texture

    def save_and_display_texture(self, output_dir="./results/"):
        """
        Saves and displays the current tex_exemplar_image and the output_image tensors that this model holds
        """
        utils.save_image_tensor(self.tex_exemplar_image.cpu(),
                                output_dir=output_dir,
                                image_name=f"exemplar_{self.tex_exemplar_name}.png").show()
        print()
        utils.save_image_tensor(self.output_image.detach().cpu(),
                                output_dir=output_dir,
                                image_name=f"synth_{self.tex_exemplar_name}.png").show()