import torch
from vgg19 import VGG19
from tqdm import tqdm
import utils
import os


class TextureSynthesisCNN:
    def __init__(self, tex_exemplar_path):
        """
        :param tex_exemplar_path: ideal texture image w.r.t which we are synthesizing our textures
        :param device: torch device - cuda or cpu
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tex_exemplar_name = os.path.splitext(os.path.basename(tex_exemplar_path))[0]

        # init VGGs
        vgg_exemplar = VGG19(freeze_weights=True)  # vgg to generate ideal feature maps
        self.vgg_synthesis = VGG19(freeze_weights=False)  # vgg whose weights will be trained

        # calculate and save gram matrices for the texture exemplar once (as this does not change)
        self.tex_exemplar_image = utils.load_image_tensor(tex_exemplar_path).to(self.device)  # image path -> image Tensor
        feature_maps_ideal = vgg_exemplar(self.tex_exemplar_image)
        self.gram_matrices_ideal = utils.calculate_gram_matrices(feature_maps_ideal)

        # set up the initial random noise image output which the network will optimize
        self.output_image = torch.randn_like(self.tex_exemplar_image).to(self.device)
        self.output_image.requires_grad = True  # set to True so that the rand noise image can be optimized

        self.optimizer = torch.optim.LBFGS([self.output_image])
        self.layer_weights = [10**9] * len(vgg_exemplar.important_layers)  # layer weights as per paper
        self.beta = 10**5  # beta as per paper
        self.losses = []

    def optimize(self, num_epochs=250, checkpoint=None, use_constraints=True):
        """
        Perform num_epochs steps of L-BFGS algorithm
        If checkpoint is not none, saves self.output_image into an intermediate_outputs folder every multiple of checkpoint
        Set use_constraints True to use the spectrum constraints
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
                # 1. passes output_img through vgg_synth, 2. returns loss
                loss = self.get_loss(use_constraints=use_constraints)
                loss.backward()
                return loss

            self.optimizer.step(closure)  # LBFGS optimizer expects loss in the form of closure function
            self.losses.append(epoch_loss)

            if checkpoint and (epoch + 1) % checkpoint == 0:
                utils.save_image_tensor(self.output_image.clone().detach().cpu(),
                                        output_dir=f"intermediate_outputs_{self.tex_exemplar_name}/",
                                        image_name=f"output_at_epoch_{epoch_offset + epoch + 1:0>6}.png")

        return self.output_image.detach().cpu()

    def get_loss(self, use_constraints=True) -> torch.Tensor:
        """
        Generates the feature maps for the current output synth image, and uses the ideal feature maps to come up
        with the loss E at one layer. All the E's are added up to return the overall loss.
        Set use_constraints True to calculate Loss_spectrum and add it to the overall loss.
        """
        loss = 0
        if use_constraints:
            i_tilde = utils.get_i_tilde(i_hat=self.output_image, i=self.tex_exemplar_image)
            i_hat = utils.get_grayscale(self.output_image)
            loss_spec = (0.5 * (i_hat - i_tilde) ** 2).sum().real
            loss += (self.beta * loss_spec)

        # calculate the gram mats for the output_image at the time the get_loss is called
        feature_maps_pred = self.vgg_synthesis(self.output_image)
        gram_matrices_pred = utils.calculate_gram_matrices(feature_maps_pred)

        for i in range(len(self.layer_weights)):
            # E_l = w_l * ||G_ideal_l - G_pred_l||^2
            # scale = torch.tensor([4*(self.N**2)*(self.M[i]**2)], dtype=torch.float32).to(self.device)
            E = self.layer_weights[i] * ((self.gram_matrices_ideal[i] - gram_matrices_pred[i]) ** 2).sum()
            loss += E

        # loss = (w1*E1 + w2*E2 + ... + w5*E5) + beta * loss_spec
        return loss

    def synthesize_texture(self, num_epochs=250, checkpoint=None, display_when_done=False, use_constraints=True):
        """
        - Can be called multiple times to generate different texture variations of the tex exemplar this model holds
        - IMPT: resets the output_image to random noise each time this is called
        - Idea: Each time the optimizer starts off from a random noise image, the network optimizes/synthesizes
          the original tex exemplar in a slightly different way - i.e. introduce variation in the synthesis.
        """
        self.output_image = torch.randn_like(self.tex_exemplar_image).to(
            self.device)  # make the last texture synth output the output image
        self.output_image.requires_grad = True
        self.optimizer = torch.optim.LBFGS([self.output_image])

        self.losses = []

        synthesized_texture = self.optimize(num_epochs=num_epochs, checkpoint=checkpoint,
                                            use_constraints=use_constraints)
        if display_when_done: utils.display_image_tensor(synthesized_texture)

        return synthesized_texture

    def save_textures(self, output_dir="./results/", display_when_done=False):
        """
        Saves and displays the current tex_exemplar_image and the output_image tensors that this model holds
        into the results directory (creates it if not yet created)
        """
        tex_exemplar = utils.save_image_tensor(self.tex_exemplar_image.cpu(),
                                               output_dir=output_dir,
                                               image_name=f"exemplar_{self.tex_exemplar_name}.png")

        if display_when_done: tex_exemplar.show()

        print()
        tex_synth = utils.save_image_tensor(self.output_image.detach().cpu(),
                                            output_dir=output_dir,
                                            image_name=f"synth_{self.tex_exemplar_name}.png")
        if display_when_done: tex_synth.show()
