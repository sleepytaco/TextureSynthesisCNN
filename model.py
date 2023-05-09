import torch
from torch import fft
from vgg19 import VGG19
from tqdm import tqdm
import utils
import os


class TextureSynthesisCNN:
    def __init__(self, tex_exemplar_path):
        """
        tex_exemplar_path: ideal texture image w.r.t which we are synthesizing our textures
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tex_exemplar_name = os.path.splitext(os.path.basename(tex_exemplar_path))[0]

        # init VGGs
        vgg_exemplar = VGG19(freeze_weights=True)  # vgg to generate ideal feature maps
        self.vgg_synthesis = VGG19(freeze_weights=False)  # vgg whose weights will be trained

        # calculate and save gram matrices for the texture exemplar once (as this does not change)
        self.tex_exemplar_image = utils.load_image_tensor(tex_exemplar_path).to(self.device)  # image path -> image Tensor
        self.gram_matrices_ideal = vgg_exemplar(self.tex_exemplar_image).get_gram_matrices()

        # set up the initial random noise image output which the network will optimize
        self.output_image = torch.sigmoid(torch.randn_like(self.tex_exemplar_image)).to(self.device)  # sigmoid to ensure values b/w 0 and 1
        self.output_image.requires_grad = True  # set to True so that the rand noise image can be optimized

        self.LBFGS = torch.optim.LBFGS([self.output_image])
        self.layer_weights = [10**9] * len(vgg_exemplar.output_layers)  # output layer weights as per paper
        self.beta = 10**5  # beta as per paper
        self.losses = []

    def synthesize_texture(self, num_epochs=250, display_when_done=False):
        """
        - Idea: Each time the optimizer starts off from a random noise image, the network optimizes/synthesizes
          the original tex exemplar in a slightly different way - i.e. introduce variation in the synthesis.
        - Can be called multiple times to generate different texture variations of the tex exemplar this model holds
        - IMPT: resets the output_image to random noise each time this is called
        """
        self.losses = []

        # reset output image to random noise
        self.output_image = torch.sigmoid(torch.randn_like(self.tex_exemplar_image)).to(self.device)
        self.output_image.requires_grad = True
        self.LBFGS = torch.optim.LBFGS([self.output_image])  # update LBFGS to hold the new output image

        synthesized_texture = self.optimize(num_epochs=num_epochs)
        if display_when_done:
            utils.display_image_tensor(synthesized_texture)

        return synthesized_texture

    def optimize(self, num_epochs=250):
        """
        Perform num_epochs steps of L-BFGS algorithm
        """
        progress_bar = tqdm(total=num_epochs, desc="Optimizing...")
        epoch_offset = len(self.losses)

        for epoch in range(num_epochs):
            epoch_loss = self.get_loss().item()
            progress_bar.update(1)
            progress_bar.set_description(f"Loss @ Epoch {epoch_offset + epoch + 1}  - {epoch_loss} ")
            self.LBFGS.step(self.LBFGS_closure)  # LBFGS optimizer expects loss in the form of closure function
            self.losses.append(epoch_loss)

        return self.output_image.detach().cpu()

    def LBFGS_closure(self):
        """
        Closure function for LBFGS which passes the curr output_image through vgg_synth, computes prediction gram_mats,
        and uses that to compute loss for the network.
        """
        self.LBFGS.zero_grad()
        loss = self.get_loss()
        loss.backward()
        return loss

    def get_loss(self):
        """
        CNN loss: Generates the feature maps for the current output synth image, and uses the ideal feature maps to come
        up with a loss E_l at one layer l. All the E_l's are added up to return the total cnn loss.
        Spectrum loss: project tex synth to tex exemplar to come up with the spectrum constraint as per paper
        Overall loss = loss_cnn + loss_spec
        """
        # calculate spectrum constraint loss using current output_image and tex_exemplar_image
        # - projects image I_hat (tex_synth) onto image I (tex_exemplar) and return I_proj (equation as per paper)
        I_hat = utils.get_grayscale(self.output_image)
        I_fourier = fft.fft2(utils.get_grayscale(self.tex_exemplar_image))
        I_hat_fourier = fft.fft2(I_hat)
        I_fourier_conj = torch.conj(I_fourier)
        epsilon = 10e-12  # epsilon to avoid div by 0 and nan values
        I_proj = fft.ifft2((I_hat_fourier * I_fourier_conj) / (torch.abs(I_hat_fourier * I_fourier_conj) + epsilon) * I_fourier)
        loss_spec = (0.5 * (I_hat - I_proj) ** 2.).sum().real

        # get the gram mats for the synth output_image by passing it to second vgg network
        gram_matrices_pred = self.vgg_synthesis(self.output_image).get_gram_matrices()

        # calculate cnn loss
        loss_cnn = 0.  # (w1*E1 + w2*E2 + ... + wl*El)
        for i in range(len(self.layer_weights)):
            # E_l = w_l * ||G_ideal_l - G_pred_l||^2
            E = self.layer_weights[i] * ((self.gram_matrices_ideal[i] - gram_matrices_pred[i]) ** 2.).sum()
            loss_cnn += E

        return loss_cnn + (self.beta * loss_spec)

    def save_textures(self, output_dir="./results/", display_when_done=False):
        """
        Saves and displays the current tex_exemplar_image and the output_image tensors that this model holds
        into the results directory (creates it if not yet created)
        """
        tex_exemplar = utils.save_image_tensor(self.tex_exemplar_image.cpu(),
                                               output_dir=output_dir,
                                               image_name=f"exemplar_{self.tex_exemplar_name}.png")
        tex_synth = utils.save_image_tensor(self.output_image.detach().cpu(),
                                            output_dir=output_dir,
                                            image_name=f"synth_{self.tex_exemplar_name}.png")
        if display_when_done:
            tex_exemplar.show()
            print()
            tex_synth.show()
