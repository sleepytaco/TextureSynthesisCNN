from model import TextureSynthesisCNN
import torch
import utils


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # set up texture exemplar
    texture_path = "data/preprocessed_images/banded_0002.jpg"
    texture_exemplar_image = utils.load_image_tensor(texture_path).to(device)  # "path" -> Tensor

    # set up output image - can be whitenoise or some color image if we want variations of textures
    output_path = "data/preprocessed_images/banded_0036.jpg"
    texture_output_image = utils.load_image_tensor(output_path).to(device)  # "path" -> Tensor
    texture_output_image_rand = torch.randn_like(texture_exemplar_image).to(device)  # white noise image

    synthesizer = TextureSynthesisCNN(texture_exemplar_image=texture_exemplar_image,
                                      texture_output_image=texture_output_image_rand,
                                      device=device)
    output_image = synthesizer.optimize(num_epochs=10,
                                        checkpoint=1)  # returns "optimized" texture_output_image Tensor

    utils.save_image_tensor(output_image,
                            image_name="final.png")  # Tensor -> ./final.png file


if __name__ == '__main__':
    main()