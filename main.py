from model import TextureSynthesisCNN
import torch


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    texture_exemplar_path = "data/preprocessed_images/banded_0002.jpg"
    synthesizer = TextureSynthesisCNN(tex_exemplar_path=texture_exemplar_path, device=device)
    output_image = synthesizer.synthesize_texture(num_epochs=10, checkpoint=1)


if __name__ == '__main__':
    main()
