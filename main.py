from model import TextureSynthesisCNN
import torch

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    texture_path = ""
    texture_exemplar = load_image_tensor(texture_path)  # "path" -> Tensor
    synthesizer = TextureSynthesisCNN(texture_exemplar_image=texture_exemplar, device=device)
    output_image = synthesizer.optimize(num_epochs=10)  # returns Tensor
    save_image_tensor(output_image)  # save Tensor as .jpg/.png file locally