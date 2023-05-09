from model import TextureSynthesisCNN


def main():
    synthesizer = TextureSynthesisCNN(tex_exemplar_path="data/cracked_0063.png")
    synthesizer.synthesize_texture(num_epochs=10)
    # synthesizer.optimize(num_epochs=500)  # can call this on an existing model object to continue optimization
    synthesizer.save_textures(output_dir="./results/",  # directory automatically is created if not found
                              display_when_done=True)  # saves exemplar and synth into the output_dir folder


if __name__ == '__main__':
    main()
