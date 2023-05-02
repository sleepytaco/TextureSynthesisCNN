from model import TextureSynthesisCNN


def main():
    model = TextureSynthesisCNN(tex_exemplar_path="data/exemplar_scenery.png")
    model.synthesize_texture(num_epochs=250, use_constraints=True)
    # synth.optimize(num_epochs=500)  # can call this on an existing model object to continue optimization
    model.save_textures(display_when_done=True)  # saves the tex exemplar and synth into results folder (creates it if folder DNE)


if __name__ == '__main__':
    main()
