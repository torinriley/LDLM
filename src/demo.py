import torch
from model import LatentDiffusionModel, ReverseTransformer


def load_model(checkpoint_path, latent_dim, encoder_model_name, decoder_model_name, device):
    """
    Load the pretrained models and the reverse transformer from a checkpoint.
    """
    model = LatentDiffusionModel(
        encoder_model_name=encoder_model_name,
        decoder_model_name=decoder_model_name
    )
    reverse_transformer = ReverseTransformer(latent_dim=latent_dim)
    
    reverse_transformer.load_state_dict(torch.load(checkpoint_path, map_location=device))
    reverse_transformer.to(device)
    reverse_transformer.eval()

    return model, reverse_transformer


def create_timestep_embedding(timestep, embedding_dim, device):
    """
    Creates sinusoidal timestep embeddings.
    """
    half_dim = embedding_dim // 2
    freqs = torch.exp(
        -torch.arange(0, half_dim, dtype=torch.float32, device=device) * (2 * torch.pi / embedding_dim)
    )
    timestep = timestep.view(-1, 1)  # Shape: (batch_size, 1)
    angles = timestep * freqs  # Shape: (batch_size, half_dim)
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  # Shape: (batch_size, embedding_dim)
    return embedding


def infer(model, reverse_transformer, input_texts, device, num_steps=1000):
    """
    Perform inference using the trained Latent Diffusion Model.
    """
    latents = model.encode(input_texts).to(device)

    noisy_latents = torch.randn_like(latents).to(device)

    for t in reversed(range(num_steps)):
        timesteps = torch.full((latents.size(0),), t, dtype=torch.float32, device=device) 
        timestep_embedding = create_timestep_embedding(timesteps, latents.size(-1), device)

        predicted_latents = reverse_transformer(noisy_latents, timestep_embedding)
        noisy_latents = predicted_latents

    decoded_texts = model.decode(noisy_latents)

    return decoded_texts


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "" # Path to checkpoint
    latent_dim = 768
    encoder_model_name = "bert-base-uncased"
    decoder_model_name = "gpt2"

    input_texts = ["What is the capital of France?", "Describe diffusion models in a sentence."]

    model, reverse_transformer = load_model(checkpoint_path, latent_dim, encoder_model_name, decoder_model_name, device)

    results = infer(model, reverse_transformer, input_texts, device)

    for i, result in enumerate(results):
        print(f"Input: {input_texts[i]}")
        print(f"Output: {result}")
        print("-" * 50)
        