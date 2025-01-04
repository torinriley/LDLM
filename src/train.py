import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import LatentDiffusionModel, ReverseTransformer, TextDataset


def train(model, reverse_transformer, dataloader, optimizer, device, epochs=5, save_dir="checkpoints", log_file="training_log.json"):
    os.makedirs(save_dir, exist_ok=True)
    model.encoder.to(device)
    reverse_transformer.to(device)

    training_log = []

    for epoch in range(epochs):
        model.encoder.eval() 
        reverse_transformer.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in pbar:
            batch = batch.to(device)
            texts = [model.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch]
            latents = model.encode(texts).to(device)

            timesteps = torch.randint(0, model.num_steps, (latents.size(0),), device=device)
            noisy_latents = model.add_noise(latents, timesteps)

            timestep_embedding = reverse_transformer.create_timestep_embedding(timesteps, latents.size(-1)).to(device)
            predicted_latents = reverse_transformer(noisy_latents, timestep_embedding)

            loss = torch.mean((latents - predicted_latents) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        training_log.append({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch {epoch + 1}/{epochs} completed. Average Loss: {avg_loss:.4f}")

        torch.save(reverse_transformer.state_dict(), os.path.join(save_dir, f"reverse_transformer_epoch_{epoch + 1}.pth"))

    print("Training completed.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = 5
    batch_size = 2
    learning_rate = 1e-4
    latent_dim = 768

    model = LatentDiffusionModel(
        encoder_model_name="bert-base-uncased",
        decoder_model_name="gpt2"
    )
    reverse_transformer = ReverseTransformer(latent_dim=latent_dim)

    tokenizer = model.tokenizer
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "What is the capital of France?",
        "Latent diffusion models are fascinating!",
        "Deep learning enables powerful applications.",
        "Transformers revolutionized natural language processing."
    ]
    dataset = TextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(reverse_transformer.parameters(), lr=learning_rate)

    train(model, reverse_transformer, dataloader, optimizer, device, epochs=epochs)