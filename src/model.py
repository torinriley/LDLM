import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import GPT2LMHeadModel, AutoModel, AutoTokenizer
import math


class LatentDiffusionModel:
    def __init__(self, encoder_model_name, decoder_model_name, num_steps=1000, beta_start=0.00085, beta_end=0.012):
        self.encoder = AutoModel.from_pretrained(encoder_model_name)
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)
        self.vocab_projection = nn.Linear(self.encoder.config.hidden_size, self.tokenizer.vocab_size)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if self.tokenizer.unk_token is None:
            self.tokenizer.add_special_tokens({'unk_token': '[UNK]'})

        self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.decoder.resize_token_embeddings(len(self.tokenizer))

        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_steps, dtype=torch.float32) ** 2
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def encode(self, input_texts):
        inputs = self.tokenizer(
            input_texts, return_tensors="pt", padding=True, truncation=True, max_length=128
        )
        print(f"Input IDs: {inputs['input_ids']}")
        print(f"Tokenizer Vocabulary Size: {len(self.tokenizer)}")

        inputs['input_ids'] = torch.clamp(inputs['input_ids'], max=len(self.tokenizer) - 1)

        with torch.no_grad():
            outputs = self.encoder(**inputs)
            latent = outputs.last_hidden_state.mean(dim=1)  # Batch size x latent_dim
        return latent

    def decode(self, latent_embeddings, num_beams=5, max_new_tokens=50):
        latent_embeddings = latent_embeddings.unsqueeze(1)  # Batch size x 1 x latent_dim

        outputs = self.decoder.generate(
            inputs_embeds=latent_embeddings,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            attention_mask=torch.ones(latent_embeddings.size()[:2], dtype=torch.long, device=latent_embeddings.device)
        )

        decoded_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outputs]
        return decoded_texts

    def add_noise(self, latent, timesteps):
        sqrt_alpha_prod = self.alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - self.alphas_cumprod[timesteps]) ** 0.5
        noise = torch.randn_like(latent)
        return sqrt_alpha_prod.unsqueeze(-1) * latent + sqrt_one_minus_alpha_prod.unsqueeze(-1) * noise


class ReverseTransformer(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.transformer = nn.Transformer(latent_dim, nhead=8, num_encoder_layers=4)
        self.projection = nn.Linear(latent_dim * 2, latent_dim)

    def create_timestep_embedding(self, timesteps, embedding_dim):
        half_dim = embedding_dim // 2
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / half_dim))
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb

    def forward(self, noisy_latent, timestep_embedding):
        noisy_latent = noisy_latent.unsqueeze(1)  # Batch size x 1 x latent_dim
        timestep_embedding = timestep_embedding.unsqueeze(1).expand(-1, noisy_latent.size(1), -1)  # Batch size x 1 x embedding_dim
        inputs = torch.cat([noisy_latent, timestep_embedding], dim=2)  # Batch size x 1 x (latent_dim + embedding_dim)
        inputs = self.projection(inputs)
        inputs = inputs.permute(1, 0, 2)  # 1 x Batch size x latent_dim
        output = self.transformer(inputs, inputs)
        output = output.permute(1, 0, 2)  # Batch size x 1 x latent_dim
        return output[:, 0, :]  # Batch size x latent_dim


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length)
        return encoded.input_ids.squeeze(0)  # Tensor of shape (max_length,)
