# Latent Diffusion Model for Language Modeling

## Overview
This implementation introduces a Latent Diffusion Model (LDM) for language generation tasks. The model leverages latent space encoding via a pretrained transformer-based encoder and reconstructs text using a pretrained GPT-2 decoder. A reverse diffusion process, created by a custom transformer, refines noisy latent representations. The approach combines diffusion probabilistic models with the expressive power of large language models (LLMs).

---

## Model Architecture

### 1. **Latent Diffusion Model**
The `LatentDiffusionModel` serves as the core of the pipeline, featuring:
- **Encoder**: A pretrained transformer (e.g., BERT) to encode input text into latent representations.
- **Decoder**: A GPT-2 model fine-tuned to decode latent embeddings back into text.
- **Latent Space Diffusion**: Gaussian noise is added to the latent space, and the reverse process removes the noise using a transformer.

#### Key Methods:
- **`encode(input_texts)`**: 
  Encodes textual inputs into latent representations. The mean of the last hidden states of the encoder is used as the latent vector.
  
- **`decode(latent_embeddings)`**:
  Decodes the latent space embeddings into textual output using the GPT-2 decoder. Utilizes beam search for more coherent text generation.

- **`add_noise(latent, timesteps)`**:
  Perturbs the latent embeddings by adding Gaussian noise scaled by the diffusion schedule.

#### Diffusion Hyperparameters:
- **Timesteps**: 1000 (default)
- **Noise Schedule**: Linear from $\beta_{\text{start}} = 0.00085$ to $\beta_{\text{end}} = 0.012$
- **Cumulative Alpha**: Maintains the product of $1 - \beta_t$ over timesteps for smooth diffusion.

---

### 2. **Reverse Transformer**
A **Transformer-based reverse diffusion module** is used to denoise the perturbed latent embeddings.

#### Components:
- **Latent Dimensionality**: $d_{\text{latent}} = 768$
- **Timestep Embedding**: 
  - Sinusoidal positional embeddings (analogous to positional encodings).
  - Supports continuous timesteps for diffusion-based denoising.

- **Transformer Layers**: 
  - 4 encoder layers with 8 attention heads.
  - Latent embedding concatenated with timestep embedding and projected back to latent space dimensionality.

#### Key Method:
- **`forward(noisy_latent, timestep_embedding)`**:
  Processes the noisy latent representations and timestep embeddings through a transformer stack. Outputs the denoised latent embeddings.

---

## Training Procedure

### Loss Function
The model minimizes the mean squared error (MSE) between the original latent representations and the denoised latents:
$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \| z_i - \hat{z}_i \|^2
$$
Where $z_i$ is the original latent, and $\hat{z}_i$ is the predicted latent after reverse diffusion.

### Training Steps
1. **Encode** input texts into latent space using the encoder.
2. **Add Noise** to latent embeddings based on the diffusion schedule.
3. **Denoise** using the reverse transformer.
4. Backpropagate the MSE loss and update parameters.

### Optimizer
Adam optimizer with a learning rate of $10^{-4}$ is used.

---

## Inference Procedure

### Steps:
1. Encode input text into latent space.
2. Initialize random Gaussian noise in the latent space.
3. Iteratively denoise using the reverse transformer for $T = 1000$ steps.
4. Decode the final latent representations into text using the GPT-2 decoder.

### Notes:
- Beam search ensures diverse and coherent text generation during decoding.
- Ensure proper handling of padding and attention masks for reliable outputs.
