# Tiny Black-and-White Generative AI

A deliberately minimal text-to-image generative AI project for teaching beginners.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dujing82-blip/tiny-black-white-generative-ai/blob/main/Tiny_Black_White_Generative_AI.ipynb)

## The entire idea

```text
WORD + RANDOM z  →  DECODER  →  16×16 BLACK-AND-WHITE IMAGE
```

The model knows only five words:

**heart · face · robot · tree · spaceship**

There are no colors and no sizes. This keeps the code and the conceptual story as direct as possible.


## Version 2: CNN-VAE for better generation

Students who already know CNNs can use the improved CNN-VAE notebook:

[![Open V2 In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dujing82-blip/tiny-black-white-generative-ai/blob/main/V2_CNN_VAE_Better_Generation.ipynb)

V2 keeps the same VAE concept but uses a convolutional encoder and decoder, 10,000 training images, an 8-dimensional latent space, Binary Crossentropy reconstruction loss, and a smaller KL weight for more recognizable pixel art.

## Why this version exists

This repository is designed for a first hands-on lesson in generative AI. It deliberately removes most of the complexity found in real text-to-image systems so students can see three ideas clearly:

1. **Embedding** — turn a word into numbers.
2. **Latent variable z** — introduce variation/randomness.
3. **Decoder** — turn those numbers into pixels.

## Dataset

Both notebooks download the pre-generated dataset [`data/icons_16x16.npz`](data/icons_16x16.npz) directly from GitHub. **No dataset generator runs during a Colab lesson.** The download is about 47 KB and is reused when its checksum matches.

The fixed dataset contains **10,000** 16×16 black-and-white images:

- 2,000 hearts
- 2,000 faces
- 2,000 robots
- 2,000 trees
- 2,000 spaceships

Examples within each class vary in position and shape. The data is identical to the original CNN notebook's `generate_dataset.py` output, with the same deterministic seeds and label order.

The archive contains `images` (uint8, shape `(10000, 16, 16)`, pixels 0 or 255), `labels` (int32, shape `(10000,)`), and `words` (the five condition words in label order). The notebooks normalize pixels to 0–1; the CNN notebook adds one channel dimension.

Dataset provenance and SHA256 are recorded in [`data/manifest.json`](data/manifest.json). For maintainers only, `python build_precomputed_dataset.py` rebuilds the archive using NumPy and Pillow. Students simply run the download and training cells. If the archive changes, update the checksum in both notebooks.

## Model

The notebook uses a tiny **conditional variational autoencoder (VAE)**.

During training:

```text
training image ────────┐
                       ├── ENCODER ──> latent z
word ──> embedding ────┘

latent z ──────────────┐
                       ├── DECODER ──> reconstructed image
word ──> embedding ────┘
```

During generation, the encoder is no longer needed:

```text
random z ──────────────┐
                       ├── DECODER ──> NEW IMAGE
word ──> embedding ────┘
```

This makes the central generative idea very easy to demonstrate: keep the word fixed, sample a new random `z`, and obtain another variation of the requested object.

## Start the lesson

Click **Open in Colab** at the top of this page and run the cells from top to bottom.

The notebook contains detailed comments intended to be read by students. No pretrained model is used.

## Files

```text
tiny-black-white-generative-ai/
├── README.md
├── LICENSE
├── data/
│   ├── icons_16x16.npz
│   └── manifest.json
├── build_precomputed_dataset.py
├── generate_dataset.py
├── generate_dataset_v2.py
├── Tiny_Black_White_Generative_AI.ipynb
└── V2_CNN_VAE_Better_Generation.ipynb
```

## Important limitation

This is not a language model. It recognizes only five predefined words.

That limitation is intentional. The goal is to make the basic mechanism visible before introducing tokenization, Transformers, diffusion models, or large text encoders.

## License

MIT License. Code and the procedurally generated teaching data may be reused for educational purposes.

