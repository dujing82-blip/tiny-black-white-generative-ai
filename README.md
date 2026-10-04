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

## Why this version exists

This repository is designed for a first hands-on lesson in generative AI. It deliberately removes most of the complexity found in real text-to-image systems so students can see three ideas clearly:

1. **Embedding** — turn a word into numbers.
2. **Latent variable z** — introduce variation/randomness.
3. **Decoder** — turn those numbers into pixels.

## Dataset

The included generator creates **5,000** 16×16 grayscale images:

- 1,000 hearts
- 1,000 faces
- 1,000 robots
- 1,000 trees
- 1,000 spaceships

Although the images are black and white, examples within each class vary in position and shape. For example, robot dimensions and antenna length vary; faces vary in radius and eye spacing; trees vary in canopy width; and spaceships vary in body and flame geometry.

The dataset is generated automatically inside Colab. Students do not need to download or upload data.

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
├── .gitignore
├── generate_dataset.py
└── Tiny_Black_White_Generative_AI.ipynb
```

## Important limitation

This is not a language model. It recognizes only five predefined words.

That limitation is intentional. The goal is to make the basic mechanism visible before introducing tokenization, Transformers, diffusion models, or large text encoders.

## License

MIT License. Code and the procedurally generated teaching data may be reused for educational purposes.
