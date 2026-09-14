# YuE2 native inference reference

This native implementation follows the published YuE2 checkpoint architecture,
not a private application implementation. Numerical reference:

- m-a-p/YuE2-3B, revision 29b3558dd46954a0cd9021dc76d5c91864a0f1c7
- yue2-infer 0.1.5, wheel SHA-256 8801e2c0d969db02df78d2994150b4ccd86077d87c24fdb8509b1f6f31462641
- m-a-p/YuE2-Vae, revision 9a94e1d0ea9f8087e98f77fa88df4a4068104d2a

Source: https://huggingface.co/m-a-p/YuE2-3B

The VAE architecture and SnakeBeta equations originate in stable-audio-tools
commit a6ae0cdf8b2eb1567a4b42ceadddec3712d99d45 (Copyright 2023 Stability AI,
MIT) and BigVGAN SnakeBeta (Copyright 2022 NVIDIA CORPORATION, MIT).
Their notices and full license texts are preserved alongside this file.
Weights remain external and retain their separate CC BY-NC 4.0 license.
