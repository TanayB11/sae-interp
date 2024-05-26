# sae-interp

I'm attempting to reproduce some results of Anthropic's [Towards Monosemanticity: Decomposing Language Models With Dictionary Learning](https://transformer-circuits.pub/2023/monosemantic-features/index.html#appendix-autoencoder-dataset) and [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html).

The goal is to train a sparse autoencoder (SAE) on the (2nd layer) MLP activations of a 2-layer model trained on [TinyStories](https://arxiv.org/pdf/2305.07759). Libraries such as [SAELens](https://github.com/jbloomAus/SAELens/) are useful for training SAEs for mechanistic interpretability, but I want to try an implementation from scratch in (mostly) plain PyTorch. If I get this to work, I'll be able to find some interpretable features in the model!

More info to come, currently *very* much a work in progress.