<p align="middle">
  <img height="40mm" src="https://dataloop.ai/wp-content/uploads/2020/03/logo.svg">
</p>

## Text Embeddings with Feed-Forward Neural-Net Language Models

### Key Features

- **Model Types**:
  - Text is mapped to embedding vectors of either 50-dimensional or 128-dimensional.
- **Current Implementation**:
  - The model is currently implemented as a Model Adapter with a Predict function.
- **Upcoming Changes**:
  - The functionality will soon migrate to generate a feature set function.

### Feature Set Naming Convention

The feature set name is generated from the model name combined with the string `feature-set`. For example:

- For the model `nlm-en-dim128`, the feature set name will be `nlm-en-dim128-feature-set`.

### Model Integration

This repo is a model integration between [Google NNLM text embeddings](https://www.kaggle.com/models/google/nnlm/tensorFlow1/en-dim128/1?tfhub-redirect=true) and [Dataloop](https://dataloop.ai/).

## Models

| Language   | 50 dimensions                                     | 128 dimensions                                      |
| ---------- | ------------------------------------------------- | --------------------------------------------------- |
| Chinese    | nnlm-zh-dim50<br>nnlm-zh-dim50-with-normalization | nnlm-zh-dim128<br>nnlm-zh-dim128-with-normalization |
| English    | nnlm-en-dim50<br>nnlm-en-dim50-with-normalization | nnlm-en-dim128<br>nnlm-en-dim128-with-normalization |
| German     | nnlm-de-dim50<br>nnlm-de-dim50-with-normalization | nnlm-de-dim128<br>nnlm-de-dim128-with-normalization |
| Indonesian | nnlm-id-dim50<br>nnlm-id-dim50-with-normalization | nnlm-id-dim128<br>nnlm-id-dim128-with-normalization |
| Japanese   | nnlm-ja-dim50<br>nnlm-ja-dim50-with-normalization | nnlm-ja-dim128<br>nnlm-ja-dim128-with-normalization |
| Korean     | nnlm-ko-dim50<br>nnlm-ko-dim50-with-normalization | nnlm-ko-dim128<br>nnlm-ko-dim128-with-normalization |
| Spanish    | nnlm-es-dim50<br>nnlm-es-dim50-with-normalization | nnlm-es-dim128<br>nnlm-es-dim128-with-normalization |
