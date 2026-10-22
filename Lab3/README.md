# Lab 3 — Transformers in the HuggingFace Ecosystem: Sentiment Analysis with DistilBERT

> 🇫🇷 A French version of this document is available in [README_FR.md](README_FR.md).

This repository contains my solution to the third laboratory of the **Deep Learning Applications** course. The goal of the lab is to learn how to adapt a pre-trained Transformer to a new task with the HuggingFace ecosystem (`transformers`, `datasets`, `peft`). The task is **binary sentiment analysis** of movie reviews. I compare three ways of using a pre-trained **DistilBERT**, from the cheapest to the most expensive:

1. as a **frozen feature extractor** followed by a classical classifier (SVM): the *stable baseline*;
2. **full fine-tuning** of the whole network;
3. **parameter-efficient fine-tuning** with **LoRA**, which trains only about 1 % of the parameters.

---

## Table of contents

- [Repository structure](#repository-structure)
- [Requirements](#requirements)
- [Background: Transformers, the [CLS] token and fine-tuning](#background-transformers-the-cls-token-and-fine-tuning)
- [Exercise 1 — Dataset, pre-trained model and stable baseline](#exercise-1--dataset-pre-trained-model-and-stable-baseline)
- [Exercise 2 — Full fine-tuning of DistilBERT](#exercise-2--full-fine-tuning-of-distilbert)
- [Exercise 3.1 — Efficient fine-tuning with LoRA](#exercise-31--efficient-fine-tuning-with-lora)
- [Summary of results](#summary-of-results)
- [Conclusion](#conclusion)

---

## Repository structure

```
.
├── Lab3-Transformer.ipynb   # Notebook with all the code, experiments and outputs
├── README.md                # This file (English)
└── README_FR.md             # French version
```

The notebook follows the order of the exercises:

| Section | Content |
|---|---|
| Exercise 1.1 | Loading and exploring the Rotten Tomatoes dataset |
| Exercise 1.2 | Loading DistilBERT and its tokenizer, inspecting their outputs |
| Exercise 1.3 | Stable baseline: `[CLS]` features + linear SVM |
| Exercise 2.1 | Tokenizing the dataset with `Dataset.map` |
| Exercise 2.2 | Building `AutoModelForSequenceClassification` |
| Exercise 2.3 | Full fine-tuning with the HuggingFace `Trainer` |
| Exercise 3.1 | Parameter-efficient fine-tuning with LoRA (PEFT) and final comparison |

This lab contains no plots: all results are reported as **tables**.

## Requirements

```bash
pip install torch transformers datasets peft scikit-learn numpy
# optional, for experiment tracking:
pip install wandb
```

A GPU is strongly recommended for Exercises 2 and 3 (training times below were measured on a CUDA GPU with `fp16=True`).

---

## Background: Transformers, the [CLS] token and fine-tuning

**DistilBERT** is a compressed version of BERT. It was trained with *knowledge distillation*, with the full BERT-base model as a teacher, on the same text corpus. It keeps 6 Transformer layers instead of 12 (hidden size 768, ~66 M parameters), and it is about 40 % smaller and 60 % faster than BERT while keeping most of its language understanding.

**Tokenization and [CLS].** The tokenizer splits a sentence into sub-word tokens, maps them to ids, and adds two special tokens: `[CLS]` at the **beginning** and `[SEP]` at the **end**. Through self-attention, every output vector depends on the whole sentence. By convention, the output of the `[CLS]` token in the last layer is used as a **summary of the whole sentence** for classification tasks.

**Three ways to adapt a pre-trained model:**

| Approach | What is trained | Cost |
|---|---|---|
| Feature extraction | Nothing in the Transformer; only an external classifier on its frozen features | Very low |
| Full fine-tuning | All the weights of the Transformer + a new classification head | High |
| LoRA (parameter-efficient) | Small low-rank matrices added to some layers + the classification head; the original weights stay frozen | Low |

**LoRA (Low-Rank Adaptation).** Instead of updating a large weight matrix $W \in \mathbb{R}^{d \times d}$, LoRA freezes $W$ and learns a low-rank update:

$$W' = W + \frac{\alpha}{r} \, B A, \qquad A \in \mathbb{R}^{r \times d},\; B \in \mathbb{R}^{d \times r},\; r \ll d$$

With $d = 768$ and $r = 8$, a 768×768 matrix (589 824 parameters) is adapted with only 2 × 8 × 768 = 12 288 parameters. $B$ is initialized to zero, so training starts exactly from the pre-trained model.

---

## Exercise 1 — Dataset, pre-trained model and stable baseline

### 1.1 Objective: explore the dataset

**Dataset: Rotten Tomatoes** (Cornell Movie Review Data). Short sentences extracted from movie reviews, labeled **positive (1)** or **negative (0)**. It contains 5 331 positive and 5 331 negative sentences, so it is perfectly balanced.

I listed the available splits with `get_dataset_split_names` and loaded them with `load_dataset("rotten_tomatoes")`:

| Split | Number of sentences | Columns |
|---|---|---|
| train | 8 530 | `text`, `label` |
| validation | 1 066 | `text`, `label` |
| test | 1 066 | `text`, `label` |

Example (train[0]): *"the rock is destined to be the 21st century's new " conan " and that he's going to make a splash even greater than arnold schwarzenegger, jean-claud van damme or steven segal."*, label 1 (positive).

> **Important observation.** The splits are **sorted by label**: all the positive sentences come first, then all the negative ones. Any subset made of the *first* N examples therefore contains a single class. In my first version of the baseline, I used only the first 100 validation examples, which were all positive. The resulting "accuracy" only measured how often the SVM predicted "positive". The final version uses the **complete splits**.

### 1.2 Objective: load DistilBERT and its tokenizer

I loaded `distilbert-base-uncased` with `AutoTokenizer` and `AutoModel`, tokenized one sentence and passed it through the model:

| What | Result |
|---|---|
| Tokens (start … end) | `['[CLS]', 'the', 'rock', 'is', 'destined']` … `['.', '[SEP]']` |
| Output keys | `last_hidden_state` |
| `last_hidden_state` shape | `(1, 47, 768)`: (batch, number of tokens, hidden size) |
| `[CLS]` vector | `last_hidden_state[:, 0, :]`, shape `(1, 768)` |

`AutoModel` returns only the hidden states of the last layer: one 768-dimensional vector **per token**. Since the tokenizer always places `[CLS]` at position 0, the sentence representation is the vector at index 0 of the token dimension. This answers the lab's question *"how can you figure out which output is the [CLS] token?"*.

### 1.3 Objective: build a stable baseline

**Method.**
1. **Feature extraction** with a HuggingFace `pipeline("feature-extraction")` running DistilBERT on the GPU. For each sentence, the pipeline returns an array of shape `(1, n_tokens, 768)`, so the `[CLS]` vector of the last layer is `output[0][0]`. Long sentences are truncated (`truncation=True`).
2. The features of the **full** train, validation and test splits are **standardized** (`StandardScaler`, fitted on train only).
3. A **linear SVM** (`LinearSVC`) is trained on the train features. The regularization parameter C is chosen on the **validation** split, and the final model is evaluated once on the **test** split.

**Why this approach.** The Transformer is not trained at all: we only test whether its pre-trained representations already contain sentiment information. It is cheap, deterministic and easy to reproduce, which is why it is called a *stable baseline*. A linear classifier is enough to measure how *linearly separable* positive and negative sentences are in the `[CLS]` space.

**Results: choice of C (validation accuracy)**

| C | 0.001 | 0.01 | 0.1 | 1.0 |
|---|---|---|---|---|
| Val accuracy | **0.8236** | 0.8208 | 0.8189 | 0.8180 |

**Final baseline (C = 0.001)**

| Split | Accuracy |
|---|---|
| Validation | **82.36 %** |
| Test | **80.11 %** |

**Analysis.**
- Without any training of the Transformer, a simple linear classifier on the `[CLS]` vector already reaches **~80 % accuracy**. The representations learned by DistilBERT during pre-training (masked language modeling) already encode a large part of the sentiment of a sentence.
- The best C is the **smallest** value tested. Strong regularization works best, which is consistent with 768 dimensions for only 8 530 examples. A larger C overfits slightly. Since 0.001 is at the edge of the grid, an even smaller C could be tested, but the curve is very flat (less than 0.6 points between all values).
- The gap between validation (82.4 %) and test (80.1 %) is partly due to C being selected on the validation split, and partly to the small size of the splits (1 066 sentences, standard error ≈ ±1.2 points).
- The limit of this approach is that the features are **generic**: they were not optimized for sentiment. That is what fine-tuning will change.

---

## Exercise 2 — Full fine-tuning of DistilBERT

### 2.1 Objective: tokenize the dataset

I used `Dataset.map(tokenize_function, batched=True)` to tokenize all the splits efficiently. The tokenizer is called with `truncation=True, max_length=256` and **without padding**: padding is done later, batch by batch, by the data collator. This avoids padding every sentence to the length of the longest sentence in the dataset.

| Step | Columns of each example |
|---|---|
| After `map` | `text`, `label`, `input_ids`, `attention_mask` (as required by the lab) |
| After cleaning | `labels`, `input_ids`, `attention_mask` (`text` removed, `label` renamed to `labels` as expected by the `Trainer`, PyTorch format) |

### 2.2 Objective: prepare the model for classification

`AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)` loads the pre-trained DistilBERT and adds a **new, randomly-initialized classification head** on top of the `[CLS]` token:

```
[CLS] vector (768) → pre_classifier (768 → 768) → ReLU → dropout → classifier (768 → 2) → logits
```

The warning *"Some weights … are newly initialized: classifier.*, pre_classifier.*"* confirms that only the head is new. Everything else comes from the pre-trained checkpoint.

### 2.3 Objective: fine-tune with the HuggingFace `Trainer`

**Setup.**

| Component | Choice |
|---|---|
| Data collator | `DataCollatorWithPadding`: pads each batch to the length of its longest sentence |
| Metrics (`compute_metrics`) | accuracy and F1 score, computed with scikit-learn from `(logits, labels)` |
| Optimizer | AdamW (Trainer default), learning rate **2e-5**, weight decay 0.01 |
| Batch size | 16 (train and eval) |
| Epochs | 3 |
| Precision | `fp16=True` (mixed precision) |
| Evaluation / saving | at the end of each epoch; the **best checkpoint according to validation accuracy** is reloaded at the end |
| Trained parameters | **all** of them: 66 955 010 |

A small learning rate (2e-5) is standard for fine-tuning BERT-like models. The pre-trained weights are already good, and large updates would destroy what was learned during pre-training ("catastrophic forgetting").

**Results**

| Method | Trained parameters | Training time | Val accuracy | Test accuracy |
|---|---|---|---|---|
| SVM on `[CLS]` (baseline) | — | — | 82.36 % | 80.11 % |
| **Full fine-tuning** | 66 955 010 | 5.6 min | **86.21 %** | **84.05 %** |

**Analysis.**
- Fine-tuning improves the test accuracy by **+3.9 points** (80.1 % → 84.1 %) and the validation accuracy by +3.9 points. Instead of using generic features, all the layers of the Transformer are adapted to the sentiment task: the `[CLS]` representation becomes specialized.
- The price is that **all 67 M parameters** are updated. This requires storing gradients and optimizer states for every weight, and a full copy of the model must be saved for each new task.
- Validation accuracy is higher than test accuracy because the best checkpoint was **selected** on the validation split, so the validation score is slightly optimistic. The test score is the unbiased estimate.

---

## Exercise 3.1 — Efficient fine-tuning with LoRA

### Objective

Fine-tune DistilBERT on Rotten Tomatoes in a more *efficient* way than full fine-tuning, and compare the cost/performance trade-off.

### Method

I used **LoRA** from the HuggingFace **PEFT** library:

| Hyperparameter | Value | Role |
|---|---|---|
| `r` | 8 | rank of the update matrices $A$ and $B$ |
| `lora_alpha` | 16 | scaling factor (the update is multiplied by α / r = 2) |
| `target_modules` | `q_lin`, `v_lin` | LoRA is applied to the **query** and **value** projections of the attention in each of the 6 layers |
| `lora_dropout` | 0.1 | dropout on the LoRA branch |
| `modules_to_save` | `pre_classifier`, `classifier` | the new classification head is fully trained |
| Learning rate | **5e-4** | higher than for full fine-tuning, because far fewer parameters are trained |

The training pipeline (`Trainer`, data collator, `compute_metrics`, 3 epochs, batch 16, fp16, best checkpoint on validation accuracy) is **identical** to Exercise 2, so the comparison is fair.

**Implementation details that matter:**
- LoRA is applied to a **fresh** `distilbert-base-uncased` model. In my first version, it was applied to the model already fully fine-tuned in Exercise 2, which made the comparison meaningless.
- By default, with `task_type="SEQ_CLS"`, PEFT only trains the module named `classifier`. DistilBERT's head also contains a randomly-initialized `pre_classifier` layer, which would have stayed **frozen at random values**. Adding it to `modules_to_save` fixes this.
- `label_names=["labels"]` is set explicitly, because PEFT wraps the model and hides the arguments of its `forward`.

### Where the trainable parameters are

| Component | Trainable parameters |
|---|---|
| LoRA adapters (6 layers × 2 matrices (`q_lin`, `v_lin`) × (8×768 + 768×8)) | 147 456 |
| `pre_classifier` (768×768 + 768) | 590 592 |
| `classifier` (768×2 + 2) | 1 538 |
| **Total** | **739 586** (1.09 % of the model) |

PEFT reports `all params: 67 694 596` instead of 66 955 010, because it keeps a frozen copy of the original head next to the trainable copy (`modules_to_save`) and adds the LoRA matrices.

### Results

| Method | Trained parameters | % of full model | Training time | Val accuracy | Test accuracy |
|---|---|---|---|---|---|
| SVM on `[CLS]` (baseline) | — | — | — | 82.36 % | 80.11 % |
| Full fine-tuning | 66 955 010 | 100 % | 5.6 min | **86.21 %** | 84.05 % |
| **LoRA (r = 8)** | 739 586 | **1.10 %** | **1.9 min** | 83.77 % | **84.43 %** |

### Analysis

- **Same performance for a fraction of the cost.** LoRA reaches the same test accuracy as full fine-tuning (84.4 % vs 84.1 %) while training only **1.1 %** of the parameters, and training is **~3× faster** (1.9 min vs 5.6 min). Fewer gradients and optimizer states are needed, which also reduces GPU memory.
- **The adapters themselves are tiny.** Most of LoRA's trainable parameters (80 %) belong to the classification head, which must be trained in any case. The LoRA adapters themselves are only **147 456 parameters (0.22 % of the model)**. To adapt DistilBERT to a new task, it is enough to store these adapters and the head (a few MB) instead of a full 67 M-parameter copy of the model.
- **Why it works.** The weight updates needed to adapt a pre-trained model to a specific task have a low "intrinsic rank": a small number of directions in parameter space is enough. Adapting the query and value projections lets the model change *what it pays attention to* for the sentiment task, without modifying the rest of the network.
- **The differences between LoRA and full fine-tuning are not significant.** LoRA is 2.4 points lower in validation but 0.4 points higher in test. With 1 066 sentences per split, the standard error is about ±1.1 points, and each method was run with a single seed. The honest conclusion is that both methods perform **equally well** on this task.
- **Another efficient approach,** suggested by the lab, which does not require changing the training pipeline: **freeze part of the network**, for example the embeddings and the first Transformer layers, and fine-tune only the last layers and the head. This reduces the number of trained parameters and the cost of backpropagation, at the price of less flexibility.

---

## Summary of results

| Method | What is trained | Trained parameters | Time | Val acc | Test acc |
|---|---|---|---|---|---|
| `[CLS]` features + linear SVM | external classifier only | — | — | 82.36 % | 80.11 % |
| Full fine-tuning | the whole network | 66 955 010 (100 %) | 5.6 min | 86.21 % | 84.05 % |
| LoRA (r = 8) | adapters + head | 739 586 (1.1 %) | 1.9 min | 83.77 % | 84.43 % |

---

## Conclusion

- **Pre-trained representations are already strong.** A linear SVM on DistilBERT's frozen `[CLS]` vectors reaches 80 % test accuracy without training the Transformer. This makes it a solid and cheap *stable baseline*. Understanding the model's outputs was essential: the `[CLS]` token is at position 0 of the last hidden state. The way the dataset is organized (sorted by label) can completely distort an evaluation if one only uses its first examples.
- **Fine-tuning brings a clear gain.** Adapting the whole network to the task improves test accuracy by about 4 points (84 %). The HuggingFace `Trainer`, together with `Dataset.map` for tokenization and `DataCollatorWithPadding` for dynamic padding, makes this pipeline short and reusable.
- **LoRA gives the best trade-off.** With only 1.1 % of the parameters trained (0.22 % for the adapters themselves) and a 3× shorter training time, LoRA matches the accuracy of full fine-tuning. Parameter-efficient methods make it possible to adapt large pre-trained models with limited resources, and to store one small adapter per task instead of one full model per task.

**Limitations and possible extensions:** each configuration was run with a single seed, and the splits are small, so differences of about 1 point are not significant. Several seeds, a wider hyperparameter search (rank `r`, target modules, learning rate), or a comparison with partial layer freezing would make the comparison more robust. Exercises 3.2 (CLIP) and 3.3 were not addressed.
