# Working with Transformers in the HuggingFace Ecosystem

## Description
This notebook constitutes the thirt laboratory of the "**Deep Learning Applications**" course. The objective is to use the HuggingFace ecosystem to adapt Transformer models (specifically DistilBERT) for sentiment analysis tasks.

## Exercise 1: Sentiment Analysis
The goal was to establish a stable baseline by using a pre-trained DistilBERT model solely as a feature extractor, without modifying its internal weights.

**Implementation**:
- Dataset: We used the Cornell Rotten Tomatoes movie review dataset, which contains 5,331 positive and 5,331 negative reviews.
- Splits: The data was divided into `train` (8,530 samples), `validation` (1,066 samples), and `test` (1,066 samples).
- Feature Extraction: Using the HuggingFace `pipeline`, we extracted the representation of the `[CLS]` token from the last transformer layer for each sentence. This resulted in a vector of dimension 768 representing the entire phrase.
- Classification: A Scikit-learn SVM (Support Vector Machine) with a linear kernel was trained on these extracted features.

**Results**:
- Validation Accuracy: `0.770`.
- Test Accuracy: `0.765`.

These results serve as our starting point. While DistilBERT has a strong general understanding of language, the features are "static" because the model was not specifically adapted to the nuances of movie review vocabulary.

## Exercise 2: Fine-tuning DistilBERT
To improve performance by re-training the entire DistilBERT model—including both the base layers and a new classification "head"—specifically for our dataset.

**Implementation**:
- Preprocessing: Texts were tokenized using `Dataset.map` with padding and truncation to ensure uniform input lengths.
- Model: We used `AutoModelForSequenceClassification`, which automatically appends a linear classification layer on top of the transformer.
- Training: We utilized the `Trainer` API, configuring the process for 3 epochs with a learning rate of 2x10<sup>-5</sup> and a weight decay of 0.01.

**Results**:

|Epoch | Training Loss | Validation Loss | Accuracy |	F1|
|:----:|:-------------:|:---------------:|:--------:|:-:|
|1     |	0.415600|	0.416507|	0.818011|	0.836700|
|2     |	0.254500|	0.411208|	0.841463|	0.845943|
|3     |	0.164300|	0.493393|	0.848968|	0.851064|

Results on test:
|loss| 0.471|
|accuracy| 0.839|
|f1| 0.844|

Full fine-tuning resulted in a significant performance boost (approximately 7% higher accuracy than the baseline) as the model learned to recognize sentiment-specific patterns in the text.

## Exercise 3.1: Efficient Fine-tuning (LoRA)
To reduce the computational cost of adaptation by using LoRA (Low-Rank Adaptation) via the PEFT library. This method updates only a small set of added parameters rather than the entire model.

**Implementation**:
- LoRA Configuration: We targeted the `q_lin` (query) and `v_lin` (value) modules with a rank $r=8$ and a scaling factor $\alpha=16$.
- Optimization: We enabled `fp16` (half-precision) to accelerate training and reduce memory usage.
- Training: The same training parameters from Exercise 2 were used, but only a fraction of the total parameters were updated.

**Results**:
|Epoch | Training Loss | Validation Loss | Accuracy |	F1|
|:----:|:-------------:|:---------------:|:--------:|:-:|
|1     |	0.142100|	0.441362|	0.845216|	0.846797|
|2     |	0.139300|	0.434065|	0.848968|	0.849110|
|3     |	0.142800|	0.437866|	0.849906|	0.849906|

Results on test:
|loss| 0.50|
|eval_accuracy| 0.85|
|eval_f1| 0.851|

- Test Accuracy: `0.843`.
- Efficiency: Training was significantly faster, requiring approximately 2 minutes for 3 epochs compared to the longer duration required for full fine-tuning.

LoRA achieved performance comparable to (and in this specific run, slightly better than) full fine-tuning. This demonstrates that parameter-efficient methods can achieve state-of-the-art results with much lower resource requirements.
