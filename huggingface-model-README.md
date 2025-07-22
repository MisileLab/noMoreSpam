---
license: mit
datasets:
- MisileLab/youtube-bot-comments-v2
language:
- ko
pipeline_tag: text-classification
tags:
- pytorch
- youtube
- spam
- detection
base_model:
- beomi/KcELECTRA-base
---
# Model Card for MisileLab/noMoreSpam

<!-- Provide a quick summary of what the model is/does. -->

A transformer-based model for detecting bot-generated spam comments on YouTube, with a focus on Korean content promoting adult content and gambling websites.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

noMoreSpam is a fine-tuned KcELECTRA model designed to identify and filter bot comments on YouTube videos. It specifically targets automated comments that promote adult content or gambling websites using repetitive patterns and specific keywords in Korean. The model uses a combination of CLS token and mean pooling strategies with custom classification layers to achieve high accuracy in distinguishing between human and bot-generated content.

- **Developed by:** MisileLab
- **Model type:** Fine-tuned KcELECTRA for sequence classification
- **Language(s) (NLP):** Korean (ko)
- **License:** MIT
- **Finetuned from model:** KcELECTRA

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/misilelab/noMoreSpam
- **Train code & result:** https://static.marimo.app/static/nomorespam-zvfn

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

This model is suitable for:
- Detecting spam bot comments in Korean YouTube content
- Filtering promotional comments for adult content and gambling websites
- Content moderation systems for Korean social media platforms
- Research on automated spam detection in Korean text

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

The model can be integrated into:
- YouTube comment moderation systems
- Content filtering pipelines for Korean platforms
- Research frameworks studying bot behavior and spam patterns
- Social media monitoring tools

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

This model should not be used for:
- General text classification tasks unrelated to spam detection
- Detection of sophisticated bots beyond the patterns it was trained on
- Applications requiring high precision in non-Korean languages
- Making decisions about content without human review
- Censorship of legitimate speech or opinions

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

- **Pattern dependency:** The model relies on specific keywords and patterns that may become outdated
- **Language specificity:** Optimized for Korean language and may not work well for other languages
- **Bot type limitation:** Focuses specifically on adult/gambling promotion bots, not all spam types
- **Temporal relevance:** Bot patterns evolve over time, potentially reducing long-term effectiveness
- **False positives:** Legitimate comments containing flagged keywords may be misclassified
- **Domain specificity:** Trained on YouTube comments which may not transfer well to other platforms

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should:
- Regularly update the model as spam techniques evolve
- Use in combination with other detection methods for robust spam filtering
- Consider both precision and recall when evaluating performance
- Ensure human review of flagged content before taking action
- Monitor for evolving bot patterns and retrain the model periodically
- Be aware of potential biases in the training data

## How to Get Started with the Model

Use the code below to get started with the model.

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("MisileLab/noMoreSpam")
model = AutoModelForSequenceClassification.from_pretrained("MisileLab/noMoreSpam")

# Prepare input
comment = "여기 방문하세요 19금 즐거움이 가득합니다"  # Example spam comment
inputs = tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=512)

# Make prediction
with torch.no_grad():
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
is_bot = predictions[0][1].item() > 0.5
probability = predictions[0][1].item()

print(f"Is bot comment: {is_bot}, Probability: {probability:.4f}")
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model was trained on the [youtube-bot-comments-v2](https://huggingface.co/datasets/MisileLab/youtube-bot-comments-v2) dataset, which contains:
- 50% human comments, 50% bot comments (balanced dataset)
- Comments collected from top South Korean YouTube videos
- Manual and regex-based classification
- Focus on identifying repetitive promotional patterns for adult and gambling websites

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing [optional]

- Comments were tokenized using the KcELECTRA tokenizer
- Texts were truncated to a maximum length of 512 tokens
- Data was split into training (80%) and validation (20%) sets
- Special tokens were added for classification

#### Training Hyperparameters

- **Training regime:** fp16 mixed precision
- **Optimizer:** AdamW
- **Learning rate:** 2e-5
- **Batch size:** 16
- **Epochs:** 3
- **Loss function:** Focal Loss (to handle class imbalance)
- **Early stopping:** Based on validation F1 score
- **Weight decay:** 0.01
- **Warmup steps:** 500

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The model was evaluated on a held-out test set (20%) from the [youtube-bot-comments-v2](https://huggingface.co/datasets/MisileLab/youtube-bot-comments-v2) dataset.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

- **Precision:** Measures the proportion of predicted bot comments that are actually bots
- **Recall:** Measures the proportion of actual bot comments that were correctly identified
- **F1 Score:** Harmonic mean of precision and recall
- **Accuracy:** Overall proportion of correct predictions

### Results

#### Summary

- **Precision:** 1
- **Recall:** 1
- **F1 Score:** 1
- **Accuracy:** 1

The model performs well on Korean YouTube comments, particularly for detecting common spam patterns promoting adult content and gambling websites.

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

Attention visualization shows the model focuses heavily on specific Korean keywords and patterns associated with spam content, such as "19금" (adult content indicator), gambling-related terms, and URL patterns.

## Technical Specifications [optional]

### Model Architecture and Objective

The model architecture includes:
- Base KcELECTRA transformer with frozen initial layers
- Custom classification head with:
  - Dropout layers (rate=0.1) for regularization
  - Combined CLS token and mean pooling strategy
  - Two fully connected layers with GELU activation
  - Binary classification output with sigmoid activation

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

```bibtex
@misc{misile2025nomorespam,
  title={noMoreSpam: Korean YouTube Bot Comment Detection Model},
  author={MisileLab},
  year={2025},
  howpublished={\url{https://huggingface.co/MisileLab/noMoreSpamYT}}
}
```

**APA:**

MisileLab. (2025). noMoreSpam: Korean YouTube Bot Comment Detection Model. https://huggingface.co/MisileLab/noMoreSpamYT

## Glossary [optional]

- **KcELECTRA:** Korean-centric ELECTRA model, a transformer-based model pre-trained on Korean text
- **Bot comment:** Automated comment typically promoting adult content or gambling websites
- **Focal Loss:** A loss function that addresses class imbalance by focusing on hard examples
- **CLS token:** Special classification token used in transformer models for sequence classification

## More Information [optional]

For more information about the project and its development, visit:
- [Project GitHub Repository](https://github.com/misilelab/noMoreSpam)
- [Marimo Notebook Demo](https://static.marimo.app/static/nomorespam-zvfn)

## Model Card Contact

For questions or issues regarding this model, please contact Misile (misile@duck.com). 