import torch
import torch.nn as nn
from transformers import ElectraModel, AutoTokenizer
from safetensors.torch import load_model

# Model classes
class SpamUserClassificationLayer(nn.Module):
  def __init__(self, encoder: ElectraModel):
    super().__init__()
    self.encoder = encoder
    self.dense1 = nn.Linear(1536, 512)
    self.layernorm1 = nn.LayerNorm(512)
    self.gelu1 = nn.GELU()
    self.dropout1 = nn.Dropout(0.4)
    self.dense2 = nn.Linear(512, 256)
    self.layernorm2 = nn.LayerNorm(256)
    self.gelu2 = nn.GELU()
    self.dropout2 = nn.Dropout(0.3)
    self._init_weights()

  def _init_weights(self):
    for module in [self.dense1, self.dense2]:
      nn.init.xavier_uniform_(module.weight)
      if module.bias is not None:
        nn.init.constant_(module.bias, 0)

  def forward(self, input_ids, attention_mask=None, token_type_ids=None):
    outputs = self.encoder(
      input_ids=input_ids,
      attention_mask=attention_mask,
      token_type_ids=token_type_ids,
      output_attentions=True
    )
    cls_output = outputs.last_hidden_state[:, 0, :]
    token_embeddings = outputs.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    mean_pooled = sum_embeddings / sum_mask
    combined_output = torch.cat([cls_output, mean_pooled], dim=1)
    x = self.dense1(combined_output)
    x = self.layernorm1(x)
    x = self.gelu1(x)
    x = self.dropout1(x)
    x = self.dense2(x)
    x = self.layernorm2(x)
    x = self.gelu2(x)
    x = self.dropout2(x)
    return x

class SpamUserClassifier(nn.Module):
  def __init__(self, pretrained_model_name="beomi/kcelectra-base"):
    super().__init__()
    self.encoder = ElectraModel.from_pretrained(pretrained_model_name)
    for i, layer in enumerate(self.encoder.encoder.layer):
      if i < 2:
        for param in layer.parameters():
          param.requires_grad = False
    self.nameLayer = SpamUserClassificationLayer(self.encoder)
    self.contentLayer = SpamUserClassificationLayer(self.encoder)
    self.dense = nn.Linear(512, 256)
    self.layernorm = nn.LayerNorm(256)
    self.gelu = nn.GELU()
    self.dropout = nn.Dropout(0.3)
    self.output_layer = nn.Linear(256, 1)
    self.sigmoid = nn.Sigmoid()
    self._init_weights()

  def _init_weights(self):
    nn.init.xavier_uniform_(self.dense.weight)
    if self.dense.bias is not None:
      nn.init.constant_(self.dense.bias, 0)

  def forward(self, name_input_ids, content_input_ids, name_attention_mask=None, name_token_type_ids=None,
             content_attention_mask=None, content_token_type_ids=None, return_logits=False, return_probs=True):
    namePrediction = self.nameLayer(name_input_ids, name_attention_mask, name_token_type_ids)
    contentPrediction = self.contentLayer(content_input_ids, content_attention_mask, content_token_type_ids)
    x = self.dense(torch.cat([namePrediction, contentPrediction], dim=1))
    x = self.layernorm(x)
    x = self.gelu(x)
    x = self.dropout(x)
    logits = self.output_layer(x)
    if return_logits:
      return logits
    probs = self.sigmoid(logits)
    return probs if return_probs else (probs > 0.9).long().squeeze(-1)

# Load model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")
model = SpamUserClassifier().to(device)
load_model(model, 'model.safetensors', device=str(device))
model.eval()

print("Spam Detection Model Loaded Successfully!")
print("Enter 'quit' to exit")
print("-" * 50)

while True:
  try:
    # Get user input
    author_name = input("Enter author name: ").strip()
    if author_name.lower() == 'quit':
      break
        
    comment = input("Enter comment: ").strip()
    if comment.lower() == 'quit':
      break
    
    # Tokenize author name
    name_encoding = tokenizer(
      author_name,
      truncation=True,
      padding="max_length",
      max_length=128,
      return_tensors="pt"
    )
    name_input_ids = name_encoding["input_ids"].to(device)
    name_attention_mask = name_encoding["attention_mask"].to(device)

    # Tokenize content
    content_encoding = tokenizer(
      comment,
      truncation=True,
      padding="max_length",
      max_length=128,
      return_tensors="pt"
    )
    content_input_ids = content_encoding["input_ids"].to(device)
    content_attention_mask = content_encoding["attention_mask"].to(device)

    # Get prediction
    with torch.no_grad():
      logits = model(
        name_input_ids=name_input_ids,
        content_input_ids=content_input_ids,
        name_attention_mask=name_attention_mask,
        content_attention_mask=content_attention_mask,
        return_logits=True
      )
      
      # Get probability
      prob = torch.sigmoid(logits.squeeze(-1)).item()
      
      # Determine label
      if prob > 0.5:
        label = "Bot/Spam"
        percentage = prob * 100
      else:
        label = "Normal User"
        percentage = (1 - prob) * 100
    
    # Print results
    print(f"Label: {label}")
    print(f"Confidence: {percentage:.2f}%")
    print("-" * 50)
    
  except KeyboardInterrupt:
    print("\nExiting...")
    break
  except Exception as e:
    print(f"Error: {e}")
    print("-" * 50)

print("Goodbye!")