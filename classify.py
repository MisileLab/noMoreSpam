import marimo

__generated_with = "0.14.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import ElectraModel, AutoTokenizer
    from torch.utils.data import DataLoader, Dataset
    from torch.optim import AdamW
    import altair as alt
    import polars as pl

    class FocalLoss(nn.Module):
        """Focal Loss for handling class imbalance - supports both binary and multi-class"""
        def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction

        def forward(self, inputs, targets):
            # Handle binary classification (single output) vs multi-class
            if inputs.dim() == 2 and inputs.size(1) == 1:
                # Binary classification: inputs shape [batch_size, 1]
                inputs = inputs.squeeze(-1)  # [batch_size]
                bce_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), reduction='none')
                pt = torch.exp(-bce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
            else:
                # Multi-class classification: inputs shape [batch_size, num_classes]
                ce_loss = F.cross_entropy(inputs, targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss

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

        def get_attention_weights(self, input_ids, attention_mask=None, token_type_ids=None):
            with torch.no_grad():
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    output_attentions=True
                )
                return outputs.attentions[-1]

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
            else:
                probs = self.sigmoid(logits)
                return probs if return_probs else (probs > 0.9).long().squeeze(-1)
    return AutoTokenizer, DataLoader, Dataset, SpamUserClassifier, pl, torch


@app.cell
def _(pl):
    comments = pl.read_avro("comments.avro")
    return (comments,)


@app.cell
def _(AutoTokenizer, Dataset):
    tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base")

    # dataset wrapper
    class YTBotDataset(Dataset):
        def __init__(self, ds, tokenizer, max_length=128):
            self.author_names = ds["author_name"].to_list()
            self.contents = ds["content"].to_list()
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.contents)

        def __getitem__(self, idx):
            author_name = self.author_names[idx]
            content = self.contents[idx]

            # Tokenize author name
            name_encoding = self.tokenizer(
                author_name,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )
            # Tokenize comment content
            content_encoding = self.tokenizer(
                content,
                truncation=True,
                padding="max_length",
                max_length=self.max_length,
                return_tensors="pt",
            )

            item = {}
            for k, v in name_encoding.items():
                item[f"name_{k}"] = v.squeeze(0)
            for k, v in content_encoding.items():
                item[f"content_{k}"] = v.squeeze(0)
            return item
    return YTBotDataset, tokenizer


@app.cell
def _(
    DataLoader,
    SpamUserClassifier,
    YTBotDataset,
    comments,
    tokenizer,
    torch,
):
    data_loader = DataLoader(YTBotDataset(comments, tokenizer), batch_size=148)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpamUserClassifier().to(device)
    return data_loader, device, model


@app.cell
def _(comments, data_loader, device, mo, model, pl, torch):
    # Evaluate model and save predictions
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in mo.status.progress_bar(data_loader, show_eta=True, show_rate=True):
            # Move batch to device for all dual inputs
            name_input_ids = batch["name_input_ids"].to(device)
            name_attention_mask = batch["name_attention_mask"].to(device)
            name_token_type_ids = batch.get("name_token_type_ids", None)
            if name_token_type_ids is not None:
                name_token_type_ids = name_token_type_ids.to(device)
            content_input_ids = batch["content_input_ids"].to(device)
            content_attention_mask = batch["content_attention_mask"].to(device)
            content_token_type_ids = batch.get("content_token_type_ids", None)
            if content_token_type_ids is not None:
                content_token_type_ids = content_token_type_ids.to(device)

            # Get predictions (as probabilities or binary class)
            batch_predictions = model(
                name_input_ids=name_input_ids,
                content_input_ids=content_input_ids,
                name_attention_mask=name_attention_mask,
                name_token_type_ids=name_token_type_ids,
                content_attention_mask=content_attention_mask,
                content_token_type_ids=content_token_type_ids,
                return_logits=False,
                return_probs=False  # Use thresholding as in training
            )
            predictions.extend(batch_predictions.cpu().numpy().tolist())

    # Add predictions to the original dataset
    processed_comments = comments.with_columns(
        pl.Series("is_bot_comment", predictions, dtype=pl.Int32)
    )

    # Save to processed.avro
    processed_comments.write_avro("processed.avro")

    print(f"Processed {len(predictions)} comments")
    print(f"Bot comments detected: {sum(predictions)}")
    print(f"Regular comments: {len(predictions) - sum(predictions)}")
    print("Results saved to processed.avro")
    return


if __name__ == "__main__":
    app.run()
