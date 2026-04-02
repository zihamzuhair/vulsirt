import torch
import torch.nn as nn
from transformers import AutoModel


def first_token_features(output):
    return output.last_hidden_state[:, 0, :]


class SourceOnlyModel(nn.Module):
    def __init__(self, model_name, dropout):
        super().__init__()
        self.source_encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.source_encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, source_input_ids, source_attention_mask, **kwargs):
        output = self.source_encoder(input_ids=source_input_ids, attention_mask=source_attention_mask)
        features = first_token_features(output)
        logits = self.classifier(self.dropout(features)).squeeze(-1)
        return {"logits": logits}


class IROnlyModel(nn.Module):
    def __init__(self, model_name, dropout):
        super().__init__()
        self.ir_encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.ir_encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, ir_input_ids, ir_attention_mask, **kwargs):
        output = self.ir_encoder(input_ids=ir_input_ids, attention_mask=ir_attention_mask)
        features = first_token_features(output)
        logits = self.classifier(self.dropout(features)).squeeze(-1)
        return {"logits": logits}


class ConcatenationModel(nn.Module):
    def __init__(self, model_name, latent_dimension, dropout):
        super().__init__()
        self.source_encoder = AutoModel.from_pretrained(model_name)
        self.ir_encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.source_encoder.config.hidden_size
        self.source_projection = nn.Linear(hidden_size, latent_dimension)
        self.ir_projection = nn.Linear(hidden_size, latent_dimension)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(latent_dimension * 2, 1)

    def forward(self, source_input_ids, source_attention_mask, ir_input_ids, ir_attention_mask, **kwargs):
        source_output = self.source_encoder(input_ids=source_input_ids, attention_mask=source_attention_mask)
        ir_output = self.ir_encoder(input_ids=ir_input_ids, attention_mask=ir_attention_mask)

        source_features = first_token_features(source_output)
        ir_features = first_token_features(ir_output)
        source_projected = self.source_projection(source_features)
        ir_projected = self.ir_projection(ir_features)

        combined = torch.cat([source_projected, ir_projected], dim=1)
        logits = self.classifier(self.dropout(combined)).squeeze(-1)
        return {"logits": logits}


class GatedFusionModel(nn.Module):
    def __init__(self, model_name, latent_dimension, dropout):
        super().__init__()
        self.source_encoder = AutoModel.from_pretrained(model_name)
        self.ir_encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.source_encoder.config.hidden_size

        self.source_projection = nn.Linear(hidden_size, latent_dimension)
        self.ir_projection = nn.Linear(hidden_size, latent_dimension)
        self.gate = nn.Linear(latent_dimension * 2, latent_dimension)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(latent_dimension, 1)

    def forward(self, source_input_ids, source_attention_mask, ir_input_ids, ir_attention_mask, **kwargs):
        source_output = self.source_encoder(input_ids=source_input_ids, attention_mask=source_attention_mask)
        ir_output = self.ir_encoder(input_ids=ir_input_ids, attention_mask=ir_attention_mask)

        source_features = first_token_features(source_output)
        ir_features = first_token_features(ir_output)

        source_projected = self.source_projection(source_features)
        ir_projected = self.ir_projection(ir_features)

        gate_input = torch.cat([source_projected, ir_projected], dim=1)
        alpha = torch.sigmoid(self.gate(gate_input))

        fused = alpha * source_projected + (1 - alpha) * ir_projected
        logits = self.classifier(self.dropout(fused)).squeeze(-1)
        return {"logits": logits, "alpha": alpha}


def build_model(baseline, config):
    baseline = baseline.lower()
    model_name = config["model"]["name"]
    dropout = config["model"]["dropout"]
    latent_dimension = config["model"]["latent_dimension"]

    if baseline == "b1":
        return SourceOnlyModel(model_name, dropout)
    if baseline == "b2":
        return IROnlyModel(model_name, dropout)
    if baseline == "b3":
        return ConcatenationModel(model_name, latent_dimension, dropout)
    if baseline == "b4":
        return GatedFusionModel(model_name, latent_dimension, dropout)
    raise ValueError("Baseline must be one of: b1, b2, b3, b4")
