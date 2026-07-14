"""The gated-fusion model with an XGBoost classifier."""

import torch
import torch.nn as nn
from transformers import AutoModel
from xgboost import XGBClassifier


class GatedFusionXGBoost(nn.Module):
    def __init__(self, source_model_name, ir_model_name, latent_dimension, dropout, gate_config=None):
        super().__init__()
        gate_config = gate_config or {}

        self.source_encoder = AutoModel.from_pretrained(source_model_name)
        self.ir_encoder = AutoModel.from_pretrained(ir_model_name)
        self.source_projection = nn.Linear(self.source_encoder.config.hidden_size, latent_dimension)
        self.ir_projection = nn.Linear(self.ir_encoder.config.hidden_size, latent_dimension)
        self.gate = nn.Linear(latent_dimension * 2, latent_dimension)

        if "bias_init" in gate_config:
            nn.init.constant_(self.gate.bias, float(gate_config["bias_init"]))
        self.gate_mode = str(gate_config.get("mode", "learned")).lower()
        self.fixed_alpha = gate_config.get("fixed_alpha")
        self.gate_temperature = max(float(gate_config.get("temperature", 1.0)), 1e-6)

        # This is the only part that differs from the original B4 model.
        self.classifier = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            n_jobs=-1,
        )

    def fused_features(self, source_input_ids, source_attention_mask, ir_input_ids, ir_attention_mask):
        source = self.source_encoder(
            input_ids=source_input_ids, attention_mask=source_attention_mask
        ).last_hidden_state[:, 0, :]
        ir = self.ir_encoder(
            input_ids=ir_input_ids, attention_mask=ir_attention_mask
        ).last_hidden_state[:, 0, :]

        source = self.source_projection(source)
        ir = self.ir_projection(ir)
        gate_input = torch.cat([source, ir], dim=1)

        if self.gate_mode == "fixed" and self.fixed_alpha is not None:
            alpha = torch.full_like(source, min(max(float(self.fixed_alpha), 0.0), 1.0))
        else:
            alpha = torch.sigmoid(self.gate(gate_input) / self.gate_temperature)

        return alpha * source + (1 - alpha) * ir, alpha

    def forward(self, source_input_ids, source_attention_mask, ir_input_ids, ir_attention_mask):
        fused, alpha = self.fused_features(
            source_input_ids, source_attention_mask, ir_input_ids, ir_attention_mask
        )
        return {"features": fused, "alpha": alpha}

    def load_b4_backbone(self, checkpoint):
        """Load every trained B4 weight except its original linear classifier."""
        state = checkpoint.get("model_state_dict", checkpoint)
        backbone_state = {key: value for key, value in state.items() if not key.startswith("classifier.")}
        self.load_state_dict(backbone_state, strict=False)
