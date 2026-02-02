import torch
import torch.nn as nn
from transformers.modeling_outputs import MaskedLMOutput
from transformers import PreTrainedModel, PretrainedConfig


class MaskedCNNConfig(PretrainedConfig):
    """Configuration class for MaskedCNN model."""
    
    def __init__(
        self,
        vocab_size=8,
        hidden_size=256,
        num_conv_layers=6,
        kernel_size=3,
        dropout_prob=0.1,
        max_length=512,
        pad_token_id=0,
        mask_token_id=5,
        cls_token_id=6,
        sep_token_id=7,
        label_smoothing=0.0,
        **kwargs
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            **kwargs
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.dropout_prob = dropout_prob
        self.max_length = max_length
        self.mask_token_id = mask_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.label_smoothing = label_smoothing


class MaskedCNN(PreTrainedModel):
    """1D CNN model for masked language modeling on DNA sequences."""
    
    config_class = MaskedCNNConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # Token embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        
        # Stack of 1D convolutional layers
        self.conv_layers = nn.ModuleList()
        for i in range(config.num_conv_layers):
            conv_layer = nn.Sequential(
                nn.Conv1d(
                    in_channels=config.hidden_size,
                    out_channels=config.hidden_size,
                    kernel_size=config.kernel_size,
                    padding=config.kernel_size // 2,  # Same padding
                    bias=False
                ),
                nn.BatchNorm1d(config.hidden_size),
                nn.ReLU(),
                nn.Dropout(config.dropout_prob)
            )
            self.conv_layers.append(conv_layer)
        
        # Residual connections every 2 layers
        self.use_residual = config.num_conv_layers > 2
        
        # Output projection to vocabulary
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        return_dict=True,
        **kwargs
    ):
        # Embed tokens
        embeddings = self.embedding(input_ids)  # [batch_size, seq_len, hidden_size]
        
        # Transpose for Conv1D: [batch_size, hidden_size, seq_len]
        x = embeddings.transpose(1, 2)
        
        # Apply convolutional layers with residual connections
        for i, conv_layer in enumerate(self.conv_layers):
            residual = x
            x = conv_layer(x)
            
            # Add residual connection every 2 layers
            if self.use_residual and i > 0 and i % 2 == 1:
                x = x + residual
        
        # Transpose back: [batch_size, seq_len, hidden_size]
        x = x.transpose(1, 2)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask to match hidden dimensions
            attention_mask = attention_mask.unsqueeze(-1).expand_as(x)
            x = x * attention_mask
        
        # Project to vocabulary
        logits = self.classifier(x)  # [batch_size, seq_len, vocab_size]
        
        loss = None
        if labels is not None:
            # Calculate loss only for masked positions (labels != -100)
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=getattr(self.config, "label_smoothing", 0.0)
            )
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        
        return MaskedLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


class DNACNNForMaskedLM(MaskedCNN):
    """Alias for compatibility with existing code."""
    pass