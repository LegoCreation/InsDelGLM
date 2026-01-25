from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.bert.modeling_bert import BertModel as TransformersBertModel
from transformers.models.bert.modeling_bert import BertForMaskedLM as TransformersBertForMaskedLM
from transformers.models.bert.modeling_bert import BertForPreTraining as TransformersBertForPreTraining
from transformers.models.bert.modeling_bert import BertPreTrainedModel
from transformers.modeling_outputs import MaskedLMOutput
#from transformers.modeling_outputs import SequenceClassifierOutput


class BertModel(TransformersBertModel):
    def __init__(self, config):
        super().__init__(config)

class BertForMaskedLM(TransformersBertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

class BertForPreTraining(TransformersBertForPreTraining):
    def __init__(self, config):
        super().__init__(config)



class DNABertForMaskedLM(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        #self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        #position_ids: Optional[torch.Tensor] = None,
        #head_mask: Optional[torch.Tensor] = None,
        #inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        #return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # get the size of input_ids
        #batch_size, seq_len = input_ids.shape

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            #position_ids=position_ids,
            #head_mask=head_mask,
            #inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        #pooled_output = outputs[1] #
        sequence_output = outputs.last_hidden_state

        #pooled_output = self.dropout(pooled_output)
        #logits = self.classifier(pooled_output)
        # Get logits for each token in vocab
        prediction_scores = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # Standard CrossEntropyLoss ignores -100 by default (used for unmasked tokens)
            # Use label smoothing if specified in config, else default to 0.0
            label_smoothing = getattr(self.config, "label_smoothing", 0.0)
            loss_fct = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            
            # Flatten logits and labels for loss calculation
            loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
            
        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions
        )