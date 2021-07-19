from transformers import ElectraModel
import torch
import torch.nn as nn

class ClassificationHeadElectra(nn.Module):
    '''Head for classifying sequence embeddings'''
    def __init__(self, hidden_size, classes, dropout=0.5):
        super().__init__()
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, classes)

    def forward(self, features):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.layer(x)
        m = nn.GELU()
        x = m(x) # gelu used by electra authors
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class ElectraUTClassifier(nn.Module):
    '''
    Elecra based universal transformer, classifier
    '''
    def __init__(self, device, hidden_size=768, classes=6, layers=12):
        super().__init__()
        self.electra = ElectraModel.from_pretrained('google/electra-base-discriminator')
        self.classifier = ClassificationHeadElectra(hidden_size, classes)
        self.layers = layers
        self.device = device
    
    def forward(self, input_ids, attention_mask):
        '''
        input_ids = [N x L], containing sequence of ids of words after tokenization
        attention_mask = [N x L], mask for attention

        N = batch size
        L = maximum sentence length
        '''

        self.input_shape = input_ids.size()
        extended_attention_mask: torch.Tensor = self.electra.get_extended_attention_mask(attention_mask, self.input_shape, self.device)
        hidden_states = self.electra.embeddings(input_ids=input_ids)

        for _ in range(self.layers):
            layer_outputs = self.electra.encoder.layer[0](hidden_states, extended_attention_mask)
            hidden_states = layer_outputs[0]
            
        logits = self.classifier(hidden_states)
        return logits



