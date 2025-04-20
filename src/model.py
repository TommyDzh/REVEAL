import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoTokenizer

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base-uncased'}


def off_diagonal(x):
    """Return a flattened view of the off-diagonal elements of a square matrix.
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def pool_sub_sentences(hidden_states, cls_indexes, table_length=None):
    pooled_outputs = []
    B = hidden_states.size(0)
    
    for i in range(B):
        cls_indices = cls_indexes[cls_indexes[:,0]==i]
        sub_sentences = []
        max_length = table_length[i] if table_length is not None else hidden_states.size(1)
        # Extract sub-sentence embeddings based on CLS tokens
        for j in range(len(cls_indices)):
            start_idx = cls_indices[j][1].item()
            end_idx = cls_indices[j+1][1].item() if j+1 < len(cls_indices) else max_length
            
            # Pooling (e.g., mean pooling) the tokens in the sub-sentence
            sub_sentence_embedding = hidden_states[i, start_idx:end_idx, :].mean(dim=0)
            sub_sentences.append(sub_sentence_embedding)
        
        pooled_outputs.append(torch.stack(sub_sentences))
    pooled_outputs = torch.cat(pooled_outputs, dim=0)
    return pooled_outputs

# Function to extract CLS token embeddings for each sub-sentence
def extract_cls_tokens(hidden_states, cls_indexes, head=False):
    cls_embeddings = []
    for i, j in cls_indexes:
        sub_sentence_cls_embeddings = hidden_states[i, 0, :] if head else hidden_states[i, j, :]
        cls_embeddings.append(sub_sentence_cls_embeddings)
    cls_embeddings = torch.stack(cls_embeddings)
    return cls_embeddings

class BertPooler(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        pooled_outputs = self.dense(hidden_states)
        pooled_outputs = self.activation(pooled_outputs)
        return pooled_outputs
    
class BertMultiPooler(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.pooler = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),  # Linear layer
        nn.ReLU(),  # Non-linear activation
        nn.LayerNorm(hidden_size)  # Optional normalization
    )
        

    def forward(self, hidden_states):

        pooled_outputs = self.pooler(hidden_states)
        return pooled_outputs





class BertMultiPairPooler(nn.Module):

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size


        self.pooler = nn.Sequential(
        nn.Linear(self.hidden_size * 2, self.hidden_size),  # Linear layer
        nn.ReLU(),  # Non-linear activation
        nn.LayerNorm(self.hidden_size)  # Optional normalization
    )

        
    def forward(self, hidden_states):
        hidden_states_first_cls = hidden_states[:, 0].unsqueeze(1).repeat(
            [1, hidden_states.shape[1], 1])

        pooled_outputs = self.pooler(
            torch.cat([hidden_states_first_cls, hidden_states], 2))

        return pooled_outputs

class BertForMultiOutputClassification(nn.Module):

    def __init__(self, hp, device='cuda', lm='roberta', col_pair='None', use_attention_mask=True, output_hidden_states=False, type_vocab_size=2, pair_output=False):
        
        super().__init__()
        self.hp = hp
        if type_vocab_size == 2:
            self.bert = AutoModel.from_pretrained(lm_mp[lm], output_hidden_states=output_hidden_states)
        else:
            self.bert = AutoModel.from_pretrained(lm_mp[lm], output_hidden_states=output_hidden_states, type_vocab_size=type_vocab_size, ignore_mismatched_sizes=True)
        self.device = device
        self.col_pair = col_pair
        self.use_attention_mask =  use_attention_mask
        self.output_hidden_states = output_hidden_states
        hidden_size = 768

        # projector
        pooler = BertPooler(hidden_size)
        with torch.no_grad():
            pooler.dense.weight.copy_(self.bert.pooler.dense.weight)
            pooler.dense.bias.copy_(self.bert.pooler.dense.bias)
            self.bert.pooler = pooler
            
        self.projector = nn.Linear(hidden_size, hp.projector)
        '''Require all models using the same CLS token'''
        self.cls_token_id = AutoTokenizer.from_pretrained(lm_mp[lm]).cls_token_id
        self.num_labels = hp.num_labels
        self.dropout = nn.Dropout(hp.hidden_dropout_prob)
        self.classifier = nn.Linear(hidden_size, self.hp.num_labels)
    
    def load_from_CL_model(self, model):
        '''load from models pre-trained with contrastive learning'''
        self.bert = model.bert
        self.projector = model.projector
        self.cls_token_id = model.cls_token_id
    
    def forward(
        self,
        input_ids=None,
        get_enc=False,
        cls_indexes=None,
        token_type_ids=None,
        get_hidden_states=False,
    ):
        if self.use_attention_mask:
            attention_mask = input_ids != 0
        else:
            attention_mask = None
        # BertModelMultiOutput
        bert_output = self.bert(input_ids, token_type_ids=token_type_ids, 
                                attention_mask=attention_mask)

        pooled_output = bert_output[1]
        pooled_output = extract_cls_tokens(pooled_output, cls_indexes)
        logits = self.classifier(pooled_output)

        if get_enc:
            outputs = (logits, pooled_output) if not get_hidden_states else (logits, pooled_output, bert_output[2])
        else:
            outputs = logits
        return outputs  # (loss), logits, (hidden_states), (attentions)







        
class VerifierMulti(nn.Module):
    # explicitly set pos/neg examples (2-class classification)
    def __init__(self, module ="ffn", dropout=0.0, norm="batch_norm", input_size=1, num_layers=None):
        super().__init__()
        hidden_size = 768
        input_size = input_size * hidden_size
        self.module = module
        if module == "ffn_fix":
            self.ffn = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
        elif module == "ffn_fix_3":
            self.ffn = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
        self.linear = nn.Linear(hidden_size, 2)               


    def forward(
        self,
        embs,
        hidden_states=None,
        return_embs=False,
    ):
        embs_dim = len(embs.size())
        if embs_dim == 3:
            B, L, D = embs.size()
            embs = embs.reshape(B*L, D)
        if hidden_states is not None:
            if len(hidden_states.size()) == 3:
                B, L, D = hidden_states.size()
                hidden_states = hidden_states.reshape(B*L, D)
            embs = torch.cat([embs, hidden_states], dim=-1)
        if "ffn" in self.module:

            embs = self.ffn(embs)
            scores = self.linear(embs)
            
            if embs_dim == 3:
                scores = scores.reshape(B, L, 1)

        if return_embs:
            return scores, embs
        else:
            return scores
