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
        self.bert = AutoModel.from_pretrained(lm_mp[lm], output_hidden_states=output_hidden_states)
        # role embedding initialization
        self.bert.config.type_vocab_size = type_vocab_size
        self.bert.embeddings = BertEmbeddings(self.bert.config)

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

from typing import List, Optional, Tuple, Union
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values_length: int = 0,
    ) -> torch.Tensor:
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # role embedding addition
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings





        
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
