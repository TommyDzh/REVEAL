import torch
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix

# import pytrec_eval


from collections import OrderedDict

BASE_DATA_PATH = '/efs/pretrain_datasets/'


task_num_class_dict = {
        "turl": 255,
        "turl-re": 121,
        "gt-semtab22-dbpedia-all": 101,
        "gt-semtab22-schema-property-all": 53,
        "gt-sotab": 91,
        "SOTAB": 91,
        "sotab": 91,
        "sotab-re": 176,
    }


def collate_fn(pad_token_id, data_only=True):
    '''padder for input batch'''

    def padder(samples):    
        data = torch.nn.utils.rnn.pad_sequence(
            [sample["data"] for sample in samples], padding_value=pad_token_id)
        if not data_only:
            label = torch.nn.utils.rnn.pad_sequence(
                [sample["label"] for sample in samples], padding_value=-1)
        else:
            label = torch.cat([sample["label"] for sample in samples])
        batch = {"data": data, "label": label}
        if "idx" in samples[0]:
            batch["idx"] = [sample["idx"] for sample in samples]
        if "cls_indexes" in samples[0]:
            cls_indexes = torch.nn.utils.rnn.pad_sequence(
                [sample["cls_indexes"] for sample in samples], padding_value=0)
            batch["cls_indexes"] = cls_indexes
        if "token_type_ids" in samples[0]:
            token_type_ids = torch.nn.utils.rnn.pad_sequence(
                [sample["token_type_ids"] for sample in samples], padding_value=1)
            batch["token_type_ids"] = token_type_ids
        if "table_embedding" in samples[0]:
            table_embeddings = [sample["table_embedding"] for sample in samples]
            batch["table_embedding"] = torch.stack(table_embeddings, dim=0)
        if "target_col_mask" in samples[0]:
            target_col_mask = torch.nn.utils.rnn.pad_sequence(
                [sample["target_col_mask"] for sample in samples], padding_value=-1)
            batch["target_col_mask"] = target_col_mask    
        return batch
        
    return padder

def veri_batch_collate_fn(pad_token_id, binary=True):
    '''padder for input batch'''
    
    def padder_binary(samples):   
        batch_index = torch.cat([torch.tensor([idx for _ in range(len(sample["label"]))]) for idx, sample in enumerate(samples)])
        label = torch.cat([sample["label"].reshape(-1) for sample in samples], dim=-1)
        batch = {"label": label, "index": batch_index}
        if  "data" in samples[0]:
            data = torch.nn.utils.rnn.pad_sequence(
                [sample["data"] for sample in samples], padding_value=pad_token_id)
            batch["data"] = data
        if "cls_indexes" in samples[0]:
            i = 0
            cls_indexes = []
            for sample in samples:
                value =  sample["cls_indexes"]
                cls_indexes.append(torch.tensor([i, value]))
                i += 1
            cls_indexes = torch.stack(cls_indexes, dim=0).long()
            batch["cls_indexes"] = cls_indexes
        if "embs" in samples[0]:
            embs = torch.cat([sample["embs"] for sample in samples], dim=0)
            batch["embs"] = embs
        if "logits" in samples[0]:
            logits = torch.stack([sample["logits"] for sample in samples], dim=0)
            batch["logits"] = logits
        return batch

    return padder_binary

def collate_fn_iter(pad_token_id, data_only=True):
    '''padder for input batch'''

    def padder(samples):    
        data = torch.nn.utils.rnn.pad_sequence(
            [sample["data"] for sample in samples], padding_value=pad_token_id)
        if not data_only:
            label = torch.nn.utils.rnn.pad_sequence(
                [sample["label"] for sample in samples], padding_value=-1)
        else:
            label = torch.cat([sample["label"] for sample in samples])
        batch = {"data": data, "label": label}
        if "idx" in samples[0]:
            batch["idx"] = [sample["idx"] for sample in samples]
        if "cls_indexes" in samples[0]:
            cls_indexes = torch.nn.utils.rnn.pad_sequence(
                [sample["cls_indexes"] for sample in samples], padding_value=0)
            batch["cls_indexes"] = cls_indexes
        if "target_col_mask" in samples[0]:
            target_col_mask = torch.nn.utils.rnn.pad_sequence(
                [sample["target_col_mask"] for sample in samples], padding_value=-1)
            batch["target_col_mask"] = target_col_mask
        if "table_embedding" in samples[0]:
            table_embeddings = [sample["table_embedding"] for sample in samples]
            batch["table_embedding"] = torch.stack(table_embeddings, dim=0)
        return batch
        
    return padder



def veri_collate_fn(pad_token_id, binary=False):
    '''padder for input batch'''
    
    def padder(samples):    
        batch = {}
        if "embs" in samples[0]:
            embs = torch.stack([sample["embs"] for sample in samples], dim=0)
            batch["embs"] = embs
        
        if "data" in samples[0]:
            data = []
            for sample in samples:
                data.extend(sample["data"])
            data =torch.nn.utils.rnn.pad_sequence(
                data, padding_value=pad_token_id)
            batch["data"] = data
        if "label" in samples[0]:
            label = torch.cat([sample["label"] for sample in samples], dim=-1)
            batch["label"] = label
        if "cls_indexes" in samples[0]:
            i = 0
            cls_indexes = []
            for sample in samples:
                for value in sample["cls_indexes"]:
                    cls_indexes.append(torch.tensor([i, value]))
                    i += 1
            cls_indexes = torch.stack(cls_indexes, dim=0).long()
            batch["cls_indexes"] = cls_indexes
        if "hidden_states" in samples[0]:
            hidden_states = torch.stack([sample["hidden_states"] for sample in samples], dim=0)
            batch["hidden_states"] = hidden_states
        return batch
    
    def padder_binary(samples):   

        label = torch.cat([sample["label"] for sample in samples], dim=-1)
        batch = {"label": label}
        if  "data" in samples[0]:
            data = torch.nn.utils.rnn.pad_sequence(
                [sample["data"] for sample in samples], padding_value=pad_token_id)
            batch["data"] = data
        if "cls_indexes" in samples[0]:
            i = 0
            cls_indexes = []
            for sample in samples:
                value =  sample["cls_indexes"]
                cls_indexes.append(torch.tensor([i, value]))
                i += 1
            cls_indexes = torch.stack(cls_indexes, dim=0).long()
            batch["cls_indexes"] = cls_indexes
        if "embs" in samples[0]:
            embs = torch.stack([sample["embs"] for sample in samples], dim=0)
            batch["embs"] = embs
        if "logits" in samples[0]:
            logits = torch.stack([sample["logits"] for sample in samples], dim=0)
            batch["logits"] = logits
        if "hidden_states" in samples[0] and samples[0]["hidden_states"] is not None:
            hidden_states = torch.stack([sample["hidden_states"] for sample in samples], dim=0)
            batch["hidden_states"] = hidden_states
        return batch
    if binary:
        return padder_binary
    else:
        return padder

def load_checkpoint(ckpt, device=None):
    """Load a model from a checkpoint.
    Args:
        ckpt (str): the model checkpoint path.

    Returns:
        SupCLforTable or UnsupCLforTable: the pre-trained model
        PretrainDataset: the dataset for pre-training the model
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hp = ckpt['hp']
    if 'table_order' not in hp:
        setattr(hp, 'table_order', 'column')

    if hp.mode in ['supcon', 'supcon_ddp']:
        model = SupCLforTable(hp, device=device, lm=hp.lm)
    else:
        model = UnsupCLforTable(hp, device=device, lm=hp.lm)
        
    model = model.to(device)
    try:
        model.load_state_dict(ckpt['model'])
    except:
        new_state_dict = OrderedDict()
        for k, v in ckpt['model'].items():
            name = k[7:]
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)

    if 'pretrain_data' not in hp:
        setattr(hp, 'pretrain_data', hp.task)
        
    print(hp)

    return model, None
        


def f1_score_multilabel(true_list, pred_list):
    conf_mat = multilabel_confusion_matrix(np.array(true_list),
                                           np.array(pred_list))
    agg_conf_mat = conf_mat.sum(axis=0)
    # Note: Pos F1
    # [[TN FP], [FN, TP]] if we consider 1 as the positive class
    p = agg_conf_mat[1, 1] / agg_conf_mat[1, :].sum()
    r = agg_conf_mat[1, 1] / agg_conf_mat[:, 1].sum()
    
    micro_f1 = 2 * p * r / (p  + r) if (p + r) > 0 else 0.
    class_p = conf_mat[:, 1, 1] /  conf_mat[:, 1, :].sum(axis=1)
    class_r = conf_mat[:, 1, 1] /  conf_mat[:, :, 1].sum(axis=1)
    class_f1 = np.divide(2 * (class_p * class_r), class_p + class_r,
                         out=np.zeros_like(class_p), where=(class_p + class_r) != 0)
    class_f1 = np.nan_to_num(class_f1)
    macro_f1 = class_f1.mean()
    return (micro_f1, macro_f1, class_f1, conf_mat)


def get_col_pred(logits, labels, idx_list, top_k=-1):
    '''for column population'''
    pred_labels = {}
    out_prob_cp = torch.autograd.Variable(logits.clone(), requires_grad=False).cpu()
    for k, row_labels in enumerate(labels):
        # print(k, len(row_labels),  sum(row_labels > -1), row_labels)
        n_pred = sum(row_labels > -1) if top_k == -1 else top_k
        pred_row_labels = out_prob_cp[k][1:].argsort(dim=0, descending=True)[:n_pred].cpu().numpy()  # out_prob[0]: blank header
        pred_row_labels = [elem+1 for elem in pred_row_labels]  # add idx to 1 (for out_prob[0])
        pred_row_labels_prob = dict(zip(pred_row_labels, map(lambda x: out_prob_cp[k][x].item(), pred_row_labels)))
        pred_labels["Q" + str(idx_list[k])] = pred_row_labels_prob
    return pred_labels


h2i_fn = {"col-popl": "/efs/task_datasets/col_popl/h2idx.json",
          "col-popl-turl": "/efs/task_datasets/col_popl_turl/h2idx.json"}


import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Load NLTK resources
import nltk
nltk.download('punkt')
nltk.download('stopwords')
text = "BM25 is a ranking function used in information retrieval. It's effective for search engines!"
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove punctuation
    # Create a translation table where all punctuation is replaced with spaces
    translation_table = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    # Replace all punctuation in the text with spaces
    text = text.translate(translation_table)
    # print(text)

    # Tokenize the text
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    processed_text = ' '.join(tokens)
    return processed_text