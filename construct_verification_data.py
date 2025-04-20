import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import math
import os
import random
from time import time
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from collections import defaultdict

# import pytrec_eval
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from accelerate import Accelerator
from copy import deepcopy

torch.backends.cuda.matmul.allow_tf32 = True

from src.dataset import (
    GittablesTablewiseIterateMMRDataset, # GitTablesDB, GitTablesSC
    SotabTablewiseIterateMMRDataset, # SOTAB-CTA
    SotabRelExtIterateMMRDataset, # SOTAB-CPA
    TURLRelExtTablewiseIterateMMRDataset, # WikiTable-CTA
    TURLColTypeTablewiseIterateMMRDataset, # WikiTable-CPA
)

from src.dataset import TableDataset, SupCLTableDataset, SemtableCVTablewiseDataset, GittablesColwiseDataset, GittablesTablewiseDataset
from src.model import BertMultiPooler, BertForMultiOutputClassification, BertForMultiOutputClassificationColPopl, Verifier
from src.model import SupCLforTable, UnsupCLforTable, lm_mp
from src.utils import load_checkpoint, f1_score_multilabel, collate_fn, get_col_pred, ColPoplEvaluator
from src.utils import task_num_class_dict
from accelerate import DistributedDataParallelKwargs
import wandb

from argparse import Namespace
import torch
import random
import pandas as pd
import numpy as np
import os
import pickle
import json
import re
import transformers
from torch.utils import data
from torch.nn.utils import rnn
from transformers import AutoTokenizer

from typing import List
from functools import reduce
import operator

from itertools import chain
import copy


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
        if "target_col_mask" in samples[0]:
            target_col_mask = torch.nn.utils.rnn.pad_sequence(
                [sample["target_col_mask"] for sample in samples], padding_value=-1)
            batch["target_col_mask"] = target_col_mask
        if "table_embedding" in samples[0]:
            table_embeddings = [sample["table_embedding"] for sample in samples]
            batch["table_embedding"] = torch.stack(table_embeddings, dim=0)
        return batch
        
    return padder



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb", type=bool, default=False)
    parser.add_argument("--cv", type=int, default=4)
    parser.add_argument("--unlabeled_train_only", type=bool, default=False)
    parser.add_argument("--context_encoding_type", type=str, default="v0")
    parser.add_argument("--pool_version", type=str, default="v0.2")
    parser.add_argument("--random_sample", type=bool, default=False)
    parser.add_argument("--comment", type=str, default="debug", help="to distinguish the runs")
    parser.add_argument(
        "--shortcut_name",
        default="bert-base-uncased",
        type=str,
        help="Huggingface model shortcut name ",
    )
    parser.add_argument(
        "--max_length",
        default=32,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--adaptive_max_length",
        default=False,
        type=bool,
    )    
    parser.add_argument(
        "--max_num_col",
        default=8,
        type=int,
    )   

    parser.add_argument(
        "--batch_size",
        default=3,
        type=int,
        help="Batch size",
    )
    parser.add_argument(
        "--epoch",
        default=1,
        type=int,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--random_seed",
        default=4649,
        type=int,
        help="Random seed",
    )

    parser.add_argument(
        "--train_n_seed_cols",
        default=-1,
        type=int,
        help="number of seeding columns in training",
    )

    parser.add_argument(
        "--num_classes",
        default=78,
        type=int,
        help="Number of classes",
    )
    parser.add_argument("--gpu",
                        type=int,
                        default=0)
    parser.add_argument("--fp16",
                        action="store_true",
                        default=False,
                        help="Use FP16")
    parser.add_argument("--best_dict_path",
                        type=str,
                        help="Path to the trained REVEAL model checkpoint.")   
    parser.add_argument("--task",
                        type=str,
                        choices=[
                            "sotab", "sotab-re", 
                            "gt-semtab22-dbpedia-all", 
                            "gt-semtab22-schema-property-all",
                            "turl", "turl-re", 
                        ],
                        help="Task names}")     
    parser.add_argument("--warmup",
                        type=float,
                        default=0.,
                        help="Warmup ratio")
    parser.add_argument("--depth",
                        type=int,
                        default=3,
                        help="gt-semtab22-dbpedia-all, gt-semtab22-schema-property-all: 3; others: 1")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--colpair",
                        action="store_true",
                        help="Use column pair embedding")
    parser.add_argument("--metadata",
                        action="store_true",
                        help="Use column header metadata")
    parser.add_argument("--from_scratch",
                        action="store_true",
                        help="Training from scratch")
    parser.add_argument("--cl_tag",
                        type=str,
                        default="None",
                        help="path to the pre-trained file")
    parser.add_argument("--dropout_prob",
                        type=float,
                        default=0.5)
    parser.add_argument("--eval_test",
                        action="store_true",
                        help="evaluate on testset and do not save the model file")
    parser.add_argument("--small_tag",
                        type=str,
                        default="semi1",
                        help="e.g., by_table_t5_v1")
    parser.add_argument("--data_path",
                        type=str,
                        default="./data")
    parser.add_argument("--pretrained_ckpt_path",
                        type=str,
                        default="./model")    

    args = parser.parse_args()
    device = torch.device(args.gpu)
    task = args.task
    if args.small_tag != "":
        args.eval_test = True

    args.num_classes = task_num_class_dict[task]
    if args.colpair:
        assert "turl-re" == task, "colpair can be only used for Relation Extraction"
    if args.metadata:
        assert "turl-re" == task or "turl" == task, "metadata can be only used for TURL datasets"
    if "col-popl":
        # metrics = {
        #     "accuracy": CategoricalAccuracy(tie_break=True),
        # }
        if args.train_n_seed_cols != -1:
            if "col-popl" in task:
                assert args.train_n_seed_cols == int(task[-1]),  "# of seed columns must match"

    print("args={}".format(json.dumps(vars(args))))

    max_length = args.max_length
    batch_size = args.batch_size
    num_train_epochs = args.epoch

    shortcut_name = args.shortcut_name

    if args.colpair and args.metadata:
        taskname = "{}-colpair-metadata".format(task)
    elif args.colpair:
        taskname = "{}-colpair".format(task)
    elif args.metadata:
        taskname = "{}-metadata".format(task)
    elif args.train_n_seed_cols == -1 and 'popl' in task:
        taskname = "{}-mix".format(task)
    else:
        taskname = "".join(task)

    if args.from_scratch:
        if "gt" in task:
            tag_name = "{}/{}-{}-{}-pool{}-max_cols{}-rand{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname,  "{}-fromscratch".format(shortcut_name), args.small_tag, args.comment, args.pool_version, args.max_num_col, args.random_sample,
                batch_size, max_length, num_train_epochs, args.dropout_prob, 
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')
        else:
            tag_name = "{}/{}-{}-{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname,  "{}-fromscratch".format(shortcut_name), args.small_tag, args.comment, 
                batch_size, max_length, num_train_epochs, args.dropout_prob, 
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')
        
    else:
        if "gt" in task:
            tag_name = "{}/{}_{}-pool{}-max_cols{}-rand{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname, args.cl_tag.replace('/', '-'),  shortcut_name, args.small_tag, args.pool_version, args.max_num_col, args.random_sample,
                batch_size, max_length, num_train_epochs, args.dropout_prob,
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')
        else:
            tag_name = "{}/{}_{}-{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname, args.cl_tag.replace('/', '-'),  shortcut_name, args.small_tag,
                batch_size, max_length, num_train_epochs, args.dropout_prob,
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')

    print(tag_name)
    file_path = os.path.join("./model", args.task, "outputs", tag_name)

    dirpath = os.path.dirname(file_path)
    if not os.path.exists(dirpath):
        print("{} not exists. Created".format(dirpath))
        os.makedirs(dirpath)

    if args.fp16:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
        
        
    # accelerator = Accelerator(mixed_precision="no" if not args.fp16 else "fp16")   
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="no" if not args.fp16 else "fp16", kwargs_handlers=[ddp_kwargs])

    ckpt_path = os.path.join(args.pretrained_ckpt_path, args.cl_tag)
    # ckpt_path = '/efs/checkpoints/{}.pt'.format(args.cl_tag)
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_hp = ckpt['hp']
    print(ckpt_hp)

    setattr(ckpt_hp, 'batch_size', args.batch_size)
    setattr(ckpt_hp, 'hidden_dropout_prob', args.dropout_prob)
    setattr(ckpt_hp, 'shortcut_name', args.shortcut_name)
    setattr(ckpt_hp, 'num_labels', args.num_classes)



    tokenizer = BertTokenizer.from_pretrained(shortcut_name)
    padder = collate_fn(tokenizer.pad_token_id)
    model = BertForMultiOutputClassification(ckpt_hp, device=device, 
                                            lm=ckpt['hp'].lm, 
                                            version="v1",
                                            use_attention_mask=True)      
    config = BertConfig.from_pretrained(lm_mp[ckpt['hp'].lm])
    model.bert.pooler = BertMultiPooler(config.hidden_size, version="v1").to(device)


    small_tag = "sp_semi1" if args.task == "gt-semtab22-schema-property-all" else "db_semi1"
    lbd = 0.5



    best_state_dict = torch.load(args.best_dict_path, map_location=device)
    model.load_state_dict(best_state_dict, strict=False)
    model = model.to(device)

    if "gt-dbpedia" in task or "gt-schema" in task or "gt-semtab" in task:
        dataset_cls = GittablesTablewiseIterateMMRDataset
        train_dataset_mmr = dataset_cls(
                                cv=args.cv,
                                    split="train",
                                    src=None,
                                    tokenizer=tokenizer,
                                    max_length=128,
                                    gt_only='all' not in task,
                                    device=device,
                                    base_dirpath=os.path.join(args.data_path, "GitTables/semtab_gittables/2022"),
                                    small_tag=small_tag,
                                    max_unlabeled=8,
                                    random_sample=True,
                                    lbd=0.5)
        padder = collate_fn(tokenizer.pad_token_id)
        train_dataloader_iter = DataLoader(train_dataset_mmr,
                                        batch_size=1,
                                    #   collate_fn=collate_fn)
                                    collate_fn=padder)

        lbd = 0.5
        veri_dataset_mmr = dataset_cls(
                                cv=args.cv,
                                    split="valid",
                                    src=None,
                                    tokenizer=tokenizer,
                                    max_length=128,
                                    gt_only='all' not in task,
                                    device=device,
                                    base_dirpath=os.path.join(args.data_path, "GitTables/semtab_gittables/2022"),
                                    small_tag=small_tag,
                                    max_unlabeled=8,
                                    random_sample=True,
                                    lbd=0.5)
        padder = collate_fn(tokenizer.pad_token_id)
        veri_dataloader_iter = DataLoader(veri_dataset_mmr,
                                        batch_size=1,
                                    #   collate_fn=collate_fn)
                                    collate_fn=padder)
        valid_dataloader_iter = DataLoader(veri_dataset_mmr,\
                                        batch_size=1,
                                    #   collate_fn=collate_fn)
                                    collate_fn=padder)

        test_dataset_mmr = dataset_cls(
                                cv=args.cv,
                                                split="test",
                                                src=None,
                                                tokenizer=tokenizer,
                                                max_length=128,
                                                gt_only='all' not in task,
                                                device=device,
                                                base_dirpath=os.path.join(args.data_path, "GitTables/semtab_gittables/2022"),
                                                small_tag=small_tag,
                                                max_unlabeled=8,
                                                random_sample=True,
                                                lbd=0.5)
        padder = collate_fn(tokenizer.pad_token_id)
        test_dataloader = DataLoader(test_dataset_mmr,
                                        batch_size=16,
                                    #   collate_fn=collate_fn)
                                    collate_fn=padder)
        test_dataloader_iter = DataLoader(test_dataset_mmr,
                                        batch_size=1,
                                    #   collate_fn=collate_fn)
                                    collate_fn=padder)

    # data check
    model.eval()
    if "popl" in task:
        ts_pred_list = {}
        ts_true_list = {}
        ts_logits_list = {}
    else:
        ts_pred_list = []
        ts_true_list = []
        ts_logits_list = []
    # Test
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            batch["data"] = batch["data"].to(device)
            batch["label"] = batch["label"].to(device)
            cls_indexes = batch["cls_indexes"].reshape(-1).to(device)
            cls_indexes = torch.stack([torch.arange(len(cls_indexes), device=device), cls_indexes], dim=1)
            if "re" in task:
                # col_mask= (batch["target_col_mask"].T < 2).long().to(device)
                col_mask= batch["target_col_mask"].T.long().to(device)
                col_mask[(col_mask>=2) | (col_mask==-1)] = 2 
            else:
                col_mask= (batch["target_col_mask"].T == 0).long().to(device)
            if "popl" in task:
                logits, = model(batch["data"].T, cls_indexes)
                labels = batch["label"].T
                logits = []
                labels_1d = []
                all_labels = []
                for _, x in enumerate(logits):
                    logits.append(x.expand(sum(labels[_]>-1), args.num_classes))
                    labels_1d.extend(labels[_][labels[_]>-1])
                    all_labels.append(labels[_][labels[_]>-1].cpu().detach().numpy())
                logits = torch.cat(logits, dim=0).to(device)
                labels_1d = torch.as_tensor(labels_1d).to(device)
                all_preds = get_col_pred(logits, labels, batch["idx"], top_k=500)#.cpu().detach().numpy()
                ts_pred_list.update(all_preds)
                
            else:
                logits = model(batch["data"].T, cls_indexes=cls_indexes, token_type_ids=col_mask).cpu()

                if "sato" in task or "gt-" in task or "sotab" in task:
                    if task == "sotab-re":
                        target_mask = (batch["label"] != -1).cpu()
                        labels_1d = batch["label"][target_mask]
                        new_logits = logits[target_mask]
                        
                        
                        ts_pred_list += new_logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        ts_true_list += labels_1d.long().cpu().detach().numpy().tolist()   
                        ts_logits_list += new_logits.cpu().detach().numpy().tolist()
                    else:
                        ts_pred_list += logits.argmax(
                            1).cpu().detach().numpy().tolist()
                        ts_true_list += batch["label"].cpu().detach().numpy(
                        ).tolist()
                        ts_logits_list += logits.cpu().detach().numpy().tolist()
                elif "turl" in task:
                    if "turl-re" in task:  # turl-re-colpair
                        all_preds = (logits >= math.log(0.5)
                                    ).int().detach().cpu().numpy()
                        all_labels = batch["label"].cpu().detach().numpy()
                        idxes = np.where(all_labels > 0)[0]
                        ts_pred_list += all_preds[idxes, :].tolist()
                        ts_true_list += all_labels[idxes, :].tolist()
                    elif task == "turl":
                        ts_pred_list += (logits >= math.log(0.5)
                                        ).int().detach().cpu().tolist()
                        ts_true_list += batch["label"].cpu().detach(
                        ).numpy().tolist()
        if "sato" in task or "gt-" in task or "sotab" in task:
            ts_micro_f1 = f1_score(ts_true_list,
                                ts_pred_list,
                                average="micro")
            ts_macro_f1 = f1_score(ts_true_list,
                                ts_pred_list,
                                average="macro")
        print("ts_micro_f1={:.4f}, ts_macro_f1={:.4f}".format(ts_micro_f1, ts_macro_f1))
        
        
    import torch.nn.functional as F
    # exactly the same as test
    import itertools

    def is_sublist(A, B):
        it = iter(B)
        return all(x in it for x in A)
    def get_permutation(x):
        new = []
        x = x.tolist()
        if len(x) == 1:
            x = x[0]
        for k in x:
            if k not in new:
                new.append(k)
        return new

    # train data construction

    train_data = defaultdict(list)
    train_logits = defaultdict(list)
    train_cls_indexes = defaultdict(list)
    train_target_col_mask = defaultdict(list)
    train_embs = defaultdict(list)
    train_target_embs = defaultdict(list)
    train_col_num = defaultdict(list)
    train_label = defaultdict(list)
    train_class = defaultdict(list)
    train_extra_mask = defaultdict(list)
    train_hard_mask = {}

    model.load_state_dict(best_state_dict, strict=False)
    model.eval()
    model = model.to(device)
    change_log = []
    score_init = []
    score_best = [] 
    start_idx = 0
    interval = 12000
    for threshold in [1.5]:
        print(f"*********************Training, Depth: {depth}****************************")
        ft_embs_test = []
        labels_test = []
        logits_test = []
        log = defaultdict(list)
        num_cols = []
        corrected = 0
        total_mistakes = 0
        num_permutations = {}
        init_permutation = {}
        init_correctness = {}
        score_init = {}
        score_permutation = defaultdict(list)
        permutation_correctness = defaultdict(list)
        with torch.no_grad():
            for batch_idx, batch in enumerate(train_dataloader_iter):
                cls_indexes = torch.LongTensor([[0, batch["cls_indexes"].cpu().item()]]).to(device)
                target_col_mask = batch["target_col_mask"].T
                label_i = batch["label"].reshape(-1).cpu()
                init_permutation_i = get_permutation(target_col_mask)
                num_permutations[batch_idx] = 0
                

                num_cols.append(len(init_permutation_i))
                labels_test.append(label_i)
                col_idx_set = target_col_mask.unique().tolist()
                assert -1 not in col_idx_set
                for r in range(len(init_permutation_i),  0, -1):
                    for x in itertools.combinations(init_permutation_i, r):
                        if 0 not in x:
                            continue

                        num_permutations[batch_idx] += 1
                        new_batch_data = []
                        new_target_col_mask = []
                        for col_i in x:
                            if col_i == 0:
                                if len(new_batch_data) == 0:
                                    cls_indexes_value = 0
                                else:
                                    cls_indexes_value = sum([len(new_batch_data[i]) for i in range(len(new_batch_data))])
                            new_batch_data.append(batch["data"].T[target_col_mask==col_i])
                            new_target_col_mask.append(target_col_mask[target_col_mask==col_i])
                        new_target_col_mask = torch.cat(new_target_col_mask, dim=-1)
                        col_mask= (new_target_col_mask == 0).long().to(device)
                        new_batch_data = torch.cat(new_batch_data, dim=-1).reshape(1, -1).to(device)
                        cls_indexes = torch.tensor([0, cls_indexes_value]).reshape(1, -1).to(device)
                        logits_temp, embs_temp = model(new_batch_data, cls_indexes=cls_indexes, get_enc=True, token_type_ids=col_mask)
                        
                        ood_score_temp = F.softmax(logits_temp.detach()).max().item()
                        score_permutation[batch_idx].append(ood_score_temp)
                        predict_temp = logits_temp.argmax().item()
                        permutation_correctness[batch_idx].append(predict_temp == label_i)
                        # v3: add more neg examples
                        if predict_temp == label_i and len(x) < max(len(init_permutation_i)-depth, 1):
                            continue
                        if predict_temp != label_i and len(x) <= max(len(init_permutation_i)-depth, 1):
                            train_extra_mask[batch_idx].append(1)
                        else:
                            train_extra_mask[batch_idx].append(0)
                        
                        
                        train_data[batch_idx].append(new_batch_data.cpu())
                        train_logits[batch_idx].append(logits_temp.detach().cpu())
                        train_cls_indexes[batch_idx].append(cls_indexes_value)
                        train_embs[batch_idx].append(embs_temp.cpu())
                        train_col_num[batch_idx].append(len(x))
                        train_label[batch_idx].append(torch.tensor(predict_temp == label_i).long()) # indicate whether the permutation is correct or not
                        train_class[batch_idx].append(label_i)
                if True not in permutation_correctness[batch_idx] or False not in permutation_correctness[batch_idx]:
                    train_hard_mask[batch_idx] = False
                else:
                    train_hard_mask[batch_idx] = True                      
                if batch_idx % 1000 == 0:
                    print(f"{batch_idx}/{len(train_dataloader_iter)}")
                    
    train_embs_pos = []
    train_embs_neg = []
    for batch_idx in train_embs:
        for i in range(len(train_embs[batch_idx])):
            if train_label[batch_idx][i].item() == 1:
                train_embs_pos.append(train_embs[batch_idx][i])
            else:
                train_embs_neg.append(train_embs[batch_idx][i])
    train_embs_pos = torch.stack(train_embs_pos, dim=0)
    train_embs_neg = torch.stack(train_embs_neg, dim=0)
    print("Train Data Statistics", len(train_embs_pos)+len(train_embs_neg), len(train_embs_pos)/(len(train_embs_pos)+len(train_embs_neg)))
                    


    veri_data = defaultdict(list)
    veri_logits = defaultdict(list)
    veri_cls_indexes = defaultdict(list)
    veri_target_col_mask = defaultdict(list)
    veri_embs = defaultdict(list)
    veri_target_embs = defaultdict(list)
    veri_col_num = defaultdict(list)
    veri_label = defaultdict(list)
    veri_class = defaultdict(list)
    veri_extra_mask = defaultdict(list)
    veri_hard_mask = {}

    model.load_state_dict(best_state_dict, strict=False)
    model.eval()
    model = model.to(device)
    change_log = []
    score_init = []
    score_best = [] 
    start_idx = 0
    interval = 12000
    for threshold in [1.5]:
        print(f"*********************Veri Data, Depth: {depth}****************************")
        ft_embs_test = []
        labels_test = []
        logits_test = []
        log = defaultdict(list)
        num_cols = []
        corrected = 0
        total_mistakes = 0
        num_permutations = {}
        init_permutation = {}
        init_correctness = {}
        score_init = {}
        score_permutation = defaultdict(list)
        permutation_correctness = defaultdict(list)
        with torch.no_grad():
            for batch_idx, batch in enumerate(veri_dataloader_iter):
                cls_indexes = torch.LongTensor([[0, batch["cls_indexes"].cpu().item()]]).to(device)
                target_col_mask = batch["target_col_mask"].T
                label_i = batch["label"].reshape(-1).cpu()
                init_permutation_i = get_permutation(target_col_mask)
                num_permutations[batch_idx] = 0
                

                num_cols.append(len(init_permutation_i))
                labels_test.append(label_i)
                col_idx_set = target_col_mask.unique().tolist()
                assert -1 not in col_idx_set
                for r in range(len(init_permutation_i),  0, -1):
                    for x in itertools.combinations(init_permutation_i, r):
                        if 0 not in x:
                            continue

                        num_permutations[batch_idx] += 1
                        new_batch_data = []
                        new_target_col_mask = []
                        for col_i in x:
                            if col_i == 0:
                                if len(new_batch_data) == 0:
                                    cls_indexes_value = 0
                                else:
                                    cls_indexes_value = sum([len(new_batch_data[i]) for i in range(len(new_batch_data))])
                            new_batch_data.append(batch["data"].T[target_col_mask==col_i])
                            new_target_col_mask.append(target_col_mask[target_col_mask==col_i])
                        new_target_col_mask = torch.cat(new_target_col_mask, dim=-1)
                        col_mask= (new_target_col_mask == 0).long().to(device)
                        new_batch_data = torch.cat(new_batch_data, dim=-1).reshape(1, -1).to(device)
                        cls_indexes = torch.tensor([0, cls_indexes_value]).reshape(1, -1).to(device)
                        logits_temp, embs_temp = model(new_batch_data, cls_indexes=cls_indexes, get_enc=True, token_type_ids=col_mask)
                        
                        ood_score_temp = F.softmax(logits_temp.detach()).max().item()
                        score_permutation[batch_idx].append(ood_score_temp)
                        predict_temp = logits_temp.argmax().item()
                        permutation_correctness[batch_idx].append(predict_temp == label_i)
                        # v3: add more neg examples
                        if predict_temp == label_i and len(x) < max(len(init_permutation_i)-depth, 1):
                            continue
                        if predict_temp != label_i and len(x) <= max(len(init_permutation_i)-depth, 1):
                            veri_extra_mask[batch_idx].append(1)
                        else:
                            veri_extra_mask[batch_idx].append(0)
                        
                        
                        veri_data[batch_idx].append(new_batch_data.cpu())
                        veri_logits[batch_idx].append(logits_temp.detach().cpu())
                        veri_cls_indexes[batch_idx].append(cls_indexes_value)
                        veri_embs[batch_idx].append(embs_temp.cpu())
                        veri_col_num[batch_idx].append(len(x))
                        veri_label[batch_idx].append(torch.tensor(predict_temp == label_i).long()) # indicate whether the permutation is correct or not
                        veri_class[batch_idx].append(label_i)
                if True not in permutation_correctness[batch_idx] or False not in permutation_correctness[batch_idx]:
                    veri_hard_mask[batch_idx] = False
                else:
                    veri_hard_mask[batch_idx] = True                      
                if batch_idx % 1000 == 0:
                    print(f"{batch_idx}/{len(veri_dataloader_iter)}")
                    

    veri_embs_pos = []
    veri_embs_neg = []
    num_hard = 0
    for batch_idx in range(len(veri_embs)):
        num_hard += 1
        for i in range(len(veri_embs[batch_idx])):
            if veri_label[batch_idx][i].item() == 1:
                veri_embs_pos.append(veri_embs[batch_idx][i])
            else:
                veri_embs_neg.append(veri_embs[batch_idx][i])
    veri_embs_pos = torch.stack(veri_embs_pos, dim=0)
    veri_embs_neg = torch.stack(veri_embs_neg, dim=0)
    print("Veri Data Statistics", (len(veri_embs_neg)+len(veri_embs_pos)), len(veri_embs_pos)/(len(veri_embs_neg)+len(veri_embs_pos)))




    train_embs_temp = torch.cat([train_embs_pos, train_embs_neg], dim=0)
    train_labels_temp = torch.cat([torch.ones(len(train_embs_pos)), torch.zeros(len(train_embs_neg))], dim=0)
    veri_embs_temp = torch.cat([veri_embs_pos, veri_embs_neg], dim=0)
    veri_labels_temp = torch.cat([torch.ones(len(veri_embs_pos)), torch.zeros(len(veri_embs_neg))], dim=0)
    veri_embs_save = torch.cat([train_embs_temp, veri_embs_temp], dim=0)
    veri_labels_save = torch.cat([train_labels_temp, veri_labels_temp], dim=0)
    print("Total Training Statistics", len(veri_embs_save), veri_labels_save.sum().item()/len(veri_labels_save))
    torch.save({ "label": veri_labels_save, "embs": veri_embs_save}, f"./data/{args.task}/veri_data_{args.cv}.pth")







    test_data = defaultdict(list)
    test_logits = defaultdict(list)
    test_cls_indexes = defaultdict(list)
    test_target_col_mask = defaultdict(list)
    test_embs = defaultdict(list)
    test_target_embs = defaultdict(list)
    test_col_num = defaultdict(list)
    test_label = defaultdict(list)
    test_class = defaultdict(list)
    test_extra_mask = defaultdict(list)

    small_tag=small_tag
    model.eval()
    model = model.to(device)
    change_log = []
    score_init = []
    score_best = [] 
    # start_idx = 24000
    # interval = 12000
    for threshold in [1.5]:
        print(f"*********************Test Data, Depth: {depth}****************************")
        ft_embs_test = []
        labels_test = []
        logits_test = []
        log = defaultdict(list)
        num_cols = []
        corrected = 0
        total_mistakes = 0
        num_permutations = {}
        init_permutation = {}
        init_correctness = {}
        score_init = {}
        score_permutation = defaultdict(list)
        permutation_correctness = defaultdict(list)
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_dataloader_iter):
                cls_indexes = torch.LongTensor([[0, batch["cls_indexes"].cpu().item()]]).to(device)
                target_col_mask = batch["target_col_mask"].T
                label_i = batch["label"].reshape(-1).cpu()
                init_permutation_i = get_permutation(target_col_mask)
                num_permutations[batch_idx] = 0
                

                num_cols.append(len(init_permutation_i))
                labels_test.append(label_i)
                col_idx_set = target_col_mask.unique().tolist()
                assert -1 not in col_idx_set
                for r in range(len(init_permutation_i),  max(len(init_permutation_i)-depth, 1)-1, -1):
                    for x in itertools.combinations(init_permutation_i, r):
                        if 0 not in x:
                            continue

                        num_permutations[batch_idx] += 1
                        new_batch_data = []
                        new_target_col_mask = []
                        for col_i in x:
                            if col_i == 0:
                                if len(new_batch_data) == 0:
                                    cls_indexes_value = 0
                                else:
                                    cls_indexes_value = sum([len(new_batch_data[i]) for i in range(len(new_batch_data))])
                            new_batch_data.append(batch["data"].T[target_col_mask==col_i])
                            new_target_col_mask.append(target_col_mask[target_col_mask==col_i])
                        new_target_col_mask = torch.cat(new_target_col_mask, dim=-1)
                        col_mask= (new_target_col_mask == 0).long().to(device)
                        new_batch_data = torch.cat(new_batch_data, dim=-1).reshape(1, -1).to(device)
                        cls_indexes = torch.tensor([0, cls_indexes_value]).reshape(1, -1).to(device)
                        logits_temp, embs_temp = model(new_batch_data, cls_indexes=cls_indexes, get_enc=True, token_type_ids=col_mask)
                        
                        ood_score_temp = F.softmax(logits_temp.detach()).max().item()
                        score_permutation[batch_idx].append(ood_score_temp)
                        predict_temp = logits_temp.argmax().item()
                        permutation_correctness[batch_idx].append(predict_temp == label_i)

                        
                        test_data[batch_idx].append(new_batch_data.cpu())
                        test_logits[batch_idx].append(logits_temp.detach().cpu())
                        test_cls_indexes[batch_idx].append(cls_indexes_value)
                        test_embs[batch_idx].append(embs_temp.cpu())
                        test_col_num[batch_idx].append(len(x))
                        test_label[batch_idx].append(torch.tensor(predict_temp == label_i).long()) # indicate whether the permutation is correct or not
                        test_class[batch_idx].append(label_i)                
                if batch_idx % 1000 == 0:
                    print(f"{batch_idx}/{len(test_dataloader_iter)}")
                    
                    
                    
    valid_data = defaultdict(list)
    valid_logits = defaultdict(list)
    valid_cls_indexes = defaultdict(list)
    valid_target_col_mask = defaultdict(list)
    valid_embs = defaultdict(list)
    valid_target_embs = defaultdict(list)
    valid_col_num = defaultdict(list)
    valid_label = defaultdict(list)
    valid_class = defaultdict(list)
    valid_extra_mask = defaultdict(list)


    model.eval()
    model = model.to(device)
    change_log = []
    score_init = []
    score_best = [] 
    # start_idx = 24000
    # interval = 12000
    for threshold in [1.5]:
        print(f"*********************Valid Data, Depth: {depth}****************************")
        ft_embs_test = []
        labels_test = []
        logits_test = []
        log = defaultdict(list)
        num_cols = []
        corrected = 0
        total_mistakes = 0
        num_permutations = {}
        init_permutation = {}
        init_correctness = {}
        score_init = {}
        score_permutation = defaultdict(list)
        permutation_correctness = defaultdict(list)
        with torch.no_grad():
            for batch_idx, batch in enumerate(valid_dataloader_iter):
                cls_indexes = torch.LongTensor([[0, batch["cls_indexes"].cpu().item()]]).to(device)
                target_col_mask = batch["target_col_mask"].T
                label_i = batch["label"].reshape(-1).cpu()
                init_permutation_i = get_permutation(target_col_mask)
                num_permutations[batch_idx] = 0
                

                num_cols.append(len(init_permutation_i))
                labels_test.append(label_i)
                col_idx_set = target_col_mask.unique().tolist()
                assert -1 not in col_idx_set
                for r in range(len(init_permutation_i),  max(len(init_permutation_i)-depth, 1)-1, -1):
                    for x in itertools.combinations(init_permutation_i, r):
                        if 0 not in x:
                            continue

                        num_permutations[batch_idx] += 1
                        new_batch_data = []
                        new_target_col_mask = []
                        for col_i in x:
                            if col_i == 0:
                                if len(new_batch_data) == 0:
                                    cls_indexes_value = 0
                                else:
                                    cls_indexes_value = sum([len(new_batch_data[i]) for i in range(len(new_batch_data))])
                            new_batch_data.append(batch["data"].T[target_col_mask==col_i])
                            new_target_col_mask.append(target_col_mask[target_col_mask==col_i])
                        new_target_col_mask = torch.cat(new_target_col_mask, dim=-1)
                        col_mask= (new_target_col_mask == 0).long().to(device)
                        new_batch_data = torch.cat(new_batch_data, dim=-1).reshape(1, -1).to(device)
                        cls_indexes = torch.tensor([0, cls_indexes_value]).reshape(1, -1).to(device)
                        logits_temp, embs_temp = model(new_batch_data, cls_indexes=cls_indexes, get_enc=True, token_type_ids=col_mask)
                        
                        ood_score_temp = F.softmax(logits_temp.detach()).max().item()
                        score_permutation[batch_idx].append(ood_score_temp)
                        predict_temp = logits_temp.argmax().item()
                        permutation_correctness[batch_idx].append(predict_temp == label_i)

                        
                        valid_data[batch_idx].append(new_batch_data.cpu())
                        valid_logits[batch_idx].append(logits_temp.detach().cpu())
                        valid_cls_indexes[batch_idx].append(cls_indexes_value)
                        valid_embs[batch_idx].append(embs_temp.cpu())
                        valid_col_num[batch_idx].append(len(x))
                        valid_label[batch_idx].append(torch.tensor(predict_temp == label_i).long()) # indicate whether the permutation is correct or not
                        valid_class[batch_idx].append(label_i)                
                if batch_idx % 1000 == 0:
                    print(f"{batch_idx}/{len(valid_dataloader_iter)}")
                    
                    
    import os
    os.makedirs(f"./data/{args.task}", exist_ok=True)
    torch.save({ "logits": test_logits, "cls_indexes": test_cls_indexes, 
                "embs": test_embs,  "col_num": test_col_num, "label": test_label, "class": test_class}, f"./data/{args.task}/test_data_{args.cv}.pth")
    import os
    os.makedirs(f"./data/{args.task}", exist_ok=True)
    torch.save({ "logits": valid_logits, "cls_indexes": valid_cls_indexes, 
                "embs": valid_embs,  "col_num": valid_col_num, "label": valid_label, "class": valid_class}, f"./data/{args.task}/valid_data_{args.cv}.pth")