import argparse
import json
import math
import os
import random
from time import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from collections import defaultdict
from copy import deepcopy
# import pytrec_eval
import torch
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from accelerate import Accelerator


torch.backends.cuda.matmul.allow_tf32 = True

from src.dataset import (
    GittablesTablewiseIterateMMRDataset, # GitTablesDB, GitTablesSC
    SotabTablewiseIterateMMRDataset, # SOTAB-CTA
    SotabRelExtIterateMMRDataset, # SOTAB-CPA
    TURLRelExtTablewiseIterateMMRDataset, # WikiTable-CTA
    TURLColTypeTablewiseIterateMMRDataset, # WikiTable-CPA
)

from src.model import BertMultiPooler, BertMultiPairPooler, BertForMultiOutputClassification
from src.model import lm_mp
from src.utils import load_checkpoint, f1_score_multilabel, collate_fn, get_col_pred
from src.utils import task_num_class_dict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--model", type=str, default="REVEAL")
    parser.add_argument("--use_attention_mask", type=bool, default=True)
    parser.add_argument("--unlabeled_train_only", type=bool, default=False)
    parser.add_argument("--pool_version", type=str, default="v1")
    parser.add_argument("--random_sample", type=bool, default=True)
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
        "--max_unlabeled",
        default=8,
        type=int,
    )   

    parser.add_argument(
        "--batch_size",
        default=16,
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
    parser.add_argument("--multi_gpu",
                        action="store_true",
                        default=False,
                        help="Use multiple GPU")
    parser.add_argument("--fp16",
                        action="store_true",
                        default=False,
                        help="Use FP16")
    parser.add_argument("--warmup",
                        type=float,
                        default=0.,
                        help="Warmup ratio")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate") # 1e-5, 2e-5
    parser.add_argument("--task",
                        type=str,
                        default='sotab',
                        choices=[
                            "sotab", "sotab-re", 
                            "gt-semtab22-dbpedia-all", 
                            "gt-semtab22-schema-property-all",
                            "turl", "turl-re", 
                        ],
                        help="Task names}")
    parser.add_argument("--colpair",
                        action="store_true",
                        help="Use column pair embedding")
    parser.add_argument("--metadata",
                        action="store_true",
                        help="Use column header metadata")
    parser.add_argument("--from_scratch",
                        default=True, 
                        action="store_true",
                        help="Training from scratch")
    parser.add_argument("--reset_pooler",
                        action="store_true",)
    parser.add_argument("--cl_tag",
                        type=str,
                        default="",
                        help="path to the pre-trained file")
    parser.add_argument("--dropout_prob",
                        type=float,
                        default=0.5)
    parser.add_argument("--eval_test",
                        action="store_true",
                        help="evaluate on testset and do not save the model file")
    parser.add_argument("--small_tag",
                        type=str,
                        default="",
                        help="e.g., by_table_t5_v1")
    parser.add_argument("--data_path",
                        type=str,
                        default="./data")
    parser.add_argument("--pretrained_ckpt_path",
                        type=str,
                        default="./model")    

    args = parser.parse_args()
    task = args.task
    if args.small_tag != "":
        args.eval_test = True
    
    args.num_classes = task_num_class_dict[task]
    if args.colpair:
        assert task in ["turl-re", "sotab-re"], "colpair can be only used for Relation Extraction"
    if args.metadata:
        assert "turl-re" == task or "turl" == task, "metadata can be only used for TURL datasets"


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
            tag_name = "{}/{}-{}-{}-pool{}-unlabeled{}-rand{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname,  "{}-fromscratch".format(shortcut_name), args.small_tag, args.comment,args.pool_version, args.max_unlabeled, args.random_sample,
                batch_size, max_length, num_train_epochs, args.dropout_prob, 
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')
        else:
            tag_name = "{}/{}-{}-{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname,  "{}-fromscratch".format(shortcut_name), args.small_tag, args.comment,
                batch_size, max_length, num_train_epochs, args.dropout_prob, 
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')
        
    else:
        if "gt" in task:
            tag_name = "{}/{}_{}-{}-pool{}-unlabeled{}-rand{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname, args.cl_tag.replace('/', '-'),  shortcut_name, args.small_tag, args.comment, args.pool_version, args.max_unlabeled, args.random_sample,
                batch_size, max_length, num_train_epochs, args.dropout_prob,
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')
        else:
            tag_name = "{}/{}_{}-{}-{}-bs{}-ml{}-ne{}-do{}{}".format(
                taskname, args.cl_tag.replace('/', '-'),  shortcut_name, args.small_tag, args.comment,
                batch_size, max_length, num_train_epochs, args.dropout_prob,
                '-rs{}'.format(args.random_seed) if args.random_seed != 4649 else '')


    print(tag_name)
    file_path = os.path.join("./model", args.model, "outputs", tag_name)

    dirpath = os.path.dirname(file_path)
    if not os.path.exists(dirpath):
        print("{} not exists. Created".format(dirpath))
        os.makedirs(dirpath)
    
    if args.fp16:
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
        
      
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')

    setattr(args, 'batch_size', args.batch_size)
    setattr(args, 'hidden_dropout_prob', args.dropout_prob)
    setattr(args, 'shortcut_name', args.shortcut_name)
    setattr(args, 'num_labels', args.num_classes)
    setattr(args, 'projector', 768) # BERT hidden size
    set_seed(args.random_seed)
    
    
    tokenizer = BertTokenizer.from_pretrained(shortcut_name)
    ts_micro_f1_all = defaultdict(list)
    ts_macro_f1_all = defaultdict(list)
    test_time = []
    valid_columns = ["trial", "best_vl_micro_f1", "best_vl_macro_f1", "best_vl_loss", 
                     "best_vl_micro_f1s_epoch", "best_vl_macro_f1s_epoch", "training_time_epoch"]
    test_columns = ["macro:macro_f1", "macro:micro_f1", "micro:macro_f1", "micro:micro_f1"]
    # ==============================repeat loop start================================
    for repeat_i in range(args.repeat):
        print("Starting loop", repeat_i)
        model = BertForMultiOutputClassification(args, device=device, 
                                                lm="bert", 
                                                use_attention_mask=args.use_attention_mask)
            


        if not args.reset_pooler and args.pool_version != 'v0':
            raise ValueError("pooler version must be v0 when not resetting")
        if args.reset_pooler:
            model.bert.pooler = BertMultiPooler(args.projector).to(device)
            print("Reset pooler layer")

        if (task == "sotab-re" or task == "turl-re") and args.colpair:
            model.bert.pooler = BertMultiPairPooler(args.projector).to(device)
            print("Use new token_type_embeddings for column-pair pooling")
        padder = collate_fn(tokenizer.pad_token_id)

        if "sotab" in task.lower():
            if "re" in task:
                dataset_cls = SotabRelExtIterateMMRDataset
            else:
                dataset_cls = SotabTablewiseIterateMMRDataset
            print("start load train dataset")
            train_dataset = dataset_cls(split="train",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        device=device,
                                        max_unlabeled=args.max_unlabeled,
                                        base_dirpath=os.path.join(args.data_path, args.task),
                                        lbd=0.5,
                                        )
            print("start load valid dataset")
            valid_dataset = dataset_cls(
                                        split="valid",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        device=device,
                                        max_unlabeled=args.max_unlabeled,
                                        base_dirpath=os.path.join(args.data_path, args.task),
                                        lbd=0.5,
                                        )

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                            sampler=train_sampler,
                                            batch_size=batch_size,
                                        collate_fn=padder)
            valid_dataloader = DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                        collate_fn=padder)
            print("start load test dataset")
            test_dataset = dataset_cls(
                                        split="test",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        device=device,
                                        max_unlabeled=args.max_unlabeled,
                                        base_dirpath=os.path.join(args.data_path, args.task),
                                        lbd=0.5,
                                        )
            test_dataloader = DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            collate_fn=padder)   
            print("dataset loaded")
            
        elif "gt-dbpedia" in task or "gt-schema" in task or "gt-semtab" in task:
            if 'dbpedia' in task:
                small_tag = 'db_semi1'
                if 'semtab22' in task:
                    src = 'dbpedia_property' # gt-dbpedia-semtab22-all0
                else:
                    src = 'dbpedia'
            else:
                small_tag = 'sp_semi1'
                if 'semtab22' in task:
                    if 'schema-property' in task:
                        src = 'schema_property'
                    else:
                        src = 'schema_class'
                else:
                    src = 'schema'

            dataset_cls = GittablesTablewiseIterateMMRDataset
            train_dataset = dataset_cls(cv=repeat_i,
                                        split="train",
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        gt_only='all' not in task,
                                        device=device,
                                        base_dirpath=os.path.join(args.data_path, args.task),
                                        small_tag=small_tag,
                                        max_unlabeled=args.max_unlabeled,
                                        lbd=0.5)
        
            valid_dataset = dataset_cls(cv=repeat_i,
                                        split="valid", 
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        gt_only='all' not in task,
                                        device=device,
                                        base_dirpath=os.path.join(args.data_path, args.task),
                                        small_tag=small_tag,
                                        max_unlabeled=args.max_unlabeled,
                                        lbd=0.5
                                        )

            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                            sampler=train_sampler,
                                            batch_size=batch_size,
                                        collate_fn=padder)
            valid_dataloader = DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                        collate_fn=padder)
            test_dataset = dataset_cls(cv=repeat_i,
                                        split="test", 
                                        tokenizer=tokenizer,
                                        max_length=max_length,
                                        gt_only='all' not in task or args.unlabeled_train_only,
                                        device=device,
                                        base_dirpath=os.path.join(args.data_path, args.task),
                                        small_tag=small_tag,
                                        max_unlabeled=args.max_unlabeled,
                                        lbd=0.5)
            test_dataloader = DataLoader(test_dataset,
                                            batch_size=batch_size//2,
                                            collate_fn=padder)    
                        
        elif "turl" in task:
            if task in ["turl"]:
                dataset_cls = TURLColTypeTablewiseIterateMMRDataset
            elif task in ["turl-re"]:
                dataset_cls = TURLRelExtTablewiseIterateMMRDataset
            else:
                raise ValueError("turl tasks must be turl or turl-re.")

            train_dataset = dataset_cls(
                                        split="train",
                            tokenizer=tokenizer,
                            max_length=max_length,
                            base_dirpath=os.path.join(args.data_path, args.task),
                            device=device,
                            max_unlabeled=args.max_unlabeled,
                            lbd=0.5)
            valid_dataset = dataset_cls(
                                        split="valid",
                            tokenizer=tokenizer,
                            max_length=max_length,
                            base_dirpath=os.path.join(args.data_path, args.task),
                            device=device,
                            lbd=0.5)

            # Can be the same
            train_sampler = RandomSampler(train_dataset)
            train_dataloader = DataLoader(train_dataset,
                                            sampler=train_sampler,
                                            batch_size=batch_size,
                                            collate_fn=padder)
            valid_dataloader = DataLoader(valid_dataset,
                                            batch_size=batch_size,
                                            collate_fn=padder)
            test_dataset = dataset_cls(
                                        split="test",
                            tokenizer=tokenizer,
                            max_length=max_length,
                            base_dirpath=os.path.join(args.data_path, args.task),
                            device=device,
                            max_unlabeled=args.max_unlabeled,
                            lbd=0.5)

            test_dataloader = DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            collate_fn=padder)
        else:
            raise ValueError("task name must be either sato or turl.")

        t_total = len(train_dataloader) * num_train_epochs
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay":
                0.0
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=5e-5, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=t_total)

        if "gt" in task or "sotab" in task:
            loss_fn = CrossEntropyLoss()
        elif "turl" in task:
            loss_fn = BCEWithLogitsLoss()
        else:
            raise ValueError("task name must be either sato or turl.")
        
        

        model = model.to(device)
        # model = model.cuda()
        # Best validation score could be zero
        best_vl_micro_f1 = -1
        best_vl_macro_f1 = -1
        best_vl_loss = 1e10
        best_vl_micro_f1s_epoch = -1
        best_vl_macro_f1s_epoch = -1
        best_vl_loss_epoch = -1
        best_model_dict = {}
        loss_info_list = []
        eval_dict = defaultdict(lambda: defaultdict(list))
        time_epochs = []
        # =============================Training Loop=============================
        for epoch in range(num_train_epochs):
            t1 = time()
            print("Epoch", epoch, "starts")
            model.train()
            tr_loss = 0.
            if "col-popl" in task:
                tr_pred_list = {}
                tr_true_list = {}
                tr_logits_list = {}
                vl_pred_list = {}
                vl_true_list = {}
            else:
                tr_pred_list = []
                tr_logits_list = []
                tr_true_list = []
                vl_pred_list = []
                vl_true_list = []

            vl_loss = 0.
            for batch_idx, batch in enumerate(train_dataloader):
                batch["data"] = batch["data"].to(device)
                batch["label"] = batch["label"].to(device)
                
                cls_indexes = batch["cls_indexes"].reshape(-1).to(device)
                cls_indexes = torch.stack([torch.arange(len(cls_indexes), device=device), cls_indexes], dim=1)
                
                if "col-popl" in task:
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
                    all_preds = get_col_pred(logits, labels, batch["idx"], top_k=-1)#.cpu().detach().numpy()
                    tr_pred_list.update(all_preds)
                    loss = loss_fn(logits, labels_1d)
                else:
                    if "re" in task:
                        col_mask= (batch["target_col_mask"].T < 2).long().to(device)
                    else:
                        col_mask= (batch["target_col_mask"].T == 0).long().to(device)
                    logits = model(batch["data"].T, cls_indexes=cls_indexes, token_type_ids=col_mask)
                    if  "gt-" in task or "sotab" in task:
                        if task == "sotab-re":
                            target_mask = batch["label"] != -1
                            labels_1d = batch["label"][target_mask]
                            new_logits = logits[target_mask]
                            
                            
                            tr_pred_list += new_logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            tr_true_list += labels_1d.long().cpu().detach().numpy().tolist()     
                            tr_logits_list += new_logits.cpu().detach().numpy().tolist()
                        
                            loss = loss_fn(new_logits, labels_1d)
                        else:
                            tr_pred_list += logits.argmax(
                                1).cpu().detach().numpy().tolist()
                            tr_true_list += batch["label"].cpu().detach().numpy().tolist()
                            tr_logits_list += logits.cpu().detach().numpy().tolist()
                            loss = loss_fn(logits, batch["label"])
                    elif "turl" in task:
                        target_mask = batch["label"].sum(1) > 0
                        batch["label"] = batch["label"][target_mask]
                        logits = logits[target_mask]    
                        if task == "turl-re":
                            all_preds = (logits >= 0.0
                                        ).int().detach().cpu().numpy()
                            all_labels = batch["label"].cpu().detach().numpy()
                            # Ignore the very first CLS token
                            tr_pred_list += all_preds.tolist()
                            tr_true_list += all_labels.tolist()
                        elif task == "turl":                      
                            tr_pred_list += (logits >= 0
                                            ).int().detach().cpu().tolist()                            
                            tr_true_list += batch["label"].cpu().detach(
                            ).numpy().tolist()                       
                        
                        loss = loss_fn(logits, batch["label"].float())

                loss.backward()
                tr_loss += loss.cpu().detach().item()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            tr_loss /= (len(train_dataset) / batch_size)


            if "gt-" in task or "sotab" in task:
                tr_micro_f1 = f1_score(tr_true_list,
                                        tr_pred_list,
                                        average="micro")
                tr_macro_f1 = f1_score(tr_true_list,
                                        tr_pred_list,
                                        average="macro")
                tr_class_f1 = f1_score(tr_true_list,
                                        tr_pred_list,
                                        average=None,
                                        labels=np.arange(args.num_classes))
            elif "turl" in task and "popl" not in task:
                tr_micro_f1, tr_macro_f1, tr_class_f1, _ = f1_score_multilabel(
                    tr_true_list, tr_pred_list)

            # ======================= Validation =======================
            model.eval()
            with torch.no_grad():
                for batch_idx, batch in enumerate(valid_dataloader):
                    batch["data"] = batch["data"].to(device)
                    batch["label"] = batch["label"].to(device)
                    cls_indexes = batch["cls_indexes"].reshape(-1).to(device)
                    cls_indexes = torch.stack([torch.arange(len(cls_indexes), device=device), cls_indexes], dim=1)
                    if "re" in task:
                        col_mask= (batch["target_col_mask"].T < 2).long().to(device)
                    else:
                        col_mask= (batch["target_col_mask"].T == 0).long().to(device)
                    if "col-popl" in task:
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
                        vl_pred_list.update(all_preds)
                        loss = loss_fn(logits, labels_1d)
                    else:
                        logits = model(batch["data"].T, cls_indexes=cls_indexes, token_type_ids=col_mask)
                        if "sato" in task or "gt-" in task or "sotab" in task: 
                            if task == "sotab-re":
                                target_mask = batch["label"] != -1
                                labels_1d = batch["label"][target_mask]
                                new_logits = logits[target_mask]
                                
                                
                                vl_pred_list += new_logits.argmax(
                                    1).cpu().detach().numpy().tolist()
                                vl_true_list += labels_1d.long().cpu().detach().numpy().tolist()       
                            
                                loss = loss_fn(new_logits, labels_1d)
                            else:
                                vl_pred_list += logits.argmax(
                                    1).cpu().detach().numpy().tolist()
                                vl_true_list += batch["label"].cpu().detach().numpy().tolist()
                                loss = loss_fn(logits, batch["label"])

                        elif "turl" in task:
                            target_mask = batch["label"].sum(1) > 0
                            batch["label"] = batch["label"][target_mask]
                            logits = logits[target_mask]                                
                            if task == "turl-re":
                                all_preds = (logits >= 0.0
                                            ).int().detach().cpu().numpy()
                                all_labels = batch["label"].cpu().detach().numpy()
                                vl_pred_list += all_preds.tolist()
                                vl_true_list += all_labels.tolist()
                            elif task == "turl":
                                vl_pred_list += (logits >= 0
                                                ).int().detach().cpu().tolist()
                                vl_true_list += batch["label"].cpu().detach(
                                ).numpy().tolist()
                            loss = loss_fn(logits, batch["label"].float())

                    vl_loss += loss.cpu().detach().item()

            vl_loss /= (len(valid_dataset) / batch_size)
            if "gt-" in task or "sotab" in task:
                vl_micro_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average="micro")
                vl_macro_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average="macro")
                vl_class_f1 = f1_score(vl_true_list,
                                        vl_pred_list,
                                        average=None,
                                        labels=np.arange(args.num_classes))
            elif "turl" in task:
                vl_micro_f1, vl_macro_f1, vl_class_f1, _ = f1_score_multilabel(
                    vl_true_list, vl_pred_list)
            
            t2 = time()
            if vl_micro_f1 > best_vl_micro_f1:
                best_vl_micro_f1 = vl_micro_f1
                model_savepath = "{}_best_f1_micro.pt".format(file_path)
                best_model_dict["f1_micro"] = deepcopy(model.state_dict())
                best_vl_micro_f1s_epoch = epoch
            if vl_macro_f1 > best_vl_macro_f1:
                best_vl_macro_f1 = vl_macro_f1
                model_savepath = "{}_best_f1_macro.pt".format(file_path)
                best_model_dict["f1_macro"] = deepcopy(model.state_dict())
                best_vl_macro_f1s_epoch = epoch
            if best_vl_loss > vl_loss:
                best_vl_loss = vl_loss
                model_savepath = "{}_best_loss.pt".format(file_path)
                best_model_dict["loss"] = deepcopy(model.state_dict())
                best_vl_loss_epoch = epoch
            loss_info_list.append([
                tr_loss, tr_macro_f1, tr_micro_f1, vl_loss, vl_macro_f1,
                vl_micro_f1
            ])
            time_epoch = t2-t1
            time_epochs.append(time_epoch)
            print(
                "Epoch {} ({}): tr_loss={:.7f} tr_macro_f1={:.4f} tr_micro_f1={:.4f} "
                .format(epoch, task, tr_loss, tr_macro_f1, tr_micro_f1),
                "vl_loss={:.7f} vl_macro_f1={:.4f} vl_micro_f1={:.4f} ({:.2f} sec.)"
                .format(vl_loss, vl_macro_f1, vl_micro_f1, time_epoch))

        # log train results
        if type(tr_class_f1) != list:
            tr_class_f1 = tr_class_f1.tolist()  
        eval_dict["train"][f"tr_class_f1"].append(tr_class_f1)
        eval_dict["train"][f"tr_macro_f1"].append(tr_macro_f1)
        eval_dict["train"][f"tr_micro_f1"].append(tr_micro_f1)
        eval_dict["train"][f"tr_loss"].append(tr_loss)
        eval_dict["train"][f"tr_pred_list"].append(tr_pred_list)
        eval_dict["train"][f"tr_true_list"].append(tr_true_list)
        eval_dict["train"][f"tr_logits_list"].append(tr_logits_list)          
                
    # ======================= Test =======================
        print("Test starts")
        model_savepath = "{}_best_last_{}.pt".format(file_path,  repeat_i)
        torch.save(model.state_dict(), model_savepath)
        for f1_name in ["f1_macro", "f1_micro"]:
            model_savepath = "{}_best_{}_{}.pt".format(file_path, f1_name, repeat_i)
            torch.save(best_model_dict[f1_name], model_savepath)
            model.load_state_dict(best_model_dict[f1_name])
            model.eval()
            if "popl" in task:
                ts_pred_list = {}
                ts_true_list = {}
                ts_logits_list = {}
            else:
                ts_pred_list = []
                ts_true_list = []
                ts_logits_list = []
            t1 = time()
            # Test
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_dataloader):
                    batch["data"] = batch["data"].to(device)
                    batch["label"] = batch["label"].to(device)
                    cls_indexes = batch["cls_indexes"].reshape(-1).to(device)
                    cls_indexes = torch.stack([torch.arange(len(cls_indexes), device=device), cls_indexes], dim=1)
                    if "re" in task:
                        col_mask= (batch["target_col_mask"].T < 2).long().to(device)
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
                            target_mask = batch["label"].sum(1) > 0
                            batch["label"] = batch["label"][target_mask]
                            logits = logits[target_mask.cpu()]                                
                            if "turl-re" in task:  # turl-re-colpair
                                all_preds = (logits >= 0.0
                                            ).int().detach().cpu().numpy()
                                all_labels = batch["label"].cpu().detach().numpy()
                                ts_pred_list += all_preds.tolist()
                                ts_true_list += all_labels.tolist()
                            elif task == "turl":
                                # ts_pred_list += (logits >= math.log(0.5)
                                #                 ).int().detach().cpu().tolist()
                                ts_pred_list += (logits >= 0
                                                ).int().detach().cpu().tolist()                            
                                ts_true_list += batch["label"].cpu().detach(
                                ).numpy().tolist()
            t2 = time()
            test_time.append(t2-t1)
            if "sato" in task or "gt-" in task or "sotab" in task:
                ts_micro_f1 = f1_score(ts_true_list,
                                    ts_pred_list,
                                    average="micro")
                ts_macro_f1 = f1_score(ts_true_list,
                                    ts_pred_list,
                                    average="macro")
                ts_class_f1 = f1_score(ts_true_list,
                                    ts_pred_list,
                                    average=None,
                                    labels=np.arange(args.num_classes))
                ts_conf_mat = confusion_matrix(ts_true_list,
                                            ts_pred_list,
                                            labels=np.arange(args.num_classes))
            elif "turl" in task:
                ts_micro_f1, ts_macro_f1, ts_class_f1, ts_conf_mat = f1_score_multilabel(
                    ts_true_list, ts_pred_list)

            # test results
            if type(ts_class_f1) != list:
                ts_class_f1 = ts_class_f1.tolist()    
            if type(ts_conf_mat) != list:
                ts_conf_mat = ts_conf_mat.tolist()    
            eval_dict[f1_name][f"ts_micro_f1"].append(ts_micro_f1)
            eval_dict[f1_name][f"ts_macro_f1"].append(ts_macro_f1)
            eval_dict[f1_name][f"ts_class_f1"].append(ts_class_f1)
            eval_dict[f1_name][f"ts_conf_mat"].append(ts_conf_mat)
            eval_dict[f1_name][f"ts_true_list"].append(ts_true_list)
            eval_dict[f1_name][f"ts_pred_list"].append(ts_pred_list)
            eval_dict[f1_name][f"ts_logits_list"].append(ts_logits_list)
            torch.cuda.empty_cache()
    output_filepath = "{}_eval.json".format(file_path)
    with open(output_filepath, "w") as fout:
        json.dump(eval_dict, fout)
