import argparse
import json
import math
import os
import random
from time import time
# import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from collections import defaultdict

# import pytrec_eval
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup



torch.backends.cuda.matmul.allow_tf32 = True


from accelerate import Accelerator, DistributedDataParallelKwargs
from src.dataset import  VerificationBinaryFastDataset
from src.model import VerifierMulti, VerifierSep, BertMultiPairPooler, BertForMultiOutputClassification
from src.model import lm_mp
from src.utils import f1_score_multilabel,  veri_collate_fn
from src.utils import task_num_class_dict
import warnings
warnings.filterwarnings("ignore")

import itertools
from copy import deepcopy

import torch
import torch.nn.functional as F

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    

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



if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Watchog")
    parser.add_argument("--loss", type=str, default="CE")
    parser.add_argument("--reg_weight", type=float, default=1.0)
    parser.add_argument("--norm", type=str, default="batch_norm") 
    parser.add_argument("--num_layers", type=int, default=None) 
    parser.add_argument("--max_list_length", type=int, default=10)
    parser.add_argument("--veri_module", type=str, default="ffn") 
    parser.add_argument("--context", type=str, default=None) 
    parser.add_argument("--data_version", type=str, default="mmr0.5_dp1_context_ln_hard") 

    parser.add_argument("--test_version", type=str, default="mmr0.5_dp1_context_ln") # None: only drop up to 2/half columns; 2: drop up to 1 columns
    parser.add_argument("--fast_eval", type=bool, default=False) 
    
    parser.add_argument("--use_attention_mask", type=bool, default=True)
    parser.add_argument("--unlabeled_train_only", type=bool, default=True)
    parser.add_argument("--pool_version", type=str, default="v0")
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
        default=128,
        type=int,
        help=
        "The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_unlabeled",
        default=0,
        type=int,
    )   

    parser.add_argument(
        "--batch_size",
        default=512,
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
        "--warmup_ratio",
        default=0.0,
        type=float,
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
                        default="sotab",
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
    parser.add_argument("--eval_interval",
                        type=int,
                        default=1)
    parser.add_argument("--small_tag",
                        type=str,
                        default="")
    parser.add_argument("--data_path",
                        type=str,
                        default="./data")
    parser.add_argument("--pretrained_ckpt_path",
                        type=str,
                        default="./model")    

    args = parser.parse_args()
    task = args.task

    
    args.num_classes = task_num_class_dict[task]
    if args.colpair:
        assert "turl-re" == task, "colpair can be only used for Relation Extraction"
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
        
      
        
    # accelerator = Accelerator(mixed_precision="no" if not args.fp16 else "fp16")   
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="no" if not args.fp16 else "fp16", kwargs_handlers=[ddp_kwargs])

    device = accelerator.device
    setattr(args, 'batch_size', args.batch_size)
    setattr(args, 'hidden_dropout_prob', args.dropout_prob)
    setattr(args, 'shortcut_name', args.shortcut_name)
    setattr(args, 'num_labels', args.num_classes)
    setattr(args, 'projector', 768) # BERT hidden size
    set_seed(args.random_seed)
    
    
    tokenizer = BertTokenizer.from_pretrained(shortcut_name)
    model = BertForMultiOutputClassification(args, device=device, 
                                            lm="bert", 
                                            use_attention_mask=args.use_attention_mask)

    if (task == "sotab-re" or task == "turl-re") and args.colpair:
        model.bert.pooler = BertMultiPairPooler(args.projector).to(device)
        print("Use new token_type_embeddings for column-pair pooling")



    
        
    with accelerator.main_process_first():
        src = None
        if args.data_version is not None or args.data_version != "None":
            veri_dataset = VerificationBinaryFastDataset(data_path=f"./data/{args.task}/veri_data_{args.data_version}.pth", pos_ratio= None, context=args.context)
        else:
            veri_dataset = VerificationBinaryFastDataset(data_path=f"./data/{args.task}/veri_data.pth", pos_ratio= None, context=args.context)

        veri_padder = veri_collate_fn(0, binary=True)
        veri_dataloader = DataLoader(
            veri_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=veri_padder, drop_last=True,
        )
        

        if args.test_version is None or args.test_version == "None":
            test_dataset = torch.load(f"./data/{args.task}/test_data.pth")
            valid_dataset = torch.load(f"./data/{args.task}/valid_data.pth")
        else:
            test_dataset = torch.load(f"./data/{args.task}/test_data_{args.test_version}.pth")
            valid_dataset = torch.load(f"./data/{args.task}/valid_data_{args.test_version}.pth")
        test_embs = test_dataset['embs']
        test_logits = test_dataset['logits']
        test_class = test_dataset['class']
        test_embs_target = test_dataset['target_embs'] if 'target_embs' in test_dataset else None
        test_col_num = test_dataset['col_num']
        
        valid_embs = valid_dataset['embs']
        valid_logits = valid_dataset['logits']
        valid_class = valid_dataset['class']
        valid_embs_target = valid_dataset['target_embs'] if 'target_embs' in valid_dataset else None
            


    verifier = VerifierMulti(module=args.veri_module, dropout=args.dropout_prob, norm=args.norm, num_layers=args.num_layers).to(device)

    t_total = len(veri_dataloader) * num_train_epochs
    optimizer = AdamW(verifier.parameters(), lr=args.lr, eps=1e-8)
    scheduler = get_cosine_with_min_lr_schedule_with_warmup(optimizer,
                                                num_warmup_steps=int(t_total*args.warmup_ratio),
                                                num_training_steps=t_total,
                                                min_lr=args.lr/100)


    if args.loss == "CE":
        loss_fn = torch.nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"loss function {args.loss} not implemented")
    
    
    model, verifier, optimizer, veri_dataloader, scheduler = accelerator.prepare(
        model, verifier, optimizer, veri_dataloader, scheduler
    )

    model = model.to(device)
    verifier = verifier.to(device)

    best_vl_micro_f1 = -1
    best_vl_macro_f1 = -1
    best_vl_loss = 1e10
    best_vl_micro_f1s_epoch = -1
    best_vl_macro_f1s_epoch = -1
    best_vl_loss_epoch = -1
    loss_info_list = []
    eval_dict = defaultdict(dict)
    time_epochs = []

    for epoch in range(num_train_epochs):
        if args.fast_eval and epoch > 50:
            continue
        t1 = time()
        print("Epoch", epoch, "starts")
        model.eval()
        verifier.train()
        tr_loss = 0.
        tr_reg_loss = 0.
        tr_bce_loss = 0.
        device = accelerator.device
        num_samples = 0
        reg_loss_list = []
        for batch_idx, batch in enumerate(veri_dataloader):

            embs = batch["embs"].to(device)

            scores, embeddings = verifier(embs, return_embs=True)

            num_samples += len(scores)

            labels = batch["label"].to(device).squeeze().float()
            bce_loss = loss_fn(scores, labels.long())                
            
            
            loss = bce_loss 
                
            accelerator.backward(loss)
            tr_loss += loss.cpu().detach().item()

            tr_bce_loss += bce_loss.cpu().detach().item()

            optimizer.step()
            current_lr = scheduler.get_lr()[-1]
            scheduler.step()
            optimizer.zero_grad()
            
        tr_loss /= num_samples
        tr_bce_loss /= num_samples
        t2 = time()
        time_epoch_train = t2-t1
        

        # ======================= Validation =======================
        if (epoch+1) % args.eval_interval == 0 or epoch > num_train_epochs//2:
            model.eval()
            verifier.eval()
            with accelerator.main_process_first():
                device = accelerator.device
                labels_valid = []
                logits_valid = []
                with torch.no_grad():
                    for batch_idx, batch in enumerate(valid_embs):
                        embs = valid_embs[batch_idx][0].reshape(-1).to(device)
                        logits = valid_logits[batch_idx][0].reshape(-1).to(device)
                        col_nums = torch.tensor(valid_col_num[batch_idx])
                        # scores_init = verifier(embs)
                        max_score = -float("inf")
                        embs_temp_all = torch.stack(valid_embs[batch_idx]).to(device)
                        logits_temp_all = torch.stack(valid_logits[batch_idx]).to(device)
                        if len(embs_temp_all.shape) == 3:
                            embs_temp_all = embs_temp_all.squeeze(1)
                        if len(logits_temp_all.shape) == 3:
                            logits_temp_all = logits_temp_all.squeeze(1)        
                        count = 0
                        for col_num in col_nums.unique():
                            mask = col_nums == col_num
                            embs_temp = embs_temp_all[mask]
                            logits_temp = logits_temp_all[mask]
                            scores_temp = F.softmax(verifier(embs_temp))[:,1].detach().cpu()

                            max_index = scores_temp.argmax().item()
                            max_score_current = scores_temp.max().item()
                            if max_score_current > max_score:
                                max_score = max_score_current
                                logits = logits_temp[max_index]
                            else:
                                break  
                            # if len(x) == 1 and 0 in x:
                            #     predict_target = predict_temp
                            #     msp_target = msp_temp
                            # # print(x, msp_temp, predict_temp)
                            # if 0 not in x and msp_temp > debias_threshold and (predict_temp != predict_target):
                            #     debias_classes.append(predict_temp)
                            #     continue
                        logits_valid.append(logits.detach().cpu())   
                        if "turl" in args.task:
                            labels_valid.append(valid_class[batch_idx][0].reshape(1,-1))
                        else:
                            labels_valid.append(torch.tensor([valid_class[batch_idx][0]]))
                                                     
                labels_valid = torch.cat(labels_valid, dim=0)
                logits_valid = torch.stack(logits_valid, dim=0)

                if "turl" in args.task:
                    preds_valid = (logits_valid>= 0.5
                                    ).int().detach().cpu()
                    tv_pred_list = preds_valid.tolist()
                    tv_true_list = labels_valid.tolist()
                    
                    tv_micro_f1, tv_macro_f1, tv_class_f1, tv_conf_mat = f1_score_multilabel(
                        tv_true_list, tv_pred_list) 
                else:
                    from sklearn.metrics import confusion_matrix, f1_score
                    tv_pred_list = logits_valid.argmax(
                                                1).cpu().detach().numpy().tolist()
                    tv_micro_f1 = f1_score(labels_valid.reshape(-1).numpy().tolist(),
                                        tv_pred_list,
                                        average="micro")
                    tv_macro_f1 = f1_score(labels_valid.reshape(-1).numpy().tolist(),
                                        tv_pred_list,
                                        average="macro")                
                
                
                labels_test = []
                logits_test = []

                with torch.no_grad():
                    for batch_idx, batch in enumerate(test_embs):
                        embs = test_embs[batch_idx][0].reshape(-1).to(device)
                        logits = test_logits[batch_idx][0].reshape(-1).to(device)
                        col_nums = torch.tensor(test_col_num[batch_idx])
                        # scores_init = verifier(embs)
                        max_score = -float("inf")
                        embs_temp_all = torch.stack(test_embs[batch_idx]).to(device)
                        logits_temp_all = torch.stack(test_logits[batch_idx]).to(device)
                        if len(embs_temp_all.shape) == 3:
                            embs_temp_all = embs_temp_all.squeeze(1)
                        if len(logits_temp_all.shape) == 3:
                            logits_temp_all = logits_temp_all.squeeze(1)        
                        count = 0
                        for col_num in col_nums.unique():
                            mask = col_nums == col_num
                            embs_temp = embs_temp_all[mask]
                            logits_temp = logits_temp_all[mask]
                            scores_temp = F.softmax(verifier(embs_temp))[:,1].detach().cpu()

                            max_index = scores_temp.argmax().item()
                            max_score_current = scores_temp.max().item()
                            if max_score_current > max_score:
                                max_score = max_score_current
                                logits = logits_temp[max_index]
                            else: # early stop
                                break    
                        logits_test.append(logits.detach().cpu())
                        if "turl" in args.task:
                            labels_test.append(test_class[batch_idx][0].reshape(1,-1))
                        else:
                            labels_test.append(torch.tensor([test_class[batch_idx][0]]))

                labels_test = torch.cat(labels_test, dim=0)
                logits_test = torch.stack(logits_test, dim=0)



                if "turl" in args.task:
                    preds_test = (logits_test>= 0.5
                                    ).int().detach().cpu()
                    ts_pred_list = preds_test.tolist()
                    ts_true_list = labels_test.tolist()
                    
                    ts_micro_f1, ts_macro_f1, ts_class_f1, ts_conf_mat = f1_score_multilabel(
                        ts_true_list, ts_pred_list)      
                else:
                    from sklearn.metrics import confusion_matrix, f1_score
                    ts_pred_list = logits_test.argmax(
                                                1).cpu().detach().numpy().tolist()
                    ts_micro_f1 = f1_score(labels_test.reshape(-1).numpy().tolist(),
                                        ts_pred_list,
                                        average="micro")
                    ts_macro_f1 = f1_score(labels_test.reshape(-1).numpy().tolist(),
                                        ts_pred_list,
                                        average="macro")

  
                t3 = time()
                time_epoch_test = t3-t2
                print(
                    "Epoch {} ({}): tr_loss={:.7f} lr={:.7f} ({:.2f} sec.)"
                    .format(epoch, task, tr_loss, current_lr, time_epoch_train),
                    "ts_macro_f1={:.4f} ts_micro_f1={:.4f} ({:.2f} sec.)"
                    .format(ts_macro_f1, ts_micro_f1, time_epoch_test))
    
                if tv_macro_f1 > best_vl_macro_f1:
                    best_vl_macro_f1 = tv_macro_f1
                    best_vl_macro_f1s_epoch = epoch
                    best_vl_macro_f1_macro = ts_macro_f1
                    best_vl_macro_f1_micro = ts_micro_f1
                    best_state_dict_macro = deepcopy(verifier.state_dict())
                    
                if tv_micro_f1 > best_vl_micro_f1:
                    best_vl_micro_f1 = tv_micro_f1
                    best_vl_micro_f1s_epoch = epoch
                    best_vl_micro_f1_macro = ts_macro_f1
                    best_vl_micro_f1_micro = ts_micro_f1
                    best_state_dict_micro = deepcopy(verifier.state_dict())

    if accelerator.is_local_main_process : 
        torch.save(best_state_dict_micro, "{}_verifier_binary_best_f1_micro.pt".format(file_path))    
        torch.save(best_state_dict_macro, "{}_verifier_binary_best_f1_macro.pt".format(file_path))     
        torch.save(verifier.state_dict(), "{}_verifier_binary_last.pt".format(file_path))
