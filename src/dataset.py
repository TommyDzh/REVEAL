
import torch
import pandas as pd
import os
import pickle
from transformers import AutoTokenizer

from functools import reduce
import operator
from .preprocessor import computeTfIdf, tfidfRowSample, preprocess, load_jsonl




from sentence_transformers import SentenceTransformer, util

from functools import reduce
import operator
from torch.utils.data import Dataset
import pickle


# map lm name to huggingface's pre-trained model names
lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased',
         'bert': 'bert-base-uncased'}

def collate_fn(samples):
    data = torch.nn.utils.rnn.pad_sequence(
        [sample["data"] for sample in samples])
    label = torch.cat([sample["label"] for sample in samples])
    batch = {"data": data, "label": label}
    if "idx" in samples[0]:
        # For debug purpose
        batch["idx"] = torch.cat([sample["idx"] for sample in samples])
    return batch


def maximal_marginal_relevance(query_embedding, answer_embeddings, top_k=3, lambda_param=0.7):
    """
    Perform Maximal Marginal Relevance (MMR) using PyTorch.
    
    Args:
    - query_embedding (Tensor): Embedding of the query, shape (d,).
    - answer_embeddings (Tensor): Embeddings of the candidate answers, shape (n, d).
    - top_k (int): Number of diverse answers to retrieve.
    - lambda_param (float): Trade-off parameter between relevance and diversity.
    
    Returns:
    - selected_indices (list): Indices of the selected answers.
    """
    # Initialize selected and remaining indices
    selected_indices = []
    remaining_indices = list(range(answer_embeddings.size(0)))

    # Compute relevance scores (similarity with the query)
    query_similarities = util.cos_sim(query_embedding, answer_embeddings)[0]

    for _ in range(top_k):
        if not remaining_indices:
            break

        mmr_scores = []
        for idx in remaining_indices:
            # Relevance score
            relevance_score = query_similarities[idx]

            # Diversity score (max similarity to already selected answers)
            if selected_indices:
                diversity_score = max(util.cos_sim(answer_embeddings[idx], answer_embeddings[selected_indices])[0])
            else:
                diversity_score = torch.tensor(0.0)

            # Compute MMR score
            mmr_score = lambda_param * relevance_score - (1 - lambda_param) * diversity_score
            mmr_scores.append(mmr_score)

        # Select the answer with the highest MMR score
        mmr_scores = torch.stack(mmr_scores)
        best_idx = remaining_indices[torch.argmax(mmr_scores)]
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

    return selected_indices



class GittablesTablewiseIterateMMRDataset(Dataset):

    def __init__(
            self,
            cv: int,
            split: str,
            tokenizer: AutoTokenizer,
            max_length: int = 256,
            gt_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "",
            small_tag: str = "",
            train_ratio: float = 1.0,
            max_unlabeled=8,
            adaptive_max_length=True,
            train_only=False,
            lbd=0.5,): # TODO
        if device is None:
            device = torch.device('cpu')
        basename = small_tag+ "_cv_{}.csv"
        encoded_df_path = os.path.join(base_dirpath, f"encoded_gittables_{split}_ml@{max_length}_unlabel@{max_unlabeled}_lbd@{lbd}_mmr.pkl")
        print(encoded_df_path)
        if os.path.exists(encoded_df_path):
            print(f"Loading already processed {split} dataset from {encoded_df_path}")
            with open(encoded_df_path, "rb") as f:
                df_dict = pickle.load(f)
            self.table_df = df_dict
        else:
            sb_model = SentenceTransformer("all-mpnet-base-v2")
            if split in ["train", "valid"]:
                df_list = []
                for i in range(5):
                    if i == cv:
                        continue
                    filepath = os.path.join(base_dirpath, basename.format(i))
                    df_list.append(pd.read_csv(filepath))
                    print(split, i)
                df = pd.concat(df_list, axis=0)
            else:
                # test
                filepath = os.path.join(base_dirpath, basename.format(cv))
                df = pd.read_csv(filepath)
                print(split)


            if gt_only:
                df = df[df["class_id"] > -1]
            if train_only and split != "train":
                df = df[df["class_id"] > -1]

            
            data_list = []
            
            df['class_id'] = df['class_id'].astype(int)
            df.drop(df[(df['data'].isna()) & (df['class_id'] == -1)].index, inplace=True)
            df['col_idx'] = df['col_idx'].astype(int)
            df['data'] = df['data'].astype(str)
            
            num_tables = len(df.groupby("table_id"))
            valid_index = int(num_tables * 0.8)
            num_train = int(train_ratio * num_tables * 0.8)        
            
            total_num_cols = 0
            self.correctness_checklist = []
            self.max_rest_num = 0
            self.max_col_num = 0
            df_group = df.groupby("table_id")
            for i, (index, group_df) in enumerate(df_group):
                if (split == "train") and ((i >= num_train) or (i >= valid_index)):
                    break
                if split == "valid" and i < valid_index:
                    continue
                #     break
                group_df.sort_values(by=['col_idx'], inplace=True)
                group_df = group_df.reset_index(drop=True)
                if len(group_df) > max_unlabeled:
                    col_embs = torch.tensor(sb_model.encode(group_df["data"].to_list()))   
                            
                labeled_columns = group_df[group_df['class_id'] > -1]
                labeled_columns.sort_values(by=['col_idx'], inplace=True)
                for index, target_column in labeled_columns.iterrows():
                    target_col_idx = target_column["col_idx"]
                    
                    if len(group_df) > max_unlabeled:
                        other_columns = group_df.drop(index)
                        selected_index =  maximal_marginal_relevance(col_embs[index], col_embs[torch.arange(len(col_embs)) != index], top_k=max_unlabeled-1, lambda_param=lbd)
                        assert len(set(selected_index)) == len(selected_index)
                        other_columns = other_columns.iloc[selected_index]
                        assert len(other_columns) == max_unlabeled - 1
                    else:
                        other_columns = group_df.drop(index)
                    target_table = pd.concat([target_column.to_frame().T, other_columns], ignore_index=True)
                    

                    
                    target_cls = target_column["class_id"]
                    target_table.sort_values(by=['col_idx'], inplace=True)
                    col_idx_list = target_table["col_idx"].tolist()
                    # target_table.sort_values(by=['col_idx'], inplace=True)

                    if max_length <= 128 and adaptive_max_length:
                        cur_maxlen = min(max_length, 512 // len(target_table) - 1)
                    else:
                        cur_maxlen = max_length
                        
                    token_ids_list = target_table["data"].apply(lambda x: tokenizer.encode(
                        tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                        )
                    token_ids = torch.LongTensor(reduce(operator.add,
                                                        token_ids_list)).to(device)

                    target_col_mask = []
                    cls_index_value = 0
                    context_id = 1
                    meet_target = False
                    for idx, col_i in enumerate(col_idx_list):
                        if col_i == target_col_idx:
                            target_col_mask += [0] * len(token_ids_list[idx])
                            meet_target = True
                        else:
                            target_col_mask += [context_id] * len(token_ids_list[idx])
                            context_id += 1
                        if not meet_target:
                            cls_index_value += len(token_ids_list[idx])
                    cls_index_list = [cls_index_value] 
                    for cls_index in cls_index_list:
                        assert token_ids[
                            cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                    cls_indexes = torch.LongTensor(cls_index_list).to(device)
                    class_ids = torch.LongTensor(
                        [target_cls]).to(device)
                    target_col_mask = torch.LongTensor(target_col_mask).to(device)
                    data_list.append(
                        [index,
                        len(target_table), token_ids.cpu(), class_ids.cpu(), cls_indexes.cpu(), target_col_mask.cpu()]) 
                if i % 1000 == 0:
                    print(i, "/", len(df_group), "num samples:", len(data_list))                
            print(split, len(data_list))
            self.table_df = pd.DataFrame(data_list,
                                        columns=[
                                            "table_id", "num_col", "data_tensor",
                                            "label_tensor", "cls_indexes", "target_col_mask"
                                        ])
            with open(encoded_df_path, "wb") as f:
                pickle.dump(self.table_df, f)


    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"],
            "label": self.table_df.iloc[idx]["label_tensor"],
            "cls_indexes": self.table_df.iloc[idx]["cls_indexes"],
            "target_col_mask": self.table_df.iloc[idx]["target_col_mask"],
        }


class SotabTablewiseIterateMMRDataset(Dataset):

    def __init__(
            self,
            split: str,
            tokenizer: AutoTokenizer,
            max_length: int = 32,
            device: torch.device = None,
            base_dirpath: str = "",
            max_unlabeled=8,
            lbd=0.5): # TODO
        if device is None:
            device = torch.device('cpu')

        encoded_df_path = os.path.join(base_dirpath, f"encoded_sb_{split}_ml@{max_length}_unlabel@{max_unlabeled}_lbd@{lbd}_mmr.pkl")
        df_csv_path = os.path.join(base_dirpath, f"{split}.csv")
        if os.path.exists(encoded_df_path):
            print(f"Loading already processed {split} dataset from {encoded_df_path}")
            with open(encoded_df_path, "rb") as f:
                df_dict = pickle.load(f)
            #Load as dataframe
            self.table_df = df_dict

        else:
            from sentence_transformers import SentenceTransformer, util
            sb_model = SentenceTransformer("all-mpnet-base-v2")            
            print(f"Processing {split} dataset into {encoded_df_path}")
            if os.path.exists(df_csv_path):
                df = pd.read_csv(df_csv_path)
                print(f"Data loaded from {df_csv_path}")
            else:
                raise FileNotFoundError(f"{df_csv_path} does not exist, please check the path")

            
            data_list = []
            
            df['class_id'] = df['label'].astype(int)
            df.drop(df[(df['data'].isna()) & (df['label'] == -1)].index, inplace=True)
            df["column_index"] = df["column_index"].astype(int)
            df['data'] = df['data'].astype(str)
            
            num_tables = len(df.groupby("table_id"))
            
            df_group = df.groupby("table_id")
            self.correctness_checklist = []
            for i, (index, group_df) in enumerate(df_group):
                group_df.sort_values(by=['column_index'], inplace=True)
                group_df = group_df.reset_index(drop=True)
                if len(group_df) > max_unlabeled:
                    col_embs = torch.tensor(sb_model.encode(group_df["data"].to_list()))   
                            
                labeled_columns = group_df[group_df['class_id'] > -1]
                labeled_columns.sort_values(by=["column_index"], inplace=True)
                for index, target_column in labeled_columns.iterrows():
                    target_col_idx = target_column["column_index"]
                    
                    if len(group_df) > max_unlabeled:
                        other_columns = group_df.drop(index)
                        selected_index =  maximal_marginal_relevance(col_embs[index], col_embs[torch.arange(len(col_embs)) != index], top_k=max_unlabeled-1, lambda_param=lbd)
                        assert len(set(selected_index)) == len(selected_index)
                        other_columns = other_columns.iloc[selected_index]
                        assert len(other_columns) == max_unlabeled - 1

                    else:
                        other_columns = group_df.drop(index)
                    target_table = pd.concat([target_column.to_frame().T, other_columns], ignore_index=True)
                    
                    
                    target_cls = target_column["class_id"]
                    target_table.sort_values(by=["column_index"], inplace=True)
                    col_idx_list = target_table["column_index"].tolist()
                    # target_table.sort_values(by=["column_index"], inplace=True)

                    if max_length <= 128:
                        cur_maxlen = min(max_length, 512 // len(target_table) - 1)
                    else:
                        cur_maxlen = max_length
                        
                    token_ids_list = target_table["data"].apply(lambda x: tokenizer.encode(
                        tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                        )
                    token_ids = torch.LongTensor(reduce(operator.add,
                                                        token_ids_list)).to(device)

                    target_col_mask = []
                    cls_index_value = 0
                    context_id = 1
                    meet_target = False
                    for idx, col_i in enumerate(col_idx_list):
                        if col_i == target_col_idx:
                            target_col_mask += [0] * len(token_ids_list[idx])
                            meet_target = True
                        else:
                            target_col_mask += [context_id] * len(token_ids_list[idx])
                            context_id += 1
                        if not meet_target:
                            cls_index_value += len(token_ids_list[idx])
                    cls_index_list = [cls_index_value] 
                    for cls_index in cls_index_list:
                        assert token_ids[
                            cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                    cls_indexes = torch.LongTensor(cls_index_list).cpu()
                    class_ids = torch.LongTensor(
                        [target_cls]).cpu()
                    target_col_mask = torch.LongTensor(target_col_mask).cpu()
                    data_list.append(
                        [index,
                        len(target_table), token_ids, class_ids, cls_indexes, target_col_mask])                
                if i % 1000 == 0:
                    print(i, "/", len(df_group), "num samples:", len(data_list))
            print(split, len(data_list))
            self.table_df = pd.DataFrame(data_list,
                                        columns=[
                                            "table_id", "num_col", "data_tensor",
                                            "label_tensor", "cls_indexes", "target_col_mask"
                                        ])
            with open(encoded_df_path, 'wb') as f:
                pickle.dump(self.table_df, f)
    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"].cpu(),
            "label": self.table_df.iloc[idx]["label_tensor"].cpu(),
            "cls_indexes": self.table_df.iloc[idx]["cls_indexes"].cpu(),
            "target_col_mask": self.table_df.iloc[idx]["target_col_mask"].cpu(),
        }


class SotabRelExtIterateMMRDataset(Dataset):
    def __init__(self,
                 split: str,
                 tokenizer: AutoTokenizer,
                 max_length: int = 32,
                 device: torch.device = None,
                 max_unlabeled=8,
                 base_dirpath="",
                 lbd=0.5
                 ):
        if device is None:
            device = torch.device('cpu')

            
        encoded_df_path = os.path.join(base_dirpath, f"encoded_sb_{split}_ml@{max_length}_unlabel@{max_unlabeled}_lbd@{lbd}_mmr.pkl")
        print(encoded_df_path)
        if os.path.exists(encoded_df_path):
            print(f"Loading already processed {split} dataset")
            with open(encoded_df_path, "rb") as f:
                df_dict = pickle.load(f)
            #Load as dataframe
            self.table_df = df_dict
            
        else:
            sb_model = SentenceTransformer("all-mpnet-base-v2")               
            self.df = pd.read_csv(os.path.join(base_dirpath, f"{split}_cpa.csv"))
            self.df['data'] = self.df['data'].astype(str)

            # For learning curve
            num_tables = len(self.df.groupby("table_id"))
            num_train = int(num_tables)

            data_list = []
            for i, (index, group_df) in enumerate(self.df.groupby("table_id")):
                                
                if i >= num_train:
                    break

                # It's probably already sorted but just in case
                group_df = group_df.sort_values("column_id")
                group_df = group_df.reset_index(drop=True)
                head_column = group_df.iloc[0]
                group_df.drop(0, inplace=True) # exclude head column
                group_df = group_df.reset_index(drop=True)
                labeled_columns = group_df[group_df['label'] > -1]
                if len(group_df) > max_unlabeled:
                    col_embs = torch.tensor(sb_model.encode(group_df["data"].to_list()))   

                
                
                
                for index, target_column in labeled_columns.iterrows():
                    
                    target_col_idx = target_column['column_id']
                    target_cls = target_column['label']
                    if len(group_df) > max_unlabeled:
                        other_columns = group_df.drop(index)
                        selected_index =  maximal_marginal_relevance(col_embs[index], col_embs[torch.arange(len(col_embs)) != index], top_k=max_unlabeled-1, lambda_param=lbd)
                        assert len(set(selected_index)) == len(selected_index)
                        other_columns = other_columns.iloc[selected_index]
                        assert len(other_columns) == max_unlabeled - 1                        
                    else:
                        other_columns = group_df.drop(index)
                    target_df = pd.concat([head_column.to_frame().T, target_column.to_frame().T, other_columns], ignore_index=True)                    
                    
                    
                    target_df.sort_values(by=['column_id'], inplace=True)
                    target_df.reset_index(drop=True, inplace=True)
                    col_idx_list = target_df['column_id'].tolist()


                    cur_maxlen = min(max_length, 512 // len(target_df) - 1)
                    token_ids_list = target_df["data"].apply(lambda x: tokenizer.encode(
                                        tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                                        )
                        
                    token_ids = torch.LongTensor(reduce(operator.add,
                                                        token_ids_list)).to(device)
                    
                    target_col_mask = [] # 1 for head column, 0 for target column, 2+ for other columns
                    cls_index_value = 0
                    context_id = 1
                    meet_target = False
                    for idx, col_i in enumerate(col_idx_list):
                        if col_i == target_col_idx:
                            target_col_mask += [0] * len(token_ids_list[idx])
                            meet_target = True
                        else:
                            target_col_mask += [context_id] * len(token_ids_list[idx])
                            context_id += 1
                        if not meet_target:
                            cls_index_value += len(token_ids_list[idx])                    
                    
                    
                    cls_index_list = [cls_index_value] 
                    for cls_index in cls_index_list:
                        assert token_ids[
                            cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                    cls_indexes = torch.LongTensor(cls_index_list).cpu()
                        
                        
                    class_ids = torch.LongTensor(
                        [target_cls]).cpu()
                    target_col_mask = torch.LongTensor(target_col_mask).cpu()
                    data_list.append(
                        [index,
                        len(target_df), token_ids.cpu(), class_ids.cpu(), cls_indexes.cpu(), target_col_mask.cpu()])    
                if i % 1000 == 0:
                    print(f"Processed {i}/{num_tables} tables")
            self.table_df = pd.DataFrame(data_list,
                                        columns=[
                                            "table_id", "num_col", "data_tensor",
                                            "label_tensor", "cls_indexes", "target_col_mask"
                                        ])
            with open(encoded_df_path, 'wb') as f:
                pickle.dump(self.table_df, f)

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"].cpu(),
            "label": self.table_df.iloc[idx]["label_tensor"].cpu(),
            "cls_indexes": self.table_df.iloc[idx]["cls_indexes"].cpu(),
            "target_col_mask": self.table_df.iloc[idx]["target_col_mask"].cpu(),
        }



import ast
class TURLColTypeTablewiseIterateMMRDataset(Dataset):

    def __init__(
            self,
            split: str,
            tokenizer: AutoTokenizer,
            max_length: int = 32,
            device: torch.device = None,
            base_dirpath: str = "",
            max_unlabeled=8,
            lbd=0.5,): 
        if device is None:
            device = torch.device('cpu')

        encoded_df_path = os.path.join(base_dirpath, f"encoded_all_sb_{split}_ml@{max_length}_unlabel@{max_unlabeled}_lbd@{lbd}_mmr.pkl")

        df_csv_path = os.path.join(base_dirpath, f"turl_cta_{split}_all.csv")
        
        if os.path.exists(encoded_df_path):
            print(f"Loading already processed {split} dataset from {encoded_df_path}")
            with open(encoded_df_path, "rb") as f:
                df_dict = pickle.load(f)
            #Load as dataframe
            self.table_df = df_dict

        else:
            from sentence_transformers import SentenceTransformer, util
            sb_model = SentenceTransformer("all-mpnet-base-v2")
            if os.path.exists(df_csv_path):
                df = pd.read_csv(df_csv_path)
            else:
                raise FileNotFoundError(f"{df_csv_path} does not exist, please check the path")

            
            data_list = []
            df.drop(df[(df['data'].isna()) & (df['label'] == -1)].index, inplace=True)
            df["column_index"] = df["column_index"].astype(int)
            df['data'] = df['data'].astype(str)
            
            num_tables = len(df.groupby("table_id"))
            
            # df.drop(df[(df['data'] == '') & (df['label'] == -1)].index, inplace=True)
            df_group = df.groupby("table_id")
            self.correctness_checklist = []
            for i, (index, group_df) in enumerate(df_group):
                group_df["label"] = group_df["label"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                group_df.sort_values(by=['column_index'], inplace=True)
                group_df = group_df.reset_index(drop=True)
                if len(group_df) > max_unlabeled:
                    col_embs = torch.tensor(sb_model.encode(group_df["data"].to_list()))   
                            
                labeled_columns = group_df[(torch.tensor(group_df['label'].tolist()).sum(dim=1) > 0).tolist()]
                labeled_columns.sort_values(by=["column_index"], inplace=True)
                for index, target_column in labeled_columns.iterrows():
                    target_col_idx = target_column["column_index"]
                    
                    if len(group_df) > max_unlabeled:
                        other_columns = group_df.drop(index)
                        selected_index =  maximal_marginal_relevance(col_embs[index], col_embs[torch.arange(len(col_embs)) != index], top_k=max_unlabeled-1, lambda_param=lbd)
                        assert len(set(selected_index)) == len(selected_index)
                        other_columns = other_columns.iloc[selected_index]
                        assert len(other_columns) == max_unlabeled - 1
                    else:
                        other_columns = group_df.drop(index)
                    target_table = pd.concat([target_column.to_frame().T, other_columns], ignore_index=True)
                    
        
                    target_cls = torch.LongTensor(target_column["label"]).reshape(1, -1)
                    target_table.sort_values(by=["column_index"], inplace=True)
                    col_idx_list = target_table["column_index"].tolist()
                    # target_table.sort_values(by=["column_index"], inplace=True)

                    if max_length <= 128:
                        cur_maxlen = min(max_length, 512 // len(target_table) - 1)
                    else:
                        cur_maxlen = max_length
                        
                    token_ids_list = target_table["data"].apply(lambda x: tokenizer.encode(
                        tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                        )
                    token_ids = torch.LongTensor(reduce(operator.add,
                                                        token_ids_list))

                    target_col_mask = []
                    cls_index_value = 0
                    context_id = 1
                    meet_target = False
                    for idx, col_i in enumerate(col_idx_list):
                        if col_i == target_col_idx:
                            target_col_mask += [0] * len(token_ids_list[idx])
                            meet_target = True
                        else:
                            target_col_mask += [context_id] * len(token_ids_list[idx])
                            context_id += 1
                        if not meet_target:
                            cls_index_value += len(token_ids_list[idx])
                    cls_index_list = [cls_index_value] 
                    for cls_index in cls_index_list:
                        assert token_ids[
                            cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                    cls_indexes = torch.LongTensor(cls_index_list).cpu()
                    target_col_mask = torch.LongTensor(target_col_mask).cpu()
                    data_list.append(
                        [index,
                        len(target_table), token_ids.cpu(), target_cls.cpu(), cls_indexes.cpu(), target_col_mask.cpu()])                
                if i % 1000 == 0:
                    print(i, "/", len(df_group), "num samples:", len(data_list))
            print(split, len(data_list))
            self.table_df = pd.DataFrame(data_list,
                                        columns=[
                                            "table_id", "num_col", "data_tensor",
                                            "label_tensor", "cls_indexes", "target_col_mask"
                                        ])
            with open(encoded_df_path, 'wb') as f:
                pickle.dump(self.table_df, f)

    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"].cpu(),
            "label": self.table_df.iloc[idx]["label_tensor"].cpu(),
            "cls_indexes": self.table_df.iloc[idx]["cls_indexes"].cpu(),
            "target_col_mask": self.table_df.iloc[idx]["target_col_mask"].cpu(),
        }

class TURLRelExtTablewiseIterateMMRDataset(Dataset):

    def __init__(
            self,
            cv: int,
            split: str,
            src: str,  # train or test
            tokenizer: AutoTokenizer,
            max_length: int = 32,
            gt_only: bool = False,
            device: torch.device = None,
            base_dirpath: str = "",
            base_tag: str = '', # blank, comma
            small_tag: str = "",
            train_ratio: float = 1.0,
            max_unlabeled=8,
            lbd=0.5,
            model=None): # TODO
        if device is None:
            device = torch.device('cpu')

        
        encoded_df_path = os.path.join(base_dirpath, f"encoded_all_sb_{split}_ml@{max_length}_unlabel@{max_unlabeled}_lbd@{lbd}_mmr.pkl")
        df_csv_path = os.path.join(base_dirpath, f"turl_cpa_{split}_all.csv")        
        if os.path.exists(encoded_df_path):
            print(f"Loading already processed {split} dataset from {encoded_df_path}")
            with open(encoded_df_path, "rb") as f:
                df_dict = pickle.load(f)
            self.table_df = df_dict

        else:
            from sentence_transformers import SentenceTransformer, util
            sb_model = SentenceTransformer("all-mpnet-base-v2")
            if os.path.exists(df_csv_path):
                df = pd.read_csv(df_csv_path)
            else:
                raise FileNotFoundError(f"{df_csv_path} does not exist, please check the path")

            
            data_list = []
            df.drop(df[(df['data'].isna()) & (df['label'] == -1)].index, inplace=True)
            df["column_index"] = df["column_index"].astype(int)
            df['data'] = df['data'].astype(str)
            
            df_group = df.groupby("table_id")
            self.correctness_checklist = []
            for i, (index, group_df) in enumerate(df_group):
                group_df["label"] = group_df["label"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
                group_df = group_df.reset_index(drop=True)
                head_column = group_df.iloc[0]
                group_df.drop(0, inplace=True) # exclude head column                
                
                group_df.sort_values(by=['column_index'], inplace=True)
                group_df = group_df.reset_index(drop=True)
                
                if len(group_df) > max_unlabeled:
                    col_embs = torch.tensor(sb_model.encode(group_df["data"].to_list()))   
                            
                labeled_columns = group_df[(torch.tensor(group_df['label'].tolist()).sum(dim=1) > 0).tolist()]
                labeled_columns.sort_values(by=["column_index"], inplace=True)
                for index, target_column in labeled_columns.iterrows():
                    target_col_idx = target_column["column_index"]
                    
                    if len(group_df) > max_unlabeled:
                        other_columns = group_df.drop(index)
                        selected_index =  maximal_marginal_relevance(col_embs[index], col_embs[torch.arange(len(col_embs)) != index], top_k=max_unlabeled-1, lambda_param=lbd)
                        assert len(set(selected_index)) == len(selected_index)
                        other_columns = other_columns.iloc[selected_index]
                        assert len(other_columns) == max_unlabeled - 1
                    else:
                        other_columns = group_df.drop(index)
                    target_table = pd.concat([head_column.to_frame().T, target_column.to_frame().T, other_columns], ignore_index=True)         
                    
        
                    target_cls = torch.LongTensor(target_column["label"]).reshape(1, -1)
                    target_table.sort_values(by=["column_index"], inplace=True)
                    col_idx_list = target_table["column_index"].tolist()


                    if max_length <= 128:
                        cur_maxlen = min(max_length, 512 // len(target_table) - 1)
                    else:
                        cur_maxlen = max(1, max_length // len(target_table)  - len(target_table))
                        
                    token_ids_list = target_table["data"].apply(lambda x: tokenizer.encode(
                        tokenizer.cls_token + " " + x, add_special_tokens=False, max_length=cur_maxlen, truncation=True)).tolist(
                        )
                    token_ids = torch.LongTensor(reduce(operator.add,
                                                        token_ids_list))

                    target_col_mask = []
                    cls_index_value = 0
                    context_id = 1
                    meet_target = False
                    for idx, col_i in enumerate(col_idx_list):
                        if col_i == target_col_idx:
                            target_col_mask += [0] * len(token_ids_list[idx])
                            meet_target = True
                        else:
                            target_col_mask += [context_id] * len(token_ids_list[idx])
                            context_id += 1
                        if not meet_target:
                            cls_index_value += len(token_ids_list[idx])
                    cls_index_list = [cls_index_value] 
                    for cls_index in cls_index_list:
                        assert token_ids[
                            cls_index] == tokenizer.cls_token_id, "cls_indexes validation"
                    cls_indexes = torch.LongTensor(cls_index_list).cpu()
                    target_col_mask = torch.LongTensor(target_col_mask).cpu()
                    data_list.append(
                        [index,
                        len(target_table), token_ids.cpu(), target_cls.cpu(), cls_indexes.cpu(), target_col_mask.cpu()])                
                if i % 1000 == 0:
                    print(i, "/", len(df_group), "num samples:", len(data_list))
            print(split, len(data_list))
            self.table_df = pd.DataFrame(data_list,
                                        columns=[
                                            "table_id", "num_col", "data_tensor",
                                            "label_tensor", "cls_indexes", "target_col_mask"
                                        ])
            with open(encoded_df_path, 'wb') as f:
                pickle.dump(self.table_df, f)
    def __len__(self):
        return len(self.table_df)

    def __getitem__(self, idx):
        return {
            "data": self.table_df.iloc[idx]["data_tensor"].cpu(),
            "label": self.table_df.iloc[idx]["label_tensor"].cpu(),
            "cls_indexes": self.table_df.iloc[idx]["cls_indexes"].cpu(),
            "target_col_mask": self.table_df.iloc[idx]["target_col_mask"].cpu(),
        }
