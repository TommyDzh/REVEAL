import os
import subprocess
import time
import pickle
from multiprocessing import Process
from multiprocessing import Semaphore

'''run training and evaluation REVEAL+ on datasets
gt-semtab22-dbpedia-all: GitTablesDB
gt-semtab22-schema-property-all: GitTablesSC
sotab: SOTAB
sotab-re: SOTAB-CPA
turl: WikiTable-CTA
turl-re: WikiTable-CPA
'''
bs = 64 # gt-semtab22-dbpedia-all, gt-semtab22-schema-property-all: 64; others: 512
n_epochs = 100
base_model = 'bert-base-uncased'
cl_tag = "None"
ckpt_path = "None"

from_scratch = True
eval_test = True
colpair = False # Set True for sotab-re (SOTAB-CPA) and turl-re (WikiTable-CPA)

warmup_ratio = 0.0

random_seed = 0
max_unlabeled = 8
gpus = '0'
dropout_prob = 0.0
norm = "batch_norm"
context = None
veri_module = "ffn_fix" # gt-semtab22-dbpedia-all, gt-semtab22-schema-property-all: "ffn_fix"; others: "ffn_fix_3"
for lr in [5e-5, 1e-4, 5e-4, 1e-3]:
    for task in ['gt-semtab22-dbpedia-all']: 
        comment = "CV1-Seed@{}-mode@{}-context@{}-lr@{}-dp@{}-norm@{}".format(random_seed,  veri_module, context,  lr, warmup_ratio, dropout_prob, norm, )
        cmd = '''CUDA_VISIBLE_DEVICES={} python supcl_ft_verifier_multi_random1_contra_fast_sotab_valid.py --wandb True  \
                    --shortcut_name {} --random_seed {}  --warmup_ratio {} --norm {}  --veri_module {} --context {} --reweight True --task {}  --use_attention_mask True --lr {} --max_unlabeled {} --batch_size {} --epoch {} \
                    --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --comment "{}" {} {} {}'''.format(
            gpus, base_model, random_seed, warmup_ratio, norm,  veri_module, context, task,  lr, max_unlabeled, bs, n_epochs, dropout_prob,
            ckpt_path, cl_tag, comment,
            '--colpair' if colpair else '',
            '--from_scratch' if from_scratch else '',        
            '--eval_test' if eval_test else ''
        )   
        # os.system('{} & '.format(cmd))
        subprocess.run(cmd, shell=True, check=True)

        
