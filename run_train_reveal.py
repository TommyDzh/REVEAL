
import subprocess

'''run training and evaluation REVEAL on datasets
gt-semtab22-dbpedia-all: GitTablesDB
gt-semtab22-schema-property-all: GitTablesSC
sotab: SOTAB
sotab-re: SOTAB-CPA
turl: WikiTable-CTA
turl-re: WikiTable-CPA
'''

ml = 128  # gt-semtab22-dbpedia-all, gt-semtab22-schema-property-all: 128; sotab, sotab-re, turl, turl-re: 32
bs = 16 
n_epochs = 50 # gt-semtab22-dbpedia-all, gt-semtab22-schema-property-all: 50; sotab, sotab-re,  turl-re: 30; turl: 20
base_model = 'bert-base-uncased'
cl_tag = "None"
ckpt_path = "./model/"
dropout_prob = 0.1
from_scratch = True
eval_test = True
colpair = False # Set True for sotab-re (SOTAB-CPA) and turl-re (WikiTable-CPA)


gpus = 0
max_unlabeled = 8
small_tag = ""

version = "v1"
repeat=5
seed=0

comment = "{}-Repeat@{}-max-unlabeled@{}".format(version, repeat, max_unlabeled)
for task in ["gt-semtab22-dbpedia-all"]:
    cmd = '''python train_reveal.py  --reset_pooler \
              --gpu {}  --shortcut_name {} --task {} --pool_version {} --repeat {}  --use_attention_mask True --max_length {} --random_sample True --max_unlabeled {} --batch_size {} --epoch {} \
                --dropout_prob {} --pretrained_ckpt_path "{}" --cl_tag {} --small_tag "{}" --comment "{}" {} {} {}'''.format(
        gpus, base_model, task, version, repeat,  ml, max_unlabeled, bs, n_epochs, dropout_prob,
        ckpt_path, cl_tag, small_tag, comment,
        '--colpair' if colpair else '',
        '--from_scratch' if from_scratch else '',        
        '--eval_test' if eval_test else ''
    )   
    # os.system('{} & '.format(cmd))
    subprocess.run(cmd, shell=True, check=True)
