import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import torch
import pandas as pd
import itertools
import os
import subprocess

#function to create onnx model for any checkpoint into a directory
# zs_chkpt = "facebook/bart-large-mnli"
# zs_model_path = 'zs_mdl_onnx'

def create_onnx_model_zs_nli(zs_chkpt,zs_onnx_mdl_dir):
    try:
        subprocess.run(['python3', '-m', 'transformers.onnx',
                    f'--model={zs_chkpt}',
                    '--feature=sequence-classification',
                    '--atol=1e-3',
                    zs_onnx_mdl_dir])
    except Exception as e:
            print(e)

#pipeline class
class ZsPipeline:

    def __init__(self,premise_list,labels,onnx_mdl,tokenizer,hypothesis,):

        self.premise_list=premise_list
        self.labels=labels

        self.tokenizer = tokenizer
        self.hypothesis = hypothesis
        self.model=onnx_mdl

        self.sequence_pairs=self.preprocess(self.premise_list,self.labels,self.hypothesis)
        self.tokenized_inputs=self.tokenize_input(self.sequence_pairs)
        self.model_input=self.make_onnx_input()
        self.model_output=self.forward()

    #combine premise & hypotheis and collect them in a list
    def preprocess(self,premise_list,labels,hypothesis):
        sequence_pairs=[]
        for premise in premise_list:
            sequence_pairs.extend([[premise,f"{hypothesis} {label}"] for label in labels])
        return sequence_pairs

    #tokenize the output from prerocess step
    def tokenize_input(self,sequence_pairs):
        tokenized_inputs=self.tokenizer(sequence_pairs,padding=True, truncation=True,return_tensors="pt")
        return tokenized_inputs

    #create input feed for the onnx model , each chkpt has different input feed
    def make_onnx_input(self):
        input_feed = {
            'input_ids': np.array(self.tokenized_inputs['input_ids']),
            'attention_mask': np.array(self.tokenized_inputs['attention_mask'])
        }
        return input_feed

    #get the logits after passing the output of make_onnx_input to onnx model
    def forward(self):
        output_logits=self.model.run(output_names=['logits'], input_feed=dict(self.model_input))[0]
        output_probs=torch.from_numpy(output_logits)[:,[0,2]].softmax(dim=1)
        return np.array(output_probs[:,1])

    #output in relevant format
    def make_zs_classification(self):
        labels=self.labels
        premise_list=self.premise_list
        model_output=self.model_output

        x=list(itertools.product(premise_list,labels))
        df = pd.DataFrame(x, columns=['sequence', 'labels'])
        df['proba'] = model_output
        df = pd.pivot_table(data=df, columns='labels', index='sequence', values='proba', aggfunc='first')
        df = df.reset_index()
        df['sum_'] = df[labels].sum(axis=1)
        for l in labels:
            df[l] = np.round(100 * df[l] / df['sum_'], 1)

        df['score'] = df[labels].max(axis=1)
        df['label_classified'] = df[labels].idxmax(axis=1)
        df = df[['sequence', 'score', 'label_classified']]
        return df


