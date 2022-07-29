from transformers import AutoTokenizer,AutoModel,AutoConfig

#make sure pretrained model name is pytorch_model.bin

chkpt='microsoft/deberta-v3-large'
model_dir='feedback-prize-mlm-model/'
model_hub='deberta_v3_large_mlm_feedback_prize'

modelConfig=AutoConfig.from_pretrained(chkpt)
modelConfig.save_pretrained(model_dir)

model = AutoModel.from_pretrained(model_dir)
tokenizer=AutoTokenizer.from_pretrained(chkpt)

model.push_to_hub(model_hub)
tokenizer.push_to_hub(model_hub)