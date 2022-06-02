import pandas as pd
import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

from transformers import AutoTokenizer,AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments,EarlyStoppingCallback

#read data
df = pd.read_csv("IMDB Dataset.csv",  engine="python",error_bad_lines=False)
df['sentiment']=df['sentiment'].map({'positive':1,'negative':0})

#load tokenizer and model
chkpt='distilbert-base-uncased'
tokenizer=AutoTokenizer.from_pretrained(chkpt)
model=AutoModelForSequenceClassification.from_pretrained(chkpt,num_labels=2)

#Prep data to convert into torch dataset
X=df['review'].tolist()
y=df['sentiment'].tolist()

X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.2)

X_train_tokenized=tokenizer(X_train,padding=True,truncation=True,max_length=512)
X_val_tokenized=tokenizer(X_train,padding=True,truncation=True,max_length=512)

# The Trainer API requires the model to be in a torch.utils.data.Dataset class.
# Hence, we would need to create a new class that inherits from the torch Dataset class.

# Also, In the inherited class, we need to have the __getitem__and __len__ method
# which allows Trainer to create batches of data and to obtain the length respectively.

# The purpose of setting the default labels parameter as None
# is so that we can reuse the class to make prediction on unseen data as these data do not have labels.

class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


train_dataset = TorchDataset(X_train_tokenized, y_train)
test_dataset = TorchDataset(X_val_tokenized, y_val)

# print(len(train_dataset))
# print(train_dataset[1])

#compute metrics for Trainer
def compute_metrics(p):
  logits,labels=p
  pred=np.argmax(logits,axis=-1)

  accuracy = accuracy_score(y_true=labels, y_pred=pred)
  recall = recall_score(y_true=labels, y_pred=pred)
  precision = precision_score(y_true=labels, y_pred=pred)
  f1 = f1_score(y_true=labels, y_pred=pred)

  return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

#creating custom trainer with our desired loss
#similarly other functions could be subclassed
class CustomTrainer(Trainer):
  def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # compute custom loss (suppose one has 2 labels with different weights)
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 0.999]))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

#creating training Arguments
args=TrainingArguments(
    output_dir='output',
    evaluation_strategy='steps',#epoch
    eval_steps=500,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    seed=0,
    load_best_model_at_end=True,
)

#instantiate our custom trainer
trainer=CustomTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# Train pre-trained model
CustomTrainer.train()

# After the model is trained, we repeat the same steps for the test data:
# Tokenize test data with pretrained tokenizer
# Create torch dataset
# Load trained model
# Define Trainer

# ----- Prediction Steps -----#
# Load test data
test_data = pd.read_csv("test.csv")
X_test = list(test_data["review"])
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=512)

# Create torch dataset
test_dataset = TorchDataset(X_test_tokenized)

# Load trained model
model_path = "output/checkpoint-50000"
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)

# Define test trainer
test_trainer = CustomTrainer(model)

# Make prediction
raw_pred, _, _ = test_trainer.predict(test_dataset)

# Preprocess raw predictions
y_pred = np.argmax(raw_pred, axis=1)