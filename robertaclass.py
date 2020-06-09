# -*- coding: utf-8 -*-
"""robertaclass.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10FuoW1KpX6tLaLtLvvSqiA0HJz2Ki8MZ
"""

!pip install transformers

import torch
import torch.nn as nn

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x) 
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

x=torch.LongTensor([    0,  1541,   964,   351,    75,   907,    42,  1966,     6,   905,
         1937,     5,   220,    65,    52, 15393,     4,     2,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1])

from transformers import RobertaForSequenceClassification, RobertaConfig, RobertaModel

from transformers.modeling_utils import create_position_ids_from_input_ids

"""model1: ok"""

class RobFirst(RobertaForSequenceClassification):

  config_class = RobertaConfig 
  pretrained_model_archive_map = {"roberta-base": "https://cdn.huggingface.co/roberta-base-pytorch_model.bin"}
  base_model_prefix = "roberta"
  
  def __init__(self,config):
      super().__init__(config) 

  def forward(self, x):  #input -- x : input_id
      return list(self.roberta.embeddings.children())[:1][0](x) #output: embedding

model1 = RobFirst.from_pretrained("roberta-base", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
    )

model1(x.unsqueeze(0))

emb=model1(x.unsqueeze(0))

"""model 2: pas ok"""

class RobSecond(RobertaForSequenceClassification):

  config_class = RobertaConfig
  pretrained_model_archive_map = {"roberta-base": "https://cdn.huggingface.co/roberta-base-pytorch_model.bin"}
  base_model_prefix = "roberta"

  def __init__(self, config):
        super().__init__(config)  

  def forward(self, x, emb): #inputs -- x : input_id  -- emb : embedding (RobFirst(x))
    padding_idx=1
    input_shape = x.size()
    seq_length = input_shape[1]
    #device cuda
    position_ids = create_position_ids_from_input_ids(x, 1)
    token_type_ids=torch.zeros(input_shape, dtype=torch.long) #, device=device

    emb2=list(self.roberta.embeddings.children())[1:][0](position_ids)
    emb3=list(self.roberta.embeddings.children())[1:][1](token_type_ids)
    ess=list(self.roberta.embeddings.children())[1:][2](emb+emb2+emb3) 
    out_1st=list(self.roberta.embeddings.children())[1:][3](ess)  #result of the whole embedding layer of roberta

    #getting result of encoder layer of roberta
    out_2nd=self.roberta.encoder.layer[:12][0](out_1st)
    for i in range(1,12):
      out_2nd=self.roberta.encoder.layer[:12][i](out_2nd[0])

    #getting result of pooler layer of roberta
    out_3nd = self.roberta.pooler(out_2nd[0])
    out_4nd=(out_2nd[0], out_3nd,) + out_2nd[1:]
    out_fin=out_4nd[0]

    #getting result of classifier layer of roberta
    out=self.classifier(out_fin) #this is equivalent to model(x)

    #criterion=torch.nn.CrossEntropyLoss()

    #loss=criterion(out,labels[0].unsqueeze(0))  #pas labels[0]

    return out #this is equivalent to model(x)

model2 = RobSecond.from_pretrained(
    "roberta-base", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
    )

model2(x.unsqueeze(0),emb)

"""true model"""

# Load BertForSequenceClassification, the pretrained BERT model with a single 
# linear classification layer on top. 
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", # Use the 12-layer BERT model, with an uncased vocab.
    num_labels = 2, # The number of output labels--2 for binary classification.
                    # You can increase this for multi-class tasks.   
    output_attentions = False, # Whether the model returns attentions weights.
    output_hidden_states = False, # Whether the model returns all hidden-states.
)
 
# Tell pytorch to run this model on the GPU.
#model.cuda()

model.roberta.embeddings.word_embeddings(x.unsqueeze(0))

model(x.unsqueeze(0))

"""décomposé: meme resultat que true model"""

y=x.unsqueeze(0)

padding_idx=1
input_shape = y.size()
seq_length = input_shape[1]
#device cuda
position_ids = create_position_ids_from_input_ids(y, 1)
token_type_ids=torch.zeros(input_shape, dtype=torch.long) #, device=device

emb2=list(model.roberta.embeddings.children())[1:][0](position_ids)
emb3=list(model.roberta.embeddings.children())[1:][1](token_type_ids)
ess=list(model.roberta.embeddings.children())[1:][2](emb+emb2+emb3) 
out_1st=list(model.roberta.embeddings.children())[1:][3](ess)  #result of the whole embedding layer of roberta

#getting result of encoder layer of roberta
out_2nd=model.roberta.encoder.layer[:12][0](out_1st)
for i in range(1,12):
  out_2nd=model.roberta.encoder.layer[:12][i](out_2nd[0])

#getting result of pooler layer of roberta
out_3nd = model.roberta.pooler(out_2nd[0])
out_4nd=(out_2nd[0], out_3nd,) + out_2nd[1:]
out_fin=out_4nd[0]

#getting result of classifier layer of roberta
out=model.classifier(out_fin) #this is equivalent to model(x)

#criterion=torch.nn.CrossEntropyLoss()

    #loss=criterion(out,labels[0].unsqueeze(0))  #pas labels[0]
print(out) #this is equivalent to model(x)