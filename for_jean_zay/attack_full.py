# -*- coding: utf-8 -*-
 
from transformers import RobertaTokenizer
import csv
import torch
from transformers import RobertaForSequenceClassification, RobertaConfig

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math as mm
import matplotlib.pyplot as plt
import random
from time import time
import pandas as pd
  
from advertorch.utils import clamp
from advertorch.utils import normalize_by_pnorm
from advertorch.utils import clamp_by_pnorm
from advertorch.utils import is_float_or_torch_tensor
from advertorch.utils import batch_multiply
from advertorch.utils import batch_clamp
from advertorch.utils import replicate_input
from advertorch.utils import batch_l1_proj

from advertorch.attacks.base import Attack
from advertorch.attacks.base import LabelMixin
from advertorch.attacks.utils import rand_init_delta

#required functions:

def sentlong(x): #real length of an input_id sentence 
  compteur = 0
  while compteur<len(x) and x[compteur]!=2:
    compteur+=1
  return compteur # compteur smallest index for which x[compteur]=2

def clean(x,emb): #zero all embeddings at indexes beyond the real sentence lenght
  size = emb.size()
  for b in range(size[0]): #batch_size
    sent_long=sentlong(x[b])
    for i in range(sent_long,size[1]): #real size of the sentence 
      emb[b,i,:]=torch.zeros(size[2]) #embedding size
  return emb

def tozero(tens,ind): #zero all indexes except ind
  tens2=torch.zeros_like(tens)
  tens2[0][ind]=tens[0][ind]
  return tens2

def tozerolist(tens,indlist): #zero all indexes except indlist
  nb=len(indlist)
  tens3=torch.zeros_like(tens)
  for k in range(nb):
    tens2=torch.zeros_like(tens)
    tens2[0][indlist[k]]=tens[0][indlist[k]]
    tens3+=tens2
  return tens3 

def my_proj_all(emb2,emb,indlist,eps):  #project all dim en cosim norm
  emb3=emb2.clone() 
  for t in indlist:
    emb3[t]=my_proj(emb2[t],emb[t],eps)
  return emb3 # size = nombre de mots x 768 

def my_proj(xt,x0,eps): #project one dim en cosim norm
  nrm = float(torch.norm(x0))
  scal = float(torch.dot(xt,x0))

  if scal/(nrm*float(torch.norm(xt)))>=eps:
    return xt
  elif scal>0:
    x0p = (nrm**2)*xt/scal - x0
    x0p *= (nrm)/(torch.norm(x0p))

    alpha = ( torch.dot(xt,x0p) + scal/mm.sqrt(1-eps**2) ) / ( mm.sqrt(1/eps**2 -1) + 1/mm.sqrt(1/eps**2-1) )

    beta = alpha * mm.sqrt(1/eps**2 - 1) 

    z = alpha * x0/nrm**2 + beta * x0p/nrm**2 

    return z
  else:
    return torch.zeros_like(xt)

def first(couple):
  a,b,c=couple
  return a

def create_position_ids_from_input_ids(input_ids, padding_idx):
    """ Replace non-padding symbols with their position numbers. Position numbers begin at
    padding_idx+1. Padding symbols are ignored. This is modified from fairseq's
    `utils.make_positions`.

    :param torch.Tensor x:
    :return torch.Tensor:
    """
    # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
    mask = input_ids.ne(padding_idx).int()
    incremental_indices = torch.cumsum(mask, dim=1).type_as(mask) * mask
    return incremental_indices.long() + padding_idx

def lenlist(l): #return list of sizes from a list of list
  res=[]
  for li in l:
    res+=[len(li)]
  return res

#replace word at index ind from sentence x
def replace(x,ind,word):
  xprime=x.clone()
  xprime[0][ind]=word
  return xprime

#replace word at index ind from sentence x
def replacelist(x,indlist,wordlist):
  xprime=x.clone()
  for t in range(len(indlist)):
    xprime[0][indlist[t]]=wordlist[t]
  return xprime

#projecting v on probability simplex
#https://gist.github.com/mblondel/6f3b7aaad90606b98f71

def projection_simplex_sort(v, z=1):
    #v=v.cpu()
    #v=v.numpy()
    n_features = v.shape[0]
    u = torch.sort(v)[::-1][1]
    cssv = torch.cumsum(u,dim=0) - z
    ind = torch.arange(n_features).to(device) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    zer=torch.zeros(n_features).to(device)
    w = torch.max(v - theta, zer)
    #w = torch.from_numpy(w) 
    #w=w.to(device)
    return w
   
#Kullback-Leibler div:
def KL(p,q):
  p=nn.Softmax(dim=0)(p)
  q=nn.Softmax(dim=0)(q)
  return float(p[0])*mm.log(float(p[0])/float(q[0]))+float(p[1])*mm.log(float(p[1])/float(q[1])) #float


def main(): 
    
    #load encoder
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base') 

    #get data and encode it
    sentences=[]
    labels=[]
    with open("./glue_data/SST-2/dev.tsv") as tsvfile:
        tsvreader = csv.reader(tsvfile, delimiter="\t")
        for i, line in enumerate(tsvreader):
          if i>0:
            sentences += [line[0]]
            labels += [int(line[1])]

    # ici juste un tour pour voir quelle est la taille max . on visera un peu pls haut par sécurité.
    max_len = 0
    # For every sentence...
    for sent in sentences:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))

    print('Max sentence length: ', max_len)

    input_ids = []

    for sent in sentences:
      encoded_dict = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = max_len+10,           # Pad & truncate all sentences.
                            truncation=True,  
                            pad_to_max_length = True,
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
      # Add the encoded sentence to the list.    
      input_ids.append(encoded_dict)

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    labels = torch.tensor(labels)

    # Print sentence 0, now as a list of IDs.
    #print('Original: ', sentences[0])
    #print('Token IDs:', input_ids[0])


    # If there's a GPU available...
    if torch.cuda.is_available():    

        # Tell PyTorch to use the GPU.    
        device = torch.device("cuda")

        print('There are %d GPU(s) available.' % torch.cuda.device_count())

        print('We will use the GPU:', torch.cuda.get_device_name(0))

    # If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        
       
    
    # Load BertForSequenceClassification, the pretrained BERT model with a single 
    # linear classification layer on top. 
    model = RobertaForSequenceClassification.from_pretrained(
        "./my_pretrained", # Use the 12-layer BERT model, with an uncased vocab.  #please rather use "roberta-base"
        num_labels = 2, # The number of output labels--2 for binary classification.
                       # You can increase this for multi-class tasks.   
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    # If there's a GPU available...
    if torch.cuda.is_available():   
      # Tell pytorch to run this model on the GPU.
      model.cuda()
      map_location=lambda storage, loc: storage.cuda()
    else: 
     map_location= 'cpu'
      
    #load saved model (which is finetuned roberta)
    model.load_state_dict(torch.load('./roberta_finetunedbeta.pt', map_location=map_location))
    model.eval()
    
    #import language model
    from transformers import RobertaForMaskedLM
    modd = RobertaForMaskedLM.from_pretrained(
     "./my_pretrained",
      output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    modd.eval()
    # If there's a GPU available...
    if torch.cuda.is_available():   
      # Tell pytorch to run this model on the GPU.
      modd.cuda()
  

    
    #define the two forward functions (we cut our model in two parts)
    def predict1(x): #ex: x = input_ids[0].unsqueeze(0).to(device)
     emb=list(model.roberta.embeddings.children())[:1][0](x) #embedding of x
     return emb
   
    def predict2(x,emb):

     #some model requirements for the calculation
     padding_idx=1
     input_shape = x.size()
     seq_length = input_shape[1]
     position_ids = create_position_ids_from_input_ids(x.to('cpu'), 1).to(device) 
     token_type_ids=torch.zeros(input_shape, dtype=torch.long, device=device)

     #model calculations:
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

     return out
    
    
    #find numb neighboors of embedd among the embedding dictionary
    def neighboors(embedd,numb):
      emb_matrix = model.roberta.embeddings.word_embeddings.weight
      normed_emb_matrix=F.normalize(emb_matrix, p=2, dim=1) 
      normed_emb_word=F.normalize(embedd, p=2, dim=0) 
      cosine_similarity = torch.matmul(normed_emb_word, torch.transpose(normed_emb_matrix,0,1))
      calc, closest_words = torch.topk(cosine_similarity,numb,dim=0)
      print(tokenizer.decode(closest_words))
      return closest_words

    #find numb neighboors of embedd among candidates
    def neighboors_np_dens_cand(embedd,rayon,candidates):  
      normed_emb_word=F.normalize(embedd, p=2, dim=0) 
      cosine_similarity = torch.matmul(normed_emb_word, torch.transpose(candidates,0,1))
      calc, closest_words = torch.topk(cosine_similarity,1,dim=0)
      compteur=0 
      if rayon<1.:
       for t in range(len(cosine_similarity)):  
         if cosine_similarity[t]>rayon:
           compteur+=1 #calculate the density around embedd, among all possible candidates
      return closest_words, compteur
    
    
    def perturb_iterative_fool_many(xvar, embvar, indlistvar, yvar, predict, nb_iter, eps, epscand, eps_iter, loss_fn,rayon,
                      delta_init=None, minimize=False, ord=np.inf,
                      clip_min=0.0, clip_max=1.0,
                       l1_sparsity=None):
      """
      Iteratively maximize the loss over the input. It is a shared method for
      iterative attacks including IterativeGradientSign, LinfPGD, etc.
      :param xvar: input data.
      :param yvar: input labels.
      :param predict: forward pass function.
      :param nb_iter: number of iterations.
      :param eps: maximum distortion.
      :param eps_iter: attack step size.
      :param loss_fn: loss function.
      :param delta_init: (optional) tensor contains the random initialization.
      :param minimize: (optional bool) whether to minimize or maximize the loss.
      :param ord: (optional) the order of maximum distortion (inf or 2).
      :param clip_min: mininum value per input dimension.
      :param clip_max: maximum value per input dimension.
      :param l1_sparsity: sparsity value for L1 projection.
                    - if None, then perform regular L1 projection.
                    - if float value, then perform sparse L1 descent from
                      Algorithm 1 in https://arxiv.org/pdf/1904.13000v1.pdf
      :return: tensor containing the perturbed input.
      """

      #will contain all words encountered during PGD
      nb=len(indlistvar)
      tablist=[]
      for t in range(nb):
        tablist+=[[]]
      fool=False

      #contain each loss on embed and each difference of loss on word nearest neighboor
      loss_memory=np.zeros((nb_iter,))
      word_balance_memory=np.zeros((nb_iter,))

      candid=[torch.empty(0)]*nb
      convers=[[]]*nb
      for u in range(nb):
        #prepare all potential candidates, once and for all
        candidates=torch.empty([0,768]).to(device)
        conversion=[]
        emb_matrix=model.roberta.embeddings.word_embeddings.weight   
        normed_emb_matrix=F.normalize(emb_matrix, p=2, dim=1) 
        normed_emb_word=F.normalize(embvar[0][indlistvar[u]], p=2, dim=0) 
        cosine_similarity = torch.matmul(normed_emb_word, torch.transpose(normed_emb_matrix,0,1))
        for t in range(len(cosine_similarity)): #evitez de faire DEUX boucles .
          if cosine_similarity[t]>epscand:
            candidates=torch.cat((candidates,normed_emb_matrix[t].unsqueeze(0)),0)
            conversion+=[t]
        candid[u]=candidates
        convers[u]=conversion
        print("nb of candidates :")
        print(len(conversion))

      #U, S, V = torch.svd(model.roberta.embeddings.word_embeddings.weight)

      if delta_init is not None:
          delta = delta_init
      else:
          delta = torch.zeros_like(embvar)
 
      #PGD
      delta.requires_grad_()
      ii=0
      while ii<nb_iter and not(fool):
          outputs = predict(xvar, embvar + delta)
          if ii==0:
             loss = loss_fn(outputs, yvar) - 0.01*modd(inputs_embeds=embvar+delta,labels=xvar)[0] - 100*sum([KL(model(inputs_embeds=((embvar)[0][kk]).unsqueeze(0).unsqueeze(0))[0][0],model(inputs_embeds=((embvar+delta)[0][kk]).unsqueeze(0).unsqueeze(0))[0][0]) for kk in indlistvar])
           else:
             loss = loss_fn(outputs, yvar) - 0.01*modd(inputs_embeds=embvar+delta,labels=replacelist(xvar,indlistvar,adverslist))[0] - 100*sum([KL(model(inputs_embeds=((embvar)[0][kk]).unsqueeze(0).unsqueeze(0))[0][0],model(inputs_embeds=((embvar+delta)[0][kk]).unsqueeze(0).unsqueeze(0))[0][0]) for kk in indlistvar])
          #loss = loss_fn(outputs, yvar)
          if minimize: 
              loss = -loss 

          loss.backward()
          if ord == np.inf:    
              grad_sign = delta.grad.data.sign()
              grad_sign = tozerolist(grad_sign,indlistvar)
              #grad_sign=torch.matmul(torch.cat((torch.matmul(grad_sign,v)[:,:,:50],torch.zeros([768-50]).to(device)),2),v.t())
              delta.data = delta.data + batch_multiply(eps_iter, grad_sign)
              delta.data = batch_clamp(eps, delta.data)
              delta.data = clamp(embvar.data + delta.data, clip_min, clip_max #à retirer?
                                 ) - embvar.data
              with torch.no_grad():  
                delta.data = tozero(delta.data,indlistvar) 
                if (ii%200)==0:
                 adverslist=[]
                 for t in range(nb):
                   advers, nb_vois =neighboors_np_dens_cand((embvar+delta)[0][indlistvar[t]],rayon,candid[t])
                   advers=int(advers[0]) 
                   advers=torch.tensor(convers[t][advers])
                   if len(tablist[t])==0:
                     tablist[t]+=[(tokenizer.decode(advers.unsqueeze(0)),ii,nb_vois)]
                   elif not(first(tablist[t][-1])==tokenizer.decode(advers.unsqueeze(0))): 
                     tablist[t]+=[(tokenizer.decode(advers.unsqueeze(0)),ii,nb_vois)]
                   adverslist+=[advers]
                 word_balance_memory[ii]=float(model(replacelist(xvar,indlistvar,adverslist),labels=1-yvar)[0])-float(model(replacelist(xvar,indlistvar,adverslist),labels=yvar)[0])
                 if word_balance_memory[ii]<0:
                   fool=True          

          elif ord == 0: 
              grad = delta.grad.data 
              grad = tozero(grad,indlistvar)   
              delta.data = delta.data + batch_multiply(eps_iter, grad)
              delta.data[0] = my_proj_all(embvar.data[0]+delta.data[0],embvar[0],indlistvar,eps) -embvar.data[0]
              delta.data = clamp(embvar.data + delta.data, clip_min, clip_max
                                 ) - embvar.data #à virer je pense
              with torch.no_grad(): 
                delta.data = tozero(delta.data,indlistvar) 
                if (ii%200)==0:
                 adverslist=[]
                 for t in range(nb):
                   advers, nb_vois =neighboors_np_dens_cand((embvar+delta)[0][indlistvar[t]],rayon,candid[t])
                   advers=int(advers[0]) 
                   advers=torch.tensor(convers[t][advers])
                   if len(tablist[t])==0:
                     tablist[t]+=[(tokenizer.decode(advers.unsqueeze(0)),ii,nb_vois)]
                   elif not(first(tablist[t][-1])==tokenizer.decode(advers.unsqueeze(0))):  
                     tablist[t]+=[(tokenizer.decode(advers.unsqueeze(0)),ii,nb_vois)]
                   adverslist+=[advers] 
                 word_balance_memory[ii]=float(model(replacelist(xvar,indlistvar,adverslist),labels=1-yvar)[0])-float(model(replacelist(xvar,indlistvar,adverslist),labels=yvar)[0])
                 if word_balance_memory[ii]<0:
                   fool=True            

          elif ord == 2:  
              grad = delta.grad.data
              grad = tozero(grad,indlistvar) 
              grad = normalize_by_pnorm(grad)
              delta.data = delta.data + batch_multiply(eps_iter, grad)
              delta.data = clamp(embvar.data + delta.data, clip_min, clip_max
                                 ) - embvar.data
              if eps is not None:
                  delta.data = clamp_by_pnorm(delta.data, ord, eps)
              with torch.no_grad(): 
                delta.data = tozero(delta.data,indlistvar) 
                if (ii%200)==0:
                 adverslist=[]
                 for t in range(nb):
                   advers, nb_vois =neighboors_np_dens_cand((embvar+delta)[0][indlistvar[t]],rayon,candid[t])
                   advers=int(advers[0]) 
                   advers=torch.tensor(convers[t][advers])
                   if len(tablist[t])==0:
                     tablist[t]+=[(tokenizer.decode(advers.unsqueeze(0)),ii,nb_vois)]
                   elif not(first(tablist[t][-1])==tokenizer.decode(advers.unsqueeze(0))):  
                     tablist[t]+=[(tokenizer.decode(advers.unsqueeze(0)),ii,nb_vois)]
                   adverslist+=[advers] 
                 word_balance_memory[ii]=float(model(replacelist(xvar,indlistvar,adverslist),labels=1-yvar)[0])-float(model(replacelist(xvar,indlistvar,adverslist),labels=yvar)[0])
                 if word_balance_memory[ii]<0:
                   fool=True     

          elif ord == 1:
              grad = delta.grad.data
              grad_sign = tozero(grad_sign,indvar) 
              abs_grad = torch.abs(grad)

              batch_size = grad.size(0)
              view = abs_grad.view(batch_size, -1)
              view_size = view.size(1)
              if l1_sparsity is None:
                  vals, idx = view.topk(1)
              else:
                  vals, idx = view.topk(
                      int(np.round((1 - l1_sparsity) * view_size)))

              out = torch.zeros_like(view).scatter_(1, idx, vals)
              out = out.view_as(grad)
              grad = grad.sign() * (out > 0).float()
              grad = normalize_by_pnorm(grad, p=1)
              delta.data = delta.data + batch_multiply(eps_iter, grad)

              delta.data = batch_l1_proj(delta.data.cpu(), eps)
              if embvar.is_cuda:
                  delta.data = delta.data.cuda()
              delta.data = clamp(embvar.data + delta.data, clip_min, clip_max
                                 ) - embvar.data
          else:
              error = "Only ord = inf, ord = 1 and ord = 2 have been implemented"
              raise NotImplementedError(error)
          delta.grad.data.zero_()
          with torch.no_grad():
            loss_memory[ii]= loss  


          ii+=1


      #plt.plot(loss_memory)
      #plt.title("evolution of embed loss")
      #plt.show() 
      #plt.plot(word_balance_memory)
      #plt.title("evolution of word loss difference")
      #plt.show() 
      emb_adv = clamp(embvar + delta, clip_min, clip_max)
      return emb_adv, word_balance_memory, loss_memory, tablist, fool
     
    #pgd attack classes
    class PGDAttack(Attack, LabelMixin):
      """
      The projected gradient descent attack (Madry et al, 2017).
      The attack performs nb_iter steps of size eps_iter, while always staying
      within eps from the initial point.
      Paper: https://arxiv.org/pdf/1706.06083.pdf
      :param predict: forward pass function.
      :param loss_fn: loss function.
      :param eps: maximum distortion.
      :param nb_iter: number of iterations.
      :param eps_iter: attack step size.
      :param rand_init: (optional bool) random initialization.
      :param clip_min: mininum value per input dimension.
      :param clip_max: maximum value per input dimension.
      :param ord: (optional) the order of maximum distortion (inf or 2).
      :param targeted: if the attack is targeted.
      """    

      def __init__(
              self, predict, loss_fn=None, eps=0.3, epscand=0.03, nb_iter=40,
              eps_iter=0.01, rayon=0.5,rand_init=True, clip_min=0., clip_max=1.,
              ord=np.inf, l1_sparsity=None, targeted=False):
          """
          Create an instance of the PGDAttack.
          """
          super(PGDAttack, self).__init__(
              predict, loss_fn, clip_min, clip_max)
          self.eps = eps
          self.epscand= epscand
          self.nb_iter = nb_iter
          self.eps_iter = eps_iter
          self.rayon= rayon
          self.rand_init = rand_init
          self.ord = ord
          self.targeted = targeted  
          if self.loss_fn is None:
              self.loss_fn = nn.CrossEntropyLoss(reduction="sum") #or no reduction
          self.l1_sparsity = l1_sparsity
          assert is_float_or_torch_tensor(self.eps_iter)
          assert is_float_or_torch_tensor(self.eps)

      def perturb_fool_many(self, x, emb, indlist, y=None): #list of ind of words to be perturbed
          """
          Given examples (x, y), returns their adversarial counterparts with
          an attack length of eps.
          :param x: input tensor.
          :param y: label tensor.
                    - if None and self.targeted=False, compute y as predicted
                      labels.
                    - if self.targeted=True, then y must be the targeted labels.
          :return: tensor containing perturbed inputs.
          """
          emb, y = self._verify_and_process_inputs(emb, y) #???

          delta = torch.zeros_like(emb) 
          delta = nn.Parameter(delta)
          if self.rand_init: 
              rand_init_delta( 
                  delta, emb, np.inf, self.eps, self.clip_min, self.clip_max)
              delta.data = clamp(
                  emb + delta.data, min=self.clip_min, max=self.clip_max) - emb  

          with torch.no_grad():
            for t in range(delta.size()[1]):
              if not(t in indlist):
                for k in range(delta.size()[2]):
                  delta[0][t][k]=0
            if self.ord == 0: 
              delta[0]=my_proj_all(emb[0]+delta[0],emb[0],indlist,self.eps)-emb[0]



          rval, word_balance_memory, loss_memory, tablist, fool = perturb_iterative_fool_many(
              x, emb, indlist, y, self.predict, nb_iter=self.nb_iter,
              eps=self.eps, epscand=self.epscand, eps_iter=self.eps_iter,
              loss_fn=self.loss_fn, minimize=self.targeted,
              ord=self.ord, clip_min=self.clip_min,
              clip_max=self.clip_max, delta_init=delta,
              l1_sparsity=self.l1_sparsity,rayon=self.rayon
          )

          return rval.data, word_balance_memory, loss_memory, tablist, fool 
     
     
    
    
    ###here we actually attack all sentences
    
    
    res_se=[] 
    res_or=[] 
    res_lw=[] 
    res_lg=[] 
    res_ne=[] 
    res_cs=[] 
    
    l1=range(125,175)
    for iid in l1:
     for eps_iter in [2.]:
      eps=0.7 #embeddings distance with norm ord
      epscand=0.35 #fix nb of candidates according to cosim
      nb_iter=8001
      ord=np.inf #norm choice
      rayon=1. #density search
      
      t0 = time()
      print("\n")
      
      mf=0

      x = input_ids[iid].unsqueeze(0).to(device)
      y = labels[iid].unsqueeze(0).to(device)
      neigh=3
      
      indlist=range(1,sentlong(x[0])) 
      nind=len(indlist)

      if float(model(x,labels=y)[0])>float(model(x,labels=1-y)[0]): #if model misclassifies
        mf+=1
        print("sentence already misclassified")  
        #return res_se, res_or,res_lw,res_lg,res_ne,res_cs
      else: #model classifies correctly so it makes sense to try to fool it

        print("original sentence:")
        print(tokenizer.decode(x[0]))
        print("is classified as:")
        if bool(y==1):
          print("positive")
        else:
          print("negative")
        print("\n")

        orig_wordlist=[]
        for u in range(nind):
          #print("let's change word number"+str(indlist[u])+"which is:")
          orig_word=tokenizer.decode(x[0][indlist[u]].unsqueeze(0))
          #print(orig_word)
          orig_wordlist+=[orig_word]
        #print("\n")

        emb=predict1(x)

        new_word=['']*nind  
        print("Does our PGD output's first neighboor fool model?:") 
        att=PGDAttack(predict2, loss_fn=None, eps=eps, epscand=epscand, nb_iter=nb_iter, 
                  eps_iter=eps_iter,rayon=rayon, rand_init=True, clip_min=-1., clip_max=1.,
                  ord=ord, l1_sparsity=None, targeted=False)                   
        rval, word_balance_memory, loss_memory, tablist, fool =att.perturb_fool_many(x,emb,indlist,y)
        print(fool)  
 
        csnlist=[0]*nind
 

        for u in range(nind):
          new_word[u]=first(tablist[u][-1]) 
        print("\n")

        for u in range(nind): 
          csnlist[u]=float(torch.matmul(F.normalize(model.roberta.embeddings.word_embeddings(torch.tensor(tokenizer.encode(new_word[u])[1]).to(device)), p=2, dim=0), torch.transpose(F.normalize(model.roberta.embeddings.word_embeddings(x[0][indlist[u]]).unsqueeze(0), p=2, dim=1),0,1)))
         
        nwi=''
        for w in new_word:
         nwi=nwi+w
        new_word=[nwi]
        
        print(tokenizer.decode(x[0]))
        print(orig_wordlist)
        print(tablist)
        print(lenlist(tablist))
        print(new_word)
        print(csnlist)
        
        if fool:
         res_se+=[tokenizer.decode(x[0])]
         res_or+=[orig_wordlist]
         res_lw+=[tablist]
         res_lg+=[lenlist(tablist)] 
         res_ne+=[new_word]
         res_cs+=[csnlist]
         
        t1 = time()
        print('function takes %f' %(t1-t0))

    df = pd.DataFrame(list(zip(res_se, res_or,res_lw,res_lg,res_ne,res_cs)), 
               columns =['sentence', 'original word', 'not-fooling words ( = path)', 'path length', 'new word','csn similarity']) 
    
    
    df.to_csv('results/results.csv', index = False)  #r
   

    #return res_se, res_or,res_lw,res_lg,res_ne,res_cs
    ###



if __name__ == '__main__':
    main()
