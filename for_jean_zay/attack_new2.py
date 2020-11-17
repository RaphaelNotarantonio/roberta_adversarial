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

def levenshtein(seq1, seq2): #from https://stackabuse.com/levenshtein-distance-and-text-similarity-in-python/
    size_x = len(seq1) + 1 
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1])

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

def tozerolist(tens,indlist): #zero all indexes except indlist #indlist= list of list . tens : batch
  batch_size=len(indlist)
  tens3=torch.zeros_like(tens)
  for ba in range(batch_size): 
    for k in range(len(indlist[ba])):
      tens2=torch.zeros_like(tens)
      tens2[ba][indlist[ba][k]]=tens[ba][indlist[ba][k]]
      tens3+=tens2
  if not(tens3.is_cuda):
    print("oulah")
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
  if not(xprime.is_cuda):
    print("oulah replacelist")
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
    model.load_state_dict(torch.load('./roberta_finetuned.pt', map_location=map_location))
    model.eval()
  

    
    #define the two forward functions (we cut our model in two parts)
    def predict1(x): #ex: x = input_ids[0].unsqueeze(0).to(device)
     emb=list(model.roberta.embeddings.children())[:1][0](x) #embedding of x #ajouter [0] avant (x) ?
     return emb
   
    def predict2(x,emb):
     return model(inputs_embeds=emb)[0]
    
    
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
      calc, closest_words = torch.topk(cosine_similarity,10,dim=0)
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
      batch_size=len(indlistvar)
      nb=[]
      tablistbatch=[]
      for ba in range(batch_size):
        nb+=[len(indlistvar[ba])]
        tablist=[]
        for t in range(nb[-1]):
          tablist+=[[]]
        tablistbatch+=[tablist]
      fool=[False]*batch_size

      #contain each loss on embed and each difference of loss on word nearest neighboor
      loss_memory=np.zeros((nb_iter,))
      word_balance_memory=np.zeros((nb_iter,))

      candidbatch=[]
      conversbatch=[]
      for ba in range(batch_size):
        candid=[torch.empty(0)]*nb[ba]
        convers=[[]]*nb[ba]
        for u in range(nb[ba]):
          #prepare all potential candidates, once and for all
          candidates=torch.empty([0,768]).to(device)
          conversion=[]
          emb_matrix=model.roberta.embeddings.word_embeddings.weight   
          normed_emb_matrix=F.normalize(emb_matrix, p=2, dim=1) 
          normed_emb_word=F.normalize(embvar[ba][indlistvar[ba][u]], p=2, dim=0) 
          cosine_similarity = torch.matmul(normed_emb_word, torch.transpose(normed_emb_matrix,0,1))
          for t in range(len(cosine_similarity)): #evitez de faire DEUX boucles .
            if cosine_similarity[t]>epscand:
              if levenshtein(tokenizer.decode(torch.tensor([xvar[ba][indlistvar[ba][u]]])),tokenizer.decode(torch.tensor([t])))!=1:
               candidates=torch.cat((candidates,normed_emb_matrix[t].unsqueeze(0)),0)
               conversion+=[t]
          candid[u]=candidates
          convers[u]=conversion
          print("nb of candidates :")
          print(len(conversion))
        candidbatch+=[candid]
        conversbatch+=[convers]

      #U, S, V = torch.svd(model.roberta.embeddings.word_embeddings.weight)

      if delta_init is not None:
          delta = delta_init
      else:
          delta = torch.zeros_like(embvar)
 
      #PGD
      delta.requires_grad_()
      ii=0
      while ii<nb_iter: #not fool
          outputs = predict(xvar, embvar + delta)
          loss = loss_fn(outputs, yvar)
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
                delta.data = tozerolist(delta.data,indlistvar) 
                if (ii%50)==0:
                 adverslistbatch=[]
                 for ba in range(batch_size):
                   if not(fool[ba]):
                     adverslist=[[] for _ in range(10)] #i choose k=10 neighboors  
                     for t in range(nb[ba]):
                      adversk, nb_vois = neighboors_np_dens_cand((embvar+delta)[ba][indlistvar[ba][t]],rayon,candidbatch[ba][t])
                      for k in range(10):
                        advers=int(adversk[k])
                        advers=torch.tensor(conversbatch[ba][t][advers])
                        adverslist[k]+=[advers]
                     adverslistbatch+=[adverslist]
                     word_balance_memory[ii]=1000 #now let's choose the best k of all ten
                     k_mem=-1
                     for k in range(10):
                       aut=float(model(replacelist(xvar[ba].unsqueeze(0),indlistvar[ba],adverslistbatch[ba][k]),labels=1-yvar[ba].unsqueeze(0))[0])-float(model(replacelist(xvar[ba].unsqueeze(0),indlistvar[ba],adverslistbatch[ba][k]),labels=yvar[ba].unsqueeze(0))[0])
                       if aut<word_balance_memory[ii]:
                          word_balance_memory[ii]=aut
                          k_mem=k 
                     if len(tablistbatch[ba][t])==0:
                             tablistbatch[ba][t]+=[(tokenizer.decode(adverslistbatch[ba][k_mem].unsqueeze(0)),ii,nb_vois)]
                     elif not(first(tablistbatch[ba][t][-1])==tokenizer.decode(adverslistbatch[ba][k_mem].unsqueeze(0))): 
                             tablistbatch[ba][t]+=[(tokenizer.decode(adverslistbatch[ba][k_mem].unsqueeze(0)),ii,nb_vois)]
                           #n'oublie pas que se posera la question de partir d'un embedding différent à chaque phrase.
                     if word_balance_memory[ii]<0:
                         fool[ba]=True  
                         
                         
                   
                        
                             

          elif ord == 0: 
              grad = delta.grad.data 
              grad = tozero(grad,indlistvar)   
              delta.data = delta.data + batch_multiply(eps_iter, grad)
              delta.data[0] = my_proj_all(embvar.data[0]+delta.data[0],embvar[0],indlistvar,eps) -embvar.data[0]
              delta.data = clamp(embvar.data + delta.data, clip_min, clip_max
                                 ) - embvar.data #à virer je pense
              with torch.no_grad(): 
                delta.data = tozero(delta.data,indlistvar) 
                if (ii%300)==0:
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
                if (ii%300)==0:
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
          #with torch.no_grad():
          #  loss_memory[ii]= loss  


          ii+=1


      #plt.plot(loss_memory)
      #plt.title("evolution of embed loss")
      #plt.show() 
      #plt.plot(word_balance_memory)
      #plt.title("evolution of word loss difference")
      #plt.show() 
      emb_adv = clamp(embvar + delta, clip_min, clip_max)
      return emb_adv, word_balance_memory, loss_memory, tablistbatch, fool
     
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
            for ba in range(delta.size()[0]):
              for t in range(delta.size()[1]):
                if not(t in indlist[ba]):
                  for k in range(delta.size()[2]):
                    delta[ba][t][k]=0
            if self.ord == 0: 
              for ba in range(delta.size()[0]):
                delta[ba]=my_proj_all(emb[ba]+delta[ba],emb[ba],indlist[ba],self.eps)-emb[ba]



          rval, word_balance_memory, loss_memory, tablistbatch, fool = perturb_iterative_fool_many(
              x, emb, indlist, y, self.predict, nb_iter=self.nb_iter,
              eps=self.eps, epscand=self.epscand, eps_iter=self.eps_iter,
              loss_fn=self.loss_fn, minimize=self.targeted,
              ord=self.ord, clip_min=self.clip_min,
              clip_max=self.clip_max, delta_init=delta,
              l1_sparsity=self.l1_sparsity,rayon=self.rayon
          )

          return rval.data, word_balance_memory, loss_memory, tablistbatch, fool 
     
     
    
    
    ###here we actually attack all sentences
    
    
    res_se=[] 
    res_or=[] 
    res_lw=[] 
    res_lg=[] 
    res_ne=[] 
    res_cs=[] 
    
    
    eps=0.7 #embeddings distance with norm ord
    epscand=0.25 #fix nb of candidates according to cosim
    nb_iter=8001
    ord=np.inf #norm choice
    rayon=1. #density search
    
    neigh=3
    mf=0 
    for uid in range(5):
      vide=True
      for vid in range(16):
        iid = uid*16+vid
        #if model misclassifies:
        if float(model(input_ids[iid].unsqueeze(0).to(device),labels=labels[iid].unsqueeze(0).to(device))[0])>float(model(input_ids[iid].unsqueeze(0).to(device),labels=1-labels[iid].unsqueeze(0).to(device))[0]): 
          mf+=1 
          print("sentence already misclassified") 
        else:
          if vide:
            x = input_ids[iid].unsqueeze(0).to(device)
            y = labels[iid].unsqueeze(0).to(device)
            vide=False
          else:
            x = torch.cat((x,input_ids[iid].unsqueeze(0).to(device)),0)
            y = torch.cat((y,labels[iid].unsqueeze(0).to(device)),0) 
            
      for eps_iter in [0.5]: 
      
        t0 = time()
        print("\n")
        
        batch_sze=x.size()[0]
        indlist=[]
        nb=[]
        
        for ba in range(batch_sze):
          indlist+=[range(1,sentlong(x[ba]))]
          nb+=[len(indlist[-1])]

          print("original sentence:")
          print(tokenizer.decode(x[ba]))
          print("is classified as:")
          if bool(y[ba]==1):
            print("positive")
          else:
            print("negative")
          print("\n")
        
        orig_wordlistbatch=[]
        for ba in range(batch_sze):      
          orig_wordlist=[]
          for u in range(nb[ba]):
            #print("let's change word number"+str(indlist[ba][u])+"which is:")
            orig_word=tokenizer.decode(x[ba][indlist[ba][u]].unsqueeze(0))
            #print(orig_word)
            orig_wordlist+=[orig_word]
          #print("\n")
          orig_wordlistbatch+=[orig_wordlist]

        emb=predict1(x)

        new_wordbatch=[]
        for ba in range(batch_sze):
          new_word=['']*nb[ba]
          new_wordbatch+=[new_word]
          
        print("Does our PGD output's first neighboor fool model?:") 
        att=PGDAttack(predict2, loss_fn=None, eps=eps, epscand=epscand, nb_iter=nb_iter, 
                  eps_iter=eps_iter,rayon=rayon, rand_init=True, clip_min=-1., clip_max=1.,
                  ord=ord, l1_sparsity=None, targeted=False)                   
        rval, word_balance_memory, loss_memory, tablistbatch, fool =att.perturb_fool_many(x,emb,indlist,y)
        print(fool)  

        csnlistbatch=[]
        for ba in range(batch_sze):
          csnlist=[0]*nb[ba]
          csnlistbatch+=[csnlist]

        for ba in range(batch_sze):
          for u in range(nb[ba]):
            new_wordbatch[ba][u]=first(tablistbatch[ba][u][-1]) 
          print("\n")
          
        for ba in range(batch_sze):
          for u in range(nb[ba]): 
            csnlistbatch[ba][u]=float(torch.matmul(F.normalize(model.roberta.embeddings.word_embeddings(torch.tensor(tokenizer.encode(new_wordbatch[ba][u])[1]).to(device)), p=2, dim=0), torch.transpose(F.normalize(model.roberta.embeddings.word_embeddings(x[ba][indlist[ba][u]]).unsqueeze(0), p=2, dim=1),0,1)))

        for ba in range(batch_sze):
          print(tokenizer.decode(x[ba]))
          print(orig_wordlistbatch[ba])
          print(tablistbatch[ba])
          print(lenlist(tablistbatch[ba]))
          print(new_wordbatch[ba])
          print(csnlistbatch[ba])

          if fool[ba]:
           res_se+=[tokenizer.decode(x[ba])]
           res_or+=[orig_wordlistbatch[ba]]
           res_lw+=[tablistbatch[ba]]
           res_lg+=[lenlist(tablistbatch[ba])] 
           res_ne+=[new_wordbatch[ba]]
           res_cs+=[csnlistbatch[ba]]

        t1 = time()
        print('function takes %f' %(t1-t0))

    df = pd.DataFrame(list(zip(res_se, res_or,res_lw,res_lg,res_ne,res_cs)), 
               columns =['sentence', 'original word', 'not-fooling words ( = path)', 'path length', 'new word','csn similarity']) 
    
    
    df.to_csv('results/results.csv', index = False)  #r
   

    #return res_se, res_or,res_lw,res_lg,res_ne,res_cs
    ###



if __name__ == '__main__':
    main()
