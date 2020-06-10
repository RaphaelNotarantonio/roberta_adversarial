# roberta_adversarial

Adversarial attacks have given very efficient results applied to image classification. Here, we work on applying adversarial attacks to some NLP tasks.

We download a RoBerta pretrained model from hugging face and then finetune it on a sequence classification task on a CoLA dataset. 
Then we use Projective Gradient Descent (PGD) on each batch to find the embedding in a defined neighborhood which maximises the loss on our model. Finally we check the accuracy of our model on this adversarial dataset.

In roberta_class, we try create two classes dividing our roberta model (to be done).
In plot_attack_rob, we plot PGD to find best adversarial maximising our model loss.
In plot_attack_roberta, we plot PGD with advertorch to find best adversarial maximising our model loss.
