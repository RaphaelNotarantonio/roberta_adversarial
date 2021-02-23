# roberta_adversarial

Adversarial attacks have given very efficient results applied to image classification. Here, we work on applying adversarial attacks to some NLP tasks.

We download a RoBerta pretrained model from hugging face and then finetune it on a sentiment classification task on a Stanford treebank dataset. 
Then we use Projective Gradient Descent (PGD) on each batch to find the embedding in a defined neighborhood which maximises the loss on our model. The last step will consist in checking the accuracy of our model on this adversarial dataset.

# jean-zay

Please go to for_jean_zay folder in order to use our algorithms.
