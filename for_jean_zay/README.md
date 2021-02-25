Once and for all:

First clone this git:

```
cd $WORK
git clone https://github.com/RaphaelNotarantonio/roberta_adversarial
```

And then download Stanford data and cola data:

```
git clone https://gist.github.com/60c2bdb54d156a41194446737ce03e2e.git download_glue_repo
python download_glue_repo/download_glue_data.py --data_dir='glue_data' --tasks='SST'

wget 'https://nyu-mll.github.io/CoLA/cola_public_1.1.zip'
unzip cola_public_1.1.zip
```

You also need to install transformers and advertorch.

You should then manually download models into a directory my_pretrained :

```
https://github.com/huggingface/transformers/issues/856
```

Then whenever you want to finetune roberta:

```
sbatch launch_train_model.py
```

Finally, after having created a directory results, you can attack any sentence with:

```
sbatch launch_attack.py  
```

Specify the sentences you want to attack in the python attack file you are launching.


You can also finetune distilbert with launch_train_model.py and change the file that is run in launch_attack.py in order to attack distilbert rather than roberta.
