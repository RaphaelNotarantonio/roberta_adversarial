Once and for all:

First clone this git:

```
cd $WORK
git clone https://github.com/RaphaelNotarantonio/roberta_adversarial
```

And then download Stanford data:

```
git clone https://gist.github.com/60c2bdb54d156a41194446737ce03e2e.git download_glue_repo

python download_glue_repo/download_glue_data.py --data_dir='glue_data' --tasks='SST'
```

You also need to install transformers and advertorch.

Then whenever you want to finetune roberta:

```
sbatch launch_model_train.py
```

Finally, you can attack any sentence with:

```
sbatch launch_attack.py  
```
