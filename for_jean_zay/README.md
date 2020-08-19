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

Then whenever you want to finetune roberta:

```
sbatch launch_model_train.py
```

Finally, you can attack any sentence with:

```
sbatch launch_attack.py --iid 0 --indlist [5,8] --eps 0.3 --epscand 0.1 --nb_iter 100 --eps_iter 0.5 --rayon 0.3 --ord np.inf
```

where iid is the index number of the sentence and indlist the list of the indexes of the words to be changed
