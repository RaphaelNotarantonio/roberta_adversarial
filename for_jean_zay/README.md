Once and for all:

First clone this git:

```
cd $WORK
git clone https://github.com/RaphaelNotarantonio/roberta_adversarial/tree/master/for_jean_zay.git
```

And then download Stanford data:

```
! test -d download_glue_repo || git clone https://gist.github.com/60c2bdb54d156a41194446737ce03e2e.git download_glue_repo

!python download_glue_repo/download_glue_data.py --data_dir='glue_data' --tasks='SST'
```

