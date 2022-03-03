# A repo for amlt yaml file

> Created by v-ziyangma on 2022.3.3.

## Usage

```bash
# available resources
amlt target info <service> #<service> = amlk8s|aml|sing

# run a job
amlt run data2vec/data2vec.yaml data2vec_train_960h_devclean
```

## Set up the multi-node multi-gpu cluster

### data2vec

1. download the code including the bash script.
    ```bash
    git clone -b v-ziyangma https://github.com/ddlBoJack/fairseq.git
    cd fairseq
    ```

2. edit submit_script/ITP_bash_scripts/data2vec/data2vec_audio_train.sh.
    ``` 
    exp:              include the model name and the exp name.
    config:           Pay attention that we use the yaml file in config/data2vec to specify the parameters.
    data:             the data path and the file name.
    compute resource: original data2vec use 16*48GB-GPU, with the max_tokens=3800000. we need to simulate it with update_freq.
    ckpt:             where to save the checkpoints and load the checkpoints.
    log:              where to save the output logs and the tensorboard logs.
    ``` 

3. go back to edit the amlt yaml file to submit the job.
    ```
    code:
      local_dir: where you git clone the upper code.
    jobs:
    - name: data2vec_960h_devclean
      sku: 2xG8
      command:
        - bash submit_script/ITP_bash_scripts/data2vec/data2vec_audio_train.sh
    ```

## Pay Attention

1. The file submit_script/ITP_bash_scripts/data2vec/data2vec_audio_conda.sh setup a conda environment for the user before running the job.
2. We use the yaml file in config/data2vec to specify the parameters. Instead, the command line parameters are not like "--max-tokens 3800000". It is like "dataset.max_tokens=3800000". Writing the low-change-frequency parameters in the yaml file is recommended.
3. we do NOT need to specify the sku_count and the aml_mpirun in the lateset version of amlt.