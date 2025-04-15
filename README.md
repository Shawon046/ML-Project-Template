# ML-Project-Template
Basic ML Project with Hydra


## Virtual Environment Creation

### Creating env from the exported env
```
cd dependencies/
conda env create -f environment.yml
```

### Creating env from a simple base
1. Create the environment from the environment.yml file:
```
conda env create -f env.yml
```
2. Activate the new environment: conda activate myenv
```
conda activate llm-nids
```
3. Exporting the env 
```
conda env export > environment.yml
```
4. Deactivating the env
```
conda deactivate
```
5. Removing the environment
```
conda remove --name llm-nids --all
```
To verify that the environment was removed, run:
```
conda info --envs
```

## Run the program
Download the nsl-kdd dataset from the drive and put it inside the nsl-kdd folder. To unzip the folder, 
```

```
Change corresponding variable in conf/config.yaml file, if needed. 
Run the main file with:
```
cd src/
nohup python main.py > dummy.log 2>&1 &
nohup python main.py model='llm-agent'> output.log 2>&1 &
```
