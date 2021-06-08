## Requirements

- Code running in python 3.6.8
- pip freeze is stored in requirements.txt. You can install all the packages using the following command in *./.../Graphs/*:

    ```shell
    $ pip install -r requirements.txt
    ```
    
## Running the code (only PPR-regression-version)

1. Running graphsage using following command in *./.../Graphs/*:
    
    ```shell
    $ python -m graphsage.supervised_train --train_prefix ./example_data/. --model gcn --ppr --identity_dim 2 --epochs 100
    ```
    

python -m graphsage.supervised_train --train_prefix ./example_data/set-querynode-7000/toy-ppi-7000 --model gcn --ppr --identity_dim 2 --epochs 500
    
    - change parameters:
      - in ..Graphs/graphsage/supervised_train 
      - or in shell command adding e.g. `--epochs 10 --dropout .1`

2. Running graphsage using a python script for loops in *./.../Graphs/Tests/*:
  
    ```shell
    $ python test_PPR.py
    ```
    - change parameters:
        - see code (simple)

## Open TensorBoard (requires logging)

1. Run following command in *./.../Graphs*:

    ```shell
    $ tensorboard --logdir tensorboard --logdir ./sup-example_data/gcn_small_0.0100/
    ```     
2. Open [TensorBoard](http://localhost:6006/)

## Recurrent neural network (outdated)
    see other files "supervised_models_..." (need to fix only_ppr; see original file (FLAG.ppr))