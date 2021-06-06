import os
os.chdir("..")
print(os.getcwd())
# n_neurons = [10]
identity_dim = [0,1, 8, 9, 100, 150]
# n_neurons = [x - 1 for x in n_neurons]
n_dropout = [.0]
# n_neurons = [10, 30, 100, 300, 1000, 10000]
# n_neurons = [x - 1 for x in n_neurons]
# n_dropout = [.0, .1, .2, .3, .4]

for i in identity_dim:
    for nd in n_dropout:
        bashcommand = "python -m graphsage.supervised_train --train_prefix ./example_data/toy-ppi --model gcn --ppr"
        bc_id = " --identity_dim " + str(i)
        bc_dropout = " --dropout " + str(nd)
        bashcommand += bc_id + bc_dropout
        print(bashcommand)
        os.system(bashcommand)
