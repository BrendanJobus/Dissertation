# FedMD
FedMD: Heterogenous Federated Learning via Model Distillation. 
Preprint on https://arxiv.org/abs/1910.03581.

## Run scripts on Google Colab

1. open a google Colab

2. Clone the project folder from Github
```
! git clone github_link
```

3. Then access the folder just created. 
```
% cd project_folder/
```

4. Run the python script in Colab. For instance 
``` 
! python FEMNIST_Balanced.py -conf conf/EMNIST_balance_conf.json
```


In order to run the project on google colab or on python 3.10, the optimizers must be set to keras.optimizers.legacy.Adam and unpack_x_y_sample weight must be gotten from keras.src.engine.data_adapter.unpack_x_y_sample_weight

