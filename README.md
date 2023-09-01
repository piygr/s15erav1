# Session 15 Assignment
To built full transformer for language translation (Only to run for 10 epochs training)


------
## models/TransformerV1Lightning.py
The file contains the Transformer Lightning model as desired in the assignment. 

Here is the summary of the network -

```
  | Name             | Type               | Params
--------------------------------------------------------
0 | encoder          | Encoder            | 12.6 M
1 | decoder          | Decoder            | 18.9 M
2 | projection_layer | ProjectionLayer    | 11.5 M
3 | src_embed        | InputEmbeddings    | 8.0 M 
4 | tgt_embed        | InputEmbeddings    | 11.5 M
5 | src_pos          | PositionalEncoding | 0     
6 | tgt_pos          | PositionalEncoding | 0     
7 | loss_fn          | CrossEntropyLoss   | 0     
--------------------------------------------------------
62.5 M    Trainable params
0         Non-trainable params
62.5 M    Total params
250.151   Total estimated model params size (MB)
```

## transforms.py
The file contains trasforms which are applied to the input dataset as per the assignment requirement.
```
t = T.Compose(
        [
            T.RandomCrop( (32, 32), padding=4, fill=(mean[0]*255, mean[1]*255, mean[2]*255) )
        ]
    )

    a = A.Compose(
        [
            A.Normalize(mean, std),
            A.HorizontalFlip(p=p),
            A.CoarseDropout(max_holes = 1,
                            max_height=8,
                            max_width=8,
                            min_holes = 1,
                            min_height=8,
                            min_width=8,
                            fill_value=mean,
                            mask_fill_value = None,
                            p=p
            )
        ]
    )
```

## dataset.py
CustomCIFAR10Dataset is created on top of CIFAR10 to take care of albumentation + torchvision transforms

## utils.py
The file contains utility & helper functions needed for training & for evaluating our model.

## S10.ipynb
The file is an IPython notebook.
The LRfinder has been used to find the Max LR. Multiple LR values were tried starting from 0.1 to 1e-8 to find that where the loss is lowest. Based on that the Max LR value is set to 3.20E-04.

**LRFinder**

<img width="705" alt="Screenshot 2023-07-14 at 11 39 09 PM" src="https://github.com/piygr/s10erav1/assets/135162847/20d5404e-d580-4b50-b246-39b1292c40ab">


```
Epoch 1 LR [3.2e-06]
Test set: Average loss: 1.4072, Accuracy: 4903/10000 (49.03%)

Epoch 2 LR [6.656000000000011e-05]
Test set: Average loss: 1.1207, Accuracy: 5979/10000 (59.79%)

Epoch 3 LR [0.0001299199999999999]
Test set: Average loss: 0.9486, Accuracy: 6603/10000 (66.03%)

Epoch 4 LR [0.00019328]
Test set: Average loss: 0.7799, Accuracy: 7267/10000 (72.67%)

Epoch 5 LR [0.0002566400000000001]
Test set: Average loss: 0.7548, Accuracy: 7336/10000 (73.36%)

Epoch 6 LR [0.00031999999999999986]
Test set: Average loss: 0.6213, Accuracy: 7901/10000 (79.01%)

Epoch 7 LR [0.0003033263157894737]
Test set: Average loss: 0.6122, Accuracy: 7897/10000 (78.97%)

Epoch 8 LR [0.0002866526315789474]
Test set: Average loss: 0.5619, Accuracy: 8130/10000 (81.30%)

Epoch 9 LR [0.0002699789473684211]
Test set: Average loss: 0.4941, Accuracy: 8326/10000 (83.26%)

Epoch 10 LR [0.0002533052631578948]
Test set: Average loss: 0.4339, Accuracy: 8541/10000 (85.41%)

Epoch 11 LR [0.0002366315789473684]
Test set: Average loss: 0.4452, Accuracy: 8496/10000 (84.96%)

Epoch 12 LR [0.00021995789473684213]
Test set: Average loss: 0.4931, Accuracy: 8367/10000 (83.67%)

Epoch 13 LR [0.0002032842105263158]
Test set: Average loss: 0.4949, Accuracy: 8346/10000 (83.46%)

Epoch 14 LR [0.00018661052631578954]
Test set: Average loss: 0.3910, Accuracy: 8699/10000 (86.99%)

Epoch 15 LR [0.0001699368421052631]
Test set: Average loss: 0.4118, Accuracy: 8601/10000 (86.01%)

Epoch 16 LR [0.00015326315789473684]
Test set: Average loss: 0.4488, Accuracy: 8537/10000 (85.37%)

Epoch 17 LR [0.0001365894736842106]
Test set: Average loss: 0.3487, Accuracy: 8860/10000 (88.60%)

Epoch 18 LR [0.00011991578947368415]
Test set: Average loss: 0.3302, Accuracy: 8902/10000 (89.02%)

Epoch 19 LR [0.0001032421052631579]
Test set: Average loss: 0.3080, Accuracy: 8962/10000 (89.62%)

Epoch 20 LR [8.656842105263165e-05]
Test set: Average loss: 0.3482, Accuracy: 8859/10000 (88.59%)

Epoch 21 LR [6.98947368421052e-05]
Test set: Average loss: 0.2918, Accuracy: 9047/10000 (90.47%)

Epoch 22 LR [5.3221052631578956e-05]
Test set: Average loss: 0.2971, Accuracy: 9041/10000 (90.41%)

Epoch 23 LR [3.6547368421052686e-05]
Test set: Average loss: 0.2621, Accuracy: 9159/10000 (91.59%)

Epoch 24 LR [1.9873684210526257e-05]
Test set: Average loss: 0.2544, Accuracy: 9170/10000 (91.70%)
```

## How to setup
### Prerequisits
```
1. python 3.8 or higher
2. pip 22 or higher
```

It's recommended to use virtualenv so that there's no conflict of package versions if there are multiple projects configured on a single system. 
Read more about [virtualenv](https://virtualenv.pypa.io/en/latest/). 

Once virtualenv is activated (or otherwise not opted), install required packages using following command. 

```
pip install requirements.txt
```

## Running IPython Notebook using jupyter
To run the notebook locally -
```
$> cd <to the project folder>
$> jupyter notebook
```
The jupyter server starts with the following output -
```
To access the notebook, open this file in a browser:
        file:///<path to home folder>/Library/Jupyter/runtime/nbserver-71178-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
     or http://127.0.0.1:8888/?token=64bfa8105e212068866f24d651f51d2b1d4cc6e2627fad41
```

Open the above link in your favourite browser, a page similar to below shall be loaded.

![Jupyter server index page](https://github.com/piygr/s5erav1/assets/135162847/40087757-4c99-4b98-8abd-5c4ce95eda38)

- Click on the notebook (.ipynb) link.

A page similar to below shall be loaded. Make sure, it shows *trusted* in top bar. 
If it's not _trusted_, click on *Trust* button and add to the trusted files.

![Jupyter notebook page](https://github.com/piygr/s5erav1/assets/135162847/7858da8f-e07e-47cd-9aa9-19c8c569def1)
Now, the notebook can be operated from the action panel.

Happy Modeling :-) 
 
