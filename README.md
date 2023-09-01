# Session 15 Assignment
To build a full transformer for language translation (Only to run for 10 epochs of training)

## Transformer Architecture
<img width="1013" alt="transformers" src="https://github.com/piygr/s15erav1/assets/135162847/610de7d6-d869-4841-bb79-ee43ba1a692e">


------
## models/TransformerV1Lightning.py
The file contains **TransformerV1LightningModel** - a Transformer model written in pytorch-lightning as desired in the assignment. 

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

## dataset.py
opus_books - HuggingFace dataset & tokenizer is used as raw dataset. The translation language pair is en (English) to it (Italian).

## S15.ipynb
The file is an IPython notebook. This was run separately in Kaggle to train the model.


```

```

## How to setup locally
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
 
