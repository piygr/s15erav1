# Session 11 Assignment - (Google Colab Notebook)

## S11.ipynb
The notebook clones mainErav1.git repo using -

```
!git clone https://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@github.com/piygr/mainErav1.git
``` 

If the repo is cloned, then move the mainErav1 folder

```
%cd mainErav1
```

To fetch the latest code from the mainErav1 repo do -

```
!git pull origin main
```

We might have to install few packages for eg.
```
!pip install torch_lr_finder
!pip install grad-cam
```

Once the packages are installed, we have to import functionalities from main.py & utils.py

```
from main import *
```

We have to call init from main, in order to get the summary of the model or want to have samples from dataset

```
init(show_model_summary=True, find_lr=False)
```

To train the model, call -
```
train_model(resume=False, num_epochs=20)
```
if **resume** is set to True, it will load the previously stored checkpoint and start training the model from that.

Finally, once the model is trained.. loading the misclassified images & gradCAM can be done using -

```
from utils import plot_missclassified_preds, plot_grad_cam

plot_missclassified_preds(dataset_mean, dataset_std, count=15)
plot_grad_cam(model, dataset_mean, dataset_std, count=15, missclassified=True)
```

Happy Modeling :-) 
 
