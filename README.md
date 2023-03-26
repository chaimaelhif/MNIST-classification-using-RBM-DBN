# MNIST-classification-using-RBM-DBN
Training a non-supervised model (Deep Belief Network(using RBMs)) to Binary AlphaDigits dataset and
evaluating the non-supervised algorithm on the MNIST database with supervised learning.

To reproduce the scores and plots in the report you need to run the main.py file. 

1. Install the required librairies using "requirements.txt"
2. Run the main file:
   1. Use **"--action action"** (action in ["RBM", "DBN", "DNN5.1", "DNN5.2.1", "DNN5.2.2"]) 
   to specify which questions you want to check 
   2. Use **"--arg1 value"** (value in ["train", "test"])
       - "train" to train a model and save it
       - "test" to import a model and test it
   3. Use **"--arg2 value"** (value in [True, False]) to specify if you want to pretrain the model or not
   4. Use **"--arg3 value"** (value in ["nb_layers", "nb_neurons", "train_size"]) to specify which analysis to make:
   compare the model with and without pretraining for different set of parameters