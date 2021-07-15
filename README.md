# UniversalTransformerTwitter
Twitter Emotion Classifier using a Universal Transformer. In this implementation of the Universal Transformer the encoder of a Transformer is used, but all the layers are forced to have identical weights. This allows for a variable number of layers in the encoder architecture.

# Objective

NLP classification of twitter tweets into one of six emotions: love, joy, fear, anger, surprise, sadness.
The dataset is described in https://www.kaggle.com/praveengovi/emotions-dataset-for-nlp?select=train.txt

The model architecture is restricted to use of the universal transformer (identical layer weights).


# Requirements

python3.4 or above

## Necessary Packages (installation with PyPI)

pip install torch, torchvision

pip install transformers


# Training

Fork the repository (and clone).

Run the _train.py_ scripts with desired arguments in your terminal. For example to train a universal transformer (UT) encoder with 12 layers:

_python ./train.py classifier12.th --B=8 --lr=0.00001 --epochs=5 --layers=12_

# Experimental Results

| Model Architecture | Test Accuracy (%) |
| ----------------- | :-----------------: |
UT (12 layers) encoder + classification head |  |

### Training Details

- Batch Size = 8
- Epochs = 
- Learning Rate = 
