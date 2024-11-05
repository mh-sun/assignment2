# Training a Transformer model for Predicting if statements


# Prerequisites
In order to run this project in your local machine. You have to follow some procedure.
## Clone the repository
```bash
 git clone https://github.com/mh-sun/assignment2.git
```
Before using this script, make sure you have the following installed on your local machine:

- Python (3.10+ recommended)
- Required Python libraries (specified in `requirements.txt`)

You can install the required libraries by running:

```bash
pip install -r requirements.txt
```

Model: Code-T5-Base

Files:
-a_model_pretrain.py: Pretrain Code-T5 on Code Search Net
-b_fine_tune.py: Finetune and evaluate Code-T5 on Code Search Net

Usage
To pre-train the model, follow these steps:

```bash
python a_model_pretrain.py
```

arguments:
 - '-e', '--epoch' : Epoch
 - '-m', '--mask' : Mask Token Portion
 - '-o', '--out' : Test Result Path

 To finetune the model, follow these steps:

```bash
python b_fine_tune.py
```

arguments:
 - '-e', '--epoch' : Epoch
 - '-l', '--learning_rate' : Learning Rate
 - '-o', '--out' : Test Result Path

## Data
The data we've collected from SEART, that data is available here [link](https://drive.google.com/drive/folders/100X2rtYo3oV4Rt9cPjkDi3z2hU9_csr7?usp=sharing) and our model is available: [link](https://drive.google.com/drive/folders/1nPAh0l4rFAgXsQsad_D096EVVKozLuA2?usp=sharing)
