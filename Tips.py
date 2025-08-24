# Import libraries and print their versions
import pandas as pd
import numpy as np
import torch
import transformers
import nltk


print("pandas version:", pd.__version__)
print("numpy version:", np.__version__)
print("torch version:", torch.__version__)
print("transformers version:", transformers.__version__)
print("nltk version:", nltk.__version__)


msg = "Roll a dice!"
print(msg)
