# Import necessary libraries
import os
# Import models trained on synthetic data
from CharacterDescriptors_Example import SVM_digits_clf, SVM_alpha_clf

## Directory with cropped characters (modify accordingly)
script_dir = os.path.abspath(__file__)
cropped_characters = os.path.join(script_dir, "../CharacterSegmentation/cropped_charcters")

