
how to setup :
install conda
conda create -n digits python=3.9
conda activate digits
pip install -r requirements.txt
dummy commit
how to run

python exp.py


feature
    - vary model hyper parameters


Meaning of failure :
    - poor model metrics
    - coding runtime/compile error

    - test model gave bad predictions on test samples during the demo

Places of randomness :
    - creating the split train test so for reproducibility freezing the data is important shuffle = False
    - weight initialisation of model