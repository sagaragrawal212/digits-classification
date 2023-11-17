
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

# pytest cache directory #

This directory contains data from the pytest's cache plugin,
which provides the `--lf` and `--ff` options, as well as the `cache` fixture.

**Do not** commit this to version control.

See [the docs](https://docs.pytest.org/en/stable/how-to/cache.html) for more information.

docker exec -it containerid bash

docker build -t digits:v1 -f docker/Dockerfile .

docker run -it digits:v1

curl 127.0.0.1:5000

export FLASK_APP=api/app

flask run

docker run -it -p 5000:5000 digits:v1 bash

docker run -it -p 5000:5000 digits:v1
docker system prune


installl azure cli
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

az login --use-device


az acr build --file docker/Dockerfile --registry sagarmlops23 --image digits .
