# Try GAN
Experiments with GANs to train de-noising auto-encoders on images extracted from video.

## Set up

Confirmed with Python 3.9.12.

Clone repo.

    git clone https://github.com/rb-and-lpx-foundation/try-gan.git
    cd try-gan

Create a virtual environment and install dependencies.

    mkvirtualenv gan
    pip install -r requirements.txt
    pip install -e .

Run tests to confirm that nothing terrible has happened.

    python test.py

Begin experiments.

    jupyter notebook
