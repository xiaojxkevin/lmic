# Disscussions on *Language Modeling is Compression*

## Installation

`pip install -r requirements.txt` will install all required dependencies.
This is best done inside a [conda environment](https://www.anaconda.com/).
To that end, install [Anaconda](https://www.anaconda.com/download#downloads).

Then, run the following commands:

```bash
# Clone the source code into a local directory:
git clone https://github.com/google-deepmind/language_modeling_is_compression.git
cd language_modeling_is_compression

# Create and activate the conda environment:
conda create --name lmic
conda activate lmic

# Install `pip` and use it to install all the dependencies:
conda install pip
pip install -r requirements.txt
```

If you have a GPU available (highly recommended for fast training), then you can install JAX with CUDA support.
```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
Note that the jax version must correspond to the existing CUDA installation you wish to use (CUDA 12 in the example above).
Please see the [JAX documentation](https://github.com/google/jax#installation) for more details.

## Usage

Before running any code, make sure to activate the conda environment and set the `PYTHONPATH`:

```bash
conda activate lmic
export PYTHONPATH=$(pwd)/..
```

If you want to compress with a language model, you need to train it first using:
```bash
python train.py
```

To evaluate the compression rates, use:
```bash
python compress.py
```

To test if we can decode the encoded text from the language model:
```bash
python test.py
```