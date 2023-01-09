# AI4Good Project Group 2A
BirdCLEF 2022 training code for group 2A.

## Setup
We recommend using a new anaconda environment with python 3.10. First clone the repo with `git clone`
and then create the environment with

```commandline
conda create -n birdclef python=3.10`
conda activate birdclef
```

## Requirements
Our pipeline was tested under several Ubuntu environments. Using Windows is not recommended
as there are problematic libraries that don't work well when using it.

Install the required modules with pip

```commandline
pip install -r requirements.txt
```



## Data Preparation
The quickest way would be to download our premade dataset directly
from [Kaggle](https://www.kaggle.com/datasets/mathiasvogel/birdclef-mel).
Put the dataset in a `datasets` folder in the project root directory.

Alternatively you can preprocess the data yourself. First download the
[original dataset](https://www.kaggle.com/competitions/birdclef-2022/data)
from Kaggle and unpack it to a separated folder in `data_processing`. We chose the folder
name `birdclef-2022`.

### Data Processing
Then change the `root` and the `out_folder` variable in `generate_specs.py` to
the `train_audio` path (found in the 2022 dataset) and your desired output folder name 
respectively. Provide also a root folder where there are background noise audio files for
data augmentation or uncomment the noise processing.

Start the dataset generation by moving to the processing directory and executing the script with

```commandline
cd data_processing
python3 generate_specs.py
```
Note that this takes a very long time. Afterwards move the generated folders inside the chosen
spectrogram root folder (data and noise) to the Kaggle
dataset folder (`birdclef-2022` in our case).

### Augmenting the Metadata
Since we changed the file structure, we must recalculate the metadata and provide group
information for the k-fold splitting. Change the directory to `model`. Then change
the root variable in the `generate_augmented_df.py` to your dataset root folder with the
csv files and the generated data and noise folders.

````commandline
cd model
python3 generate_augmented_df.py
````

The dataset is now ready for training.

## Training
Run the training with the following command
````commandline
python3 train.py --data_path ../datasets/birdclef-2022 --load_weights False
````
the ``--data_path`` argument is the relative path to the dataset. The `--load_weights` is for
loading precalculated weights during pretraining on another dataset like the 2021 data. For 3-fold
models we have example weights that match the default config.

All the necessary configurations can be set in the ``config`` file.

