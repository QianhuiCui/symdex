# Symmetry Dexterity (SYMDEX)
This repository provides a PyTorch implementation of the paper [Morphologically Symmetric Reinforcement Learning for Ambidextrous Bimanual Manipulation](https://arxiv.org/abs/2505.05287).

---

- [:books: Citation](#citation)
- [:gear: Installation](#installation)
    - [Install IsaacLab](#install_isaac)
    - [Install Third Party Package](#install_third)
    - [Install SYMDEX](#install_symdex)
- [:scroll: Usage](#usage)
    - [:pencil2: Logging](#usage_logging)
    - [:bookmark: Run with Random Actions](#usage_random)
    - [:bulb: Train with SYMDEX](#usage_symdex)
    - [:floppy_disk: Saving and Loading](#usage_saving_loading)


## :books: Citation

```
@article{li2025morphologically,
  title={Morphologically Symmetric Reinforcement Learning for Ambidextrous Bimanual Manipulation},
  author={Li, Zechu and Jin, Yufeng and Apraez, Daniel Ordonez and Semini, Claudio and Liu, Puze and Chalvatzaki, Georgia},
  journal={arXiv preprint arXiv:2505.05287},
  year={2025}
}
```

## :gear: Installation

### Install IsaacLab <a name="install_isaac"></a>

> **Note**
> For reproducibility, we use an old version of IsaacLab.

1. Clone IsaacLab from [repo](https://github.com/supersglzc/IsaacLab.git)

2. Create Conda environment and install isaacsim 4.5.0:
    ```bash
    conda create -n symdex python=3.10
    conda activate symdex
    pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
    pip install "isaacsim[all,extscache]==4.5.0" --extra-index-url https://pypi.nvidia.com
    ```
3. Install IsaacLab
    ```bash
    cd IsaacLab
    ./isaaclab.sh --install
    ```
    
### Install Third Party Package <a name="install_third"></a>

1. Clone the MorphoSymm:

    ```bash
    git clone https://github.com/Danfoa/MorphoSymm
    cd MorphoSymm
    ```

2. Install the package:

    ```bash
    pip install -e .
    ```
    
### Install SYMDEX <a name="install_symdex"></a>

1. Clone the package:

    ```bash
    git clone git@github.com:symdex.git
    cd symdex
    ```

2. Install the package:

    ```bash
    pip install -e .
    ```

## :scroll: Usage

### :pencil2: Logging <a name="usage_logging"></a>

We use Weights & Biases (W&B) for logging. 

1. Get a W&B account from https://wandb.ai/site

2. Get your API key from https://wandb.ai/authorize

3. set up your account in terminal
    ```bash
    export WANDB_API_KEY=$API Key$
    ```

### :bookmark: Run with Random Actions <a name="usage_random"></a>

1. Download the [asset](https://drive.google.com/file/d/1R3xygLI2OqXtpEcva_L_ORngKajAE8S6/view?usp=sharing) folder and place it in the root.

2. Run drawer-insert task with random actions

```bash
python random_actions.py num_envs=1 task=insertDrawer
```


### :bulb: Train with SYMDEX <a name="usage_symdex"></a>
> **Note**
> The available tasks include `insertDrawer`, `boxLift`, `pickObject`, `stirBowl`, `threading`, `handover`.

Run SYMDEX on drawer-insert task.

```bash
python train.py task=insertDrawer save_model=True
```


### :floppy_disk: Saving and Loading <a name="usage_saving_loading"></a>

Checkpoints are automatically saved as W&B [Artifacts](https://docs.wandb.ai/ref/python/artifact).

To load and visualize the policy, run

```bash
python visualize.py task=insertDrawer num_envs=4 artifact=$team-name$/$project-name$/$run-id$/$version$
```
