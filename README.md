# LPR_FLAIR

## Quick Start

Execute the following commands to set up the Conda environment and run the project:

```bash
git clone -b gcp-environment-setup https://github.com/RasaraThathsarana/LPR_FLAIR.git
cd LPR_FLAIR
bash setup.sh
tmux new -s flair
conda activate FLAIRHUB
bash run.sh
```

## Hardware Requirements & Training Details
* GPU Requirement: Assumes training on an NVIDIA A100 80GB GPU.

* Storage Requirement: Minimum 350GB storage required.

* Training Configuration: Batch size is set to 32, and gradient checkpointing is disabled.

* Execution Behavior:

  * This will run 5 training processes consecutively if not terminated.

  * Each training process is initialized with a random seed value.
