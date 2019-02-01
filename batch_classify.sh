#!/bin/bash

#SBATCH -t 0-02:00 # Runtime in D-HH:MM
#SBATCH -p gpu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=15000
#SBATCH --account=comsm0018       # use the course account
#SBATCH -J classify    # name
#SBATCH -o classify_%N_%j.out # File to which STDOUT will be written
#SBATCH -e classify_%N_%j.err # File to which STDERR will be written
module add libs/tensorflow/1.2
module load languages/anaconda2/5.0.1
pip install --user librosa
python classify_music.py
