# Comparing Shallow Versus Deep Neural Network Architectures For Automatic Music Genre Classification
In [this report](Deep_Learning_Report.pdf) we examine [the paper by A. Schindler, T. Lidy and A. Rauber](https://pdfs.semanticscholar.org/4614/0d548991b4fc9d03dcc412d398bf9b17dae9.pdf). Firstly detailing similar works and their relation to the paper, then re-implementing the architecture described in the paper in order to reproduce the results found. We then propose 2 further adaptations and show how these improve upon the results found.

## Dataset
The dataset should be placed in the same directory and named 'music_genres_dataset.pkl'.

## BlueCrystal Imports
Ensure numpy and tensorflow are on the correct versions.
```
module load languages/anaconda2/5.0.1
module add libs/tensorflow/1.2
```

## Running

```
python classify_music.py [FLAG OPTIONS]
```

| Flag            | Options         | Meaning |
| ----------------|-----------------|------------------------------------------------|
| --decay         | [const, exp]    | Learning rate - constant or exponential decay. |
| --repr-func     | [mel, cqt_note] | Spectrogram representation - mel or CQT.       |
| --net-depth     | [shallow, deep] | Network depth - shallow or deep.               |
| --max-steps     | [integer > 0]   | Number of epochs to run for.                   |
| --learning-rate | [float > 0]     | Learning rate (initial rate if exp. decay used)|

## Class Indices:
|Name     |Index|
|---------|-----|
|blues    | 0   |
|classical| 1   |
|country  | 2   |
|disco    | 3   |
|hiphop   | 4   |
|jazz     | 5   |
|metal    | 6   |
|pop      | 7   |
|reggae   | 8   |
|rock     | 9   |
