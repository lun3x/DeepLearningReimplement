# Comparing Shallow Versus Deep Neural Network Architectures For Automatic Music Genre Classification

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

## Dataset
The dataset should be placed in the same directory and named 'music_genres_dataset.pkl'.

## Running
./classify_music.py [FLAG OPTIONS]

| Flag          | Options         | Meaning |
| ------------- |-----------------|------------------------------------------------|
| --decay       | [const, exp]    | Learning rate - constant or exponential decay. |
| --repr-func   | [mel, cqt_note] | Spectrogram representation - mel or CQT.       |
| --net-depth   | [shallow, deep] | Network depth - shallow or deep.               |