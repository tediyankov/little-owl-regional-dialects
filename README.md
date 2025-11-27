# Little Owl Regional Dialect Analysis

Welcome to my bioacoustic CNN project! In this code, data on Little Owls (_Athene noctua_) are analysed using a Convolutional Neural Network (CNN) to investigate whether they exhibit regional (country-level) vocal variation. 

First, clone this repo, and run the requirements.txt file in a virtual environment with Python 3.9 or higher.

Then, in order to run the code, please make sure you add the following folders into the project directory after cloning it: 

```
- ./data
- ./spectrograms
```

Now, make sure to add your Xeno Canto API key as an environment variable.

Afterwards, the code is structured as follows:

1. **fetch_data.py**: calls the Xeno-Canto API to give all quality A recordings of our target species - Athene noctua - since our specified date (1 Jan 2000)
2. **data_preprocess.py**: converts extracted audiofiles into spectrograms for analysis
3. **model_train.py**: trains a shallow CNN on countries label-encoded into categorical classes.
4. **eval.py**: evaluates precision, recall, f1-score and accuracy of the model predictions on a 20% validation set.

**TO-DO**

- data-preprocessing
  - expand training sample to worse quality
  - object detection model to snip the recordings to contain the calls
  - separate single call spectrograms and multiple call spectrograms
  - add a feature counting how many calls.
- re-train after augmentation changes
- deal with class imbalances, and increase threshold of representation 
