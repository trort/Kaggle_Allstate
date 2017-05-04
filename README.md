# Kaggle: Allstate claims severity

This is the first [competition](https://www.kaggle.com/c/allstate-claims-severity) I participated on Kaggle.

### About the data

The data is provided by Allstate, with 188318 entries.
The features are all anonymous, with 14 continuous features scaled to be between 0 and 1 with mean around 0.5, in addition to 116 category features, 72 of which have only two types.

The goal for this competition is to predict the total loss of each claim. However, it is not a good idea to predict the loss value directly since it is highly skewed with a minimum of $0.67 and a maximum of $121012.25.

![Figure 1. Original loss value distribution](/img/orig_loss_distr.png "Figure 1. Original loss value distribution")

One trick that significantly improved prediction accuracy is to train and predict with the log of the loss value (with an offset before log transformation). This will make the distribution more similar to a normal distribution.

![Figure 2. Loss value distribution after log transformation](/img/log_loss_distr.png "Figure 2. Loss value distribution after log transformation")

### About the code

- `plot_data.py`: This is the initial step to get a rough idea about the feature/label value distributions. Noticeably, the loss is very skewed and needs a log-transformation. Depending on the models used, some features are also skewed and need special treatments. Some continuous features are highly correlated.

- `local_train_random_forest.py`: Train a random forest regression model. The part commented out are for the grid search step, used to find the optimal (in terms of C-V error) meta-parameters in the model. The final run only uses the optimal values. This file can be used for other regression models provided in the `sklearn` package with little changes.

- `local_train_xgboost.py`: Similar to the previous file, this trains a xgboost model and finds the optimal parameter set using grid search. `XGBoost` stands for the [eXtreme Gradient Boosting](http://xgboost.readthedocs.io/en/latest/model.html) model, which is similar to the gradient boost model but much more efficient. This code is different from the previous one because xgboost has an interface slightly different from the sklearn interface.

- `ensemble_generate_features.py` and `ensemble_level2_train.py`: Use the base models trained in previous steps to build a second level [stacking](https://en.wikipedia.org/wiki/Ensemble_learning#Stacking) model, which will have performance better than any of the base models. The file `ensemble_generate_features.py` reads the .pkl base model files to generate the training and prediction features for the level-2 model, and the file `ensemble_level2_train.py` actually trains the model and generates the final prediction.

### About the results

For this competition, I used 3 xgboost models, 1 neural network model, and 1 random forest model as the base models, and the xgboost model as the level-2 model. My final result is a mean absolute error of $1112 for the public test set and $1125 for the private test set, both are around $16 more than the top participant in the competition. How can I improve my score? According to the models shared by the top teams, the most effective way is to train more base models. All top teams have trained between 30 to 100 base models to build a 2-level or even 3-level model. This helps to reduce the bias in each individual model and yields a final model with less overfitting. Of cause, the base models need to be as diverse as possible, and their performance should be close to each other. As shown in the following graph, the best base model I can get is still too biased.

![Figure 3. Distribution of log-transformed loss value in training set (blue) and in test prediction (red).](/img/train_test_distr.png "Figure 3. Distribution of log-transformed loss value in training set (blue) and in test prediction (red).")
