# Predicting Mountain vs. Beach Vacation Preferences

**Jibril Hussaini** - CS405/CS505 Data Mining, Bishop's University

## Abstract

This project builds a classifier that predicts whether a person prefers a beach or a mountain
vacation from demographic and lifestyle features. Three models are compared with 10-fold
cross-validation: a decision tree, k-nearest neighbours, and a neural network. The decision tree and
the neural network both reach about 99.6-99.8% accuracy, while a default KNN trails at 92% until its
neighbourhood size is tuned. An overfitting analysis shows the fully grown tree memorises the
training set, and limiting its depth keeps the same accuracy with a much simpler model. A look at the
tree's feature importances shows that a few features (proximity to mountains and beaches, and the
preferred activity) explain most of the preference.

## Step 1: Choice of classifiers

I chose three classifiers from the list, picked to cover different model families:

The **decision tree** is a white-box model. It handles the mixed numeric and categorical features
directly and reports feature importances, which is exactly what the project asks for when it says to
analyse the features that influence the preference. **K-nearest neighbours** is a simple,
non-parametric baseline that makes no assumption about the shape of the decision boundary; because it
depends on distances it is sensitive to scaling and to the number of features, so it is a good
contrast to the other two. The **neural network** (a multi-layer perceptron) is a flexible,
high-capacity model that tells us whether extra modelling power actually beats a simple tree here.

I left out the SVM because a kernel SVM scales roughly with the square of the number of samples, and
with 52,444 rows that is impractical, whereas all three chosen models train in seconds.

## Dataset

The dataset has 52,444 records and 13 input features: age, gender, income, education level, travel
frequency, preferred activity, vacation budget, location, proximity to mountains, proximity to
beaches, favourite season, whether the person has pets, and an environmental-concern flag. The target
is the preference (0 = beach, 1 = mountain). The classes are imbalanced: about 75% of people prefer
beaches and 25% prefer mountains. Because of that imbalance, accuracy alone is not enough, so I also
report precision, recall and F1 for the mountain class.

The only transformation applied is in preprocessing: the five categorical columns are one-hot encoded
and the numeric columns are standardised, both fit on the training data only inside a
`ColumnTransformer`. The data is split 80% training / 20% test with stratification on the target.

## Step 2: Validation and comparison

Each model was wrapped in a pipeline with the preprocessing and assessed with 10-fold stratified
cross-validation on the training set, then evaluated once on the held-out test set.

| Model | 10-fold CV accuracy | Test accuracy | Test precision | Test recall | Test F1 |
|-------|--------------------:|--------------:|---------------:|------------:|--------:|
| Decision Tree | 0.995 | 0.996 | 0.993 | 0.992 | 0.992 |
| KNN (default) | 0.923 | 0.920 | 0.889 | 0.780 | 0.831 |
| Neural Net (MLP) | 0.998 | 0.998 | 0.999 | 0.993 | 0.996 |

The neural network is the strongest, with the decision tree a close second; both classify almost
every test case correctly. KNN is clearly behind, mostly on recall for the mountain class: with the
default of five neighbours and one-hot features it misses many of the minority cases. The
cross-validation and test numbers agree closely for every model, which means the comparison is
trustworthy and none of the models is just lucky on the test split.

### Overfitting

The decision tree was the model most at risk of overfitting. Grown without a depth limit it reaches
100% training accuracy, while its cross-validation accuracy is a little lower, which is the textbook
signature of overfitting. Plotting accuracy against `max_depth` shows the cross-validation accuracy
rising quickly and then flattening from about depth 8 to 10, while training accuracy keeps climbing
to 1.0. A tree limited to `max_depth = 10` matches the full tree's test accuracy (0.996) with far
fewer rules, so the depth limit removes the overfitting at no cost. The neural network did not show
this problem at the size used.

## Step 3: Improving the scores

KNN was the weakest model, so I tuned it with a grid search over the number of neighbours and the
weighting scheme. The best setting (15 neighbours with distance weighting) raised the test F1 from
0.831 to 0.865 and accuracy to 0.939. That improvement confirms the weakness was the default `k`
rather than the data being hard for KNN. The decision tree and the network were already near the
ceiling, so the gain there came from controlling the tree's depth rather than from raising accuracy.

The decision tree's feature importances make the result interpretable. The preference is driven
mostly by **proximity to mountains** and **proximity to beaches** (together about two-thirds of the
importance), followed by the **preferred activity** (swimming and sunbathing point to the beach). A
handful of features therefore explain almost the entire preference, which is why even a shallow tree
classifies so well.

## Step 4: Conclusion

The project delivered three validated classifiers and a fair comparison of them. The decision tree
and the neural network predicted the beach-vs-mountain preference with very high accuracy and F1,
while KNN needed tuning to become competitive. Cross-validation kept the comparison honest, the
depth study made the tree's overfitting explicit and fixable, and the feature-importance analysis
identified the variables that matter.

The main drawback is the data itself. It is large but imbalanced, so accuracy had to be read
alongside F1, and the near-perfect scores suggest the dataset is strongly separable and probably
partly synthetic, which makes the task easier than a real preference-prediction problem. The natural
next steps would be class-weighting or resampling for the imbalance, an ensemble such as a random
forest or gradient boosting, and a wider hyperparameter search, though the remaining headroom is
small because the scores are already close to the ceiling.
