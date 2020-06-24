# PUBG-Dataset-Prediction
predict the final placement of the player from the final game stats

Dataset Link :- https://www.kaggle.com/c/pubg-finish-placement-prediction/overview

This dataset provided a large number of anonymized PUBG game stats, and each row contains one player's post-game stats. The data comes from matches of all types: solos, duos, squads, and custom; there is no guarantee of there being 100 players per match, nor at most 4 players per group.

Firts we do exploratory data analysis. Afterwards, we perform feature engineering, creating more insightful features that better predict the target variable. We try a variety of models for predictions: Linear Regression, Random Forests, Multilayer Perceptrons, and Gradient Boosting LightGBM. We perform postprocessing of our data to decrease our error. 
