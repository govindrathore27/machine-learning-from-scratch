# Naieve Baeyes for Categorical Dataset
## Mushroom Classification:
![](https://miro.medium.com/max/6190/1*39U1Ln3tSdFqsfQy6ndxOA.png)


In this project we use Naive Baeyes Classifier to solve this bi-class classification problem. In this problem I have written the code from scratch for 
Naive baeyes algorithm.
Libraries used:
- Numpy 
- Scipy
- Pandas
In this type of data where each value is categorical and using any pdf is impossible then we use \
 the probabilities calculated and use them instead to calculate Likelihood Probability. \
**BUT**\
Multiplying so many values will reach beyond the capability of python(lowest value possible is of 10^(-36) so we will calculate the probabilties by converting them into **Negative Log Probability** .

## Laplace Smoothening :
It is a smoothing technique that helps tackle the problem of zero probability in the Naïve Bayes machine learning algorithm.In terms of Machine Learning the formula for Laplace smoothening,
![](https://miro.medium.com/max/700/1*1Fe0yGNqyNa9_EihzSZ4DQ.png)
- C_k : Total number of observations
- K : Value of unique features present in the dataset
- alpha : It is usually set to 1
