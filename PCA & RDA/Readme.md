# PCA & RDA
- ## PCA :
(Principal Components Analysis) gives us our ideal set of features. 
It creates a set of principal components that are rank ordered by variance (
the first component has higher variance than the second, the second has higher variance than the third, and so on), uncorrelated, and low in number (w
e can throw away the lower ranked components as they contain little signal).

![](https://miro.medium.com/max/1200/1*ba0XpZtJrgh7UpzWcIgZ1Q.jpeg)


- ## RDA :
-  Regularized discriminant analysis uses the same general setup as LDA and QDA but estimates the covariance in a new way, which combines the covariance of QDA using (Σk) and with the covariance of LDA (Σ) using tuning parameter λ.

![rda_1]()

Other interpretation of this will be :
![rda_2]()
Both γ and λ can be thought of as mixing parameters, as they both take values between 0 and 1. For the four extremes of γ and λ, the covariance structure reduces to special cases:
- (γ=0,λ=0): QDA - individual covariance for each group.
- (γ=0,λ=1): LDA - a common covariance matrix.
- (γ=1,λ=0): Conditional independent variables - similar to Naive Bayes, but variable variances within group (main diagonal elements) are all equal.
- (γ=1,λ=1): Classification using euclidean distance - as in previous case, but variances are the same for all groups. Objects are assigned to group with nearest mean.

Link to the research for [RDA](https://web.stanford.edu/~hastie/Papers/RDA-6.pdf)
