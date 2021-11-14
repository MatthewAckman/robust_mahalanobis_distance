# robust_mahalanobis_distance
A modular application of mahalanobis distance using robust estimation of mean and covariance for n-dimensional outlier rejection.

This project presents a modular version of the mahalanobis distance algoroithm. This is similar in principle to the `mahalanobis()` module found in SciPy distributions, however this is further adapated to allow for robust estimates of mean and covariance, greatly improving the algorithm's performance in select cases. 

Shown here is a case where a distribution of observations contain a very large amount of outliers belonging to a much wider distribution. Graphically (below), the desired distribution is the cluster spread horizontally, whereas the contaminant distribution is the cluster spread vertically. In this example, both distributions contain the same number of observartions.

The middle chart presents the data as viewed by a non-robust mahalnobis distance algorithm, where it is unsuccessful as distinguishing the target and contaminant distributions. In a real-world application, this would result in very few of the outliers being rejected from the sample.

The rightmost chart presents the data as viewed by a robust mahalanobis algorithm, where the two distributions are much more accurately distinguished. In a real-world application, would would mean that a very large amount of noise can safely be rejected from the dataset.

![alt text](https://github.com/MatthewAckman/robust_mahalanobis_distance/blob/43c97eb4abd8f5f63f5422b469d70e28579b0391/example.svg?raw=True)
