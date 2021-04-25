import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics


def main():
    # the goal is to build a model that will tell if a company should focus more on the mobile app or the website

    # dataframe with data
    customers = pd.read_csv('Ecommerce Customers')

    # information about data
    print(customers.head())
    print(customers.info())
    print(customers.describe())

    # jointplot comparing the Time on Website and Yearly Amount Spent columns
    sns.set_palette("GnBu_d")
    sns.set_style('whitegrid')
    sns.jointplot(x = 'Time on Website', y = 'Yearly Amount Spent', data = customers)

    # the same but with the Time on App column instead
    sns.jointplot(x = 'Time on App', y = 'Yearly Amount Spent', data = customers)

    # jointplot of a 2D hex bin plot comparing Time on App and Length of Membership
    sns.jointplot(x = 'Time on App', y = 'Length of Membership', kind = 'hex', data = customers)

    # relationships across the entire data set
    sns.pairplot(customers)

    # Based off this plot, Length of Membership looks to be the most correlated feature with Yearly Amount Spent

    # a linear model plot of Yearly Amount Spent vs. Length of Membership
    sns.lmplot(x = 'Length of Membership', y = 'Yearly Amount Spent', data = customers)

    # X equals to the numerical features of the customers and y equals to the "Yearly Amount Spent" column
    X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
    y = customers['Yearly Amount Spent']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 101)

    # instance of a LinearRegression() model
    lm = LinearRegression()

    # training/fitting lm on the training data
    lm.fit(X_train, y_train)

    # coefficients of the model
    print(lm.coef_)

    # predicting off the X_test set of the data
    predictions = lm.predict(X_test)

    # scatterplot of the real test values versus the predicted values
    plt.scatter(y_test, predictions)
    plt.xlabel('Y test')
    plt.ylabel('Predicted Y')

    # Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error
    print(metrics.mean_absolute_error(y_test, predictions))
    print(metrics.mean_squared_error(y_test, predictions))
    print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    # histogram of the residuals
    sns.displot(y_test - predictions, bins = 50)

    # dataframe with weight coefficients for Avg. Session Length, Time on App, Time on Website and Length of Membership
    coeffDF = pd.DataFrame(data = lm.coef_, index = X.columns, columns = ['Coefficient'])
    print(coeffDF)
    #                       Coeffecient
    # Avg. Session Length	25.981550
    # Time on App	        38.590159
    # Time on Website	    0.190405
    # Length of Membership	61.279097

    # The company should focus more on Mobile app if everything stays the same. On website if they want to improve the
    # website so that it catches up with the mobile app.

    plt.show()


if __name__ == '__main__':
    main()