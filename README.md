# Linear Regression
 
Ecommerce company based in New York City sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
The company is trying to decide whether to focus their efforts on their mobile app experience or their website. Customer information is fake.

Features provided are: Email, Address, Avatar, Avg. Session Length, Time on App, Time on Website, Length of Membership, Yearly Amount Spent.

First, data analysis is performed. Yearly Amount Spent is compared against Time on Website and Time on App. Jointplot of Time on App and Length of Membership is shown. Correlation of all the features is examined using seaborn's pairplot, which showed that the most correlated feature with Yearly Amount Spent is the Length of Membership. Data is then splitted into training (70%) and test set (30%) and fitted to a linear regression model. Test set is used for prediction and true values are plotted against predicted values. For error calculation, MAE, MSE and RMSE are used. Histogram of the residuals is plotted. The biggest effect on Yearly Amount Spent is determined to have Length of Membership, and then Time on App.
