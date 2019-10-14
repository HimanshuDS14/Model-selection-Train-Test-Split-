import pandas as pd
from sklearn import model_selection


data = pd.read_csv("Salary_Data.csv")
print(data.head(10))

independent_x = data.iloc[:,:-1].values
dependent_y = data.iloc[:,1].values

train_x , test_x , train_y , test_y = model_selection.train_test_split(independent_x , dependent_y , test_size=0.3 , random_state=0)


#test_size = This parameter decides the size of the data
# that has to be split as the test dataset.
# This is given as a fraction.
# For example, if you pass 0.5 as the value, the dataset will be split 50% as the test dataset.
# If you’re specifying this parameter, you can ignore the next parameter.

#train_size = You have to specify this parameter only if you’re not specifying the test_size.
# This is the same as test_size, but instead you tell the class what percent of the dataset you want to split as the training set.

#random_state = Here you pass an integer, which will act as the seed for the random number generator during the split.
# Or, you can also pass an instance of the RandomState class, which will become the number generator.
# If you don’t pass anything, the RandomState instance used by np.random will be used instead.