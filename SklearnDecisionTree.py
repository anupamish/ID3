#Team Members:
# Dhruv Bajpai - dbajpai - 6258833142
# Anupam Mishra - anupammi - 2053229568
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


def main():
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    combined_data = pd.concat([train_df,test_df])
    # To one-hot encode simultaneously. There may be values in the test set not present in the training set.
    # Good practice to concat and do this.

    # One-Hot-Encoding the data
    cd = pd.get_dummies(combined_data,columns=combined_data.columns.drop("Enjoy"))

    cd = cd.loc[:,cd.columns!="Enjoy"]
    train_data = cd.iloc[:-1,:]

    test_data = cd.iloc[-1,:]

    test_data = test_data.values.reshape(1,17)

    train_y = train_df["Enjoy"]

    le = LabelEncoder()
    train_y = le.fit_transform(train_y)


    # Forming the decision Tree with train data
    dtree = DecisionTreeClassifier(criterion="entropy",random_state=100)
    dtree.fit(train_data.as_matrix(),train_y)

    # Testing
    print ("Prediction on test data : Enjoy == {}".format(le.inverse_transform(dtree.predict(test_data))))

    # Printing the decision tree
    tree.export_graphviz(dtree,out_file="tree1.dot")


if __name__ == '__main__':
    main()

