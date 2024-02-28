##Allows me to download some data 
from ucimlrepo import fetch_ucirepo 
##Other imports 
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import itertools
from sklearn.ensemble import RandomForestClassifier

##Testing the best value of C to use for logistic regression
def setC(X, y, y_test, X_test, model):
    C = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    scores = []
    hAcc = 0 
    for i in C:
        model.set_params(C=i)
        model.fit(X,y)
        #scores.append(model.score(X, y))
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        if accuracy > hAcc:
            hAcc = i
        scores.append(accuracy)
    print(f"highest accuracy comes from {hAcc}")
    return(scores) ##No longer needed but good for checking tht logic is working properly 

##Function shows the best columns you can use to train the model 
def chooseFactors(X, y):
    colNum = X.shape[1]
    best_avg = 0
    best_combination = None
    best_y_pred = None
    best_y_test = None

    # Loop over all combinations of columns
    for r in range(1, colNum + 1):
        for columns in itertools.combinations(X.columns, r):
            X_subset = X[list(columns)]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)

            # Create and train the model
            model = LogisticRegression(C=1)
            model.fit(X_train, y_train.values.ravel())

            # Make predictions
            y_pred = model.predict(X_test)

            # Calculate metrics
            Accuracy = accuracy_score(y_test, y_pred)
            Precision = metrics.precision_score(y_test, y_pred, average='micro')
            Sensitivity_recall = metrics.recall_score(y_test, y_pred, average='micro')
            Specificity = metrics.recall_score(y_test, y_pred, pos_label=0, average='micro')
            F1_score = metrics.f1_score(y_test, y_pred, average='micro')

            # Calculate average score
            avg = (Accuracy + Precision + Sensitivity_recall + Specificity + F1_score) / 5

            # If this combination is the best so far, save it
            if avg > best_avg:
                best_avg = avg
                best_combination = columns
                best_y_pred = y_pred
                best_y_test = y_test

    print(f"Best average score is {best_avg} with columns {best_combination}\n\n\n\n\n\n\n\n")
    return X[list(best_combination)], best_y_pred, best_y_test

## Cleans NAs and collapses heart disease samples together    
def NACleanup(X, y):
    # Combine X and y into a single DataFrame
    data = pd.concat([X, y], axis=1)

    # Drop the rows where at least one element is missing
    data = data.dropna()

    # Split the data back into X and y
    X_clean = data[X.columns]
    y_clean = data[y.columns]
    #Collapsing 1,2 and 3 together because they all have heart disease, now we have presence and absense 
    y_clean['num'] = y_clean['num'].apply(lambda x: 1 if x in [1, 2, 3] else 0)
    return(X_clean, y_clean)

## Creats train/test split and fits linear regression
def trainTest(X,y):
    regr = linear_model.LinearRegression().fit(X, y)
    regr.fit(X, y) ##Fitting the values via linear regression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return(X_train, X_test, y_train, y_test)

## Fitting the model after doing train/test split returns confusion matrix
def fitModel(X_train, X_test, y_train, y_test):
    model = LogisticRegression()

    # Train the model
    model.set_params(C=2)
    model.fit(X_train, y_train.values.ravel())

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_pred)

    #print(f'Accuracy of logistic regression: {accuracy}')
    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    return(confusion_matrix)
    

##--------------------------------------------------------------------------------------
## MAIN
def main():
    heart_disease = fetch_ucirepo(id=45) 
    print("STARTING MACHINE LEARNING\n\n\n\n\n")

    # data (as pandas dataframes) 
    X = heart_disease.data.features 
    y = heart_disease.data.targets 

    Cleanupbaseline = NACleanup(X,y)
    Trainbaseline = trainTest(Cleanupbaseline[0], Cleanupbaseline[1])
    baseline_confusion = fitModel(Trainbaseline[0], Trainbaseline[1], Trainbaseline[2], Trainbaseline[3])
    factorResults = chooseFactors(Cleanupbaseline[0], Cleanupbaseline[1])
    confusion_matrix_factor = metrics.confusion_matrix(factorResults[1], factorResults[2])

    clf = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)

    # Train the classifier
    clf.fit(Trainbaseline[0], Trainbaseline[2].values.ravel())

    # Get feature importances
    importances = clf.feature_importances_
    # Create a dataframe for features and their importance scores
    features_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})

    # Filter the dataframe to only include features with an importance score above 0.1
    important_features_df = features_df[features_df['Importance'] > 0.1]

    # Create a new dataframe X_important that only includes the important features
    X_important = X[important_features_df['Feature']]

    trees = NACleanup(X_important, y)
    trees = trainTest(trees[0], trees[1])
    trees_confusion = fitModel(trees[0], trees[1], trees[2], trees[3])

    # If you want to visualize the confusion matrix
    sns.heatmap(baseline_confusion, annot=True, fmt='d')
    plt.xlabel('Predicted_Baseline')
    plt.ylabel('True')
    plt.savefig('baseline_confusion.png')
    plt.clf()

    sns.heatmap(confusion_matrix_factor, annot=True, fmt='d')
    plt.xlabel('Predicted_Factor')
    plt.ylabel('True')
    plt.savefig('confusion_matrix_factor.png')
    plt.clf()

    sns.heatmap(trees_confusion, annot=True, fmt='d')
    plt.xlabel('Predicted_Trees')
    plt.ylabel('True')
    plt.savefig('trees_confusion.png')

if __name__ == "__main__":
    main()

