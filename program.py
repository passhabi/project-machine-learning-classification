import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from learnt.classification import model_evaluation as evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import itertools
import string

# Import the data frame:
#   Lets call our dataset df. as we see, df contains a qid, a question column and a target column.
df = pd.read_csv('train.csv')

# preprocessing
#   for some pre processing operations, first we need to remove punctuation for question_text columns
#   there is question mark with sentence that we should remove, for this we use regular expressions:
df['question_text'] = df['question_text'].str.replace('[^\w\s]', '')
#   No need to qid, so I drop it.
df = df.drop(['qid'], axis=1)

# Sample data to balance classes:
#   Now, we write some code to compute below the percentage of +1 and 0 target in the dataset.
#   One way to combat class imbalance is to undersample the larger class until the class distribution is approximately half and half. Here, we will undersample the larger class in order to balance out our dataset. This means we are throwing away many data points.
plus = df[df['target'] == +1]
neutral = df[df['target'] == 0]
print("Number of +1  : %s" % len(plus))
print("Number of 0 : %s" % len(neutral))

#   Since there are fewer +1s than 0s, we find the ratio of the sizes and we undersample the 0s.
neutral = neutral[:len(plus)]

#   Append  0 to with the down sampled version of safe_loans
df = plus.append(neutral)

# Split data into training and test sets
# Let's perform a train/test split with 90% of the data in the training set and 20% of the data in the test set.
train, test = train_test_split(df, test_size=0.1)
train, validation = train_test_split(train, test_size=0.2)

# First we are going to use tf_idf for some sort of feature selection too. after running this process value 10204 has chosen for the max number of features. since tf_idf is descending order. we have some sort of lasso operation but in a different way.
cut_dict = {}
for cut in np.linspace(9000, 20000, 10):
    cut = int(cut)
    vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b', stop_words='english', max_features=cut)
    train_x = vectorizer.fit_transform(train['question_text'])
    validation_x = vectorizer.transform(validation['question_text'])

    logisticRegression = LogisticRegression()
    logisticRegression.fit(train_x, train['target'])
    pred = logisticRegression.predict(validation_x)
    cut_dict[cut] = logisticRegression.score(validation_x, validation['target'])  # compute accuracy

plt.plot(cut_dict.values(), '.-')
plt.show()

# we find the best value for the maximum number of features:
max_feature = max(cut_dict.items(), key=lambda x: x[1])[0]
print('Optimal maximum of features:', max_feature)

# TF_IDF representing data
#   We will now compute the TF_IDF for each word that appears in the reviews. For this reason, scikit-learn and many other tools use matrices to store a collection of word count vectors.
vectorizer = TfidfVectorizer(token_pattern=r'\b\w+\b', stop_words='english', max_features=10240)
# Use this token pattern to keep single-letter words
# First, learn vocabulary from the training data and assign columns to words
# Then convert the training data into a sparse matrix
train_x = vectorizer.fit_transform(train['question_text'])
validation_x = vectorizer.transform(validation['question_text'])
test_x = vectorizer.transform(test['question_text'])


# we write a function to find the best tuning parameter values for different part of section of models.
def model_tuning(tuple_df_x, tuple_df_y, label_model, model_type='l2_penalty'):
    """

    :param tuple_df_x:
    :param tuple_df_y:
    :param label_model:
    :param model_type: l1_penalty, l2_penalty, adaboost, decision_tree
    :return:
    """
    train_x = tuple_df_x[0]
    validation_x = tuple_df_x[1]
    test_x = tuple_df_x[2]

    train_y = tuple_df_y[0]
    validation_y = tuple_df_y[1]
    test_y = tuple_df_y[2]

    score_dict = {}
    if model_type == 'l1_penalty' or model_type == 'l2_penalty':
        for tuning in np.logspace(0, 7, 16):
            if model_type == 'l1_penalty':
                model = LogisticRegression(penalty='l1', C=tuning)
            else:
                model = LogisticRegression(C=tuning)

            model.fit(train_x, train_y)

            y_hat = model.predict(validation_x)
            score_dict[tuning] = model.score(validation_x, validation_y)  # compute accuracy

        plt.plot(score_dict.values(), '.-')
        plt.show()

        # we find the best tuning parameter:
        tuning = max(score_dict.items(), key=lambda x: x[1])[0]
        print('[', label_model, ']')
        print('Tuning parameter:', tuning)

    # lets now use the found tuning parameter to evaluate model on the test data:
    if model_type == 'l1_penalty':
        model = LogisticRegression(penalty='l1', C=tuning)
    elif model_type == 'l2_penalty':
        model = LogisticRegression(C=tuning)
    elif model_type == 'adaboost':
        print('[', label_model, ']')
        model = GradientBoostingClassifier()
    elif model_type == 'decision_tree':
        print('[', label_model, ']')
        model = DecisionTreeClassifier()
    else:
        raise Exception("model type is not correct")

    model.fit(train_x, train_y)
    y_hat = model.predict(test_x)

    print('Accuracy on train:', model.score(train_x, train_y))
    print('Accuracy on test: ', model.score(test_x, test_y))

    # Compute confusion matrix
    #   confusion is simple matrix that tells a lot about the model, as follow:
    #    left_top: # of True positive      right_top: # of False Negative
    #   left_down: # of False positive    right_down: # of True negative
    cnf_matrix = confusion_matrix(test_y, y_hat)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    # plt.figure()
    # evaluation.plot_confusion_matrix(cnf_matrix, title='Confusion matrix, without normalization')
    # plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    evaluation.plot_confusion_matrix(cnf_matrix, normalize=True, title='Normalized confusion matrix')
    plt.show()

    # lets see about precision and recall of our model:
    if model_type != 'decision_tree':
        plt.figure()
        evaluation.plot_precision_recall(model, test_x, test_y)
        plt.show()

    return model


tuple_x = [train_x, validation_x, test_x]
tuple_y = [train['target'], validation['target'], test['target']]

# Lasso Logistic Regression
# What that we want in this section is to reduce number of features that we want to include in our model. This method reduces the chance of overfitting and reduce our computations.
lasso_model = model_tuning(tuple_x, tuple_y, 'Lasso Logistic Regression', 'l1_penalty')
# result show low accuracy on test data. so we need some changes e.g. the way we split the data or the way shuffle the data. Since the output is not supposed to be stored. We pass from this stage.

# Now lets to exclude features with weight 0 from features list.
col_features = vectorizer.get_feature_names()
col_coef = lasso_model.coef_[0]

# store it in data frame:
df_features = pd.DataFrame(data={'features': col_features, 'weights': col_coef})

# remove sparsity:
df_features = df_features[df_features['weights'] != 0]
# save it in list to use it in tf_idf again:
features_list = df_features.features.iloc[:].values

# fit it throw TF_IDF for the last time:
vectorizer2 = TfidfVectorizer(token_pattern=r'\b\w+\b', vocabulary=features_list)
# Then convert the training data into a sparse matrix
train_x = vectorizer2.fit_transform(train['question_text'])
validation_x = vectorizer2.transform(validation['question_text'])
test_x = vectorizer2.transform(test['question_text'])

tuple_x = [train_x, validation_x, test_x]

# Ridge Logistic Regression
# Now that we have just features that we need, lets continue:
ridge_model = model_tuning(tuple_x, tuple_y, label_model='Ridge Logistic Regression')

# two remaining method (Decision Tree and Ada boost) is not really sensible way to classify these kind of dataset. (at least in this way).
# Decision Tree
decision_tree_model = model_tuning(tuple_x, tuple_y, 'Decision Tree', 'decision_tree')

# Ada boost
adaboost_model = model_tuning(tuple_x, tuple_y, 'Ada boost', 'adaboost')
