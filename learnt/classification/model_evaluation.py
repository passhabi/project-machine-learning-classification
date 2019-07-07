import matplotlib.pyplot as plt
import itertools
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.utils.fixes import signature
from sklearn.metrics import average_precision_score


def accuracy(y_true, y_pred):
    number_of_accurate = 0
    n = len(y_pred)

    # count number of right prediction:
    for i in range(n):
        if y_pred[i] == y_true[i]:
            number_of_accurate += 1

    # return accuracy and error:
    number_of_errors = n - number_of_accurate
    return number_of_accurate / n, number_of_errors / n


# def precision(y_ture, y_predic):
#     pass
#
#
# def recall():
#     pass


def plot_confusion_matrix(cm, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    classes = [1, 0]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # print("Normalized confusion matrix")
    # else:
    #     print('Confusion matrix, without normalization')

    # print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def plot_precision_recall(model, test_x, test_y):
    y_score = model.decision_function(test_x)
    precision, recall, _ = precision_recall_curve(test_y, y_score)

    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])

    average_precision = average_precision_score(test_y, y_score)

    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
