import numpy as np
from sklearn.metrics import confusion_matrix


def compute_precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, class_index: int) -> tuple[float, float, float]:
    # Count true positives, false positives, true negatives, false negatives
    tp = fp = tn = fn = 0
    for i in range(len(y_pred)):
        if y_pred[i] == class_index:
            if y_true[i] == class_index:
                tp += 1
            else:
                fp += 1
        else:
            if y_true[i] == class_index:
                fn += 1
            else:
                tn += 1

    # Calculate precision, recall, F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1_score


def print_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_labels: list[str]):
    # Count predictions in confusion matrix
    num_classes = len(class_labels)
    cm = confusion_matrix(y_true, y_pred, labels=range(num_classes))

    # Determine column width
    col_width = max(len(label) for label in class_labels) + 1
    
    # Print confusion matrix header
    print('Confusion matrix (predicted as columns, actual as rows):')
    print('--------------------------------------------------------')

    # Print class labels for X axis
    print(' ' * col_width, end='')
    for label in class_labels:
        print(f'{label:>{col_width}}', end='')
    print()

    # Print each row of the confusion matrix
    for i in range(num_classes):
        print(f'{class_labels[i]:>{col_width}}', end='')
        for j in range(num_classes):
            print(f'{cm[i, j]:>{col_width}}', end='')
        print()
