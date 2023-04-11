import numpy as np
import pandas as pd
import pathlib
import data_loading as dl
from tree_class import DecisionTree
import naive_bayes as nb
from knn import knn_classifier
import matplotlib.pyplot as plt
import LogisticRegression as lr


def selective_zscore(data, affected_cols=None, t_mean=None, t_std=None, metrics=False):
    if t_mean is None:
        mean = np.mean(data, axis=0)
    else:
        mean = t_mean
    if t_std is None:
        std = np.std(data, axis=0, ddof=1)
    else:
        std = t_std
    std[std == 0.0] = dl.NUMERIC_STABILITY
    if affected_cols:
        data[:, affected_cols] = ((data - mean) / std)[:, affected_cols]
    else:
        data = ((data - mean) / std)
    if metrics == True:
        return data, mean, std
    else:
        return data


def load_and_encode(rng, filePath, shuffle=True):
    df = pd.read_csv(filePath)
    df.dropna(axis=0, inplace=True)
    df.drop(columns='id', inplace=True)
    # Using a library for dummy encoding  
    df2 = pd.get_dummies(df, drop_first=True)
    label = df2.pop('stroke')
    df2.insert(0, 'stroke', label)
    raw_data = df2.to_numpy(copy=True)
    if shuffle:
        # Randomly shuffle
        rng.shuffle(raw_data)
    return raw_data


def split_and_zscore_features(raw_data):
    # Get bool array for which columns have more than 2 unique values
    nonbinary_features = [np.unique(x).shape[0] > 2 for x in raw_data.T]
    train, val, s_idx = dl.get_train_val_split(raw_data)
    # Compare the zscoring between the two methods    
    all_cols = raw_data.shape[1]
    ztrain, t_mean, t_std = selective_zscore(train, nonbinary_features, metrics=True)
    zmetrics = {'mean': t_mean,
                'std': t_std}
    zval = selective_zscore(val, nonbinary_features, t_mean=t_mean, t_std=t_std)
    return ztrain, zval, zmetrics


def tree_classifier(train, val):
    train_y = train[:, 0].reshape(-1, 1)
    train_x = train[:, 1:]
    val_y = val[:, 0].reshape(-1, 1)
    val_x = val[:, 1:]

    train_x, train_y = dl.split_x_y(train, 0)
    val_x, val_y = dl.split_x_y(val, 0)

    t_mean = np.mean(train_x, axis=0)
    t_std = np.std(train_x, axis=0, ddof=1)

    ### Not using zscored trainning for a test
    # nonbinary_features = [np.unique(x).shape[0]>2 for x in raw_data.T]
    # ztrain_x, t_mean, t_std = selective_zscore(train_x,affected_cols=nonbinary_features, metrics=True)
    # zval_x = selective_zscore(val_x, t_mean=t_mean, t_std=t_std)
    myTree = DecisionTree(train_x, train_y, t_mean, t_std)
    myTree.fit(myTree.bin_x, train_y)
    val_bin = myTree.threshold_features(val_x)
    val_pred = myTree.predict(val_bin)
    metrics_dict = nb.eval_binary_model(val_y, val_pred.reshape(-1, 1))
    return metrics_dict


def naive_bayes(train, val):
    train_y = train[:, 0].reshape(-1, 1)
    train_x = train[:, 1:]
    val_y = val[:, 0].reshape(-1, 1)
    val_x = val[:, 1:]
    nonbinary_features = [np.unique(x).shape[0] > 2 for x in train_x.T]
    ztrain_x, t_mean, t_std = selective_zscore(train_x, affected_cols=nonbinary_features, metrics=True)

    zval_x = selective_zscore(val_x, affected_cols=nonbinary_features, t_mean=t_mean, t_std=t_std)

    nbmetrics = nb.run_naive_bayes(train_x, train_y, val_x, val_y)
    return nbmetrics


def run_knn(train_x, train_y, val_x, val_y, k=1):
    # full_train = np.concatenate((train_x, train_y),axis=1)
    knn = knn_classifier(k)
    predictions = knn.make_predictions(train_x, train_y, val_x)

    # predictions=[]
    # for idx in range(len(val_x)):
    #     predictions.append(knn.predict(full_train, val_x[idx]))    
    # accuracy = knn.evaluate_accuracy(val_y, predictions)
    return evaluate_binary_model(val_y, predictions)


def run_random_forests(X_train, Y_train, X_validate, Y_validate, T=10, M=100):
    rng = np.random.default_rng(seed=1968)
    mean_train = np.mean(X_train, axis=0)
    std_train = np.std(X_train, axis=0, ddof=1)
    predictions = []
    for t in range(T):
        indices = rng.choice(X_train.shape[0], size=M, replace=True)
        tree_t = DecisionTree(X_train[indices], Y_train[indices], mean_train, std_train, random_forest=True)
        tree_t.fit(tree_t.bin_x, Y_train[indices])
        X_val_bin = tree_t.threshold_features(X_validate)
        predictions.append(tree_t.predict(X_val_bin))
    predictions = np.array(predictions)
    Yhat = np.mean(predictions, axis=0)
    # choosing .15 as a threshold because the max average was consistently around .34 or so
    Yhat[Yhat < .15] = 0
    Yhat[Yhat >= .15] = 1
    return evaluate_binary_model(Y_validate, Yhat)


def iterate_knn(train_x, train_y, val_x, val_y):
    """ 
    step through some values for k and see if any perform better
    """
    metrics_log = []
    for k in range(1, 15):
        knn = knn_classifier(k)
        predictions = knn.make_predictions(train_x, train_y, val_x)
        metrics = evaluate_binary_model(val_y, predictions)
        metrics_log.append(
            np.asarray([metrics["precision"], metrics["recall"], metrics["fscore"], metrics["accuracy"]]))
    metrics_log = np.asarray(metrics_log)

    plt.plot(range(1, 15), metrics_log[:, 0], label="precision")
    plt.plot(range(1, 15), metrics_log[:, 1], label="recall")
    plt.plot(range(1, 15), metrics_log[:, 2], label="fscore")
    plt.plot(range(1, 15), metrics_log[:, 3], label="accuracy")
    plt.legend()
    plt.title("KNN metrics for various values for k")
    plt.show()

    return metrics_log


def evaluate_binary_model(Y, Yhat):
    tp = np.sum(np.logical_and(Y == 1, Yhat == 1))
    fp = np.sum(np.logical_and(Y == 0, Yhat == 1))
    fn = np.sum(np.logical_and(Y == 1, Yhat == 0))
    tn = np.sum(np.logical_and(Y == 0, Yhat == 0))

    precision = tp / (tp + fp + dl.NUMERIC_STABILITY)
    recall = tp / (tp + fn + dl.NUMERIC_STABILITY)
    fscore = (2 * precision * recall) / (precision + recall + dl.NUMERIC_STABILITY)
    accuracy = np.sum(Yhat == Y) / Y.shape[0]
    metrics = {"precision": precision,
               "recall": recall,
               "fscore": fscore,
               "accuracy": accuracy}
    return metrics


def print_metrics(metrics, decimal_places=8, prefix=""):
    print(f"{prefix}precision: {metrics['precision']:.{decimal_places}f}")
    print(f"{prefix}recall:    {metrics['recall']:.{decimal_places}f}")
    print(f"{prefix}fscore:    {metrics['fscore']:.{decimal_places}f}")
    print(f"{prefix}accuracy:  {metrics['accuracy']:.{decimal_places}f}")


def runLogisticRegression(zscored_x, train_y, zscored_valx, val_y):
    train_y = np.reshape(train_y, (len(train_y), 1))
    weights = lr.initializeWeights(len(zscored_x.T))
    w = weights
    weights = np.reshape(w, (len(zscored_x.T), 1))
    wt, y_hat_train = lr.logisticRegression(zscored_x, train_y, weights, 0.01)
    predictions = lr.predict(zscored_valx, wt, 0.01)
    transformed_predictions = lr.transformPredictions(0.6, predictions)
    val_y = np.reshape(val_y, (len(val_y), 1))
    metric_dict = evaluate_binary_model(val_y, transformed_predictions)
    return metric_dict


if __name__ == '__main__':
    rng = np.random.default_rng(seed=1968)
    cwd = pathlib.Path().cwd()
    raw_data = load_and_encode(rng, cwd.joinpath('data', "healthcare-dataset-stroke-data.csv"), shuffle=True)
    raw_train, raw_val = dl.get_train_val_split(raw_data)

    train_y = raw_train[:, 0]
    train_x = raw_train[:, 1:]
    val_y = raw_val[:, 0]
    val_x = raw_val[:, 1:]
    ztrain_x, t_mean, t_std = selective_zscore(train_x, metrics=True)
    zval_x = selective_zscore(val_x, t_mean=t_mean, t_std=t_std)
    # Experiment with different values for KNN to gauge results
    # iterate_knn(ztrain_x, train_y, zval_x, val_y)

    ### Various models ###
    DT_metrics = tree_classifier(raw_train, raw_val)
    RF_metrics = run_random_forests(train_x, train_y, val_x, val_y, T=100, M=2000)
    NB_metrics = naive_bayes(raw_train, raw_val)
    LogReg_metrics = runLogisticRegression(ztrain_x,train_y,zval_x,val_y)
    KNN_metrics = run_knn(ztrain_x, train_y, zval_x, val_y, k=4)

    ## comparisons between performances ##
    print("\nDecision Tree metrics")
    print_metrics(DT_metrics, prefix="\t")
    print("\nRandom Forest metrics")
    print_metrics(RF_metrics, prefix="\t")
    print("\nNaive Bayes metrics")
    print_metrics(NB_metrics, prefix="\t")
    print("\nk Nearest Neighbors metrics")
    print_metrics(KNN_metrics, prefix="\t")
    print("\nLogistic Regression metrics")
    print_metrics(LogReg_metrics, prefix="\t")
