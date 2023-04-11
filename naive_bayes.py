import numpy as np
import data_loading as dl
import pathlib
import matplotlib.pyplot as plt


THIS_FILE_PARENT = pathlib.Path(__file__).parent.absolute()


NUMERIC_STABILITY = 10**(-7)
def group_by_class(x_arr, y_arr):
    """
    returns a dictionary of the x_arr arg split by class aligned with y_arr.
    Dict keys are then used for tracking truth
    """
    spam = []
    non_spam = []
    
    for idx, val in enumerate(x_arr):
        if y_arr[idx] == 1:
            spam.append(val)
        else:
            non_spam.append(val)
    grouped = {0:np.asarray(non_spam), 
               1:np.asarray(spam)}
    return grouped

def get_data_stats(data):  
        
    dstat = [(np.mean(column), np.std(column,ddof=1), len(column)) for column in zip(*data)]
    return dstat

def get_stats_by_class(data_dict):    
    # sticking with using dictionaries for now. Might use arrays eventually. 
    class_stats = dict()
    for class_, rows in data_dict.items():
        class_stats[class_] = get_data_stats(rows)
    return class_stats

def get_gaussian_probability(x, fmean, fstd):
    if fstd == 0.0:
        fstd=1
        
    exponent = (-(x - fmean)**2)/(2*(fstd**2))
    pdf = (1/(fstd*np.sqrt(2*np.pi)))* np.exp(exponent)
    return pdf
    
def get_class_probabilities(class_dict, row, total_rows):
    probs = dict()
    for cls_val, summary in class_dict.items():
        probs[cls_val] = class_dict[cls_val][0][2]/float(total_rows)
        for i in range(len(summary)):
            mean, stdev, _ = summary[i]
            x=  get_gaussian_probability(row[i],mean, stdev)
            probs[cls_val] *=x
    return probs
    
def predict(class_dict, row, trows):
    probabilities = get_class_probabilities(class_dict, row, trows)
    # lsumexp
    label_guess, best_probability = None, -1
    for class_label, prob in probabilities.items():
        if label_guess is None or prob > best_probability:
            best_probability = prob
            label_guess = class_label
    return label_guess

def make_predictions(class_dict, test_data, trows):
    predictions = []
    for row in test_data:
        predictions.append(predict(class_dict, row, trows))
    return predictions

def get_accuracy(predictions, truth):
    wrong_label = (truth != predictions ).sum()
    accuracy = wrong_label/len(predictions)
    return accuracy

# This expects a 1d np array for both Y and yhat
def eval_binary_model(Y, yhat=np.array([])):
    if type(yhat) == list:
        yhat = np.asarray(yhat)
        yhat = yhat.reshape(-1,1)
    if Y.shape != yhat.shape:
        Y = Y.reshape(-1,1)
        yhat = yhat.reshape(-1,1)
    tp = np.sum(np.logical_and(Y==1, yhat==1))
    fp = np.sum(np.logical_and(Y==0, yhat==1))
    fn = np.sum(np.logical_and(Y==1, yhat==0))
    tn = np.sum(np.logical_and(Y==0, yhat==0))
               
    precision = tp/(tp+fp+NUMERIC_STABILITY)
    recall = tp/(tp+fn+NUMERIC_STABILITY)
    fscore = (2*precision*recall)/(precision+recall+NUMERIC_STABILITY)
    accuracy = (yhat == Y).sum() / len(Y)

    metrics = {"precision":precision,
                "recall":recall,
                "fscore":fscore,
                "accuracy":accuracy}
    return metrics

def report_metrics(metric_dict):
    string_list = []
    for key in metric_dict.keys():
        some_string = f"Model {key} = {metric_dict[key]}"
        print(some_string)
        
def run_naive_bayes(zscored_x, train_y, zscored_valx, val_y):
    grouped = group_by_class(zscored_x, train_y)
    class_stats = get_stats_by_class(grouped)
    predictions = make_predictions(class_stats, zscored_valx, len(zscored_x))
    metric_dict = eval_binary_model(val_y, predictions)
    return metric_dict

if __name__ == '__main__':
    print("no main")
    # rng = np.random.default_rng(seed=1968)
    # data_path = THIS_FILE_PARENT.joinpath('spambase.data')
    # data = dl.load_dataset(data_path,rng)
    # train, val, split_idx = dl.get_train_val_split(data)
    # train_x, train_y = dl.split_x_y(train, -1)
    # ztx, meanx, stdX = dl.zscore(train_x)
    
    # val_x, val_y = dl.split_x_y(val,-1)
    # scored_valx, _, _ = dl.zscore(val_x, meanx, stdX)
    
    # grouped = group_by_class(ztx, train_y)
    # class_stats = get_stats_by_class(grouped)
    
    # predictions = make_predictions(class_stats, scored_valx, len(train_x))
    # metric_dict = eval_binary_model(val_y, predictions)
    # report_metrics(metric_dict)
    
    
    
    
    
    