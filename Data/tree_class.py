from posixpath import split
import numpy as np
import pandas as pd
from node import Node
import naive_bayes as nb


NUMERIC_STABILITY = 10**(-7)

class DecisionTree():
    ## This version pushes data into a dataframe!
    def __init__(self, x_data, y_data, meanx, stdX, random_forest=False):
        self.X = x_data
        self.y = y_data
        self.bin_x = np.zeros_like(x_data)
        
        self.n_classes = len(np.unique(y_data))
        self.n_samples, self.n_features = x_data.shape
        # array of means from zscoring, which we aren't using after all?
        self.col_means = meanx
        # array of stdDevs from zscoring
        self.col_stddev = stdX
        self.total_entropy = 0
        self.class_entropy = dict()
        self.mapped_features = []

        self.random_forest = random_forest
        self.rng = np.random.default_rng(seed=1968)
        
        self.threshold_features()
        # self.attach_label_column()
        self.Tree = None
    
    
    def attach_label_column(self):
        """
        reworking this class into dataframes means I can isolate the 
        training class labels as needed in functions. Keeping them with the features 
        helps with maintaining the relationship

        adding it as it's own function for clarity
        """
        self.bin_x = pd.DataFrame(self.bin_x)
        self.bin_x['label'] = self.y
        return
    
    def threshold_vector(self, vector, threshold):
        # take mean of each feature, set 1 if greater than mean and 0 less than 
        # means are passed in
        newvec = np.zeros_like(vector)
        newvec[vector >= threshold] = 1
        newvec[vector < threshold] = 0
        return newvec  
    
    def threshold_features(self, indata=None):
        if indata is None:
            for col in range(self.n_features):
                thresh = self.col_means[col]
                self.bin_x[:, col] = self.threshold_vector(self.X[:,col], thresh)
            return
        else:
            outdata = np.zeros_like(indata)
            for col in range(self.n_features):
                thresh = self.col_means[col]
                outdata[:, col] = self.threshold_vector(indata[:,col], thresh)
            return outdata
            
    def get_total_entropy(self, X, y):
        # find entropy of entire dataset
        class_entropy = dict()
        total_entropy  = 0
        # get entropy per class and total for the dataset
        for c in range(self.n_classes):
            class_count = (y == c).sum()
            prob_class = class_count/X.shape[0]
            # ratio of rows of class vs total
            # print(f"Probability for class {c} = {prob_class}")
            c_e = - (prob_class)*np.log2(prob_class+NUMERIC_STABILITY)
            class_entropy[c] = c_e
            total_entropy += c_e
        return total_entropy
    
    def get_node_probabilities(self,  y):
        prob_list = []
        for c in range(self.n_classes):
            class_count = y[y == c].shape[0]
            
            prob_class = class_count/y.shape[0]
            prob_list.append(prob_class)
        return np.asarray(prob_list)

    def get_feature_entropy(self, feature_rows, y):
        #feature_rows should only contain 1 value of the features, such as 1 or 0
        entropy = 0
        feature_rows.shape[0] # all rows of the feature with specific value
        for c in range(self.n_classes):
            class_count = feature_rows[y.flatten() == c].shape[0]
            
            prob_class = class_count/feature_rows.shape[0]
            # ratio of rows of class vs total
            c_e = - (prob_class)*np.log2(prob_class+NUMERIC_STABILITY)
            entropy +=c_e
        return entropy
            
    def get_feature_info_gain(self, X, y, col_num):
        f_info = 0.0
        
        for fval in range(self.n_classes):
            # rows_with_fval = self.bin_x[self.bin_x[:,col_num]==fval]
            f_column = X[:, col_num]
            fval_rows = X[f_column==fval]
            f_ent =0
            entropy_of_class =0
            if fval_rows.shape[0]!=0:
                #one class = 0 entropy
     
                f_ent = self.get_feature_entropy(fval_rows, y[f_column == fval])
                fprob = fval_rows.shape[0]/X.shape[0]
                entropy_of_class =fprob*f_ent
            # unsure about this, built from https://machinelearningmastery.com/information-gain-and-mutual-information/ 
            f_info += entropy_of_class
        return self.get_total_entropy(X,y) - f_info
            
    
    def get_max_info_feature(self, X,y):
        """
        return feature that provides the best information gain
        """
        full_entropy =  self.get_total_entropy(X, y)
        f_info_gain = np.zeros((X.shape[1])) #removing the labels column from the count
        ents = []
        for column in range(X.shape[1]): #removing the labels column from the count
            info = self.get_feature_info_gain(X,y,column)
            f_info_gain[column] = info
        best_feature_idx = f_info_gain.argmax()
        return best_feature_idx
        

    def get_split_params(self, X, y):
        if self.random_forest:
            m = int(np.sqrt(X.shape[1]))
            indices_to_check = self.rng.choice(X.shape[1], replace=False, size=m)
            split_idx = indices_to_check[self.get_max_info_feature(X[:,indices_to_check], y)]
        else:
            split_idx = self.get_max_info_feature(X, y)
        split_col = X[:,split_idx]
        x_left = X[split_col < 0.5]
        x_right = X[split_col > 0.5]
        y_left = y[split_col < 0.5]
        y_right = y[split_col > 0.5]
        return split_idx, x_left, x_right, y_left, y_right
        
        
    def build_tree(self, X, y, node):
        # Well now that I have to trim features I'm realizing Idk how track the removed columns correctly with numpy
        # Going to convert over to pandas           
        if X.shape[0] != 0:
            
            if np.unique(y).shape[0] ==1:
                node.is_leaf = True
                return
            split_idx, x_left, x_right, y_left, y_right = self.get_split_params(X,y)
            if split_idx is None:
                node.is_leaf = True
                return
            if x_left.shape[0] ==0 or x_right.shape[0] == 0:
                node.is_leaf = True
                return
            node.f_column = split_idx
            node.left=Node()
            node.right=Node()
            node.left.probabilities = self.get_node_probabilities(y_left)
            node.right.probabilities= self.get_node_probabilities(y_right)
            # recursion should handle the branching?
            self.build_tree(x_right,y_right, node.right )
            self.build_tree(x_left,y_left, node.left) 
    
    def fit(self, X, y):
        self.Tree=Node()
        self.build_tree(X, y, self.Tree)

    def predict_observation(self, obs,  node):
        # eval one observation
        
        if node.is_leaf ==True:
            return node.probabilities
        if obs[node.f_column] == 1:
            prob = self.predict_observation(obs, node.right)
        else:
            prob = self.predict_observation(obs, node.left)
        return prob

    def predict(self, testX):
        predictions = []
        for x in testX:
            one_pred = np.argmax(self.predict_observation(x, self.Tree))
            predictions.append(one_pred)
        return np.asarray(predictions)