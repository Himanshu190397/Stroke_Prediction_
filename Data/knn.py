import numpy as np
# import data_loading as dl
from scipy.stats import mode

class knn_classifier():
    
    def __init__(self, k) -> None:
        super().__init__()
        self.k = k
    
    def euclidean_square(self, r1, r2):
        dist = 0.0
        d2 = np.linalg.norm(r1-r2, axis=1)
        # for i in range(len(r1)-1):
        #     dist +=(r1[i]-r2[i])**2
        return d2
    
    def predict(self, train_X, train_Y, test_point):
        """
        test_point should be a row corresponding to one observation
        
        train_data should NOT have the class variable removed!
        """
        # face dataset identifier in first column
        
        # get distances from test point
        # distances = []
        distances = self.euclidean_square(train_X, test_point)
        sort_indices = np.argsort(distances)
        sorted_Y = train_Y[sort_indices]
        neighbor_classes = sorted_Y[:self.k]
        guess = mode(neighbor_classes)[0][0]

        # for idx in range(len(train_X)):
        #     dist = self.euclidean_square(train_X[idx][:-1], test_point)
        #     distances.append((idx, dist))
        
        # # sort distances in place
        # distances.sort(key=itemgetter(1))
        # # select k neighbors
        # neighbors = []
        # for i in range(self.k):
        # # only take the training sample idx from the tuple
        #     neighbors.append(train_X[distances[i][0]])
        # # get the class values 
        # class_count = Counter()
        # for obs in neighbors:
        #     class_count[obs[-1]] +=1
        
        return guess#mode(neighbor_classes)#class_count.elements())

    def make_predictions(self, train_X, train_Y, test_data):
        predictions = []
        for row in test_data:
            predictions.append(self.predict(train_X, train_Y, row))
        return np.asarray(predictions)
 
    def evaluate_accuracy(self, truth, predictions):
        correct = 0
        for real, pred in zip(truth,predictions):
            if real == pred:
                correct +=1
        return correct/len(truth)
