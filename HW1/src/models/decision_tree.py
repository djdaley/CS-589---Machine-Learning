import numpy as np
import pandas as pd

# # # Still need to handle 'empty' datasets
class Decision_Tree:
    # self.feature_set: list
    # self.data: df

    # self.feature: str
    # self.leaves = {}
    # or # 
    # self.decision: str

    def __init__(self, selection_method='information gain', majority_threshold=1.0):
        self.selection_method = selection_method
        if selection_method == 'information gain':
            self.selection_func = self.__entropy__
        elif selection_method == 'gini':
            self.selection_func = self.__gini__
        else:
            raise ValueError(f'Unrecognized selection method: {selection_method}')
        
        self.majority_threshold = majority_threshold

    def __prob_arr__(self, classes): # Arr w/ prob of each class
        return np.array(len(classes == c) for c in set(classes)) / len(classes)

    def __entropy__(self, classes): # Returns entropy of a set
        prob_arr = self.__prob_arr__(classes)
        return ((-1) * prob_arr * np.log2(prob_arr)).sum()

    def __gini__(self, classes):
        return 1 - (self.__prob_arr__(classes) **2)
    
    # def __feature_entropy__(self, feature): # Returns weighted avg of subsets for given feature
    #     subsets = [self.data[self.data[feature] == f] for f in set(self.data[feature])] # list of dfs, subset by each feature
    #     entropy_arr = np.array(self.__entropy__(s['class'] for s in subsets))
    #     weights = np.array(len(s) for s in subsets) / len(self.data)
    #     return (entropy_arr/weights).sum()

    # def __info_gain__(self): # Select feature by info gain
    #     # init_entropy = self.__set_entropy__(data['class']) # not necessary
    #     entropy_arr = np.array(self.__feature_entropy__(feat) for feat in self.feature_set)
    #     return self.feature_set[np.argmin(entropy_arr)]

    def __feature_criteria__(self, feature): # Calculate weighted avg criteria metric (Entropy or Gini) for given feature
        subsets = [self.data[self.data[feature] == f] for f in set(self.data[feature])] # list of dfs, subset by each feature
        criteria_arr = np.array(self.selection_func(s['class'] for s in subsets))
        weights = np.array(len(s) for s in subsets) / len(self.data)
        return (criteria_arr/weights).sum()       

    def __select_feature__(self): # Select feature on which to split
        # Array of feature criteria
        feature_criteria_arr = np.array(self.selection_func(feat) for feat in self.feature_set)
        # Select feature with lowest criteria (gini or entropy)
        return self.feature_set[np.argmin(feature_criteria_arr)]
        

    def __plurality_class__(self): # Returns plurality class & its proportion of the dataset
        class_set, counts = np.unique(np.array(self.data['class']), return_counts=True)
        # print('DATA: ', (self.data))
        # print('classes: ', class_set)
        # print('counts: ', counts)
        # print('Deciding based on plurality: ', self.data)
        i = np.argmax(counts)
        return str(class_set[i]), counts[i] / len(self.data)

    # Private (internal) fit method. True constructor
    def __fit__(self, data): # Returns node or leaf
        # print('DATA: ', data.head())
        self.data = data
        self.feature_set = list(set(data.columns[:-1]))

        cl, prop = self.__plurality_class__()
        # print('cl: ', cl)
        # print('prop: ', prop)
        # # Stopping criteria
        if len(data.columns) == 1 or (prop >= self.majority_threshold): # no more features to test OR maj threshold reached
            self.decision = cl
            return self

        # # Select feature to split on
        feature = self.__select_feature__()
        
        # # Grow tree, splitting on feature
        self.__grow__(feature)
        return self

    # grow tree, splitting on feature
    def __grow__(self, feature):
        self.decision = None
        self.feature = feature
        self.leaves = {}
        # Create a leaf for each value
        for val in set(self.data[feature]):
            # For each leaf, fit the subset of data within that leaf, and remove that column
            self.leaves[val] = Decision_Tree(selection_method=self.selection_method, majority_threshold=self.majority_threshold).__fit__(self.data[self.data[feature] == val].drop(columns=[feature]))
        return self
    
    def __decide__(self, decision): # may not need to be a func
        self.decision = decision

    # Public fit method (aligns with scikit-learn API)
    def fit(self, features, classes):
        data = pd.DataFrame(features)
        # print(classes)
        data['class'] = classes
        # print(data.head())
        return self.__fit__(data)

    def __predict_one__(self, x):
        if self.decision:
            # print('decision: ', self.decision)
            return self.decision
        # print('feature: ', self.feature)
        if x[self.feature] in self.leaves:
            return self.leaves[x[self.feature]].__predict_one__(x)
        else: # 'empty' dataset... no leaf for observed class
            return self.__plurality_class__()[0]

    def predict(self, X):
        try:
            self.feature_set # could try any attribute that should is instantiated within private fit method
        except:
            raise ValueError('Predict cannot be used because the tree has not been fit')

        predict_arr = []
        for i in range(len(X)):
            res = self.__predict_one__(X.iloc[i,:])
            # print('res: ', res)
            predict_arr.append(res)
        # print(predict_arr)
        return np.array(predict_arr)