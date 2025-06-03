import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


class kFP:
    def __init__(self, args):
        super(kFP, self).__init__()
        self.parameters = {
            'n_estimators': [20, 50, 100],
            'max_depth': [10, 30, None],
            'max_features': ['sqrt', 'log2'],
            'oob_score': [True, False]
        }
        self.rf = GridSearchCV(RandomForestClassifier(), self.parameters, n_jobs=-1, cv=3, verbose=2)
        self.train_src = []
        self.train_tgt = []
    
    def add_train_data(self, src, tgt):
        self.train_src.extend(src)
        self.train_tgt.extend(tgt)

    def clear_train_data(self):
        self.train_src = []
        self.train_tgt = []
    
    def train(self):
        src, tgt = np.array(self.train_src), np.array(self.train_tgt)
        self.rf.fit(src, tgt)
        print('Best parameters', self.rf.best_params_)
        y_pred = self.rf.predict(src)
        return tgt, y_pred
    
    def test(self, src):
        src = np.array(src)
        y_pred = self.rf.predict(src)
        return y_pred