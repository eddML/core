import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import numpy as np



	

def train_and_tune_SVC(X,y):
        ############# SVC ################ 
	model = svm.SVC(gamma=1e-8, C=100)
        c_range = np.linspace(100, 100, 1)
        g_range = np.logspace(-8, -3, 20)
        param_grid = dict(gamma=g_range, C=c_range)        
	#################################
	fold = 10
	model, score = tune_model(model, X, y, param_grid, fold, n_jobs=-1, scoring='accuracy')
        score_train_test(model , X, y)
	return model



def train_and_tune_RFC(X,y):
        ############# Random Forest Classifier  ################
        #model = RandomForestClassifier(n_estimators=100, random_state=0)
        model = RandomForestClassifier(n_estimators=100)
        n_range = range(1,100)
        param_grid = dict(n_estimators=n_range)
        ########################################################
        fold = 10
        model, score = tune_model(model, X, y, param_grid, fold, n_jobs=-1, scoring='accuracy')
	score_train_test(model , X, y)
        return model


def tune_model(model, X, y, param_grid, fold, **kwargs):
        ##################### parameter tuning  #######################
        grid = GridSearchCV(model, param_grid, cv=fold, **kwargs)
        grid.fit(X,y)
        #print('grid_scores: ',grid.grid_scores_)
        grid_mean_score = [result.mean_validation_score for result in grid.grid_scores_]

        #plt.plot(param_grid['gamma'], grid_mean_score)
        #plt.show(block=True)
        print('grid.best_score: ', grid.best_score_)
        #print('grid.best_params: ', grid.best_params_)
        #print('grid.best_estimator: ', grid.best_estimator_)
        ###############################################################
        model = grid
	return model, grid.best_score_




def score_train_test(model, X, y, **kwargs):
	##################  train-test-validation  ####################
	X_train, X_test, y_train, y_test = train_test_split(X , y, **kwargs)
	model.fit(X_train, y_train)
	tt_score = model.score(X_test, y_test)
	#print('Evaluation Using split: ', tt_score)

	accuracy = metrics.accuracy_score(y_test, model.predict(X_test))
	precision = metrics.precision_score(y_test, model.predict(X_test))
	recall = metrics.recall_score(y_test, model.predict(X_test))
	f1 = metrics.f1_score(y_test, model.predict(X_test))
	cm = metrics.confusion_matrix(y_test, model.predict(X_test))
	auc = metrics.roc_auc_score(y_test, model.predict(X_test))
	print('################ metrics ################')
	print('accuracy: ', accuracy)
	print('precision: ', precision)
	print('recall: ', recall)
	print('f1: ', f1)
	print('auc: ', auc)
	print('confusion matrix: ', cm)
	#plt.imshow(cm, cmap='Blues', interpolation='nearest')
	#plt.grid(False)
	#plt.xlabel('predicted')
	#plt.ylabel('true')
	#plt.show()
	print('#########################################')
	###############################################################
	return tt_score




def score_cross_val(model, X, y, fold, **kwargs):
 	###################### cross validation #######################
	cv = cross_val_score(model, X, y, cv=fold, **kwargs)
	cv_score = cv.mean()
	print('Evaluation Using Cross-Validation ('+str(fold)+' fold): ', cv_score)
	###############################################################
	return cv_score

