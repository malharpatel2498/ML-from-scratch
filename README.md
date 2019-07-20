# ML-from-scratch
ML library from scratch for implementing Adaboost, Decision Tree, Random Forest, Logistic Regression, Naive Bayes and Stacking

DecisionTree(max_depth,split_val_metric,min_info_gain,split_node_criterion)

Max_depth = The maximum depth upto which the tree is allowed to grow.
Split_val_metric = Mean or Median. This is the value of the column selected on which you make a binary split.
Min_info_gain = If the information gain made by the split is lesser than this value, the node does not split.
Split_node_criterion = Gini or Entropy.

LogisticRegression(regulariser,lambda,num_steps,learning_rate,initial_wts)

	Regulariser = L1 or L2.
	Lambda = Hyperparameter for regulariser term.
	Num_steps = Number of iterations for which Gradient Descent will run.
	Learning_rate = alpha. The size of step taken in gradient descent.
	Initial_wts = A list of n+1 values, where n is number of features. This is initial value of the 	coefficients. Either the user sends a list or you assign it randomly in range normal (0,1).

NaiveBayesClassifier(type,prior)
	
	Type = Gaussian, Multinomial. For numerical and categorical datasets respectively.
	Prior = Prior Probabilities of the classes.


RandomForest(n_trees, max_depth,split_val_metric,min_info_gain,split_node_criterion, max_features, bootstrap=True, n_cores)

	N_tress = Number of estimator trees in the forest.
	Split_val_metric, min_info_gain, split_node_criterion = all same as Decision Tree
	Max_features = Number of features to consider for each tree.
	Bootstrap = Whether bootstrap samples are to be used. If False, the whole dataset is   
	used to build each tree.
	N_cores = Number of cores on which to run this learning process. Write Parallelised
	code.

AdaBoost(n_trees, learning_rate)
	
	N_trees = Number of estimator trees.
	Learning_rate = Used in calculating estimator weights.

Stack( [ (clf,count) ] )

	The argument is a list of tuples.
	First element of each tuple is a classifier object.
	Second element of each tuple is how many times it is to be considered.


train(X, y)
	
	This function can be called from each of the above classifier object. It takes a training set 
	of X and y values and fits the data. Similar to fit() function in sklearn.

predict(X_test)
	
This function can be called from each of the above trained classifier object. It takes 
	The test X values and does the prediction.
