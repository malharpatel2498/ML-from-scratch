import numpy as np
import pandas as pd
from copy import deepcopy
from multiprocessing import Process, current_process, Pool
import multiprocessing
from random import randrange
from scipy import stats

class AdaboostClassifier:
    def __init__(self, max_depth = 1, n_trees = 10, learning_rate = 0.1):
        self.max_depth = max_depth
        self.classifier = DecisionTreeSW(self.max_depth, split_val_metric = 'mean')
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.trees_ = list()
        self.tree_weights_ = np.zeros(self.n_trees)
        self.tree_errors_ = np.ones(self.n_trees)
      
    def compute(self, X, y, sample_weight):
        local_tree = deepcopy(self.classifier)
        
        #y = y[0]
        
        local_tree.build_tree(X, y, sample_weights=sample_weight)
         
        y_pred = local_tree.predict_new(X)    
        
        misclassified = y != y_pred
        misclassified = np.asarray(misclassified)
        local_tree_error = np.dot(misclassified, sample_weight)/np.sum(sample_weight, axis=0)
        
        #print(local_tree_error)
        
        #if local_tree_error >= 1 - (1/self.n_classes_):
            #print("Hello")
            #return None, None, None
            
        local_tree_weight = self.learning_rate * np.log(float(1 - local_tree_error)/float(local_tree_error)) + np.log(self.n_classes_ - 1)
        
        #print(local_tree_weight)
        
        if local_tree_weight <= 0:
            return None, None, None

        sample_weight = sample_weight * np.exp(local_tree_weight * misclassified)

        sample_weight_sum = np.sum(sample_weight, axis=0)
        
        if sample_weight_sum <= 0:
            return None, None, None

        sample_weight /= sample_weight_sum

        self.trees_.append(local_tree)
        #print(len(self.trees_))

        return sample_weight, local_tree_weight, local_tree_error
      
    def fit(self, X, y):
        #y = y[0]
        list_classes = sorted(set(y.values))
        self.n_classes_ = len(list_classes)
        #list_classes = sorted(list(set(y)))
        self.classes_ = np.array(list_classes)
        #self.n_classes_ = len(self.classes_)
        self.n_samples = X.shape[0]
        
        for tree in range(self.n_trees):
            if tree == 0:
                sample_weight = np.ones(self.n_samples) / self.n_samples
            sample_weight, tree_weight, tree_error = self.compute(X, y, sample_weight)
            if tree_error == None:
                break

            self.tree_errors_[tree] = tree_error
            self.tree_weights_[tree] = tree_weight

            if tree_error <= 0:
                break

        return self
      
    def predict(self, X):
        n_classes = self.n_classes_
        self.classes_ = np.array(self.classes_)
        #print(self.classes_)
        classes = self.classes_[:, np.newaxis]
        #print(classes)
        pred = None
        
        
        #for tree in self.trees_:
          #print(tree.predict_new(X))
          #print(tree.predict_new(X)==classes)
          
        
        pred = sum((tree.predict_new(X) == classes).T * w for tree, w in zip(self.trees_, self.tree_weights_))
        print(pred)

        pred = pred/self.tree_weights_.sum()
        if n_classes == 2:
            pred[:, 0] = pred[:, 0]*(-1)
            pred = pred.sum(axis=1)
            return self.classes_.take(pred > 0, axis=0)
        #print(pred)
        predictions = self.classes_.take(np.argmax(pred, axis=1), axis=0)
        #print(predictions)
        return predictions
      
class DecisionTreeSW:
		def __init__(self,max_depth = 1,split_val_metric = 'mean',min_info_gain = 0.0,split_node_criterion = 'gini'):
			self.max_depth = max_depth
			self.split_val_metric = split_val_metric
			self.min_info_gain = min_info_gain
			self.split_node_criterion = split_node_criterion
			self.root = None
		
		# Calculate the Gini index for a split dataset
		def gini_index(self,groups, classes):
			# count all samples at split point
			total_weight = float(sum([sum(group[2]) for group in groups]))
			# sum weighted Gini index for each group
			gini = 0.0
			for group in groups:
				total_weight_group = float(sum(group[2]))
				
				if total_weight_group==0:
					continue
				
				score = 0.0
				for class_val in classes:
					weight = 0
					for i in range(len(group[0])):
					  if group[1][i]==class_val:
					    weight += group[2][i]
					
					p = float(weight)/float(total_weight_group)
					score += p**2
				
				gini += (1.0 - score) * (total_weight_group/total_weight)
			return gini

			# Split a dataset based on an attribute and an attribute value
		def test_split(self,index, value, X_train,y_train,sample_weights):
			X_left,y_left,s_left,X_right,y_right,s_right = list(), list(), list(), list(),list(), list()
			for i in range(len(X_train)):
				if X_train[i][index]<value:
					X_left.append(X_train[i])
					y_left.append(y_train[i])
					s_left.append(sample_weights[i])
				else:
					X_right.append(X_train[i])
					y_right.append(y_train[i])
					s_right.append(sample_weights[i])
					
			return [X_left,y_left,s_left],[X_right,y_right,s_right]
			
		def information_gain(self,groups,classes):
			#n_instances = float(sum([len(group[0]) for group in groups]))
			total_weight = float(sum([sum(group[2]) for group in groups]))
			
			total = []
			wt = []
			for group in groups:
				total+=group[1]
				wt+=group[2]

			ent = 0.0
			weight = 0.0
			for class_val in classes:
				for i in range(len(total)):
					if total[i]==class_val:
					  weight += wt[i]
				
				p = float(weight)/float(total_weight)
				ent+= (-1.0 * p * np.log2(p))
				
			score = ent
		
			for group in groups:
				total_weight_group = float(sum(group[2]))
				# avoid divide by zero
				if total_weight_group == 0:
					continue
					
				ent = 0.0
				# score the group based on the score for each class
				for class_val in classes:
					weight = 0
					for i in range(len(group[0])):
					  if group[1][i]==class_val:
					    weight += group[2][i]
					    
					p = float(weight)/float(total_weight_group)
					ent+= (-1.0 * p * np.log2(p))
				# weight the group score by its relative size
				score -= ent * (total_weight_group/total_weight)
			return score

		# Select the best split point for a dataset
		def get_split(self,X_train,y_train,sample_weights):
			#X_train = np.array(X_train.values)
			#class_values = y_train.values
			self.class_values = set(y_train)
			b_index, b_value, b_score, b_groups = 999, 999, 999, None
			if self.split_node_criterion == 'entropy':
				b_score = self.min_info_gain
				#b_score = -999
			for index in range(len(X_train[0])):
				if self.split_val_metric=='mean':
					fval = np.mean(np.array(X_train)[:,index])
				else:
					fval = np.median(np.array(X_train)[:,index])
				#print(fval)
				for row in X_train:
					#print(index)
					groups = self.test_split(index, fval, X_train,y_train,sample_weights)
					if self.split_node_criterion=='gini':
					  gini = self.gini_index(groups, self.class_values)
					  if gini < b_score:
					    b_index, b_value, b_score, b_groups = index, fval, gini, groups
					    
					elif self.split_node_criterion=='entropy':
					  info_gain = self.information_gain(groups,self.class_values)
					  if info_gain >= b_score:
					    b_index, b_value, b_score, b_groups = index, fval, info_gain, groups
					  
			return {'index':b_index, 'value':b_value, 'groups':b_groups}

		def to_terminal(self,group_class):
			outcomes = group_class
			return max(set(outcomes), key=outcomes.count)

		def split(self,node, depth):
			left, right = node['groups']

			#print(left[0],left[1],right[1])
			del(node['groups'])
			# check for a no split
			if not left[0] or not right[0]:
				node['left'] = node['right'] = self.to_terminal(left[1] + right[1])
				return
			# check for max depth
			if depth >= self.max_depth:
				node['left'], node['right'] = self.to_terminal(left[1]), self.to_terminal(right[1])
				return
			# process left child
			if len(left) <= 1:
				node['left'] = self.to_terminal(left[1])
			else:
				node['left'] = self.get_split(left[0],left[1],left[2])
				if node['left']['groups']==None:
					node['left'] = self.to_terminal(left[1])
				else:
					self.split(node['left'], depth+1)
			# process right child
			if len(right) <= 1:
				node['right'] = self.to_terminal(right[1])
			else:
				node['right'] = self.get_split(right[0],right[1],right[2])
				if node['right']['groups']==None:
					node['right'] = self.to_terminal(right[1])
				else:
					self.split(node['right'], depth+1)
			#print(depth)

			# Build a decision tree
		def build_tree(self,X_train,y_train,sample_weights = []):
			if len(sample_weights)==0:
				sample_weights = np.ones(X_train.shape[0])
			X_train = np.array(X_train.values)
			y_train = y_train.values
			root = self.get_split(X_train,y_train,sample_weights)
			self.split(root, 1)
			self.root = root
			#self.print_tree(self.root)
			return root

		def print_tree(self,node, depth=0):
			if isinstance(node, dict):
				print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
				self.print_tree(node['left'], depth+1)
				self.print_tree(node['right'], depth+1)
			else:
				print('%s[%s]' % ((depth*' ', node)))

		def predict(self,node, row):
			if row[node['index']] < node['value']:
				if isinstance(node['left'], dict):
					return self.predict(node['left'], row)
				else:
					return node['left']
			else:
				if isinstance(node['right'], dict):
					return self.predict(node['right'], row)
				else:
					return node['right']
			
		def predict_new(self,X_test):
			X_test = np.array(X_test.values)
			predictions = []
			for row in X_test:
				prediction = self.predict(self.root,row)
				predictions.append(prediction)
			return predictions
			
		def predict_util(self,node,X_test):
			X_test = np.array(X_test.values)
			preds = []
			for row in X_test:
				prediciton = self.predict(node,row)
				print(np.array(prediction).shape)
				predictions.append(prediction)
			return predictions
    
class Logistic_Regression:
    def __init__(self, regulariser = 'L2', lamda = 0, num_steps = 100000, learning_rate = 0.01, initial_wts = None, verbose=False):
        self.regulariser = regulariser
        self.lamda = lamda
        self.num_steps = num_steps
        self.learning_rate = learning_rate
        self.initial_wts = initial_wts
        self.verbose = verbose
    
    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
      
    def __loss(self, h, y, wt, reg):
        epsilon = 1e-5
        if reg == 'L2':
          return (np.sum(-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)) + ((self.lamda/2) * np.dot(wt.T,wt)))/y.size
        else:
          return (np.sum(-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon)) + ((self.lamda) * np.sum(np.absolute(wt))))/y.size

    def __sgn_matrix(self, W):
      ls = []
      for i in W:
        if i>0:
          ls.append(1)
        elif i<0:
          ls.append(-1)
        else:
          ls.append(0)
      ls = np.asarray(ls)
      return ls
    
    def __generate_random(self):
      list1 = np.random.rand(1)
      for i in list1:
        num1 = i
      return num1
    
    def __weight_generator(self, X, initial_wts):
      if initial_wts == None:
        theta_temp = np.random.rand(X.shape[1])
        bias_temp = self.__generate_random()
      else:
        theta_temp = []
        for i in range(1, initial_wts.size):
          theta_temp.append(initial_wts[i])
        theta_temp = np.asarray(theta_temp)
        bias_temp = initial_wts[0]
      return theta_temp, bias_temp
    
    def __binary_logreg(self, X, y, theta_par, bias_par):
      theta = theta_par
      bias = bias_par
      for i in range(self.num_steps):
        z = np.dot(X, theta) + bias
        h = self.__sigmoid(z)
           
        if self.regulariser == 'L2':
          gradient_w = (np.dot(X.T, (h - y))+(self.lamda*theta)) / y.size
        else:
          gradient_w = (np.dot(X.T, (h - y))+(self.lamda*self.__sgn_matrix(theta))) / y.size
            
        gradient_b = np.sum(h-y)/y.size

        theta -= self.learning_rate * gradient_w
        bias -= self.learning_rate * gradient_b

        z = np.dot(X, theta) + bias
        h = self.__sigmoid(z)
        loss1 = self.__loss(h, y, theta, self.regulariser)
              
        if(self.verbose ==True and i % 1000 == 0):
          print('loss after'+ str(i) + ': ' + str(loss1) +' \t')   
          
      return theta, bias
          
    def fit(self, X, y):
        self.theta, self.bias = self.__weight_generator(X, self.initial_wts)  

        self.classes_ = np.array(sorted(list(set(y))))
        self.n_classes_ = len(self.classes_)
          
        if self.n_classes_ == 2:
          self.theta, self.bias = self.__binary_logreg(X, y, self.theta, self.bias)
        else:
          self.para_theta_ = {}
          self.para_bias_ = {}
          for i in self.classes_:
            y_new = np.copy(y)
            for j in range(y.size):
              if y[j] == i:
                y_new[j] = 1
              else:
                y_new[j] = 0
                
            theta1, bias1 = self.__binary_logreg(X, y_new, self.theta, self.bias)
            self.para_theta_[i] = np.copy(theta1)
            self.para_bias_[i] = np.copy(bias1)  
    
    def predict_prob(self, X):
        if self.n_classes_ == 2: 
          return self.__sigmoid(np.dot(X, self.theta) + self.bias).round()
        else:
          self.hypothesis_ = {}
          mx = []
          for i in self.classes_:
            self.hypothesis_[i] = self.__sigmoid(np.dot(X, self.para_theta_[i]) + self.para_bias_[i])
            mx.append(self.hypothesis_[i])
          df = pd.DataFrame(mx)
          search_index = df.idxmax().values
          predictions = [self.classes_[search_index[i]] for i in range(len(search_index))]
          return predictions
    
    def predict(self, X):
      return self.predict_prob(X)
    
class DecisionTree:
  
  def __init__(self,max_depth = 5,split_val_metric = 'mean',min_info_gain = 0.0,split_node_criterion = 'gini'):
    self.max_depth = max_depth
    self.split_val_metric = split_val_metric
    self.min_info_gain = min_info_gain
    self.split_node_criterion = split_node_criterion
    self.root = None
    
  # Calculate the Gini index for a split dataset
  def gini_index(self,groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group[0]) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
      size = float(len(group[0]))
      # avoid divide by zero
      if size == 0:
        continue
      score = 0.0
      # score the group based on the score for each class
      for class_val in classes:
        p = (group[1].count(class_val) / size)
        score += p**2
      # weight the group score by its relative size
      gini += (1.0 - score) * (size / n_instances)
    return gini

  # Split a dataset based on an attribute and an attribute value
  def test_split(self,index, value, X_train,y_train):
    X_left,y_left,X_right,y_right = list(), list(), list(), list()
    for i in range(len(X_train)):
      if X_train[i][index]<value:
        X_left.append(X_train[i])
        y_left.append(y_train[i])
      else:
        X_right.append(X_train[i])
        y_right.append(y_train[i])
        
    return [X_left,y_left],[X_right,y_right]
      
  def information_gain(self,groups,classes):
    n_instances = float(sum([len(group[0]) for group in groups]))
    
    total = []
    for group in groups:
      total+=group[1]

    ent = 0.0
    for class_val in classes:
      p = total.count(class_val)/float(len(total))
      ent+= (-1.0 * p * np.log2(p))
      
    score = ent
    
    for group in groups:
      size = float(len(group[0]))
      # avoid divide by zero
      if size == 0:
        continue
      ent = 0.0
      # score the group based on the score for each class
      for class_val in classes:
        p = (group[1].count(class_val) / size)
        ent+= (-1.0 * p * np.log2(p))
      # weight the group score by its relative size
      score -= ent * (size/n_instances)
    return score

  # Select the best split point for a dataset
  def get_split(self,X_train,y_train):
    #X_train = np.array(X_train.values)
    #class_values = y_train.values
    self.class_values = set(y_train)
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    if self.split_node_criterion == 'entropy':
      b_score = self.min_info_gain
      #b_score = -999
    for index in range(len(X_train[0])):
      if self.split_val_metric=='mean':
        fval = np.mean(np.array(X_train)[:,index])
      else:
        fval = np.median(np.array(X_train)[:,index])
      #print(fval)
      for row in X_train:
        #print(index)
        groups = self.test_split(index, fval, X_train,y_train)
        if self.split_node_criterion=='gini':
          gini = self.gini_index(groups, self.class_values)
          if gini < b_score:
            b_index, b_value, b_score, b_groups = index, fval, gini, groups
            
        elif self.split_node_criterion=='entropy':
          info_gain = self.information_gain(groups,self.class_values)
          if info_gain >= b_score:
            b_index, b_value, b_score, b_groups = index, fval, info_gain, groups
          
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

  def to_terminal(self,group_class):
    outcomes = group_class
    return max(set(outcomes), key=outcomes.count)

  def split(self,node, depth):
    left, right = node['groups']

    #print(left[0],left[1],right[1])
    del(node['groups'])
    # check for a no split
    if not left[0] or not right[0]:
      node['left'] = node['right'] = self.to_terminal(left[1] + right[1])
      return
    # check for max depth
    if depth >= self.max_depth:
      node['left'], node['right'] = self.to_terminal(left[1]), self.to_terminal(right[1])
      return
    # process left child
    if len(left) <= 1:
      node['left'] = self.to_terminal(left[1])
    else:
      node['left'] = self.get_split(left[0],left[1])
      if node['left']['groups']==None:
        node['left'] = self.to_terminal(left[1])
      else:
        self.split(node['left'], depth+1)
    # process right child
    if len(right) <= 1:
      node['right'] = self.to_terminal(right[1])
    else:
      node['right'] = self.get_split(right[0],right[1])
      if node['right']['groups']==None:
        node['right'] = self.to_terminal(right[1])
      else:
        self.split(node['right'], depth+1)
    #print(depth)

      # Build a decision tree
  def build_tree(self,X_train,y_train):
    X_train = np.array(X_train.values)
    y_train = y_train.values
    root = self.get_split(X_train,y_train)
    self.split(root, 1)
    self.root = root
    #self.print_tree(self.root)
    return root

  def print_tree(self,node, depth=0):
    if isinstance(node, dict):
      print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
      self.print_tree(node['left'], depth+1)
      self.print_tree(node['right'], depth+1)
    else:
      print('%s[%s]' % ((depth*' ', node)))

  def predict(self,node, row):
    if row[node['index']] < node['value']:
      if isinstance(node['left'], dict):
        return self.predict(node['left'], row)
      else:
        return node['left']
    else:
      if isinstance(node['right'], dict):
        return self.predict(node['right'], row)
      else:
        return node['right']
  
  def predict_new(self,X_test):
    X_test = np.array(X_test.values)
    predictions = []
    for row in X_test:
      prediction = self.predict(self.root,row)
      predictions.append(prediction)
    return predictions
  
  def predict_util(self,node,X_test):
    X_test = np.array(X_test.values)
    preds = []
    for row in X_test:
      prediciton = self.predict(node,row)
      print(np.array(prediction).shape)
      predictions.append(prediction)
    return predictions
  
class RandomForest:
		def __init__(self,n_trees=10, max_depth=5,split_val_metric='mean',min_info_gain=0.0,split_node_criterion='gini', max_features=5, bootstrap=True, n_cores=1):
		  self.n_trees = n_trees
		  self.max_depth = max_depth
		  self.split_val_metric = split_val_metric
		  self.min_info_gain = min_info_gain
		  self.split_node_criterion = split_node_criterion
		  self.max_features = max_features
		  self.bootstrap = bootstrap
		  self.n_cores = n_cores
		  self.trees = None
		  
		def subsample(self,X_train,y_train, ratio):
		  sample_X,sample_y = list(),list()
		  n_sample = round(len(X_train) * ratio)
		  while len(sample_X) < n_sample:
		    index = np.random.randint(len(X_train))
		    sample_X.append(X_train[index])
		    sample_y.append(y_train[index])
		  
		  return sample_X,sample_y

		def train(self,X_train,y_train):
		  features = []  
		  cols = X_train.columns
		  while len(features) < self.max_features:
		    index = np.random.randint(len(cols))
		    if cols[index] not in features:
		      features.append(cols[index])
		      
		  print(features)
		  X_train = X_train[features]
		  
		  X_train = np.array(X_train.values)
		  y_train = y_train.values
		  trees = list()
		  sample_size = np.random.rand()
		  for i in range(self.n_trees):
		    if self.bootstrap:
		      sample_X,sample_y = self.subsample(X_train,y_train, sample_size)
		    else:
		      sample_X,sample_y = X_train,y_train
		      
		    sample_X,df = pd.DataFrame(sample_X),pd.DataFrame(sample_y)
		    sample_y = df[0]
		    #print(sample_y.values)

		    dt = DecisionTree(self.max_depth,self.split_val_metric,self.min_info_gain,self.split_node_criterion)
		    tree = dt.build_tree(sample_X, sample_y)
		    trees.append(tree)

		  self.trees = trees
		  return self.trees
		
		def fit_predict(self,X_train,y_train,X_test):
		  #features = []  
		  #cols = X_train.columns
		  #while len(features)!=len(cols) and len(features) < self.max_features:
		    #print(len(features))
		    #index = np.random.randint(len(cols))
		    #if cols[index] not in features:
		      #features.append(cols[index])
		      
		  #print(features)
		  #X_train = X_train[features]
		  #X_train = np.array(X_train.values)
		  #y_train = y_train.values
		  y_train_new = y_train.values
		  trees = list()
		  cols = X_train.columns
		  sample_size = np.random.rand()
		  predictions = []
		  for i in range(self.n_trees):
		    
		    if self.bootstrap:
		      features = []
		      i=0
		      while len(features)!=len(cols) and len(features) < self.max_features:
		        #print(len(features))
		        index = np.random.randint(len(cols))
		        if cols[index] not in features:
		          features.append(cols[index])
		        if len(features)>=1 and i == self.max_features+2:
		          break
		        i+=1

		      #print(features)
		      X_train_new = X_train[features]
		      X_train_new = np.array(X_train_new.values)
		    
		      sample_X,sample_y = self.subsample(X_train_new,y_train_new, sample_size)
		      sample_X,df = pd.DataFrame(sample_X),pd.DataFrame(sample_y)
		      sample_y = df[0]
		    else:
		      sample_X,sample_y = X_train,y_train
		      
		    
		    #print(sample_y.values)

		    dt = DecisionTree(self.max_depth,self.split_val_metric,self.min_info_gain,self.split_node_criterion)
		    dt.build_tree(sample_X, sample_y)
		    prediction = dt.predict_new(X_test)
		    predictions.append(prediction)
		    
		  #print(np.array(predictions).shape)
		  
		  df = np.array(predictions)
		  print(df.shape)
		  df = pd.DataFrame(df)
		  final_predictions = df.mode(axis = 0).values[0]
		  
		  return final_predictions
    
class Naive_Bayes(object):
    def __init__(self, type = "Gaussian", prior = []):
        self.type = type
        self.prior = prior
    
    def fit(self, X, y):
        if((self.type).lower() == "multinomial"):
            count_sample = X.shape[0]
            separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
            if len(self.prior)==0:
                self.class_log_prior_ = [np.log(len(i) / count_sample) for i in separated]
            else:
                self.class_log_prior_ = self.prior
            count = np.array([np.array(i).sum(axis=0) for i in separated]) + 1.0
            self.feature_log_prob_ = np.log(count / count.sum(axis=1)[np.newaxis].T)
            return self
        if((self.type).lower() == "gaussian"):
            separated = [[x for x, t in zip(X, y) if t == c] for c in np.unique(y)]
            self.model = np.array([np.c_[np.mean(i, axis=0), np.std(i, axis=0)]
                    for i in separated])
        return self

    def _prob(self, x, mean, std):
        if((self.type).lower() == "gaussian"):
            exponent = np.exp(- ((x - mean)**2 / (2 * std**2)))
            return np.log(exponent / (np.sqrt(2 * np.pi) * std))

    def predict_log_proba(self, X):
        if((self.type).lower() == "multinomial"):
            return [(self.feature_log_prob_ * x).sum(axis=1) + self.class_log_prior_
                    for x in X] 
        if((self.type).lower() == "gaussian"):
            return [[sum(self._prob(i, *s) for s, i in zip(summaries, x))
                    for summaries in self.model] for x in X]

    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)
      
class Stacking:
    def __init__(self, tuples = [(DecisionTree(), 2)]):
        self.tuples = tuples

    def cross_validation_split(self, X,y, folds=3):
        dataset_split = list()
        dataset_splity = list()
        dataset_copy = list(X)
        dataset_copyy = list(y)
        fold_size = int(len(X) / folds)
        for i in range(folds):
            fold = list()
            foldy = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy)
                foldy.append(dataset_copyy)
            dataset_split.append(fold)
            dataset_splity.append(foldy)
        
        fold_size = len(X) - int(float(len(X))/float(folds))
        
        X_split,y_split = X.iloc[:fold_size],y.iloc[:fold_size]
        dataset_split = X_split
        dataset_splity = y_split
        #print(len(dataset_split),len(dataset_splity))
        return dataset_split, dataset_splity

    def fit(self,X_train, X_test, y_train):  
        Predictions = []
        for i in range(len(self.tuples)): 
            Xtrain, ytrain = self.cross_validation_split(X_train, y_train, folds=self.tuples[i][1])
            #print(np.array(Xtrain).shape)
            for j in range(self.tuples[i][1]):
                print(self.tuples[i][0])
                if isinstance(self.tuples[i][0], DecisionTree):
                    ytrain_new = ytrain[0]
                    self.tuples[i][0].build_tree(Xtrain , ytrain_new)
                elif isinstance(self.tuples[i][0], RandomForest):
                    preds = self.tuples[i][0].fit_predict(Xtrain , ytrain, X_test)
                else:
                    self.tuples[i][0].fit(Xtrain , ytrain)
                
                if isinstance(self.tuples[i][0], DecisionTree):
                    preds=self.tuples[i][0].predict_new(X_test)
                elif isinstance(self.tuples[i][0], RandomForest):
                    preds=self.tuples[i][0].fit_predict(Xtrain,ytrain,X_test)
                else:
                    preds=self.tuples[i][0].predict(X_test)
                
                Predictions.append(preds)
        return Predictions

    def predict(self,Predictions):
        pred = stats.mode(Predictions)
        pred = np.array(pred.mode[0])
        return pred
