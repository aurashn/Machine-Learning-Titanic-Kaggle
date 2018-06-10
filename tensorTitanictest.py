import glob
import io
import math
import os


from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.python.data import Dataset


tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format



def parse_labels_and_features(dataset):
	labels = dataset[1]
	features = dataset[[2,4,5,6,7,9,11]]
	features = features.fillna(features.mean())
	features.columns = ['Pclass','Sex','Age','SibSp','Parch', 'Fare', 'Embarked']
	lb_make = LabelEncoder()
	features["SexCode"] = lb_make.fit_transform(features["Sex"])
	features["EmbarkCode"] = lb_make.fit_transform(features["Embarked"])
	features = features.drop(['Sex','Embarked'],axis = 1)
	print(features)
	return labels, features


titanic_dataframe = pd.read_csv(
  io.open("/home/aurash/titanic_kaggle/train.csv", "r"),
  sep=",",
  header=None,skiprows=1)

# Use just the first 10,000 records for training/validation.
#titanic_dataframe = titanic_dataframe.head(10)
#mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))
train_labels, train_features = parse_labels_and_features(titanic_dataframe[:650])
val_labels, val_features = parse_labels_and_features(titanic_dataframe[651:891])

def construct_feature_columns():
	return set([tf.feature_column.numeric_column('Pclass'),tf.feature_column.numeric_column('Age'),tf.feature_column.numeric_column('SibSp'),tf.feature_column.numeric_column('Parch'),tf.feature_column.numeric_column('Fare'),tf.feature_column.numeric_column('SexCode'),tf.feature_column.numeric_column('EmbarkCode')])


def create_training_input_fn(features, labels, batch_size, num_epochs = None, shuffle = True):
	def _input_fn(num_epochs=None,shuffle = True):
		idx =  np.random.permutation(features.index)
		raw_features = {"Pclass": features["Pclass"].values ,"Age": features["Age"].values ,"SibSp":features["SibSp"].values  ,"Parch":  features["Parch"].values,"Fare": features["Fare"].values ,"SexCode": features["SexCode"].values ,"EmbarkCode":features["EmbarkCode"].values}
		raw_targets = np.array(labels)

		ds = Dataset.from_tensor_slices((raw_features,raw_targets))
		ds = ds.batch(batch_size)
		feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
		return feature_batch,label_batch
	return _input_fn
	

def train_linear_classification_model(learning_rate,steps,batch_size, training_examples,training_targets, validation_examples, validation_targets):

	periods = 50
	steps_per_period = steps/periods
	predict_training_input_fn = create_training_input_fn(training_examples, training_targets, batch_size)	
	predict_validation_input_fn = create_training_input_fn(validation_examples,validation_targets,batch_size)	
	training_input_fn = create_training_input_fn(training_examples,training_targets,batch_size)

	my_optimizer = tf.train.AdagradOptimizer(learning_rate = learning_rate)
	my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer,5.0)
	classifier = tf.estimator.LinearClassifier(feature_columns = construct_feature_columns(),n_classes = 2,optimizer = my_optimizer, config = tf.estimator.RunConfig(keep_checkpoint_max =1))

	print "training model ...."
	print "LogLoss error (on validation data):"
	training_errors = []
	validation_errors = []
	for period in range(0,periods):
		classifier.train(input_fn = training_input_fn,steps = steps_per_period)

		training_predictions = list(classifier.predict(input_fn = predict_training_input_fn))
		training_probabilities = np.array([item['probabilities'] for item in training_predictions])
		training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
		training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id,2)

		validation_predictions = list(classifier.predict(input_fn = predict_validation_input_fn))
		validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
		validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
		validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id,2)
		

		training_log_loss = metrics.log_loss(training_targets,training_pred_one_hot)
		validation_log_loss =  metrics.log_loss(validation_targets,validation_pred_one_hot)

		print " period %02d : %.02f" % (period, validation_log_loss)

		training_errors.append(training_log_loss)
		validation_errors.append(validation_log_loss)

	print "Model training finished"
	
	_ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

	final_predictions = classifier.predict(input_fn= predict_validation_input_fn)
	final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

	accuracy = metrics.accuracy_score(validation_targets,final_predictions)
	print "Final Accuracy (on validation data): %.02f" % accuracy
	return classifier
	

classifier = train_linear_classification_model(
             learning_rate=0.02,
             steps=100,
             batch_size=100,
             training_examples=train_features,
             training_targets=train_labels,
             validation_examples=val_features,
             validation_targets=val_labels)	