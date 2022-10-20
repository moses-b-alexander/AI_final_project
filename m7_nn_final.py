# Moses Alexander
# 10/19/2022

# imports
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
from time import perf_counter, sleep

""" SETUP """
# set path to data file
filename = str(os.path.join(os.getcwd(), "car1.csv"))
# add whitespace for readability of output
print("\n\n\n\n", filename, "\n\n\n\n")
# read data from .csv
df = pd.read_csv(filename)
# seed for reproducibility of randomness
seed = 1
# define columns as specified in .csv header
# target
target_col = ["acceptability"]
# features
feature_cols = [c for c in df.columns.values if c not in target_col]
# combine target and features
cols = feature_cols + target_col
# number of rows in data
df_size = df.shape[0]
# number of target classes
num_classes = df[target_col[0]].unique().shape[0]

""" DATA INFO """
# add whitespace for readability of output
print("\n\nDATA:\n\n")
# print data
print(df)
# add whitespace for readability of output
print("\n\nFEATURE DISTRIBUTIONS:\n\n")
# print distributions of columns
[print(df[col].value_counts()) for col in feature_cols]
# add whitespace for readability of output
print("\n\nTARGET DISTRIBUTION:\n\n")
[print(df[col].value_counts()) for col in target_col]
# add whitespace for readability of output
print("\n\nNUMBER OF TARGET CLASSES:\n\n")
# print number of target classes
print(num_classes)
# add whitespace for readability of output
print("\n\nSAMPLE OF DATA:\n\n")
# get random sample of data
sample = df.sample(20)
# print sample of data
print(sample)
# add whitespace for readability of output
print("\n\nSHAPE OF DATA:\n\n")
# print shape of data
print(df.shape)
# add whitespace for readability of output
print("\n\n\n\n\n")

""" MODEL HYPERPARAMETERS """
# number of training iterations
epochs = 200
# batch size per training step
# 64 performed better on this model than 4, 8, 16, 32
# batches larger than 64 are invalid for this model
batch_size = 64
# determines frequency of weight updates per training iteration
# 0.01 performed better on this model than 0.1, 0.05, 0.01, 0.005, 0.001
lr = 0.01
# objective loss function to minimize
# since class labels are encoded as integers and there are more than 2 target classes
loss = "sparse_categorical_crossentropy"
# exponentially decay learning rate over training steps, gives better performance for this model
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-2, decay_steps=10000, decay_rate=0.9)
# optimizer for minimization
# Adam performed better on this model than SGD and Adamax
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
# convert target classes to integer indices, with no out of vocabulary items
target_lookup = \
  layers.StringLookup(vocabulary=df[target_col[0]].unique(), mask_token=None, num_oov_indices=0)

""" HELPER FUNCTIONS """
# create tf dataset from .csv
def get_dataset_from_csv(csv_file_path):
    # initialize tf dataset from .csv file with batch size 64 replicated 3 times
    # shuffle dataset with seed, use .csv header, use acceptability as target column
    # add "NA" as default value and replace missing values with "NA"
    # map target lookup onto target column and store it in dataset
    dataset = tf.data.experimental.make_csv_dataset(
        csv_file_path, batch_size=batch_size, column_names=cols, num_epochs=1, 
        column_defaults=["NA" for _ in range(len(cols))], label_name=target_col[0], 
        header=True, na_value="NA", shuffle=True, shuffle_seed=seed
        ).map(lambda features, target: (features, target_lookup(target)))
    # return
    return dataset.cache()

# tf autograph cannot transform lambda functions currently
@tf.autograph.experimental.do_not_convert
# split dataset into training and validation and test datasets using indices of shuffled dataset
def split(x):
  # return 70/20/10 split for training/validation/test
  return \
    x.enumerate().filter(lambda x, y: x % 10 < 7).map(lambda x,y: y), \
    x.enumerate().filter(lambda x, y: x % 10 > 7).map(lambda x,y: y), \
    x.enumerate().filter(lambda x, y: x % 10 == 7).map(lambda x,y: y)

# define function for plotting loss or accuracy over epochs
def plot_f(model, s):
  # verify "loss" or "accuracy" are the keys passed to the function to plot
  if s not in ["loss", "accuracy"]:  print("Invalid plot label."); return None
  # loss plot y axis limit
  if s == "loss":  y_lim = 1.5
  # accuracy plot y axis limit
  if s == "accuracy":  y_lim = 1.5
  # plot loss or accuracy over epochs on training data using model history
  plt.plot(model.history[s], label=s)
  # plot loss or accuracy over epochs on validation data using model history
  plt.plot(model.history[f"val_{s}"], label=f"val_{s}")
  # label X axis
  plt.xlabel("EPOCHS")
  # label Y axis
  plt.ylabel(s.upper())
  # set plot X axis bounds
  plt.xlim(0, epochs + (0.2 * epochs))
  # set plot Y axis bounds
  plt.ylim(0, y_lim)
  # create plot legend
  plt.legend(["train", "val"], loc="upper left")
  # add gridlines to plot
  plt.grid(True)
  # save plot as .png with random filename to prevent overwriting
  plt.show()
  # clear plots
  plt.close()

# create feature embeddings
def embed_features(inputs):
  # initialize embedded features list
  embedded_features = []
  # iterate through features in df
  for col in inputs:
    # get count of classes for each feature
    ct = df[col].unique().shape[0]
    # convert strings to integer indices, with no out of vocabulary items
    lookup = layers.StringLookup(vocabulary=df[col].unique(), num_oov_indices=0)
    # create embedding for each feature and add to embedded features list
    # input dim = number of classes for catching other values
    # output dim = square root of number of classes
    embedding = layers.Embedding(ct, int(sqrt(ct)))
    # lookup and embed feature and add to embedded features list
    embedded_features.append(embedding(lookup(inputs[col])))
  # concatenated embedded features into single embedded feature tensor
  embedded_features = layers.Concatenate()(embedded_features)
  # return
  return embedded_features

""" MODEL DEFINITIONS """
# decision tree model class
class NeuralDecisionTree(tf.keras.Model):
  # initialize decision tree model
  def __init__(self, depth, num_features, select_features, num_classes):
    # inherit from tf.keras.Model
    super().__init__()
    # depth of tree
    self.depth = depth
    # number of leaves is 2 ^ tree depth
    self.num_leaves = 2 ** depth
    # number of classes for target
    self.num_classes = num_classes
    # select number of features to use in this tree
    num_selected_features = int(num_features * select_features)
    # create identity matrix for features
    one_hot = np.eye(num_features)
    # select random features to use in this tree with one hot vectors of randomly selected features
    self.selected_features_mask = \
      one_hot[np.random.choice(np.arange(num_features), num_selected_features, replace=False)]
    # initialize weights for class distribution of tree
    # represents class distribution of leaves of tree
    self.pi = tf.Variable(
      initial_value=tf.random_normal_initializer()(shape=[self.num_leaves, self.num_classes]), 
      dtype="float32", trainable=True)
    # layer outputting routing probabilities (probability of traveling to each leaf)
    # units = number of leaves, activation function = sigmoid
    self.decision = layers.Dense(units=self.num_leaves, activation="sigmoid", name="decision")
  # call decision tree model
  def call(self, inputs):
    # apply mask and get selected random features
    # transpose selected features mask before multiplication
    # shape = (batch size, number of selected features)
    features = tf.matmul(inputs, self.selected_features_mask, transpose_b=True)
    # compute routing probabilities by applying fully-connected layer to selected random features
    # creates 3d tensor by adding 1 additional dimension of size 1 ("depth")
    # shape = (batch size, number of leaves, 1)
    decisions = tf.expand_dims(self.decision(features), axis=2)
    # concatenate routing probabilities and their complements in 3d tensor
    # complement: probability of going to any other leaf in tree
    # shape = (batch size, number of leaves, 2)
    decisions = layers.concatenate([decisions, 1-decisions], axis=2)
    # initialize probabilities of input data reaching leaves (all 1s)
    mu = tf.ones([batch_size, 1, 1])
    # initialize starting and ending indices
    stt, end = 1, 2
    # breadth-first tree traversal
    # iterate through tree one level at a time
    for i in range(self.depth):
      if mu.shape[0] == batch_size:
        # reshape mu into 3d tensor
        # shape: (batch size, 2 ^ tree level, 1)
        mu = tf.reshape(mu, [batch_size, -1, 1])
        # replicate 3d tensor and add 1 to "depth"
        # # shape: (batch size, 2 ^ tree level, 2) 
        mu = tf.tile(mu, (1, 1, 2))
        # get routing probabilities for all nodes in current tree level
        # shape: (batch size, 2 ^ tree level, 2)
        tree_level_decisions = decisions[:, stt:end, :]
        # multiply routing probabilities of input data batch by 
        # probabilities of routing to each node in current tree level
        # shape: (batch size, 2 ^ tree level, 2)
        mu *= tree_level_decisions
        # set start index to node of first index of next tree level
        # since slicing is not inclusive at the end: [start: end)
        stt = end
        # set ending index to node of first index of next tree level's next tree level
        # since slicing is not inclusive at the end: [start: end)
        end = stt + 2 ** (i + 1)
    # reshape routing probabilities into 2d tensor
    # ie multiplying the replicated "depth" by the number of nodes in the 2nd-to-last level
    # 2 * 2 ^ final tree level - 1 = number of nodes in final tree level = number of leaves
    # shape: (batch size, number of leaves)
    mu = tf.reshape(mu, [batch_size, self.num_leaves])
    # apply softmax to get class distribution for leaf
    probabilities = tf.keras.activations.softmax(self.pi)
    # multiply routing probabilities for each leaf by class distribution for each leaf
    outputs = tf.matmul(mu, probabilities)
    # return
    return outputs

# decision forest model class
class NeuralDecisionForest(tf.keras.Model):
  # initialize decision forest model
  def __init__(self, num_trees, depth, num_features, select_features, num_classes):
    # inherit from tf.keras.Model
    super().__init__()
    # number of classes
    self.num_classes = num_classes
    # create list of neural decision tree objects
    # each tree will split on a set of randomly selected features
    self.trees = [NeuralDecisionTree(depth, num_features, select_features, num_classes) 
        for _ in range(num_trees)]
  # call decision forest model
  def call(self, inputs):
    # initialize outputs with a zero matrix
    # shape: (batch size, number of classes)
    tree_outputs = tf.zeros([batch_size, self.num_classes])
    # aggregate decision tree outputs for all trees in forest
    for tree in self.trees:  tree_outputs += tree(inputs)
    # get average of aggregated tree outputs
    tree_outputs /= len(self.trees)
    # return
    return tree_outputs

""" DECISION FOREST HYPER PARAMETERS """
# number of trees in forest
# 20 performed better on this model than 5, 10, 40
num_trees = 20
# depth of trees
# 10 performed better on this model than 5, and 20 requires too much memory
depth = 10
# proportion of features used in each tree
# 0.7 performed better on this model than 0.3 and 0.5
select_features = 0.7

# define model in functional form
# define input layers for each feature
inputs = {col: layers.Input(name=col, shape=(), dtype=tf.string) for col in feature_cols}
# create feature embeddings
embedded_features = embed_features(inputs)
# batch normalize embedded input data features
features = layers.BatchNormalization()(embedded_features)
# create neural decision forest model with previously specified decision forest hyperparameters
forest = NeuralDecisionForest(num_trees, depth, features.shape[1], select_features, num_classes)
# get forest model outputs
outputs = forest(features)
# define final model
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# show model summary
model.summary()

""" COMPILATION, TRAINING, EVALUATION """
# create tf dataset from .csv
dataset = get_dataset_from_csv(filename)
# training: 70%, validation: 20%, test: 10%
training, validation, test = split(dataset)
# compile model with previously specified hyperparameters
model.compile(optimizer=opt, loss=loss, metrics=["accuracy"])
# start timing
start = perf_counter()
# fit model on training dataset for 100 epochs
model.fit(training, epochs=epochs, validation_data=validation)
# end timing
end = perf_counter()
# plot training loss over epochs
plot_f(model.history, "loss")
# plot training accuracy over epochs
plot_f(model.history, "accuracy")
# print model training time in seconds
print(f"\nModel took {round(end-start, 3)} seconds to train for {epochs} epochs.\n")
# evaluate model on test dataset
results = model.evaluate(test)
# add whitespace for readability of output
print("\n\n\n\n")
# print test loss and accuracy
print(f"\nTEST SET LOSS: {results[0]}\nTEST SET ACCURACY: {results[1]}\n")
# get model predictions on test set
predictions = model.predict(test)
# get test set labels from tensorflow dataset
labels = [i[1] for i in tfds.as_numpy(test)]
# concatenate labels from 2d array into 1d array
labels = np.concatenate(labels)
# get index of class with highest predicted probability
predictions = [np.argmax(p) for p in predictions]
# calculate confusion matrix
confusion_matrix = tf.math.confusion_matrix(labels, predictions, num_classes=num_classes)
# add whitespace for readability of output
print("\n\nCONFUSION MATRIX:\n\n")
# print confusion matrix
print(confusion_matrix)
# add whitespace for readability of output
print("\n\nTOTAL NUMBER OF ITEMS CLASSIFIED:\n\n")
# print sum of confusion matrix diagonal
print(len(labels))
# add whitespace for readability of output
print("\n\nNUMBER CORRECTLY CLASSIFIED:\n\n")
# print sum of confusion matrix diagonal
print(int(tf.linalg.trace(confusion_matrix)))
# add whitespace for readability of output
print("\n\nNUMBER INCORRECTLY CLASSIFIED:\n\n")
# print length of test label list minus sum of confusion matrix diagonal
print(len(labels) - int(tf.linalg.trace(confusion_matrix)))

# add whitespace for readability of output
print("\n\n\n\n")
""" END """
# end
print("\n\ndone.\n\n")

