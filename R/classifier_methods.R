library(rvinecopulib) # copulas
library(keras) # Machine Learning and Neural Networks
library(tensorflow) # Neural Networks

#' Performs a train test split for the original data
#' @param data The data set
#' @param train_perc Defaults to 0.8. Percentage to use for training.
#' @returns A list with 2 entries orig_train, orig_test
train_test_split_orig <- function(data, train_perc){
  num_train <- floor(train_perc * nrow(data))
  train_idx <- sample(nrow(data),num_train)
  data_train <- data[train_idx,]
  data_test <- data[-train_idx,]
  return(list(data_train, data_test))
}

#' Performs a train test split with original data already split in train and test
#' Gives all datapoints in orig_data a label of 1, all datapoints in simplified data a label of 0
#' @param orig_data_train The training set of the real data, which gets labels of 1
#' @param orig_data_test The test set of the real data, which gets labels of 1
#' @param simplified_data Synthetic data set, simulated from a simplified
#' vine copula, gets labels of 0
#' @returns A list with 4 entries, x_train, x_test, y_train, y_test in that order.
train_test_split <- function(orig_data_train, orig_data_test, simplified_data){
  train_perc <- nrow(orig_data_train) / (nrow(orig_data_train) + nrow(orig_data_test))
  y_train_orig <- as.matrix(rep(1L, nrow(orig_data_train)))
  y_test_orig <- as.matrix(rep(1L, nrow(orig_data_test)))
  simplified_labels <- as.matrix(rep(0L, nrow(simplified_data)))
  num_train_simp <- floor(train_perc*nrow(simplified_data))
  train_simp_idx <- sample(nrow(simplified_data), num_train_simp)
  x_train_simp <- simplified_data[train_simp_idx,]
  y_train_simp <- as.matrix(simplified_labels[train_simp_idx,])
  x_test_simp <- simplified_data[-train_simp_idx,]
  y_test_simp <- as.matrix(simplified_labels[-train_simp_idx,])
  # combine the data into one dataset and shuffle
  x_train <- rbind(orig_data_train, x_train_simp)
  y_train <- rbind(y_train_orig, y_train_simp)
  shuffle_idx_train <- sample(nrow(x_train), nrow(x_train))
  x_train <- x_train[shuffle_idx_train,]
  y_train <- y_train[shuffle_idx_train,]
  x_test <- rbind(orig_data_test, x_test_simp)
  y_test <- rbind(y_test_orig, y_test_simp)
  shuffle_idx_test <- sample(nrow(x_test), nrow(x_test))
  x_test <- x_test[shuffle_idx_test,]
  y_test <- y_test[shuffle_idx_test,]
  return(list(x_train,
              x_test,
              y_train,
              y_test))
}

#' learning rate scheduler halves the learning rate every 30 epochs
#' argument to a keras function
#' @param epoch current epoch
#' @param lr current learning rate
#' @returns lr: the updated learning rate
lr_schedule_fun <- function(epoch, lr) {
  if((epoch + 1) %% 30 == 0){
    return(lr/2)
  } else {
    return(lr)
  }
}

# Run Python code to define and register the metric
reticulate::py_run_string("
import tensorflow as tf
from tensorflow import keras

@keras.saving.register_keras_serializable(name='binary_accuracy_from_logits')
def binary_accuracy_from_logits(y_true, y_pred):
    y_pred = tf.nn.sigmoid(y_pred)
    return keras.metrics.binary_accuracy(y_true, y_pred)
")

# Retrieve the registered Python function into R
binary_accuracy_from_logits <- reticulate::py$binary_accuracy_from_logits

#' Defines and compiles a neural network for binary classification.
#' Activation function is leaky_relu, output is one layer with no activation
#' (to train a binary classifier with from_logits=TRUE option)
#' @param input_dim dimension of the input, defaults to 5
#' @param hidden_units vector containing the number of units per layer.
#' Defaults to c(20,10)
#' @param initial_lr initial learing rate to use, defaults to 0.01
#' (íf no lr_scheduler is used during training, this stays the same during
#' the whole training process)
#' @param leaky_relu_alpha slope in the negative part of the relu. Defaults to 0.01.
#' @returns the compiled neural network model
build_model <- function(
    input_dim=5,
    hidden_units=c(20,10),
    initial_lr = 0.01,
    use_tanh=FALSE,
    leaky_relu_alpha=0.1){
  model <- -1 # reset the model variable.
  # definition of the model
  if(use_tanh){
    model <- keras_model_sequential() %>%
      layer_dense(units = hidden_units[1], input_shape = input_dim, activation='tanh') %>%
      layer_dense(units = hidden_units[2], activation='tanh') %>%
      layer_dense(units = 1)
  } else {
    model <- keras_model_sequential() %>%
      layer_dense(units = hidden_units[1], input_shape = input_dim) %>%
      layer_activation_leaky_relu(alpha = leaky_relu_alpha) %>%
      layer_dense(units = hidden_units[2]) %>%
      layer_activation_leaky_relu(alpha = leaky_relu_alpha) %>%
      layer_dense(units = 1)
  }

  # compile the model, define optimizer, loss and metrics
  model %>% compile(
    optimizer = keras$optimizers$Adam(learning_rate=initial_lr),
    loss = keras$losses$BinaryCrossentropy(from_logits=TRUE),
    metrics = list(binary_accuracy_from_logits)
  )
  return(model)
}

#' function to train the model
#' @param model The NN model to train
#' @param x_train training data
#' @param y_train labels for the training data
#' @param lr_schedule Function for learning rate scheduling. Defaults to lr_schedule_fun.
#' @param num_epochs Integer, defaults to 500.
#' Number of epochs for which the model should be trained.
#' @param batch_size Integer, defaults to 128.
#' Batch size to use to compute gradients during training
#' @param val_split Number between 0 and 1. Fraction of data used for validation.
#' @param verbose Whether output should be printed. Can be 0,1 or 2.
#' @returns list(model,history) a list of the trained model in the first position
#' and the training history (loss, accuracy and learing_rate per epoch)
train_model <- function(
    model,
    x_train,
    y_train,
    lr_schedule = lr_schedule_fun,
    num_epochs = 300,
    batch_size=128,
    val_split = 0.2,
    verbose=1
){
  # create the keras object required for learning rate scheduling
  lr_scheduler <- callback_learning_rate_scheduler(schedule = lr_schedule)

  # train the model
  history <- model %>% fit(
    x_train, y_train,
    epochs = num_epochs,
    batch_size = batch_size,
    validation_split=val_split, # use 20 percent of training data as validation data
    verbose = verbose,
    callbacks=list(lr_scheduler)
  )
  return(list(model, history))
}

#' Calculate the correction factors, i.e. the factors lambda, with which the
#' simplified copula needs to be multiplied to get the non-parametric copula.
#' @param model The classifier trained to distinguish non-simplified data
#' with labels 1 from simplified data with labels 0
#' @param obs the observations, for which the factors should be created.
#' @param nu The fraction of noise to real samples
#' @param verbose Whether output should be printed. Can be 0,1 or 2.
#' @returns The correction factors
correction_factors <- function(model, obs, nu=1, verbose=1){
  predictions <- model %>% predict(obs, verbose=verbose)
  # cast outputs of neural net to numeric, just to be sure
  cor_factors <- exp(as.numeric(predictions) + log(nu))
  return(cor_factors)
}

#' compute the value of the non-parametric copula obtained from the simplified fit,
#' together with the classifier
#' @param model The classifier trained to distinguish non-simplified data
#' with labels 1 from simplified data with labels 0
#' @param fitted_vine The vine copula fitted to the observed data.
#' @param obs the observations, for which the non-parametric copula should be created.
#' @param nu The fraction of noise to real samples
#' @param verbose Whether output should be printed. Can be 0,1 or 2.
#' @returns The non-parametric copula evaluated at the points obs.
NCE_cop <- function(model, fitted_vine, obs, nu=1, verbose=1){
  cor_facs <- correction_factors(model=model, obs=obs, nu=nu, verbose=verbose)
  c_np <- cor_facs * dvinecop(obs, fitted_vine)
  return(c_np)
}

#' Count number of neural network parameters
#' @param weights The weights of a neural network, accessible via model$weights
#' for a tensorflow neural network called model
#' @returns num_params (integer) total number of parameters in the network
count_NN_params <- function(weights) {
  num_params <- sum(sapply(weights, function(w) {
    shape <- w$shape$as_list()
    prod(unlist(shape))
  }))
  return(num_params)
}

#' Get number of copula parameters
#' @param fitted_vine A copula model (from rvinecopulib)
#' @returns num_params (integer) total number of parameters in the vine copula model
get_num_cop_params <- function(fitted_vine){
  # Assume `fit` is your vinecop model from rvinecopulib
  num_params <- sum(sapply(fitted_vine$pair_copulas, function(pc) {
    # Each pc is a matrix (upper triangular) of pair copula objects
    sum(sapply(pc, function(cop) {
      if (is.null(cop)) return(0)
      length(cop$parameters)
    }))
  }))
  return(num_params)
}

#' Monte Carlo integral of NCE_cop (using importance sampling)
#' @param model The classifier trained to distinguish non-simplified data
#' with labels 1 from simplified data with labels 0
#' @param fitted_vine The vine copula fitted to the observed data.
#' @param n_samples Number of samples to create to compute the integral
#' @param data_dim_if_unif Data dimensionality. If this is given (and !=-1),
#' then the samples are drawn from a uniform distribution on [0,1]^d.
#' Otherwise the samples are drawn from the simplified vine
#' @param user_info Whether to display an information, which steps are currently running.
#' @returns The Monte Carlo integral approximation.
compute_integral <- function(model,
                             fitted_vine,
                             n_samples,
                             nu,
                             data_dim_if_unif=-1,
                             user_info=FALSE){
  samples <- 0
  if (data_dim_if_unif != -1){
    if (user_info){
      print("Start noise sampling (uniform)")
    }
    # simulate random uniform samples on [0,1]^d
    samples <- matrix(runif(n_samples * data_dim_if_unif),
                      ncol=data_dim_if_unif)
  } else {
    if (user_info){
      print("Start noise sampling (simplified vine)")
    }
    # simulate samples from the simplified vine copula
    samples <- rvinecop(n_samples, fitted_vine)
  }
  if(user_info){
    print("Noise samples created")
  }
  p_simp <- 1
  if(data_dim_if_unif==-1){
    # If samples are drawn from the simplified vine copula,
    # the density needs to be evaluated
    if(user_info){
      print("Evaluating noise density (simplified vine)")
    }
    p_simp <- dvinecop(samples, fitted_vine)
  }
  if(user_info){
    print("Evaluating neural network output")
  }
  verbose <- 0
  if(user_info){
    verbose <- 1
  }
  p_non_param <- NCE_cop(fitted_vine=fitted_vine, model=model, obs=samples, nu=nu, verbose=verbose)
  return(mean(p_non_param/p_simp))
}

#' Compute the g values as defined in the thesis
#' @param model Neural Network for binary classification
#' @param orig_data The observed data.
#' @param nu Defaults to 1. Fraction of noise samples to real samples that
#' was used during training of the neural network.
#' @returns A vector containing the values g_t (see thesis)
compute_gvals <- function(
    model,
    orig_data,
    nu=1){
  predictions <- model %>% predict(orig_data)
  # here a classifier on the logit level h(u) was trained on the data. to get the values g(u),
  # as defined in the thesis, g(u) = log(nu * p(u)/(1-p(u))) = log(nu * exp(h(u))) = h(u) + log(nu)
  # needs to be computed, where nu is the fraction of number of noise samples to observed samples
  g_vals <- as.numeric(predictions) + log(nu)
  return(g_vals)
}

