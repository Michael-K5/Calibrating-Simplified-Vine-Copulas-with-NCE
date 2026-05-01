# Script for applying the method to an actual dataset
# source("R/classifier_methods.R")
# source("R/plotting_methods.R")
library(keras)
library(rvinecopulib)
# devtools::install_github("Michael-K5/vineNCE")
library(vineNCE)
library(ggplot2)
library(dplyr)
library(patchwork)
library(tibble)
library(ggplot2)
library(tidyr)
# Load Data set
num_cores <- 4
dataset_name <- "Abalone" # "Abalone" or "Magic"
if(dataset_name == "Abalone"){
  abalone <- read.table("data/abalone/abalone.data",
                        sep=",",
                        header=FALSE,
                        stringsAsFactors=FALSE)
  colnames(abalone) <- c("Sex",
                         "Length",
                         "Diameter",
                         "Height",
                         "Whole_weight",
                         "Shucked_weight",
                         "Viscera_weight",
                         "Shell_weight",
                         "Rings")
  # abalone_relevant_cols <- c("Length",
  #                            "Diameter",
  #                            "Height",
  #                            "Whole_weight",
  #                            "Shucked_weight",
  #                            "Viscera_weight",
  #                            "Shell_weight"
  #                            )
  # original_data <- abalone[, abalone_relevant_cols]
  original_data <- abalone[,2:(ncol(abalone)-1)] # First is categorical, last is integers
} else if(dataset_name =="Magic"){
  file_path <- "data/magicGammaTelescope/magic04.data"
  magic <- read.table(file_path,
                      sep = ",",
                      header = FALSE,
                      stringsAsFactors = FALSE)
  original_data <- magic[,1:(ncol(magic) - 1)]
}
head(original_data)
n_samples <- nrow(original_data)
n_vars <- ncol(original_data)
n_samples
n_vars
nu <- 5
train_perc <- 0.8
# count unique values
count_unique <- function(data) {
  unique_summary <- sapply(seq_len(ncol(data)), function(j) {
    n_unique <- length(unique(data[, j]))
    return(n_unique)
  })
  unique_df <- as.data.frame(as.list(unique_summary))
  colnames(unique_df) <- colnames(data)
  return(unique_df)
}

unique_df <- count_unique(original_data)
unique_df
# add jitter to values for Abalone "Height"
if (dataset_name == "Abalone") {
  cols_for_jittering <- c("Height")  # Abalone
  for(c in cols_for_jittering){
    original_data[[c]] <- kde1d::equi_jitter(ordered(original_data[[c]]))
  }
}
# Transform to copula scale with kernel
to_copula_scale <- function(data) {
  column_names <- colnames(data)
  n <- ncol(data)
  U <- matrix(NA, nrow = nrow(data), ncol = n)
  for (j in 1:n) {
    fit <- kde1d::kde1d(data[, j])
    U[, j] <- kde1d::pkde1d(data[, j], fit)
  }
  U <- as.data.frame(U)
  colnames(U) <- column_names
  return(U)
}

data_cop_scale <- to_copula_scale(original_data)
# pairs plot of original data
pair_plot_orig_object <- copula_pairs_ggplot(data=data_cop_scale[1:(min(10000,n_samples)),])
pair_plot_orig <- ggplotify::as.ggplot(pair_plot_orig_object)
pair_plot_orig
# Save the pair plots
# ggsave(
#   filename = paste0("PairPlotOrigApplicationExample",
#                     dataset_name,
#                     "20251003",
#                     ".png"),
#   plot = pair_plot_orig,
#   width = 15,
#   height = 15,
#   dpi = 100,
#   bg="white"
# )
# train test split of original data
train_test_split_orig_output <- train_test_split_orig(
  data= data_cop_scale,
  train_perc = train_perc
)
data_cop_scale_train <- train_test_split_orig_output[[1]]
data_cop_scale_test <- train_test_split_orig_output[[2]]
# fit a simplified vine
fitted_vine_app_ex <- vinecop(
  data_cop_scale_train,
  family_set="all",
  cores = num_cores
)
# save the copula model
#copula_path <- paste0("models/", dataset_name, "ExampleCopula_","20251003",".rds")
#saveRDS(fitted_vine_app_ex, file = copula_path)
#fitted_vine_app_ex <- readRDS(file=copula_path)
fitted_vine_app_ex$npars
print.data.frame(summary(fitted_vine_app_ex),digit=2)
fitted_vine_details_df <- summary(fitted_vine_app_ex)
# round parameters and tau to two decimal places
fitted_vine_details_df[c("parameters", "tau")] <- lapply(
  fitted_vine_details_df[c("parameters", "tau")],
  function(x) {
    if (is.list(x)) {
      lapply(x, function(y) {
        if(is.numeric(y)) {
          return(round(y,2))
        } else {
          return(y)
        }
      })
    } else if (is.numeric(x)){
      round(x, 2)
    } else {
      x
    }
  }
)
# round log-likelihood and degrees of freedom to one decimal place
fitted_vine_details_df[c("loglik", "df")] <- lapply(
  fitted_vine_details_df[c("loglik", "df")], function(x) round(x,1)
)
fitted_vine_details_df <- fitted_vine_details_df %>%
  select(-var_types)
fitted_vine_details_df[c("parameters", "tau", "loglik")]

# Sample from the simplified vine
simp_samples_app_ex <- rvinecop(n_samples * nu,
                         fitted_vine_app_ex)
# Plot simplified samples (only first 10000)
pair_plot_simplified_app_ex_object <- copula_pairs_ggplot(data=simp_samples_app_ex[1:(min(10000, nrow(simp_samples_app_ex))),])
pair_plot_simplified_app_ex <- ggplotify::as.ggplot(pair_plot_simplified_app_ex_object)
pair_plot_simplified_app_ex
# ggsave(
#   filename = paste0("PairPlotSimplifiedApplicationExample",
#                     dataset_name,
#                     "20251003",
#                     ".png"),
#   plot = pair_plot_simplified_app_ex,
#   width = 15,
#   height = 15,
#   dpi = 100,
#   bg="white"
# )
# Train test split
split_output <- train_test_split(
  orig_data_train = as.matrix(data_cop_scale_train),
  orig_data_test = as.matrix(data_cop_scale_test),
  simplified_data = as.matrix(simp_samples_app_ex)
)
x_train <- split_output[[1]]
x_test <- split_output[[2]]
y_train <- split_output[[3]]
y_test <- split_output[[4]]
# Train a classifier
model <- build_model(
  input_dim=n_vars,
  hidden_units=c(20,10), # 2 hidden layers
  initial_lr = 0.01,
  use_tanh=FALSE, # Use leaky_relu, not tanh
  leaky_relu_alpha=0.1)
train_model_output <- train_model(
  model=model,
  x_train=x_train,
  y_train=y_train,
  lr_schedule=lr_schedule_fun,
  num_epochs=200,
  verbose=1
)
model <- train_model_output[[1]]
history <- train_model_output[[2]]
# Monte Carlo integral, as a sanity check
# int_val <- compute_integral(model,fitted_vine_app_ex,n_samples=20000, nu=nu,data_dim_if_unif=n_vars)
# print(int_val)
# plot binary cross entropy loss and accuracy
plot(history, metrics = c("loss", "binary_accuracy_from_logits"))
# evaluate the model on the test set
loss_and_metrics <- model %>% evaluate(x_test, y_test)
loss_and_metrics_train <- model %>% evaluate(x_train, y_train)
print(loss_and_metrics_train[["loss"]])
print(loss_and_metrics_train[["binary_accuracy_from_logits"]])
print(paste0(
  "Base Accuracy for Prior: ",
  nrow(simp_samples_app_ex) / (nrow(simp_samples_app_ex) + nrow(data_cop_scale))
)
)
# save the model for reusing it later
# current_date <- "2025-10-03"
# # Construct the file name with the date
# model_file_name <- paste0("models/AppExample",
#                           dataset_name,
#                           "_NN_",
#                           current_date,
#                           ".keras")
# keras$Model$save(model, filepath=model_file_name)
# model <- load_model_hdf5(filepath=model_file_name)
# evaluate log likelihoods of original data (on train and test set fractions of orig_data)
log_lik_simp_train_orig_vec <- log(dvinecop(x_train[y_train==1,], fitted_vine_app_ex))
log_lik_simp_train_orig <- sum(log_lik_simp_train_orig_vec)
log_lik_NCE_train_orig_vec <- log(NCE_cop(
  model=model,
  fitted_vine=fitted_vine_app_ex,
  obs=x_train[y_train==1,],
  nu=nu))
log_lik_NCE_train_orig <- sum(log_lik_NCE_train_orig_vec)
log_lik_simp_test_orig_vec <- log(dvinecop(x_test[y_test==1,], fitted_vine_app_ex))
log_lik_simp_test_orig <- sum(log_lik_simp_test_orig_vec)
log_lik_NCE_test_orig_vec <- log(NCE_cop(
  model=model,
  fitted_vine=fitted_vine_app_ex,
  obs=x_test[y_test==1,],
  nu=nu))
log_lik_NCE_test_orig <- sum(log_lik_NCE_test_orig_vec)
log_lik_simp <- log_lik_simp_train_orig + log_lik_simp_test_orig
log_lik_NCE <- log_lik_NCE_train_orig + log_lik_NCE_test_orig
log_lik_simp / nrow(data_cop_scale)
log_lik_NCE / nrow(data_cop_scale)
marg_normal_summand <- sum(log(dnorm(qnorm(as.matrix(data_cop_scale)))), na.rm=TRUE)
marg_normal_log_lik_simp <- log_lik_simp + marg_normal_summand
marg_normal_log_lik_NCE <- log_lik_NCE + marg_normal_summand
# get number of model parameters and simplified vine parameters
NN_num_params <- count_NN_params(weights=model$weights)
simp_cop_num_params <- fitted_vine_app_ex$npars
# compute AIC and BIC for the NN and simplified model
AIC_NN <- 2*(NN_num_params + simp_cop_num_params) - 2* (log_lik_NCE_train_orig + log_lik_NCE_test_orig)
AIC_simp <- 2*simp_cop_num_params - 2*(log_lik_simp_train_orig + log_lik_simp_test_orig)
BIC_NN <- log(n_samples)*(NN_num_params + simp_cop_num_params) - 2* (log_lik_NCE_train_orig + log_lik_NCE_test_orig)
BIC_simp <- log(n_samples)*simp_cop_num_params - 2*(log_lik_simp_train_orig + log_lik_simp_test_orig)
# log correction factors
log_cor_facs <- log(correction_factors(model, obs=x_train[y_train==1,], nu=nu))
log_cor_facs_test <- log(correction_factors(
  model=model, obs=x_test[y_test==1,], nu=nu
))

loglik_AIC_df <- data.frame(
  rbind(
    cbind(
      "$c_{noise}$",
      round(log_lik_simp / n_samples,3),
      round(marg_normal_log_lik_simp / n_samples,3),
      round(AIC_simp,3),
      round(BIC_simp,3)),
    cbind(
      "$c_{model}$",
      round(log_lik_NCE / n_samples,3),
      round(marg_normal_log_lik_NCE / n_samples, 3),
      round(AIC_NN,3),
      round(BIC_NN,3))
  )
)
colnames(loglik_AIC_df) <- c("model", "loglik", "marg_normal_loglik", "AIC", "BIC")
loglik_AIC_df

# Visualizations
cor_facs <- correction_factors(model, obs=as.matrix(data_cop_scale), nu=nu)
# how many samples to remove
remove_top <- floor(0.02 * length(cor_facs))
# remove the highest observations, by first sorting
# and then removing the last remove_top observations
cor_facs_no_outliers <- (sort(cor_facs)[1:(length(cor_facs) - remove_top)])
# depending on the dataset it can make sense to use cor_facs_no_outliers in the following plot
cor_facs_hist_KDE_plot <- ggplot(data.frame(x = cor_facs_no_outliers), aes(x = x)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.1, fill = "lightblue", color = "black", alpha = 0.5) +
  geom_density(color = "darkblue", linewidth = 1) +
  geom_vline(xintercept=1, color="red") +
  labs(title = "Histogram and KDE of the Correction Factors",
       x = "Correction Factors", y = "Density") +
  theme_minimal()
cor_facs_hist_KDE_plot
log_cor_facs_hist_KDE_plot <- ggplot(data.frame(
  x = log(cor_facs)) , aes(x = x)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.1, fill = "lightblue", color = "black", alpha = 0.5) +
  geom_density(color = "darkblue", linewidth = 1) +
  geom_vline(xintercept=0, color="red") +
  labs(title = "Histogram and KDE of the Log Correction Factors",
       x = "Log Correction Factors",
       y= "Density") +
  theme_minimal()
log_cor_facs_hist_KDE_plot

# Noise Likelihoods
noise_likelihoods <- dvinecop(as.matrix(data_cop_scale), fitted_vine_app_ex)
noise_log_likelihoods <- log(noise_likelihoods)
log_cor_facs <- log(cor_facs)
df_temp <- data.frame(noise_log_lik = noise_log_likelihoods, log_cor_factors = log_cor_facs)

hex_plot_log_noise_log_cor_facs <- ggplot(df_temp, aes(x = noise_log_lik, y = log_cor_factors)) +
  geom_hex(binwidth = c(1, 0.25)) +
  geom_abline(slope=0, intercept=0, col="red") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title= "Log-likelihood c_noise vs Log Correction Factors",
    x = "Log-likelihood c_noise",
    y = "Log Correction Factors",
    fill = "Count"
  ) +
  theme_minimal()
hex_plot_log_noise_log_cor_facs

model_likelihoods <- NCE_cop(model=model,
                             fitted_vine=fitted_vine_app_ex,
                             obs=as.matrix(data_cop_scale),
                             nu=nu)
model_log_likelihoods <- log(model_likelihoods)
df_temp <- data.frame(noise_log_lik = noise_log_likelihoods,
                      model_log_lik = model_log_likelihoods)
noise_vs_model_log_lik_plot <- ggplot(df_temp, aes(x = noise_log_lik, y = model_log_lik)) +
  geom_hex(binwidth = c(0.5, 0.5)) +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  geom_abline(intercept = 0, slope = 1, color = "red") +
  labs(
    title= "Log-likelihood c_noise vs c_model",
    x = "Log-likelihood c_noise",
    y = "Log-likelihood c_model",
    fill = "Count"
  ) +
  theme_minimal()
noise_vs_model_log_lik_plot

# Distance from Center against model and noise log-likelihoods
orig_data_dist_center <- sqrt(rowSums((as.matrix(data_cop_scale)-0.5)^2))
log_cor_facs <- log(cor_facs)
dist_cor_fac_df <- data.frame(dist = orig_data_dist_center,
                              log_dens_diff = log_cor_facs)
dist_cor_fac_ordered_df <- dist_cor_fac_df[order(dist_cor_fac_df$dist),]
quantile_distances <- 0.05 # 0.05 quantiles
rows_per_block <- nrow(dist_cor_fac_df) * quantile_distances
quantile_breaks <- quantile(dist_cor_fac_ordered_df$dist, probs=seq(0,1,by=quantile_distances))
dist_cor_fac_ordered_df$block <- cut(
  dist_cor_fac_ordered_df$dist,
  breaks=quantile_breaks,
  labels=FALSE,
  include.lowest=TRUE
)
dist_cor_fac_ordered_df$quantile_level <- rep(
  quantile_distances * 1:(1/quantile_distances),
  each=rows_per_block,
  length.out=nrow(dist_cor_fac_ordered_df)
)
results_by_block <- dist_cor_fac_ordered_df %>%
  group_by(quantile_level) %>%
  summarise(
    total_entries = n(),
    NCE_better = sum(log_dens_diff > 0),
    fraction_NCE_better = NCE_better / total_entries * 100
  )
#print(results_by_block)
emp_dist_quantile_NCE_better_plot <- ggplot(
  data=results_by_block, aes(x=quantile_level, y=fraction_NCE_better)) +
  geom_point(size=1.5, color ="blue") +
  geom_line(linewidth=0.7, color="darkblue") +
  geom_abline(slope=0, intercept=50, color="red", linewidth=0.5) +
  theme_minimal() +
  labs(x="Empirical Quantile of Distances to Center",
       y="% of Observations",
       title="% of Observations for which c_model is better") +
  scale_y_continuous(
    limits = c(0, 100), # Set the lower and upper limits
    breaks = seq(0, 100, by = 10) # Add ticks at every 0.1
  ) +
  scale_x_continuous(
    limits=c(0.0,1.0),
    breaks = seq(0.0,1.0, by=0.1)
  )
emp_dist_quantile_NCE_better_plot
df_temp <- data.frame(
  u_dist_center = orig_data_dist_center,
  noise_log_lik = noise_log_likelihoods,
  model_log_lik = model_log_likelihoods
)
# Reshape to long format
df_temp_long <- df_temp %>%
  tidyr::pivot_longer(
    cols = c(noise_log_lik, model_log_lik),
    names_to = "variable",
    values_to = "value"
  )

# Plot
dist_center_log_lik_plot <- ggplot(df_temp_long, aes(x = u_dist_center, y = value, color = variable, fill = variable)) +
  geom_point(size = 1, show.legend = TRUE, alpha=0.2, shape=46) +
  geom_smooth(method = "loess", alpha = 0.2) +
  scale_color_manual(values = c(noise_log_lik = "orange", model_log_lik = "blue"),
                     labels = c(noise_log_lik = "Log-likelihood c_noise", model_log_lik="Log-likelihood c_model")) +
  scale_fill_manual(values = c(noise_log_lik = "orange", model_log_lik = "blue"),
                    labels = c(noise_log_lik = "Log-likelihood c_noise", model_log_lik="Log-likelihood c_model")) +
  theme_minimal() +
  labs(x = "Distance from Center", y = "value", title = "Log-likelihood c_model and c_noise")
dist_center_log_lik_plot

# Using the Paper of Huk: c/(1+c) = P[u_i is from c versus independent]
simp_vs_ind <- noise_likelihoods / (1+noise_likelihoods)
model_vs_ind <- model_likelihoods / (1+model_likelihoods)
df_temp <- data.frame(simplified_vs_indep = simp_vs_ind, model_vs_indep = model_vs_ind)
simp_ind_vs_model_ind <- ggplot(df_temp, aes(x = simplified_vs_indep, y = model_vs_indep)) +
  geom_hex(binwidth = c(0.02, 0.02)) +
  geom_abline(slope=1, intercept=0, col="red") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title= "Plot of c/(1+c) for c_noise and c_model",
    x = "c_noise versus independent",
    y = "c_model versus independent",
    fill = "Count"
  ) +
  theme_minimal()
simp_ind_vs_model_ind
simp_ind_vs_model_ind_zoom_topright <- ggplot(df_temp, aes(x = simplified_vs_indep, y = model_vs_indep)) +
  geom_hex(binwidth = c(0.005, 0.005)) +
  geom_abline(slope=1, intercept=0, col="red") +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title= "Plot of c/(1+c) for c_noise and c_model",
    x = "c_noise versus independent",
    y = "c_model versus independent",
    fill = "Count"
  ) +
  theme_minimal() +
  scale_y_continuous(
    limits=c(0.8,1.0)
  ) +
  scale_x_continuous(
    limits=c(0.8,1.0)
  )
simp_ind_vs_model_ind_zoom_topright
all_plots_in_one_except_bayes_classifier <- (cor_facs_hist_KDE_plot | log_cor_facs_hist_KDE_plot) /
  (hex_plot_log_noise_log_cor_facs + noise_vs_model_log_lik_plot) /
  (emp_dist_quantile_NCE_better_plot|dist_center_log_lik_plot)
all_plots_in_one_except_bayes_classifier
# ggsave(
#   filename = paste0("ApplicationExample",
#                     dataset_name,
#                     "AllExamplePlotsInOne.png"),
#   plot = all_plots_in_one_except_bayes_classifier,
#   width = 12,
#   height = 15,
#   dpi = 300,
#   bg="white"
# )
# bayes_classifier_plots <- (simp_ind_vs_model_ind | simp_ind_vs_model_ind_zoom_topright)
# bayes_classifier_plots
# ggsave(
#   filename = paste0("ApplicationExample",
#                     dataset_name,
#                     "BayesClassifierPlots.png"),
#   plot = bayes_classifier_plots,
#   width = 10,
#   height = 5,
#   dpi = 300,
#   bg="white"
# )

