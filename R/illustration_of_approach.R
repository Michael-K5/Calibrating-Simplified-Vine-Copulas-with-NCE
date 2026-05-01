# Fit a simplified vine copula and train a classifier on the data, to distinguish between
# Non-Simplified and Simplified Vine Copula data.
library(rvinecopulib)
library(keras)
library(tensorflow)
# source("R/classifier_methods.R")
# source("R/plotting_methods.R")
library(vineNCE)
# Parameters to determine, which data to load.
data_simulation_date <- "2025-06-25"
data_dim <- 5
# fraction of noise to true samples (to determine how many noise samples to create)
nu <- 1
# load data
csv_filename <- paste0("data/non_simplified_sim_",data_dim,"d_",data_simulation_date,".csv")
orig_data <- as.matrix(read.csv(csv_filename))
orig_data <- unname(orig_data) #remove col- and rownames
num_rows <- nrow(orig_data)
non_simplified_pairs_plot <- copula_pairs_ggplot(orig_data[(1:min(10000, num_rows)),])
non_simplified_pairs_plot <- ggplotify::as.ggplot(non_simplified_pairs_plot)
non_simplified_pairs_plot

# fit a simplified vine
orig_data_split <- train_test_split_orig(orig_data, train_perc=0.8)
orig_data_train <- orig_data_split[[1]]
orig_data_test <- orig_data_split[[2]]
fitted_vine <-vinecop(orig_data_train,family_set="parametric", cores = 4)
print.data.frame(summary(fitted_vine),digit=2)
fitted_vine_details_df <- summary(fitted_vine)
fitted_vine_details_df
# readr::write_csv(fitted_vine_details_df,
#           file="simplified_vine_20250625_details.csv")
# simulate from the simplified vine, to train a classifier later.
num_samples <- num_rows * nu
simplified_samples <- rvinecop(num_samples, fitted_vine)
# plot at most first 10 k for faster rendering
#pairs_copula_data(simplified_samples[(1:min(10000, nrow(num_samples))),])
simplified_pairs_plot <- copula_pairs_ggplot(simplified_samples[(1:min(10000, nrow(num_samples))),])
simplified_pairs_plot <- ggplotify::as.ggplot(simplified_pairs_plot)
simplified_pairs_plot
# ggplot2::ggsave(
#   filename = "simplified_samples_5d_20250625_plot.png",
#   plot = simplified_pairs_plot,
#   width = 6,
#   height = 6,
#   dpi = 300,
#   bg="white"
# )
split_output <- train_test_split(
  orig_data_train=orig_data_train,
  orig_data_test=orig_data_test,
  simplified_data =simplified_samples)
x_train <- split_output[[1]]
x_test <- split_output[[2]]
y_train <- split_output[[3]]
y_test <- split_output[[4]]

# nrow(x_train[y_train==1,])
# nrow(x_test[y_test==1,])
# nrow(x_train)
model <- build_model(
    input_dim=5,
    hidden_units=c(32,16), # 2 hidden layers with 20 and 10 units respectively.
    initial_lr = 0.01,
    use_tanh=FALSE, # Use leaky_relu, not tanh
    leaky_relu_alpha=0.1)
train_model_output <- train_model(
  model=model,
  x_train=x_train,
  y_train=y_train,
  lr_schedule=lr_schedule_fun,
  num_epochs=200
)
model <- train_model_output[[1]]
history <- train_model_output[[2]]

# training and validation loss and accuracy
plot(history, metrics = c("loss", "binary_accuracy_from_logits"))
# evaluate the model on the test set
loss_and_metrics <- model %>% evaluate(x_test, y_test)
print(loss_and_metrics[["loss"]])
print(loss_and_metrics[["binary_accuracy_from_logits"]])
loss_and_metrics_train <- model %>% evaluate(x_train, y_train)
print(loss_and_metrics_train[["loss"]])
print(loss_and_metrics_train[["binary_accuracy_from_logits"]])
print(paste0(
  "Base Accuracy for Prior: ",
  nrow(simplified_samples) / (nrow(simplified_samples) + nrow(orig_data))
  )
)
int_val <- compute_integral(model, fitted_vine, n_samples=20000, nu=nu,data_dim_if_unif=ncol(orig_data),user_info=TRUE)
int_val
# save the model for reusing it later
# # Get the current date in YYYY-MM-DD format
# current_date <- Sys.Date()
# # Construct the file name with the date
# model_file_name <- paste0("models/NN_", ncol(x_train), "d_",current_date, ".keras")
# keras$Model$save(model, filepath=model_file_name)

log_lik_simp_train_orig <- sum(log(dvinecop(x_train[y_train==1,], fitted_vine)))
log_lik_NN_train_orig <- sum(log(NCE_cop(
  model=model,
  fitted_vine=fitted_vine,
  obs=x_train[y_train==1,],
  nu=nu)))
log_lik_simp_test_orig <- sum(log(dvinecop(x_test[y_test==1,], fitted_vine)))
log_lik_NN_test_orig <- sum(log(NCE_cop(
  model=model,
  fitted_vine=fitted_vine,
  obs=x_test[y_test==1,],
  nu=nu)))
NN_num_params <- count_NN_params(weights=model$weights)
simp_cop_num_params <- fitted_vine$npars
# Compute AIC and BIC
AIC_NN <- 2*(NN_num_params+simp_cop_num_params) - 2* (log_lik_NN_train_orig + log_lik_NN_test_orig)
BIC_NN <- log(nrow(orig_data))*(NN_num_params+ simp_cop_num_params) - 2* (log_lik_NN_train_orig + log_lik_NN_test_orig)
AIC_simp <- 2*simp_cop_num_params - 2*(log_lik_simp_train_orig + log_lik_simp_test_orig)
BIC_simp <- log(nrow(orig_data))*simp_cop_num_params - 2*(log_lik_simp_train_orig + log_lik_simp_test_orig)
eval_data_frame <- data.frame(
  MC_Integral = int_val,
  train_accuracy= round(loss_and_metrics_train[["binary_accuracy_from_logits"]],4),
  train_loss = round(loss_and_metrics_train[["loss"]],4),
  test_accuracy = round(loss_and_metrics[["binary_accuracy_from_logits"]],4),
  test_loss = round(loss_and_metrics[["loss"]],4),
  log_lik_simplified_train = round(log_lik_simp_train_orig,2),
  log_lik_NN_train = round(log_lik_NN_train_orig,2),
  log_lik_simplified_test = round(log_lik_simp_test_orig,2),
  log_lik_NN_test = round(log_lik_NN_test_orig,2),
  AIC_NN = AIC_NN,
  BIC_NN=BIC_NN,
  AIC_simp = AIC_simp,
  BIC_simp = BIC_simp,
  NN_num_params = NN_num_params,
  simp_cop_num_params = simp_cop_num_params
)
print(AIC_NN)
print(AIC_simp)
print(BIC_NN)
print(BIC_simp)
print(log_lik_NN_train_orig)
print(log_lik_simp_train_orig)
print(log_lik_NN_test_orig)
print(log_lik_simp_test_orig)

library(ggplot2)
library(dplyr)
library(tibble)
library(dplyr)
library(ggplot2)
library(tidyr)
library(patchwork)

cor_facs <- correction_factors(model, obs=orig_data, nu=nu)
# how many samples to remove
remove_top <- floor(0.02 * length(cor_facs))
# remove the highest observations, by first sorting
# and then removing the last remove_top observations
cor_facs_no_outliers <- (sort(cor_facs)[1:(length(cor_facs) - remove_top)])
cor_facs_hist_KDE_plot <- ggplot(data.frame(x = cor_facs_no_outliers), aes(x = x)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 1, fill = "lightblue", color = "black", alpha = 0.5) +
  geom_density(color = "darkblue", linewidth = 1) +
  geom_vline(xintercept=1, color="red") +
  labs(title = "Histogram and KDE of the Correction Factors",
       x = "Correction Factors", y = "Density") +
  theme_minimal()
cor_facs_hist_KDE_plot
# ggsave(
#   filename = "CorrectionFactorsHistKDE20250625.png",
#   plot = cor_facs_hist_KDE_plot,
#   width = 12,
#   height = 8,
#   dpi = 300,
#   bg="white"
# )
log_cor_facs_hist_KDE_plot <- ggplot(data.frame(
  x = log(cor_facs)) , aes(x = x)) +
  geom_histogram(aes(y = after_stat(density)), binwidth = 0.2, fill = "lightblue", color = "black", alpha = 0.5) +
  geom_density(color = "darkblue", linewidth = 1) +
  geom_vline(xintercept=0, color="red") +
  labs(title = "Histogram and KDE of the Log Correction Factors",
       x = "Log Correction Factors",
       y= "Density") +
  theme_minimal()
log_cor_facs_hist_KDE_plot

# Noise Likelihoods
noise_likelihoods <- dvinecop(orig_data, fitted_vine)
noise_log_likelihoods <- log(noise_likelihoods)
log_cor_facs <- log(cor_facs)
df_temp <- data.frame(log_noise_dens = noise_log_likelihoods, log_cor_factors = log_cor_facs)

hex_plot_log_noise_log_cor_facs <- ggplot(df_temp, aes(x = log_noise_dens, y = log_cor_factors)) +
  geom_hex(binwidth = c(0.5, 0.5)) +
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
# ggsave(
#   filename = "LogNoiseDensVsLogCorFacs20250702.png",
#   plot = hex_plot_log_noise_log_cor_facs,
#   width = 8,
#   height = 8,
#   dpi = 300,
#   bg="white"
# )

model_likelihoods <- NCE_cop(model=model, fitted_vine=fitted_vine, obs=orig_data, nu=nu)
model_log_likelihoods <- log(model_likelihoods)
df_temp <- data.frame(log_noise_dens = noise_log_likelihoods, log_model_dens = model_log_likelihoods)
noise_vs_model_log_lik_plot <- ggplot(df_temp, aes(x = log_noise_dens, y = log_model_dens)) +
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
# ggsave(
#   filename = "LogNoiseVsLogModelDens20250702.png",
#   plot = noise_vs_model_log_lik_plot,
#   width = 10,
#   height = 10,
#   dpi = 300,
#   bg="white"
# )

# Distance from center against model and noise log likelihoods
orig_data_dist_center <- sqrt(rowSums((orig_data-0.5)^2))
log_cor_facs <- log(cor_facs)
print(log_cor_facs[1:5])
print(model_log_likelihoods[1:5] - noise_log_likelihoods[1:5])
dist_cor_fac_df <- data.frame(dist = orig_data_dist_center,
                              log_dens_diff = log_cor_facs)
dist_cor_fac_ordered_df <- dist_cor_fac_df[order(dist_cor_fac_df$dist),]
# head(dist_cor_fac_df)
# head(dist_cor_fac_ordered_df)
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
print(results_by_block)
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
    limits = c(45, 100), # Set the lower and upper limits
    breaks = seq(50, 100, by = 10) # Add ticks at every 0.1
  ) +
  scale_x_continuous(
    limits=c(0.0,1.0),
    breaks = seq(0.0,1.0, by=0.1)
  )
emp_dist_quantile_NCE_better_plot
# ggsave(filename="EmpQuantileDistCenterVsFractionNCEBetter20250721.png",
#        plot = emp_dist_quantile_NCE_better_plot,
#        width=10,
#        height=10,
#        dpi=300,
#        bg="white")
#
# df_temp <- data.frame(
#   u_dist_center = orig_data_dist_center,
#   log_noise_dens = noise_log_likelihoods,
#   log_model_dens = model_log_likelihoods
# )
# # Reshape to long format
# df_temp_long <- df_temp %>%
#   tidyr::pivot_longer(cols = c(log_noise_dens, log_model_dens), names_to = "variable", values_to = "value")

# Plot
# dist_center_log_lik_plot <- ggplot(df_temp_long, aes(x = u_dist_center, y = value, color = variable, fill = variable)) +
#   geom_point(size = 1, show.legend = TRUE, alpha=0.2, shape=46) +
#   geom_smooth(method = "loess", alpha = 0.2) +
#   scale_color_manual(values = c(log_noise_dens = "orange", log_model_dens = "blue"),
#                      labels = c(log_noise_dens = "Log-likelihood c_noise", log_model_dens="Log-likelihood c_model")) +
#   scale_fill_manual(values = c(log_noise_dens = "orange", log_model_dens = "blue"),
#                     labels = c(log_noise_dens = "Log-likelihood c_noise", log_model_dens="Log-likelihood c_model")) +
#   theme_minimal() +
#   labs(x = "Distance from Center", y = "value", title = "Log-likelihood c_model and c_noise")
# dist_center_log_lik_plot

# new plot including true loglik
tau_upper = 0.92
tau_lower=-0.92
struct_mat <- matrix(c(2,3,2,1,1,
                       3,2,1,2,0,
                       1,1,3,0,0,
                       4,4,0,0,0,
                       5,0,0,0,0), ncol=5, byrow=TRUE)
family_test <- list(list("frank", "clayton","gaussian","frank"),
                    list("frank","gaussian","joe"),
                    list("gaussian", "gumbel"),
                    list("gaussian"))
params_test <- list(c(ktau_to_par(family=family_test[[1]][[1]], tau=-0.2)),
                    c(ktau_to_par(family=family_test[[1]][[2]], tau=0.3)),
                    c(ktau_to_par(family=family_test[[1]][[3]], tau=-0.1)),
                    c(ktau_to_par(family=family_test[[1]][[4]], tau=0.1)))
param_cond_funcs_test <- list(
  list(u_to_param_linear(c(1),
                         tau_lower=tau_lower,
                         tau_upper=tau_upper),
       u_to_param_linear(c(1),
                         tau_lower=tau_lower,
                         tau_upper=tau_upper),
       u_to_param_linear(c(1),
                         tau_lower=tau_lower,
                         tau_upper=tau_upper)),
  list(u_to_param_linear(c(0.7,0.3),
                         tau_lower=tau_lower,
                         tau_upper=tau_upper),
       u_to_param_linear(c(0.4,0.6),
                         tau_lower=tau_lower,
                         tau_upper=tau_upper)),
  list(u_to_param_linear(c(0.2,0.5,0.3),
                         tau_lower=tau_lower,
                         tau_upper=tau_upper)))
true_log_lik <- log_likelihood_non_simplified(
  u_data =orig_data,
  struct = struct_mat,
  families=family_test,
  params = params_test,
  param_cond_funcs = param_cond_funcs_test,
  rotations = list(list(0,0,0,0),list(0,0,0), list(0,0), list(0)),
  return_vector=TRUE
)
df_temp <- data.frame(
  u_dist_center = orig_data_dist_center,
  log_noise_dens = noise_log_likelihoods,
  log_model_dens = model_log_likelihoods,
  loglik_true = pmax(-5, true_log_lik)
)
# Reshape to long format
df_temp_long <- df_temp %>%
  tidyr::pivot_longer(cols = c(log_noise_dens, log_model_dens, loglik_true), names_to = "variable", values_to = "value")

# Plot
dist_center_log_lik_plot <- ggplot(df_temp_long, aes(x = u_dist_center, y = value, color = variable, fill = variable)) +
  geom_point(size = 1, show.legend = TRUE, alpha=0.2, shape=46) +
  geom_smooth(method = "loess", alpha = 0.2) +
  scale_color_manual(values = c(log_noise_dens = "orange", log_model_dens = "blue", loglik_true="black"),
                     labels = c(log_noise_dens = "Log-likelihood c_noise",
                                log_model_dens="Log-likelihood c_model",
                                loglik_true="Log-likelihood c_true")) +
  scale_fill_manual(values = c(log_noise_dens = "orange", log_model_dens = "blue", loglik_true="black"),
                    labels = c(log_noise_dens = "Log-likelihood c_noise",
                               log_model_dens="Log-likelihood c_model",
                               loglik_true="Log-likelihood c_true")) +
  theme_minimal() +
  labs(x = "Distance from Center", y = "value", title = "Log-likelihood c_model, c_noise and c_true")
dist_center_log_lik_plot
# ggsave(
#   filename = "DistFromCenterVsLogDensitiesPlot20250702.png",
#   plot = dist_center_log_lik_plot,
#   width = 10,
#   height = 8,
#   dpi = 300,
#   bg="white"
# )

# Bayes Classifier Plots
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
# ggsave(
#   filename = "HukSimpIndVsModelInd20250721.png",
#   plot = simp_ind_vs_model_ind,
#   width = 10,
#   height = 10,
#   dpi = 300,
#   bg="white"
# )
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
#   filename = "AllExamplePlotsInOne20250916.png",
#   plot = all_plots_in_one_except_bayes_classifier,
#   width = 12,
#   height = 15,
#   dpi = 300,
#   bg="white"
# )
# bayes_classifier_plots <- (simp_ind_vs_model_ind | simp_ind_vs_model_ind_zoom_topright)
# bayes_classifier_plots
# ggsave(
#   filename = "BayesClassifierPlots20250916.png",
#   plot = bayes_classifier_plots,
#   width = 10,
#   height = 5,
#   dpi = 300,
#   bg="white"
# )
