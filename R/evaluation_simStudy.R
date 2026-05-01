# Script for evaluating the "simple tests"
library(dplyr)
library(readr)
library(tidyr)
library(ggplot2)
library(patchwork)
# Create a vector of file names
n_samples <- 10000
noise_method <- "all"
all_filenames <- paste0("data/simulation_study/", noise_method, "20250808NSamples", n_samples, "Repl", 1:10, ".csv")
print(all_filenames)

# Read and combine all CSVs into one data frame
all_results <- bind_rows(lapply(all_filenames, read_csv))
# For the paper: Filter for only dimension 5
all_results <- all_results %>% filter(dim==5)
head(all_results[,c("log_lik_true_train",
                    "log_lik_true_test",
                    "log_lik_simp_train_orig",
                    "log_lik_simp_test_orig",
                    "log_lik_NN_train_orig",
                    "log_lik_NN_test_orig")])
tests_summary_df <- all_results %>%
  group_by(num_samples, dim, tau_max, param_cond_func) %>%
  summarise(
    log_lik_simp_train_orig_avg = mean(log_lik_simp_train_orig, na.rm=TRUE),
    log_lik_simp_train_orig_stderr = sd(log_lik_simp_train_orig, na.rm=TRUE) / sqrt(n()),
    log_lik_NN_train_orig_avg = mean(log_lik_NN_train_orig, na.rm=TRUE),
    log_lik_NN_train_orig_stderr = sd(log_lik_NN_train_orig, na.rm=TRUE) / sqrt(n()),
    log_lik_true_train_avg = mean(log_lik_true_train, na.rm=TRUE),
    log_lik_true_train_stderr = sd(log_lik_true_train, na.rm=TRUE) / sqrt(n()),
    log_lik_simp_test_orig_avg = mean(log_lik_simp_test_orig, na.rm=TRUE),
    log_lik_simp_test_orig_stderr = sd(log_lik_simp_test_orig, na.rm=TRUE) / sqrt(n()),
    log_lik_NN_test_orig_avg = mean(log_lik_NN_test_orig, na.rm=TRUE),
    log_lik_NN_test_orig_stderr = sd(log_lik_NN_test_orig, na.rm=TRUE) / sqrt(n()),
    log_lik_true_test_avg = mean(log_lik_true_test, na.rm=TRUE),
    log_lik_true_test_stderr = sd(log_lik_true_test, na.rm=TRUE) / sqrt(n()),
    AIC_NN_avg = mean(AIC_NN, na.rm=TRUE),
    AIC_NN_stderr = sd(AIC_NN, na.rm=TRUE) / sqrt(n()),
    AIC_simp_avg = mean(AIC_simp, na.rm=TRUE),
    AIC_simp_stderr = sd(AIC_simp, na.rm=TRUE) / sqrt(n()),
    BIC_NN_avg = mean(BIC_NN, na.rm=TRUE),
    BIC_NN_stderr = sd(BIC_NN, na.rm=TRUE) / sqrt(n()),
    BIC_simp_avg = mean(BIC_simp, na.rm=TRUE),
    BIC_simp_stderr = sd(BIC_NN, na.rm=TRUE) / sqrt(n()),
    #The num_params fields should be constant, so mean or median or max or min should all yield the same
    NN_num_params_avg = mean(NN_num_params, na.rm=TRUE),
    simp_cop_num_params = mean(NN_num_params, na.rm=TRUE),
    .groups = 'drop'  # ungroup after summarising
  )

head(tests_summary_df)
# likelihoods
likelihoods_df <- tests_summary_df %>%
  select("num_samples", "dim", "tau_max", "param_cond_func",
                                      "log_lik_simp_train_orig_avg",
                                      "log_lik_simp_train_orig_stderr",
                                      "log_lik_NN_train_orig_avg",
                                      "log_lik_NN_train_orig_stderr",
                                      "log_lik_true_train_avg",
                                      "log_lik_true_train_stderr",
                                      "log_lik_simp_test_orig_avg",
                                      "log_lik_simp_test_orig_stderr" ,
                                      "log_lik_NN_test_orig_avg",
                                      "log_lik_NN_test_orig_stderr",
                                      "log_lik_true_test_avg",
                                      "log_lik_true_test_stderr")
val_cols <- names(likelihoods_df)[grepl("(_avg|_stderr)$", names(likelihoods_df))]
likelihoods_df[val_cols] <- lapply(likelihoods_df[val_cols], function(x) {
  round(x,1)
})

likelihoods_summarized_df <- all_results %>%
  group_by(num_samples, dim, tau_max, param_cond_func) %>%
  summarise(
    log_lik_simp_avg = mean(log_lik_simp_train_orig + log_lik_simp_test_orig, na.rm=TRUE),
    log_lik_simp_stderr = sd(log_lik_simp_train_orig + log_lik_simp_test_orig, na.rm=TRUE) / sqrt(n()),
    log_lik_NN_avg = mean(log_lik_NN_train_orig + log_lik_NN_test_orig, na.rm=TRUE),
    log_lik_NN_stderr = sd(log_lik_NN_train_orig + log_lik_NN_test_orig, na.rm=TRUE) / sqrt(n()),
    log_lik_true_avg = mean(log_lik_true_train + log_lik_true_test, na.rm=TRUE),
    log_lik_true_stderr = sd(log_lik_true_train + log_lik_true_test, na.rm=TRUE) / sqrt(n()),
    .groups = 'drop'  # ungroup after summarising
  )
val_cols <- names(likelihoods_summarized_df)[grepl("(_avg|_stderr)$", names(likelihoods_summarized_df))]
likelihoods_summarized_df[val_cols] <- lapply(likelihoods_summarized_df[val_cols], function(x) {
  round(x,1)
})
# Likelihoods Mean Squared Error
log_likelihoods_MSE_df <- all_results %>%
  group_by(num_samples, dim, tau_max, param_cond_func) %>%
  summarise(
    MSE_simp_true_avg = mean(0.8 * MSE_loglik_simp_true_train + 0.2 * MSE_loglik_simp_true_test, na.rm=TRUE),
    MSE_simp_true_stderr = sd(0.8 * MSE_loglik_simp_true_train + 0.2 * MSE_loglik_simp_true_test, na.rm=TRUE) / sqrt(n()),
    MSE_NN_true_avg = mean(0.8 * MSE_loglik_NN_true_train + 0.2 * MSE_loglik_NN_true_test, na.rm=TRUE),
    MSE_NN_true_stderr = sd(0.8 * MSE_loglik_NN_true_train + 0.2 * MSE_loglik_NN_true_test, na.rm=TRUE) / sqrt(n()),
    .groups = 'drop'
  )
val_cols <- names(log_likelihoods_MSE_df)[grepl("(_avg|_stderr)$", names(log_likelihoods_MSE_df))]
log_likelihoods_MSE_df[val_cols] <- lapply(log_likelihoods_MSE_df[val_cols], function(x) {
  round(x,3)
})
# Summarize into one table:
likelihoods_all_df <- likelihoods_summarized_df %>%
  full_join(
    log_likelihoods_MSE_df,
    by=c("num_samples", "dim", "tau_max", "param_cond_func")
  ) %>%
  select(-num_samples)


# likelihoods per sample
likelihoods_per_sample_summarized_df <- all_results %>%
  group_by(num_samples, dim, tau_max, param_cond_func) %>%
  summarise(
    log_lik_simp_avg = mean((log_lik_simp_train_orig + log_lik_simp_test_orig) / n_samples, na.rm=TRUE),
    log_lik_simp_stderr = sd((log_lik_simp_train_orig + log_lik_simp_test_orig)/n_samples, na.rm=TRUE) / sqrt(n()),
    log_lik_NN_avg = mean((log_lik_NN_train_orig + log_lik_NN_test_orig)/n_samples, na.rm=TRUE),
    log_lik_NN_stderr = sd((log_lik_NN_train_orig + log_lik_NN_test_orig)/n_samples, na.rm=TRUE) / sqrt(n()),
    log_lik_true_avg = mean((log_lik_true_train + log_lik_true_test)/n_samples, na.rm=TRUE),
    log_lik_true_stderr = sd((log_lik_true_train + log_lik_true_test)/n_samples, na.rm=TRUE) / sqrt(n()),
    .groups = 'drop'
  )
val_cols <- names(likelihoods_per_sample_summarized_df)[grepl("(_avg|_stderr)$",
                                                              names(likelihoods_per_sample_summarized_df))]
likelihoods_per_sample_summarized_df[val_cols] <- lapply(likelihoods_per_sample_summarized_df[val_cols], function(x) {
  round(x,3)
})
head(likelihoods_per_sample_summarized_df)
log_likelihoods_MSE_df <- all_results %>%
  group_by(num_samples, dim, tau_max, param_cond_func) %>%
  summarise(
    MSE_simp_true_avg = mean(0.8 * MSE_loglik_simp_true_train + 0.2 * MSE_loglik_simp_true_test, na.rm=TRUE),
    MSE_simp_true_stderr = sd(0.8 * MSE_loglik_simp_true_train + 0.2 * MSE_loglik_simp_true_test, na.rm=TRUE) / sqrt(n()),
    MSE_NN_true_avg = mean(0.8 * MSE_loglik_NN_true_train + 0.2 * MSE_loglik_NN_true_test, na.rm=TRUE),
    MSE_NN_true_stderr = sd(0.8 * MSE_loglik_NN_true_train + 0.2 * MSE_loglik_NN_true_test, na.rm=TRUE) / sqrt(n()),
    .groups = 'drop'
  )
val_cols <- names(log_likelihoods_MSE_df)[grepl("(_avg|_stderr)$", names(log_likelihoods_MSE_df))]
log_likelihoods_MSE_df[val_cols] <- lapply(log_likelihoods_MSE_df[val_cols], function(x) {
  round(x,3)
})
# Summarize into one table
likelihoods_per_sample_all_df <- likelihoods_per_sample_summarized_df %>%
  full_join(
    log_likelihoods_MSE_df,
    by=c("num_samples", "dim", "tau_max", "param_cond_func")
  ) %>%
  select(-num_samples)

# AIC, BIC
AIC_BIC_summary_df <- tests_summary_df %>%
  select("num_samples","dim", "tau_max", "param_cond_func",
         "AIC_simp_avg", "AIC_simp_stderr",
         "AIC_NN_avg", "AIC_NN_stderr",
         "BIC_simp_avg", "BIC_simp_stderr",
         "BIC_NN_avg","BIC_NN_stderr"
)
val_cols <- names(AIC_BIC_summary_df)[grepl("(_avg|_stderr)$", names(AIC_BIC_summary_df))]
AIC_BIC_summary_df[val_cols] <- lapply(AIC_BIC_summary_df[val_cols], function(x) {
  round(x,1)
})
# kableExtra::kbl(AIC_BIC_summary_df, format = "latex", booktabs = TRUE, align = "c") %>%
#   kableExtra::add_header_above(c(" " = 4, "AIC simp" = 2, "AIC NN" = 2, "BIC simp" = 2, "BIC NN"=2)) %>%
#   kableExtra::kable_styling(latex_options = c("hold_position"))

# Boxplots
plot_likelihoods_df <- all_results %>%
  select("num_samples", "dim", "tau_max","param_cond_func", "log_lik_true_test", "log_lik_true_train",
         "log_lik_NN_train_orig", "log_lik_NN_test_orig", "log_lik_simp_train_orig", "log_lik_simp_test_orig")
plot_likelihoods_df$log_lik_true <- (plot_likelihoods_df$log_lik_true_train +
  plot_likelihoods_df$log_lik_true_test) / n_samples
plot_likelihoods_df$log_lik_model <- (plot_likelihoods_df$log_lik_NN_train_orig +
  plot_likelihoods_df$log_lik_NN_test_orig) / n_samples
plot_likelihoods_df$log_lik_simp <- (plot_likelihoods_df$log_lik_simp_train_orig +
  plot_likelihoods_df$log_lik_simp_test_orig) / n_samples
plot_likelihoods_df <- plot_likelihoods_df %>%
  select("num_samples", "dim", "tau_max","param_cond_func", "log_lik_true", "log_lik_simp", "log_lik_model")
head(plot_likelihoods_df)
plot_likelihoods_df_long <- plot_likelihoods_df %>%
  pivot_longer(
    cols = c("log_lik_true", "log_lik_simp", "log_lik_model"),
    names_to = "valType",
    values_to = "value"
  )

p_loglik <- ggplot(plot_likelihoods_df_long, aes(x = param_cond_func, y = value, fill = valType)) +
  geom_boxplot(position = position_dodge(width = 0.8),
               linewidth = 0.2,
               outlier.size=0.4) +  # side-by-side boxes for val1/val2
  facet_grid(rows = vars(dim), cols = vars(tau_max), scales = "free_y", labeller = labeller(
    dim = label_both,      # shows "dim: 1", "dim: 2", etc.
    tau_max = label_both   # shows "tau_max: 0.1", etc.
  )) +
  scale_x_discrete(labels = function(x) substring(x, 2)) +
  scale_fill_discrete(labels = c(
    "log_lik_simp" = "c_noise",
    "log_lik_model" = "c_model",
    "log_lik_true" = "c_true")) +
  theme_bw(base_size = 14) +
  theme(
    strip.background = element_rect(fill = "grey90"),
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.placement = "outside",
    legend.position="bottom"
  ) +
  labs(
    x = "Parameter Function",
    y = "Log-likelihood",
    fill = "Log-likelihoods",
    title = "Log-likelihoods per sample"
  )

p_loglik

plot_MSE_df <- all_results %>%
  select("num_samples", "dim", "tau_max","param_cond_func",
         "log_lik_true_test",
         "log_lik_true_train",
         "MSE_loglik_NN_true_train",
         "MSE_loglik_NN_true_test",
         "MSE_loglik_simp_true_train",
         "MSE_loglik_simp_true_test"
         )
plot_MSE_df$MSE_loglik_NN_true <- 0.8 * plot_MSE_df$MSE_loglik_NN_true_train +
  0.2 * plot_MSE_df$MSE_loglik_NN_true_test
plot_MSE_df$MSE_loglik_simp_true <- 0.8 * plot_MSE_df$MSE_loglik_simp_true_train +
  0.2 * plot_MSE_df$MSE_loglik_simp_true_test
plot_MSE_df <- plot_MSE_df %>%
  select("num_samples", "dim", "tau_max","param_cond_func",
         "MSE_loglik_NN_true",
         "MSE_loglik_simp_true")
head(plot_MSE_df)
plot_MSE_df_long <- plot_MSE_df %>%
  pivot_longer(
    cols = c("MSE_loglik_NN_true", "MSE_loglik_simp_true"),
    names_to = "valType",
    values_to = "value"
  )


p_MSE <- ggplot(plot_MSE_df_long, aes(x = param_cond_func, y = value, fill = valType)) +
  geom_boxplot(position = position_dodge(width = 0.8),
               linewidth = 0.2,
               outlier.size=0.4) +  # side-by-side boxes for val1/val2
  facet_grid(rows = vars(dim), cols = vars(tau_max), scales = "free_y", labeller = labeller(
    dim = label_both,      # shows "dim: 1", "dim: 2", etc.
    tau_max = label_both   # shows "tau_max: 0.1", etc.
  )) +
  scale_x_discrete(labels = function(x) substring(x, 2)) +
  scale_fill_discrete(labels = c(
    "MSE_loglik_simp_true" = "c_noise",
    "MSE_loglik_NN_true" = "c_model"
    )) +
  theme_bw(base_size = 14) +
  theme(
    strip.background = element_rect(fill = "grey90"),
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.placement = "outside",
    legend.position="bottom"
  ) +
  labs(
    x = "Parameter Function",
    y = "MSE log-likelihood",
    fill = "Model",
    title = "Mean Squared Errors of Log-likelihoods"
  )

p_MSE

# AIC and BIC
plot_AIC_df <- all_results %>%
  select("num_samples", "dim", "tau_max","param_cond_func",
         "log_lik_true_test",
         "log_lik_true_train",
         "AIC_NN",
         "AIC_simp",
         "BIC_NN",
         "BIC_simp"
  )
head(plot_AIC_df)
plot_AIC_df_long <- plot_AIC_df %>%
  pivot_longer(
    cols = c("AIC_NN", "AIC_simp", "BIC_NN", "BIC_simp"),
    names_to = "valType",
    values_to = "value"
  )


p_AIC <- ggplot(plot_AIC_df_long, aes(x = param_cond_func, y = value, fill = valType)) +
  geom_boxplot(position = position_dodge(width = 0.8),
               linewidth = 0.2,
               outlier.size=0.4) +  # side-by-side boxes for val1/val2
  facet_grid(rows = vars(dim), cols = vars(tau_max), scales = "free_y", labeller = labeller(
    dim = label_both,      # shows "dim: 1", "dim: 2", etc.
    tau_max = label_both   # shows "tau_max: 0.1", etc.
  )) +
  scale_x_discrete(labels = function(x) substring(x, 2)) +
  scale_fill_discrete(labels = c(
    "AIC_simp" = "AIC c_noise",
    "AIC_NN" = "AIC c_model",
    "BIC_simp"= "BIC c_noise",
    "BIC_NN" = "BIC c_model"
  )) +
  theme_bw(base_size = 14) +
  theme(
    strip.background = element_rect(fill = "grey90"),
    strip.text = element_text(face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    strip.placement = "outside",
    legend.position="bottom"
  ) +
  labs(
    x = "Parameter Function",
    y = "AIC/ BIC",
    fill = "Value",
    title = "AIC and BIC scores"
  )

p_AIC

# First run all the code specifying n_samples = 1000 at the top, then run the following line
combined_plots_1000 <- p_loglik / p_MSE / p_AIC
combined_plots_1000
# Before running the following line, rerun all the code above with n_samples = 10000, except the one with combined_plots_paper_1000
combined_plots_10000 <- p_loglik / p_MSE / p_AIC
combined_plots_10000

combined_plots <- combined_plots_1000 | combined_plots_10000
combined_plots

ggsave(
  filename = paste0("combined_plots.png"),
  plot = combined_plots,
  width = 20,
  height = 20,
  dpi = 100,
  bg="white"
)
