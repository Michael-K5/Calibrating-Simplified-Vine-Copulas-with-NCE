# Script to run many simulations
library(keras)
library(rvinecopulib)
library(vineNCE)
# Parameters for the simulation.
# 2 different number of samples.
num_samples <- list(1000,10000)
# fraction of number of noise to true samples. Higher, if only 1000 samples given.
nus <- list(10,4)
# dimensions of the simulated data.
dims <- list(3,4,5)
# each list element contains first the lower, then the upper limit
tau_limits <- list(c(0.001,0.3), c(0.001,0.6), c(0.001,0.9))

# Define 3d parameters
struct_mat_3d <- matrix(c(1,1,1,
                         2,2,0,
                         3,0,0), ncol=3, byrow=TRUE)
families_3d <- list(list("frank", "gaussian"), list("frank"))
params_3d <- list(c(ktau_to_par(family=families_3d[[1]][[1]], tau=0.2)),
                  c(ktau_to_par(family=families_3d[[1]][[2]], tau=0.4)))
rotations_3d <- list(list(0,0),0)
# Define 4d parameters
struct_mat_4d <- matrix(c(3,3,3,3,
                         2,2,2,0,
                         4,4,0,0,
                         1,0,0,0), ncol=4, byrow=TRUE)
families_4d <- list(list("frank", "gaussian","gaussian"),
                   list("gaussian","frank"),
                   list("frank"))
params_4d <- list(c(ktau_to_par(family=families_4d[[1]][[1]], tau=0.2)),
                 c(ktau_to_par(family=families_4d[[1]][[2]], tau=0.3)),
                 c(ktau_to_par(family=families_4d[[1]][[3]], tau=0.7)))
rotations_4d <- list(list(0,0,0),list(0,0), list(0))
# Define 5d parameters
struct_mat_5d <- matrix(c(2,3,2,1,1,
                       3,2,1,2,0,
                       1,1,3,0,0,
                       4,4,0,0,0,
                       5,0,0,0,0), ncol=5, byrow=TRUE)
families_5d <- list(list("frank", "gaussian","gaussian","frank"),
                    list("frank","gaussian","gaussian"),
                    list("gaussian", "frank"),
                    list("gaussian"))
params_5d <- list(c(ktau_to_par(family=families_5d[[1]][[1]], tau=0.2)),
                  c(ktau_to_par(family=families_5d[[1]][[2]], tau=0.3)),
                  c(ktau_to_par(family=families_5d[[1]][[3]], tau=0.4)),
                  c(ktau_to_par(family=families_5d[[1]][[4]], tau=0.1)))
rotations_5d <- list(list(0,0,0,0),list(0,0,0), list(0,0), list(0))
# Summarize parameters by dimension
# struct_mats <- list(struct_mat_3d, struct_mat_4d, struct_mat_5d)
# families <- list(families_3d, families_4d, families_5d)
# initial_params <- list(params_3d, params_4d, params_5d)
# rotations <- list(rotations_3d, rotations_4d, rotations_5d)

#' Run the simulations
#' @param num_samples: list of integers, that determine for what different
#' number of samples drawn from a non-simplified vine copula the experiments should be run.
#' @param nus: list of numbers: Determines how many noise (simplified) samples should be simulated.
#' The i-th entry contains the values nu = n_{noise}/n_{true} for
#' the number of samples specified in num_samples[[i]]
#' @param dims: list of integers, that determine for what dimensions to run the tests
#' (currently implemented for 3, 4 and 5)
#' @param tau_limits: list of 2 dimensional vectors: For each of those vectors,
#' the first entry determines, what the lower threshold of kendalls tau is,
#' the second entry determines what the upper threshold of kendalls tau is in the
#' conditional copulas
#' @param struct_mats: List of regular vine matrices: Determine what structure to use.
#' The entry in position i needs to have the same dimension as dims[[i]].
#' @param families: List of (list of list of string): The i-th element
#' contains the copula families corresponding to the copulas defined with struct_mats[[i]]
#' @param initial_params: List of vectors: The i-th entry contains the parameters
#' for the i-th unconditioned copula in the first tree, defined by struct_mats[[i]]
#' @param rotations: List of (list of list of int): The i-th element contains the
#' rotations of the copulas specified in struct_mats[[i]]
#' @param lower_q_levs: Vector of float: defaults to seq(0.01,0.25,0.01). The lower
#' quantiles that should be estimated in the quantile regression
#' @param upper_q_levs: Vector of float: defaults to seq(0.75,0.99,0.01). The upper
#' quantiles that should be estimated in the quantile regression
#' @param int_bottom_threshold: Float, defaults to 0.6: If the monte carlo integral of the
#' non-parametric model is less than int_bottom_threshold, then the same combination of parameters
#' is run again (at most max_retries times), to try to find a more appropriate model.
#' @param int_top_threshold: Float, defaults to 1.8: If the monte carlo integral of the
#' non-parametric model is greater than int_top_threshold, then the same combination of parameters
#' is run again (at most max_retries times), to try to find a more appropriate model.
#' @param max_retries: Integer, defaults to 3: How often to retry evaluating the model if
#' the monte carlo integral value is not close enough to 1 (as measured by int_dev_threshold)
#' @param noise_method: String, defaults to "parametric". Which copula families to use for
#' the noise distribution of NCE ("parametric" or "all" are recommended choices.)
#' @param filename: string, defaults to "": If not "", then the results are written to
#' a csv file, with name specified as filename
#' @param save_data_repl: integer, defaults to -1: If not -1, saves the non-simplified samples
#' along with the likelihoods under c_noise, c_model and c_true to a dataframe, with a specific name and
#' as appendix save_data_repl in the filename.
#' @returns result_df: A Dataframe with the results of the experiments.
run_simulations <- function(num_samples,
                            dims,
                            nus,
                            tau_limits,
                            struct_mats,
                            families,
                            initial_params,
                            rotations,
                            NN_verbose =0,
                            lower_q_levs = seq(0.01,0.25,0.01),
                            upper_q_levs = seq(0.75,0.99,0.01),
                            bottom_q_NN_train = c(0.01,0.05,0.1,0.15,0.2,0.25),
                            top_q_NN_train=c(0.75,0.8,0.85,0.9,0.95,0.99),
                            int_bottom_threshold = 0.6,
                            int_top_threshold = 1.9,
                            max_retries = 3,
                            noise_method="parametric",
                            filename="",
                            save_data_repl = -1){
  # Run big simulation
  total_runs <- 0
  # Initialize an empty list to store the results of each inner loop iteration
  all_results <- list()
  for(sample_idx in 1:length(num_samples)){
    # Check, if any of the dimensions are not implemented. If so stop execution and print.
    incorrect_dimensions <- setdiff(unlist(dims), c(3,4,5))
    if (length(incorrect_dimensions) > 0) {
      print("The following dimensions are not implemented: ")
      print(incorrect_dimensions)
      print("Please remove these dimensions, then try to execute this function again.")
      break
    }
    nu_var <- nus[[sample_idx]]
    for(tau_idx in 1:length(tau_limits)){
      tau_lower=tau_limits[[tau_idx]][1]
      tau_upper=tau_limits[[tau_idx]][2]
      for(dim_idx in 1:length(dims)){
        # initialize param_cond_funcs (overwritten below)
        param_cond_func_var <- -1
        param_cond_func_names <- c("0constpf", "1linpf", "2quadpf", "3cubpf")
        # Define the conditional parameter functions
        if(dims[[dim_idx]] ==3){
          param_cond_func_3d_const <- list(
            list(u_to_param_constant(tau_lower=tau_lower, tau_upper=tau_upper))
          )
          param_cond_func_3d_lin <- list(
            list(u_to_param_linear(c(1), tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_func_3d_quad <- list(
            list(u_to_param_quadratic(c(1,1),
                                    tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_func_3d_cubic <- list(
            list(u_to_param_cubic(c(1,1.2,0.5),
                                          tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_func_var <- list(
            param_cond_func_3d_const,
            param_cond_func_3d_lin,
            param_cond_func_3d_quad,
            param_cond_func_3d_cubic
          )
        } else if(dims[[dim_idx]] ==4){
          param_cond_funcs_4d_const <- list(
            list(u_to_param_constant(tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_constant(tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_constant(tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_funcs_4d_lin <- list(
            list(u_to_param_linear(c(1), tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_linear(c(1), tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_linear(c(0.4,0.6), tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_funcs_4d_quad <- list(
            list(
              u_to_param_quadratic(c(1,1.4),
                                 tau_lower=tau_lower, tau_upper=tau_upper),
              u_to_param_quadratic(c(1.2,0.9),
                                 tau_lower=tau_lower, tau_upper=tau_upper)),
            list(
              u_to_param_quadratic(c(-0.7,1.5,0.9,-1.2,1.5),
                                 tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_funcs_4d_cubic <- list(
            list(u_to_param_cubic(c(0.7,1.2,-0.8),
                                          tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_cubic(c(1.4,-1.8,1.0),
                                          tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_cubic(c(0.7,1.8, -0.9,1.4,0.8,1.1,-1.2,-1.3,1.7),
                                          tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_func_var <- list(
            param_cond_funcs_4d_const,
            param_cond_funcs_4d_lin,
            param_cond_funcs_4d_quad,
            param_cond_funcs_4d_cubic)
        } else if(dims[[dim_idx]] ==5){
          param_cond_funcs_5d_const <- list(
            list(u_to_param_constant(tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_constant(tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_constant(tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_constant(tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_constant(tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_constant(tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_funcs_5d_lin <- list(
            list(u_to_param_linear(c(1), tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_linear(c(1), tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_linear(c(1), tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_linear(c(0.7,0.3), tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_linear(c(0.4,0.6), tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_linear(c(0.2,0.5,0.3), tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_funcs_5d_quad <- list(
            list(
              u_to_param_quadratic(c(1,1),
                                 tau_lower=tau_lower, tau_upper=tau_upper),
              u_to_param_quadratic(c(1,0.7),
                                 tau_lower=tau_lower, tau_upper=tau_upper),
              u_to_param_quadratic(c(1,2),
                                 tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_quadratic(c(0.7,0.5, 0.9,-1.2,1.5),
                                    tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_quadratic(c(0.4,-0.8,1.0,0.7,2.0),
                                    tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_quadratic(c(0.7,0.4,-0.9,1.3,0.75,1.3,-0.6,0.5,-1.1),
                                    tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_funcs_5d_cubic <- list(
            list(u_to_param_cubic(c(1.0,1.0,0.5),
                                          tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_cubic(c(1.0,0.7,0.3),
                                          tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_cubic(c(1.0,2.0,1.1),
                                          tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_cubic(c(0.7,0.8, -0.9,0.4,0.8,1.1,-1.2,-1.3,1.0),
                                          tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_cubic(c(0.4,0.6,1,0.7,2, -1.1,1.2,-0.9,-0.3),
                                          tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_cubic(c(0.7, 0.5, -1.3,
                                    1,   1.4, 1  ,-1.4,0.7,-0.8,
                                    1.1,-1.4,0.7,-0.3,0.4,
                                    0.8, -0.7,1.4,-1.2,-0.6),
                                    tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_func_var <- list(
            param_cond_funcs_5d_const,
            param_cond_funcs_5d_lin,
            param_cond_funcs_5d_quad,
            param_cond_funcs_5d_cubic
          )
        }
        # Define the other necessary parameters for simulation
        struct_mat_var <- struct_mats[[dim_idx]]
        families_var <- families[[dim_idx]]
        params_var <- initial_params[[dim_idx]]
        rotations_var <- rotations[[dim_idx]]
        for(par_idx in 1:length(param_cond_func_var)){
          total_runs <- total_runs+1
          print(paste0("Executing... ",
                       total_runs,
                       " of a total of ",
                       length(dims)*length(num_samples)*length(tau_limits)*length(param_cond_func_var)))
          non_simp_data <- simulate_non_simplified(n_samples = num_samples[[sample_idx]],
                                                      struct = struct_mat_var,
                                                      families=families_var,
                                                      params = params_var,
                                                      param_cond_funcs = param_cond_func_var[[par_idx]],
                                                      rotations = rotations_var)
          orig_data <- as.matrix(non_simp_data)
          orig_data <- unname(orig_data)
          # train test split of original data
          train_test_split_orig_output <- train_test_split_orig(
            data=orig_data,
            train_perc = 0.8
          )
          orig_data_train <- train_test_split_orig_output[[1]]
          orig_data_test <- train_test_split_orig_output[[2]]
          fitted_vine <-vinecop(orig_data_train,
                                family_set=noise_method)
          # number of samples to create from the fitted vine
          n_noise <- nu_var * nrow(orig_data)
          # simulate from the simplified vine, to train a classifier
          simplified_samples <- rvinecop(n_noise, fitted_vine)
          split_output <- train_test_split(
            orig_data_train=orig_data_train,
            orig_data_test=orig_data_test,
            simplified_data=simplified_samples
          )
          x_train <- split_output[[1]]
          x_test <- split_output[[2]]
          y_train <- split_output[[3]]
          y_test <- split_output[[4]]
          int_val <- int_bottom_threshold - 1
          retry_num <- 0
          model <- 0
          train_set_eval <- -1
          test_set_eval <- -1
          # Loop to discard a certain value, if the integral deviates too much from 1
          while((int_val < int_bottom_threshold || int_val > int_top_threshold)
                && retry_num < max_retries){
            retry_num <- retry_num +1
            # define new model
            model <- build_model(input_dim=ncol(x_train), use_tanh=FALSE)
            train_model_output <- train_model(model,
                                              x_train,
                                              y_train,
                                              num_epochs=200,
                                              verbose=NN_verbose)
            model <- train_model_output[[1]]
            test_set_eval <- model %>% evaluate(x_test, y_test)
            train_set_eval <- model %>% evaluate(x_train, y_train)
            int_val <- compute_integral(model=model,
                                        fitted_vine = fitted_vine,
                                        nu=nu_var,
                                        data_dim_if_unif = ncol(orig_data),
                                        n_samples=20000,
                                        user_info=FALSE)
          }
          log_lik_true_train_vec <- log_likelihood_non_simplified(
            u_data=x_train[y_train==1,],
            struct = struct_mat_var,
            families=families_var,
            params = params_var,
            param_cond_funcs = param_cond_func_var[[par_idx]],
            rotations = rotations_var,
            return_vector=TRUE)
          log_lik_true_train <- sum(log_lik_true_train_vec)
          log_lik_true_test_vec <- log_likelihood_non_simplified(
            u_data=x_test[y_test==1,],
            struct = struct_mat_var,
            families=families_var,
            params = params_var,
            param_cond_funcs = param_cond_func_var[[par_idx]],
            rotations = rotations_var,
            return_vector=TRUE)
          log_lik_true_test <- sum(log_lik_true_test_vec)
          # evaluate log likelihoods of original data (on train and test set fractions of orig_data)
          log_lik_simp_train_orig_vec <- log(dvinecop(x_train[y_train==1,], fitted_vine))
          log_lik_simp_train_orig <- sum(log_lik_simp_train_orig_vec)
          log_lik_NN_train_orig_vec <- log(NCE_cop(
            model=model,
            fitted_vine=fitted_vine,
            obs=x_train[y_train==1,],
            nu=nu_var))
          log_lik_NN_train_orig <- sum(log_lik_NN_train_orig_vec)
          log_lik_simp_test_orig_vec <- log(dvinecop(x_test[y_test==1,], fitted_vine))
          log_lik_simp_test_orig <- sum(log_lik_simp_test_orig_vec)
          log_lik_NN_test_orig_vec <- log(NCE_cop(
            model=model,
            fitted_vine=fitted_vine,
            obs=x_test[y_test==1,],
            nu=nu_var))
          log_lik_NN_test_orig <- sum(log_lik_NN_test_orig_vec)
          if(save_data_repl != -1){
            # Optionally save the non-simplified data
            train_indicator <- c(rep(1, nrow(x_train[y_train==1,])), rep(0, nrow(x_test[y_test==1,])))
            orig_samples_dataframe_train <- cbind(x_train[y_train==1,],
                                                  log_lik_true_train_vec,
                                                  log_lik_simp_train_orig_vec,
                                                  log_lik_NN_train_orig_vec)
            orig_samples_dataframe_test <- cbind(x_test[y_test==1,],
                                                 log_lik_true_test_vec,
                                                 log_lik_simp_test_orig_vec,
                                                 log_lik_NN_test_orig_vec)
            orig_samples_dataframe <- as.data.frame(cbind(rbind(orig_samples_dataframe_train,
                                                                orig_samples_dataframe_test),
                                                          train_indicator))
            dataframe_orig_samples_name <- paste0(
              "data/", noise_method,
              "NSamples", nrow(orig_data),
              "Tau", tau_upper,
              "Dim", dims[[dim_idx]],
              param_cond_func_names[par_idx],
              "Repl", save_data_repl,
              ".csv")
            readr::write_csv(orig_samples_dataframe, file=dataframe_orig_samples_name)
          }
          MSE_loglik_NN_true_train <- mean((log_lik_NN_train_orig_vec - log_lik_true_train_vec)^2)
          MSE_loglik_NN_true_test <- mean((log_lik_NN_test_orig_vec - log_lik_true_test_vec)^2)
          MSE_loglik_simp_true_train <- mean((log_lik_simp_train_orig_vec - log_lik_true_train_vec)^2)
          MSE_loglik_simp_true_test <- mean((log_lik_simp_test_orig_vec - log_lik_true_test_vec)^2)
          # get number of model parameters and degrees of freedom
          NN_num_params <- count_NN_params(weights=model$weights)
          simp_cop_num_params <- fitted_vine$npars
          # compute AIC and BIC for the NN and simplified model
          AIC_NN <- 2*(NN_num_params + simp_cop_num_params) - 2* (log_lik_NN_train_orig + log_lik_NN_test_orig)
          AIC_simp <- 2*simp_cop_num_params - 2*(log_lik_simp_train_orig + log_lik_simp_test_orig)
          BIC_NN <- log(nrow(orig_data))*(NN_num_params + simp_cop_num_params) - 2* (log_lik_NN_train_orig + log_lik_NN_test_orig)
          BIC_simp <- log(nrow(orig_data))*simp_cop_num_params - 2*(log_lik_simp_train_orig + log_lik_simp_test_orig)
          log_cor_facs <- log(correction_factors(model, obs=orig_data, nu=nu_var))
          # Store the results of the current iteration
          current_result <- c(noise_method,
                              num_samples[[sample_idx]],
                              nus[[sample_idx]],
                              dims[[dim_idx]],
                              tau_limits[[tau_idx]], # this is a vector with 2 elements
                              param_cond_func_names[par_idx],
                              train_set_eval[[1]],
                              train_set_eval[[2]],
                              test_set_eval[[1]],
                              test_set_eval[[2]],
                              int_val,
                              log_lik_simp_train_orig,
                              log_lik_NN_train_orig,
                              log_lik_true_train,
                              log_lik_simp_test_orig,
                              log_lik_NN_test_orig,
                              log_lik_true_test,
                              MSE_loglik_NN_true_train,
                              MSE_loglik_NN_true_test,
                              MSE_loglik_simp_true_train,
                              MSE_loglik_simp_true_test,
                              NN_num_params,
                              simp_cop_num_params,
                              AIC_NN,
                              AIC_simp,
                              BIC_NN,
                              BIC_simp
                              )
          # append to the results list, as a list, to keep the rows separated
          all_results <- append(all_results, list(current_result))
        }
      }
    }
  }
  # Summarize the results in a dataframe
  results_df <- as.data.frame(do.call(rbind, all_results))
  colnames(results_df) <- c("noise_method",
                            "num_samples",
                            "nu",
                            "dim",
                            "tau_min",
                            "tau_max",
                            "param_cond_func",
                            "train set loss",
                            "train set accuracy",
                            "test set loss",
                            "test set accuracy",
                            "MC_Integral",
                            "log_lik_simp_train_orig",
                            "log_lik_NN_train_orig",
                            "log_lik_true_train",
                            "log_lik_simp_test_orig",
                            "log_lik_NN_test_orig",
                            "log_lik_true_test",
                            "MSE_loglik_NN_true_train",
                            "MSE_loglik_NN_true_test",
                            "MSE_loglik_simp_true_train",
                            "MSE_loglik_simp_true_test",
                            "NN_num_params",
                            "simp_cop_num_params",
                            "AIC_NN",
                            "AIC_simp",
                            "BIC_NN",
                            "BIC_simp"
                            )
  if(filename != ""){
    readr::write_csv(results_df, file = filename)
    print(paste("Data saved to:", filename))
  }
  return(results_df)
}
# Run simulations
# number of samples. Can be different values in a list
num_samples <- list(1000)
# fraction of number of noise to true samples
nus <- list(10)
# which simplified vine copula families to use for NCE
noise_method = "all" # "parametric" or "all"
# dimensions of the simulated data.
dims <- list(3,5)
num_replications <- 10
struct_mats <- list(struct_mat_3d, struct_mat_5d)
families <- list(families_3d, families_5d)
initial_params <- list(params_3d, params_5d)
rotations <- list(rotations_3d, rotations_5d)

# Run Simulations sequentially. Alternative: Run in parallel, see code below.
# for(i in 1:num_replications){
#   print(paste0("Replication ", i, " of ", num_replications))
#   replication_name <- paste0("Repl",i)
#   # Current data (Alternative: Use Sys.Date())
#   current_date <- "20250808"
#   # Construct the file name with the date
#   num_samples_filename <- paste0(unlist(num_samples), collapse = ",")
#   results_filename <- paste0("data/",
#                              noise_method,
#                              current_date,
#                              "NSamples",
#                              num_samples_filename,
#                              replication_name,
#                              ".csv")
#   result_df <- run_simulations(
#     num_samples=num_samples,
#     dims=dims,
#     nus=nus,
#     tau_limits=tau_limits,
#     struct_mats=struct_mats,
#     families=families,
#     initial_params=initial_params,
#     rotations=rotations,
#     noise_method=noise_method,
#     filename=results_filename,
#     save_data_repl =i # -1 to not save the original samples; i to save the original samples
#   )
# }


# Run several simulations in parallel
library(parallel)
# change number of clusters depending on computer capabilities
cl <- makeCluster(2)
clusterEvalQ(cl, {
  library(rvinecopulib)
  library(keras)
  library(vineNCE)
})
# with the vineNCE package the following is no longer necessary
# clusterEvalQ(cl, {
#   source("R/classifier_methods.R")
#   source("R/simulate_non_simplified_vine.R")
# })
clusterExport(cl, c("run_simulations",
                    "num_samples", "dims", "nus", "tau_limits",
                    "struct_mats", "families", "initial_params",
                    "rotations", "noise_method"))
clusterEvalQ(cl, ls())
# pbapply to display a progress bar
output_list <- pbapply::pblapply(1:num_replications, function(i) {
  set.seed(42+i) # set seed to ensure non-identical results for the different workers.
  replication_name <- paste0("Repl",i)
  # Current data (Alternative: Use Sys.Date())
  current_date <- "20250808"
  # Construct the file name with the date
  num_samples_filename <- paste0(unlist(num_samples), collapse = ",")
  results_filename <- paste0("data/",
                             noise_method,
                             current_date,
                             "NSamples",
                             num_samples_filename,
                             replication_name,
                             ".csv")
  result_df <- run_simulations(
    num_samples=num_samples,
    dims=dims,
    nus=nus,
    tau_limits=tau_limits,
    struct_mats=struct_mats,
    families=families,
    initial_params=initial_params,
    rotations=rotations,
    noise_method=noise_method,
    filename=results_filename,
    save_data_repl =i # -1 to not save the original samples; i to save the original samples
  )
}, cl=cl)
# stop the cluster, to free resources on computer
stopCluster(cl)
# END OF PARALLEL SIMULATIONS

# Plot Pairs copula data
source("R/plotting_methods.R")
#' plot pairs copula data
#' @param num_samples: list of integers, that determine for what different
#' number of samples drawn from a non-simplified vine copula the experiments should be run.
#' @param dims: list of integers, that determine for what dimensions to run the tests
#' (currently implemented for 3, 4 and 5)
#' @param tau_limits: list of 2 dimensional vectors: For each of those vectors,
#' the first entry determines, what the lower threshold of kendalls tau is,
#' the second entry determines what the upper threshold of kendalls tau is in the
#' conditional copulas
#' @param struct_mats: List of regular vine matrices: Determine what structure to use.
#' The entry in position i needs to have the same dimension as dims[[i]].
#' @param families: List of (list of list of string): The i-th element
#' contains the copula families corresponding to the copulas defined with struct_mats[[i]]
#' @param initial_params: List of vectors: The i-th entry contains the parameters
#' for the i-th unconditioned copula in the first tree, defined by struct_mats[[i]]
#' @param rotations: List of (list of list of int): The i-th element contains the
#' rotations of the copulas specified in struct_mats[[i]]
#' @returns result_df: A Dataframe with the results of the experiments.
plot_data <- function(num_samples,
                            dims,
                            tau_limits,
                            struct_mats,
                            families,
                            initial_params,
                            rotations
                            ){
  # Run big simulation
  total_runs <- 0
  # Initialize an empty list to store the results of each inner loop iteration
  all_plots <- list()
  for(sample_idx in 1:length(num_samples)){
    # Check, if any of the dimensions are not implemented. If so stop execution and print.
    incorrect_dimensions <- setdiff(unlist(dims), c(3,4,5))
    if (length(incorrect_dimensions) > 0) {
      print("The following dimensions are not implemented: ")
      print(incorrect_dimensions)
      print("Please remove these dimensions, then try to execute this function again.")
      break
    }
    for(tau_idx in 1:length(tau_limits)){
      tau_lower=tau_limits[[tau_idx]][1]
      tau_upper=tau_limits[[tau_idx]][2]
      for(dim_idx in 1:length(dims)){
        # initialize param_cond_funcs (overwritten below)
        param_cond_func_var <- -1
        # Define the conditional parameter functions
        if(dims[[dim_idx]] ==3){
          param_cond_func_3d_lin <- list(
            list(u_to_param_linear(c(1), tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_func_3d_quad <- list(
            list(u_to_param_quadratic(c(1,1),
                                      tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_func_3d_cubic <- list(
            list(u_to_param_cubic(c(1,1.2,0.5),
                                  tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_func_var <- list(
            param_cond_func_3d_lin,
            param_cond_func_3d_quad,
            param_cond_func_3d_cubic)
        } else if(dims[[dim_idx]] ==4){
          param_cond_funcs_4d_lin <- list(
            list(u_to_param_linear(c(1), tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_linear(c(1), tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_linear(c(0.4,0.6), tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_funcs_4d_quad <- list(
            list(
              u_to_param_quadratic(c(1,1.4),
                                   tau_lower=tau_lower, tau_upper=tau_upper),
              u_to_param_quadratic(c(1.2,0.9),
                                   tau_lower=tau_lower, tau_upper=tau_upper)),
            list(
              u_to_param_quadratic(c(-0.7,1.5,0.9,-1.2,1.5),
                                   tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_funcs_4d_cubic <- list(
            list(u_to_param_cubic(c(0.7,1.2,-0.8),
                                  tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_cubic(c(1.4,-1.8,1.0),
                                  tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_cubic(c(0.7,1.8, -0.9,1.4,0.8,1.1,-1.2,-1.3,1.7),
                                  tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_func_var <- list(
            param_cond_funcs_4d_lin,
            param_cond_funcs_4d_quad,
            param_cond_funcs_4d_cubic)
        } else if(dims[[dim_idx]] ==5){
          param_cond_funcs_5d_lin <- list(
            list(u_to_param_linear(c(1), tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_linear(c(1), tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_linear(c(1), tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_linear(c(0.7,0.3), tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_linear(c(0.4,0.6), tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_linear(c(0.2,0.5,0.3), tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_funcs_5d_quad <- list(
            list(
              u_to_param_quadratic(c(1,1),
                                   tau_lower=tau_lower, tau_upper=tau_upper),
              u_to_param_quadratic(c(1,0.7),
                                   tau_lower=tau_lower, tau_upper=tau_upper),
              u_to_param_quadratic(c(1,2),
                                   tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_quadratic(c(0.7,0.5, 0.9,-1.2,1.5),
                                      tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_quadratic(c(0.4,-0.8,1.0,0.7,2.0),
                                      tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_quadratic(c(0.7,0.4,-0.9,1.3,0.75,1.3,-0.6,0.5,-1.1),
                                      tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_funcs_5d_cubic <- list(
            list(u_to_param_cubic(c(1.0,1.0,0.5),
                                  tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_cubic(c(1.0,0.7,0.3),
                                  tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_cubic(c(1.0,2.0,1.1),
                                  tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_cubic(c(0.7,0.8, -0.9,0.4,0.8,1.1,-1.2,-1.3,1.0),
                                  tau_lower=tau_lower, tau_upper=tau_upper),
                 u_to_param_cubic(c(0.4,0.6,1,0.7,2, -1.1,1.2,-0.9,-0.3),
                                  tau_lower=tau_lower, tau_upper=tau_upper)),
            list(u_to_param_cubic(c(0.7, 0.5, -1.3,
                                    1,   1.4, 1  ,-1.4,0.7,-0.8,
                                    1.1,-1.4,0.7,-0.3,0.4,
                                    0.8, -0.7,1.4,-1.2,-0.6),
                                  tau_lower=tau_lower, tau_upper=tau_upper)))
          param_cond_func_var <- list(
            param_cond_funcs_5d_lin,
            param_cond_funcs_5d_quad,
            param_cond_funcs_5d_cubic)
        }
        # Define the other necessary parameters for simulation
        struct_mat_var <- struct_mats[[dim_idx]]
        families_var <- families[[dim_idx]]
        params_var <- initial_params[[dim_idx]]
        rotations_var <- rotations[[dim_idx]]
        for(par_idx in 1:length(param_cond_func_var)){
          total_runs <- total_runs+1
          print(paste0("Executing... ",
                       total_runs,
                       " of a total of ",
                       length(dims)*length(num_samples)*length(tau_limits)*length(param_cond_func_var)))
          non_simp_data <- simulate_non_simplified(n_samples = num_samples[[sample_idx]],
                                                      struct = struct_mat_var,
                                                      families=families_var,
                                                      params = params_var,
                                                      param_cond_funcs = param_cond_func_var[[par_idx]],
                                                      rotations = rotations_var)
          orig_data <- as.matrix(non_simp_data)
          orig_data <- unname(orig_data)
          param_cond_func_name <- "linpf"
          if(par_idx == 2){
            param_cond_func_name <- "quadpf"
          } else if (par_idx ==3){
            param_cond_func_name <- "cubpf"
          }
          all_plots[[length(all_plots)+1]] <- ggplotify::as.ggplot(
            copula_pairs_ggplot(orig_data, title=paste0(param_cond_func_name, ", tau_max= ", tau_upper))
          )
        }
      }
    }
  }

  return(all_plots)
}
# PAIRS COPULA PLOT
n_samples_plot <- list(10000)

# dimensions of the simulated data.
dims <- list(3)
# each list element contains first the lower, then the upper limit
tau_limits <- list(c(0.001,0.3), c(0.001,0.6), c(0.001,0.9))
# Define 3d parameters
struct_mat_3d <- matrix(c(1,1,1,
                          2,2,0,
                          3,0,0), ncol=3, byrow=TRUE)
families_3d <- list(list("frank", "gaussian"), list("frank"))
params_3d <- list(c(ktau_to_par(family=families_3d[[1]][[1]], tau=0.2)),
                  c(ktau_to_par(family=families_3d[[1]][[2]], tau=0.4)))
rotations_3d <- list(list(0,0),0)
# Define 5d parameters
struct_mat_5d <- matrix(c(2,3,2,1,1,
                          3,2,1,2,0,
                          1,1,3,0,0,
                          4,4,0,0,0,
                          5,0,0,0,0), ncol=5, byrow=TRUE)
families_5d <- list(list("frank", "gaussian","gaussian","frank"),
                    list("frank","gaussian","gaussian"),
                    list("gaussian", "frank"),
                    list("gaussian"))
params_5d <- list(c(ktau_to_par(family=families_5d[[1]][[1]], tau=0.2)),
                  c(ktau_to_par(family=families_5d[[1]][[2]], tau=0.3)),
                  c(ktau_to_par(family=families_5d[[1]][[3]], tau=0.4)),
                  c(ktau_to_par(family=families_5d[[1]][[4]], tau=0.1)))
rotations_5d <- list(list(0,0,0,0),list(0,0,0), list(0,0), list(0))

struct_mats <- list(struct_mat_3d)
families <- list(families_3d)
initial_params <- list(params_3d)
rotations <- list(rotations_3d)
if(dims==5){
  struct_mats <- list(struct_mat_5d)
  families <- list(families_5d)
  initial_params <- list(params_5d)
  rotations <- list(rotations_5d)
}
result_plots <- plot_data(num_samples = n_samples_plot,
                          dims=dims,
                          tau_limits=tau_limits,
                          struct_mats = struct_mats,
                          families=families,
                          initial_params=initial_params,
                          rotations=rotations)
library(patchwork)
length(result_plots)
all_pairs_plots <- (result_plots[[1]] | result_plots[[2]] | result_plots[[3]]) /
  (result_plots[[4]] | result_plots[[5]] | result_plots[[6]]) /
  (result_plots[[7]] | result_plots[[8]] | result_plots[[9]])
# all_pairs_plots
pairs_plots_file_name <- paste0(dims, "d_non_simplified_all_configs_", n_samples_plot, "_plot.png")
print(pairs_plots_file_name)
# ggplot2::ggsave(
#   filename = pairs_plots_file_name,
#   plot = all_pairs_plots,
#   width = 15,
#   height = 15,
#   dpi = 100,
#   bg="white"
# )
# END PAIRS COPULA PLOT

