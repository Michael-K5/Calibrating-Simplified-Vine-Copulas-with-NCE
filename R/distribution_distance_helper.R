source("R/simulate_non_simplified_vine.R")
source("R/classifier_methods.R")
# Parameters for the simulation.
# 2 different number of samples.
num_samples <- list(1000,10000)
# dimensions of the simulated data.
dims <- list(3,5)
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

#' For the different configurations first simulate from a non-simplified vine copula,
#' then fit a simplified vine copula with the same structure, to determine the distribution
#' difference, to get an idea "how non-simplified" the configuration is.
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
run_dist_difference_helper_fun <- function(
    num_samples,
    dims,
    tau_limits,
    struct_mats,
    families,
    initial_params,
    rotations,
    save_data_repl = 1
    ){
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
          # simulate non-simplified data
          non_simp_data <- simulate_non_simplified(
            n_samples = num_samples[[sample_idx]],
            struct = struct_mat_var,
            families=families_var,
            params = params_var,
            param_cond_funcs = param_cond_func_var[[par_idx]],
            rotations = rotations_var
          )
          orig_data <- as.matrix(non_simp_data)
          orig_data <- unname(orig_data)
          log_lik_true <- log_likelihood_non_simplified(
            orig_data,
            struct = struct_mat_var,
            families=families_var,
            params = params_var,
            param_cond_funcs = param_cond_func_var[[par_idx]],
            rotations = rotations_var,
            return_vector=TRUE
          )
          # OPTION 1: Fit a vine copula with the correct structure and onepar family
          fitted_vine <-vinecop(
            orig_data,
            structure=struct_mat_var,
            family_set = "onepar")
          # simulate from the simplified vine
          simplified_samples_fitted <- rvinecop(num_samples[[sample_idx]], fitted_vine)
          log_lik_simplified_fitted <- log(dvinecop(simplified_samples_fitted, fitted_vine))
          log_lik_orig_data_simp_fitted <- log(dvinecop(orig_data, fitted_vine))
          log_lik_simp_samples_fitted_non_simp_vine <- log_likelihood_non_simplified(
            simplified_samples_fitted,
            struct = struct_mat_var,
            families=families_var,
            params = params_var,
            param_cond_funcs = param_cond_func_var[[par_idx]],
            rotations = rotations_var,
            return_vector=TRUE
          )
          # OPTION 2: Just use the constpf parameter function to measure distance
          simp_samples_constpf <- simulate_non_simplified(
            n_samples = num_samples[[sample_idx]],
            struct = struct_mat_var,
            families=families_var,
            params = params_var,
            param_cond_funcs = param_cond_func_var[[1]], # constant parameter function
            rotations = rotations_var
          )
          log_lik_simp_samples_constpf <- log_likelihood_non_simplified(
            simp_samples_constpf,
            struct = struct_mat_var,
            families=families_var,
            params = params_var,
            param_cond_funcs = param_cond_func_var[[1]], # constant parameter function
            rotations = rotations_var,
            return_vector=TRUE
          )
          log_lik_orig_data_constpf <- log_likelihood_non_simplified(
            orig_data,
            struct = struct_mat_var,
            families=families_var,
            params = params_var,
            param_cond_funcs = param_cond_func_var[[1]], # constant parameter function
            rotations = rotations_var,
            return_vector=TRUE
          )
          log_lik_constpf_samples_non_simp_vine <- log_likelihood_non_simplified(
            simp_samples_constpf,
            struct = struct_mat_var,
            families=families_var,
            params = params_var,
            param_cond_funcs = param_cond_func_var[[par_idx]],
            rotations = rotations_var,
            return_vector=TRUE
          )
          all_non_simp_data <- as.data.frame(cbind(
            orig_data,
            log_lik_true,
            log_lik_simp_samples_fitted_non_simp_vine,
            log_lik_constpf_samples_non_simp_vine))
          readr::write_csv(
            all_non_simp_data,
            file=paste0("data/DistributionDistanceData/20250821OrigDim",
                        dims[[dim_idx]],
                        "Tau", tau_upper,
                        "ParFun", param_cond_func_names[[par_idx]],
                        "Repl", save_data_repl,
                        ".csv")
          )
          all_fitted_simp_data <- as.data.frame(cbind(
            simplified_samples_fitted,
            log_lik_orig_data_simp_fitted,
            log_lik_simplified_fitted
          ))
          readr::write_csv(
            all_fitted_simp_data,
            file=paste0("data/DistributionDistanceData/FittedSimpDim",
                        dims[[dim_idx]],
                        "Tau", tau_upper,
                        "ParFun", param_cond_func_names[[par_idx]],
                        "Repl", save_data_repl,
                        ".csv")
          )
          all_constpf_data <- as.data.frame(cbind(
            simp_samples_constpf,
            log_lik_orig_data_constpf,
            log_lik_simp_samples_constpf
          ))
          readr::write_csv(
            all_constpf_data,
            file=paste0("data/DistributionDistanceData/ConstpfDim",
                        dims[[dim_idx]],
                        "Tau", tau_upper,
                        "ParFun", param_cond_func_names[[par_idx]],
                        "Repl", save_data_repl,
                        ".csv")
          )
        }
      }
    }
  }
  return(all_results)
}
# dimensions of the simulated data.
num_samples= list(1000)
dims <- list(3,5)
num_replications <- 10

struct_mats <- list(struct_mat_3d, struct_mat_5d)
families <- list(families_3d, families_5d)
initial_params <- list(params_3d, params_5d)
rotations <- list(rotations_3d, rotations_5d)

for(i in 1:num_replications){
  print(paste0("Replication ", i, " of ", num_replications))
  result_df <- run_dist_difference_helper_fun(
    num_samples=num_samples,
    dims=dims,
    tau_limits=tau_limits,
    struct_mats=struct_mats,
    families=families,
    initial_params=initial_params,
    rotations=rotations,
    save_data_repl=i
  )
}
