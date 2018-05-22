require (caret);
require (rpart);


###########################################
####             Training             #####
####################################################################################################
#### Function  : kfhe_train                                                                     ####
#### Arguments :                                                                                #### 
####             X                : A matrix representing the input dataset                     ####
####             Y                : A matrix representing the target classes                    ####
####             max_iter         : Maximum ensemble iterations                                 ####
####             rpart_control    : An object returned by 'rpart::rpart.control', or NA         ####
####             blend_with_class : If TRUE,  then the measurement updates will be done after   ####
####                                          converting to classes.                            ####
####                                If FALSE, then the measurement updates will be done on the  #### 
####                                          predicted scores                                  ####
####             early_stop       : If TRUE,  stop training after the Kalman gain is 1          ####
####                                If FALSE, continue training until 'max_iter' ensemble       ####
####                                          iterations                                        ####
####             reset_dist       : If TRUE,  then reset the training weights to uniform if     ####
####                                          the measurement returns error more                ####
####                                          than '1 / (1 - nclass)'                           ####
####                                If FALSE, do not reset the training weights                 ####
####             feedback         : Print the states of the parameters while training           ####
####                                                                                            ####
#### Return    : An object of type kfhe_m. Consists of kf-m. Other components are for           ####
####             debugging and analysis purpose                                                 ####
####################################################################################################
kfhe_train <- function (X, Y, max_iter, rpart_control = NA, blend_with_class = TRUE, early_stop = TRUE, reset_dist = TRUE, feedback = TRUE)
{
  print_prec <- 8;  # Number of decimal places to round when printing feedback
  
  # Kalman filter variables for the model Kalman Filter, KF-m
  kf_m             <- list (); 
  kf_m$P           <- c ();    # Posterior variance/error
  kf_m$V           <- c ();    # Measurement variance/error
  kf_m$K           <- c ();    # Kalman Gain
  kf_m$h_list      <- list (); # List of learnt models
  kf_m$init_h      <- c ()     # The initial model.
  kf_m$train_blend <- c ();    # Internal state
    
  # Kalman filter variables for the distribution KF, KF-d
  kf_d        <- list ();
  kf_d$P_dist <- c ();    # Posterior variance/error
  kf_d$V_dist <- c ();    # Measurement variance/error
  kf_d$K_dist <- list (); # Kalman Gain
  kf_d$D      <- c ();    # Distribution
  
  # Debugging information
  debug_info                 <- list ();
  debug_info$train_accuracy  <- c (); # Accuracy on the train blend error
  debug_info$train_f         <- c (); # F-Score  on the train blend error
  debug_info$state_err       <- list (); # Error function evaluation of the train blend with respect to ground truth
  debug_info$blend_err_arr   <- c ();    # Uniformly weighted training blend error
  debug_info$D               <- list (); # Store distribution of all iterations
  debug_info$reset_pt        <- rep (FALSE, max_iter); # Trace in which iteration a distribution reset occurred
  debug_info$blend_err_delta <- c();
  
  
  # Final model to pack, stamp and return
  retmodel <- list ();
  
  if (is.na (rpart_control))
  {
    rpart_control = rpart.control(minsplit = 1, cp = -1, maxdepth = 30); # Default
  }
    
  # Initialise the KF-m statetrain_blend, this_err, this_blend_err
  kf_m$init_h      <- rpart (formula = Y ~ ., data = cbind (X, Y), control = rpart_control);
  kf_m$train_blend <- predict (kf_m$init_h, X, type = "prob");
  
  unwt_comp_err    <- err_fun (kf_m$train_blend, Y, NULL); # Find the per datapoint error. This is a vector.
  uniwt_comp_err   <- unwt_comp_err * (1/nrow (X));        # Get a uniformly weighted version. This is a weighted vector.
  
  # Initialise the state variance for model KF (KF-m)
  kf_m$P[1] <- 1.0; # No confidence
  
  # Initialise the state variance for distribution KF (KF-d)
  # Optionally we can consider this as a vector and consider each component of
  # the distribution being managed by the KF
  kf_d$D              <- rep (1 / nrow (X), nrow (X));
  kf_d$P_dist         <- 1.0;
  
  debug_info$D[[1]]   <- kf_d$D; #initial training distribution
  
  this_blend_err      <- sum (err_fun(kf_m$train_blend, Y, NULL) * (1/nrow (X)));
  
  
  ###################################
  #### Start ensemble iterations ####
  ###################################
  for (t in 1:max_iter)
  {
    # Print feedback
    if (feedback == TRUE) { cat ("\nIter = ", formatC (t, print_prec), ""); }
    
    ############################
    #### Time update kf-m   ####
    ############################
    # Based on present formulation, this is identity
    proc_noise <- 0;
    kf_m$train_blend <- kf_m$train_blend;
    kf_m$P[t]        <- kf_m$P[t] + proc_noise;

    
    #################################
    #### Measurement update kf-m ####
    #################################
    
    # Measurement process
    # Retry loop, if the error is more than a threshold then recompute
    dist_reset_flag <- FALSE;
    while (1)
    {
      # NOTE: Can lead to infinite loop
      resample_flag <- TRUE;
      while (resample_flag == TRUE)
      {
        bsamp <- sample (1:nrow (X), nrow (X), replace = TRUE, prob = kf_d$D / sum (kf_d$D));
        resample_flag <- sum (table(Y[bsamp]) == 0) != 0;
        # cat ("resample. \n");
        resample_flag <- FALSE;
      }
      # Using rpart
      kf_m$h_list[[t]] <- rpart (formula = Y ~ ., data = cbind (X, Y)[bsamp,], control = rpart_control);
      
      # Measurement state this_pred
      this_pred   <- predict (kf_m$h_list[[t]], X, type = "prob");
      this_pred   <- get_fixed_matrix (this_pred, levels (Y));
      
      # First transform the probabilities to classes, and then use these classes for the computation.
      if (blend_with_class == TRUE)
      {
        this_cls    <- factor (levels (Y)[apply (this_pred, 1, which.max)], levels = levels (Y));
        
        dummy_pred  <- matrix (0, nrow = nrow (this_pred), ncol = ncol (this_pred));
        colnames (dummy_pred) <- colnames (Y);
        for (i in 1:nrow(dummy_pred))
        {
          dummy_pred[i,this_cls[i]] <- 1;
        }
        
        this_pred <- dummy_pred;
      }
      # Now we have this_pred
      
      # For KF-m, calculate the measurement and the related error. This is a heuristic and can be computed in different ways.
      this_measurement <- (this_pred + kf_m$train_blend)/2;
      # this_measurement <- this_pred;
      
      # Get the error for "this_measurement"
      unwt_comp_err  <- err_fun (this_measurement, Y, NULL); # Find the per datapoint error. This is a vector.
      uniwt_comp_err <- unwt_comp_err * (1/nrow (X));        # Get a uniformly weighted version. This is a weighted vector.
      

      # Also just the prediction error is computed for the learned model in 
      # this iteration, used to reset weights for datapoints
      this_pred_err       <- err_fun (this_pred, Y, NULL);
      wtd_this_pred_err   <- this_pred_err * kf_d$D;
      uniwt_this_pred_err <- this_pred_err * (1/nrow (X));
      
      this_m_err     <- uniwt_comp_err;
      this_d_err     <- uniwt_comp_err;
      
      
      # Measurement error for the model KF-m. This is a heuristic and can be computed in different ways.
      kf_m$V[t]          <- sum (this_m_err);
      
      if (reset_dist == TRUE)
      {
        # Retry with uniform distribution if the error is more than (1 - 1/nclass)
        # Reset only if it was not reset in the immediately previous attempt. Saves from infinite loops.
        if ((sum (uniwt_this_pred_err) >= (1 - 1/length (levels(Y)))) && (dist_reset_flag == FALSE))
        {
          if (feedback == TRUE)
          {
            cat (", dreset = YES");
          }
          kf_d$D <- rep (1/nrow (X), nrow (X));
          kf_d$P_dist <- 1.0;
          
          dist_reset_flag <- TRUE;
          debug_info$reset_pt[t] <- dist_reset_flag;
          next;
        }
        else
        {
          if (dist_reset_flag == FALSE)
          {
            if (feedback == TRUE) 
            {
              cat (", dreset =  NO");
            }
          }
          break;
        }
      }
      else
      {
        if (feedback == TRUE) 
        {
          cat (", dreset =  NA");
        }
        break;
      }
    } # End of retry loop
    
    # Compute the Kalman gain for the KF-m
    kf_m$K[t]        <- kf_m$P[t] / (kf_m$P[t] + kf_m$V[t] + .Machine$double.xmin);
    # Blending the training predictions. This is not required for training, as we only need to store the kalman gains.
    # Update internal state for KF-m
    kf_m$train_blend <- kf_m$train_blend + kf_m$K[t] * (this_measurement - kf_m$train_blend);
    prev_blend_err   <- this_blend_err;
    
    # Update internal error for the KF-m
    P_t_pred <- kf_m$P[t] - kf_m$P[t] * kf_m$K[t];
    
    kf_m$P[t+1] <- P_t_pred;
    
    # Compute the actual error for the internal state
    debug_info$state_err[[t]]   <- err_fun(kf_m$train_blend, Y, NULL);
    this_blend_err              <- sum (err_fun(kf_m$train_blend, Y, NULL) * (1/nrow (X)));
    debug_info$blend_err_arr[t] <- this_blend_err;
    
    train_pred_cls               <- factor (colnames (kf_m$train_blend)[apply (kf_m$train_blend, 1, which.max)], levels = levels (Y));
    debug_info$train_accuracy[t] <- confusion_metric (train_pred_cls, Y, "A");
    debug_info$train_f[t]        <- confusion_metric (train_pred_cls, Y, "F");
    
    
    ##################################################################################################
    
    
    ############################
    #### Time update kf-d   ####
    ############################
    # Based on present formulation, this is identity
    proc_noise <- 0;
    kf_d$D           <- kf_d$D;
    kf_d$P_dist      <- kf_d$P_dist + proc_noise;
    
    
    #################################
    #### Measurement update kf-d ####
    #################################
    # Measurement of state vector of the distribution KF-d
    dtemp           <- unwt_comp_err;
    dtemp[dtemp==0] <- -1;
    kf_d$D_t_obs    <- kf_d$D * exp (dtemp);
    kf_d$D_t_obs    <- kf_d$D_t_obs / sum (kf_d$D_t_obs);
    
    # Measurement error V_dist for the distribution KF-d
    kf_d$V_dist        <- sum (this_d_err);
 
    # Compute the Kalman gain for the distribution KF-d
    kf_d$K_dist[[t]]   <- kf_d$P_dist / (kf_d$P_dist + kf_d$V_dist + .Machine$double.xmin);

    # Update iternal state for the distribition KF-d
    kf_d$D             <- kf_d$D + kf_d$K_dist[[t]] * (kf_d$D_t_obs - kf_d$D); # Fishy
      
    # Update internal error for the distribution KF
    P_dist_old <- kf_d$P_dist;
    kf_d$P_dist      <- kf_d$P_dist - kf_d$P_dist * kf_d$K_dist[[t]];
    
    # Stopping is both the distribution measurement and process noises are 0.
    # if ((kf_d$P_dist == 0) && (kf_d$V_dist == 0))
    # {
    #   max_iter <- t;
    #   break;
    # }

    # Compute the change in the internal state actual error
    debug_info$blend_err_delta[t] <- prev_blend_err - this_blend_err;

    # Error feedback
    if (feedback == TRUE) 
    { 
      cat (", blend_err = ", formatC (this_blend_err, digits = print_prec, format = "f"), 
           "diff (", formatC (debug_info$blend_err_delta[t], digits = print_prec, format = "f"), ")", 
           "V_t = ", formatC (kf_m$V[t], digits = print_prec, format = "f"), ", ",
           "P_t = ", formatC (kf_m$P[t], digits = print_prec, format = "f"), ", ",
           "K_t = ", formatC (kf_m$K[t], digits = print_prec, format = "f"), ""); 
    }
  
    
    # Normalise distribution
    kf_d$D                   <- kf_d$D / sum (kf_d$D);
    if (feedback == TRUE)
    {
      cat (", ", "V_d = ", kf_d$V_dist, ", P_d = ", P_dist_old, "K_d = ", kf_d$K_dist[[t]]);
    }
    
    debug_info$D[[t]]        <- kf_d$D;
    
    #if ((early_stop == TRUE) && (this_blend_err < 10e-10))
    if ((early_stop == TRUE) && (kf_m$K[t] == 0)) # Similar in this case. Although practically it is better to stop on validation set.
    {
      max_iter <- t;
      break;
    }
  }

  # Pack
  retmodel$kf_m             <- kf_m;
  retmodel$kf_d             <- kf_d;
  retmodel$debug_info       <- debug_info;
  retmodel$max_iter         <- max_iter;
  retmodel$cls_lvls         <- levels (Y);
  retmodel$blend_with_class <- blend_with_class;
  
  # Stamp
  class (retmodel)  <- "kfhe_m";
  
  # Send
  return (retmodel);
}



###########################################
####              Predict             #####
####################################################################################################
#### Function  : predict.kfhe_m                                                                 ####
#### Arguments :                                                                                #### 
####             model            : An object of type 'kfhe_m' returned by 'kfhe_train'         ####
####             X                : A matrix representing the datapoints to predict             ####
####             type             : If "prob", return the predicted per class scores            ####
####                                If "class", return the predicted class                      ####
####             feedback         : Print the states of the parameters while training           ####
####                                                                                            ####
#### Return    : Prediced scores or classes                                                     ####
####################################################################################################
predict.kfhe_m <- function (model, X, type = "prob", feedback = TRUE, test_Y = NULL)
{
  test_blend <- predict (model$kf_m$init_h, X, type = "prob");
  
  for (t in 1:model$max_iter)
  {
    if (feedback == TRUE) { cat ("\rPred iter = ", t, "      "); }
    this_pred  <- predict (model$kf_m$h_list[[t]], X, type = "prob");
    this_pred  <- get_fixed_matrix (this_pred, model$cls_lvls);
    
    # First transform the probabilities to classes, and then use these classes for the computation.
    if (model$blend_with_class == TRUE)
    {
      this_cls    <- factor (model$cls_lvls[apply (this_pred, 1, which.max)], levels = model$cls_lvls);
      
      dummy_pred  <- matrix (0, nrow = nrow (this_pred), ncol = ncol (this_pred));
      colnames (dummy_pred) <- model$cls_lvls;
      for (i in 1:nrow(dummy_pred))
      {
        dummy_pred[i,this_cls[i]] <- 1;
      }
      this_pred <- dummy_pred;
    }
    
    # This should be the same heuristic which is used during the training.
    this_measurement <- (test_blend + this_pred)/2;
    test_blend <- test_blend + model$kf_m$K[t] * (this_measurement - test_blend);
  }
  
  if (type == "class")
  {
    this_pred  <- apply (test_blend, 1, which.max);
    pred_cls   <- factor (colnames (test_blend)[this_pred], levels = model$cls_lvls);
    test_blend <- pred_cls;
  }
  
  return (test_blend);
}


###########################################
####         Utility functions         ####
###########################################

# In case of missing levels in the predicted thing, the number of columns will be less
# To fix that, we make a template and copy the result.
# This is real pain. Double check if things are good.
get_fixed_matrix <- function (mat_to_fix, target_levels)
{
  template              <- as.data.frame (matrix (0, nrow = nrow (mat_to_fix), ncol = length (target_levels))); 
  colnames (template)   <- target_levels;
  temp_names            <- colnames (mat_to_fix);
  template[,temp_names] <- mat_to_fix;
  return (template);
}

err_fun <- function (pred, target, wt)
{
  if (!is.data.frame(target))
  {
    target         <- as.data.frame (target);
    one_hot_target <- model.matrix (~ ., data = target, contrasts.arg = lapply (target, contrasts, contrasts = FALSE));
    one_hot_target <- one_hot_target[,-1];
    target         <- one_hot_target;
  }
  else if (ncol (target) > 1)
  {
    
  }
  if (is.null (wt) == TRUE) # Just don't weight
  {
    wt <- 1;
  }
  final_err <- wt * (apply (pred, 1, which.max) != apply (target, 1, which.max));
  
  return (final_err);
}

get_one_hot <- function (classes)
{
  classes <- as.data.frame (classes);
  one_hot <- model.matrix (~ ., data = classes, contrasts.arg = lapply (classes, contrasts, contrasts = FALSE));
  one_hot <- one_hot[,-1];
  
  return (one_hot);
}


confusion_metric <- function (pred_vec, target_vec, metric)
{
  pred_one_hot_mat   <- to_one_hot (pred_vec);
  colnames (pred_one_hot_mat)   <- LETTERS[1:length(levels(pred_vec))];
  
  target_one_hot_mat <- to_one_hot (target_vec);
  colnames (target_one_hot_mat) <- LETTERS[1:length(levels(target_vec))];
  
  # NOTE: The returned values are all *macro average*
  fobj <- F.measure.single.over.classes (target_one_hot_mat, pred_one_hot_mat);
  return (fobj$average[metric]);
}


to_one_hot <- function (one_cool_vec)
{
  total_lvls <- length (levels(one_cool_vec));
  one_cool_vec <- as.numeric (one_cool_vec);
  one_hot_matrix <- matrix (0, nrow = length (one_cool_vec), ncol = total_lvls, byrow = TRUE);
  for (i in 1:length (one_cool_vec))
  {
    one_hot_matrix[i,one_cool_vec[i]] <- 1;
  }
  
  return (one_hot_matrix);
}

induce_class_noise <- function (X, nfrac = 0.05, cidx = ncol (X))
{
  stopifnot ((cidx <= ncol (X)) && (cidx > 0));
  stopifnot ((nfrac >= 0) && (nfrac <= 1));
  
  if (nfrac == 0)
  {
    cat ("No noise induced\n");
    return (X);
  }
  
  cls_vals <- unique (X[,cidx]);
  
  noise_tgt_idx <- sample (1:nrow (X), round (nrow (X) * nfrac));
  
  for (this_tgt_idx in noise_tgt_idx)
  {
    this_cls_sample_pool <- cls_vals[(cls_vals != X[this_tgt_idx,cidx])];
    this_noise_cls       <- sample (this_cls_sample_pool, 1);
    X[this_tgt_idx,cidx] <- this_noise_cls;
  }
  
  return (X)
}
