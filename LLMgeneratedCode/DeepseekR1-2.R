library(tidyverse)
library(glmnet)
library(pROC)

# Function to process data and fit model
analyze_outcome <- function(collect_wk_thresh, delivery_wk_thresh, outcome_name) {
  # Process training metadata
  train_meta_processed <- read.csv("training_metadata.csv") %>%
    filter(collect_wk < collect_wk_thresh) %>%
    group_by(participant_id) %>%
    slice_max(collect_wk, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(outcome = delivery_wk < delivery_wk_thresh)
  
  # Process test metadata
  test_meta_processed <- read.csv("validation_metadata.csv") %>%
    filter(collect_wk < collect_wk_thresh) %>%
    group_by(participant_id) %>%
    slice_max(collect_wk, n = 1, with_ties = FALSE) %>%
    ungroup() %>%
    mutate(outcome = delivery_wk < delivery_wk_thresh)
  
  # Prepare feature data
  train_features <- read.csv("training_species_abundance.csv")
  test_features <- read.csv("validation_species_abundance.csv")
  
  # Get common features
  common_features <- intersect(colnames(train_features), colnames(test_features)) %>%
    setdiff("specimen")
  
  # Create feature matrices
  train_feat_matrix <- train_features %>%
    filter(specimen %in% train_meta_processed$specimen) %>%
    select(all_of(common_features)) %>%
    as.matrix()
  
  test_feat_matrix <- test_features %>%
    filter(specimen %in% test_meta_processed$specimen) %>%
    select(all_of(common_features)) %>%
    as.matrix()
  
  # Get outcomes
  train_outcome <- train_meta_processed$outcome
  test_outcome <- test_meta_processed$outcome
  
  # Train model
  set.seed(123)
  cv_model <- cv.glmnet(x = train_feat_matrix, y = train_outcome,
                        family = "binomial", type.measure = "auc")
  
  # Predict on test set
  pred <- predict(cv_model, newx = test_feat_matrix,
                  s = "lambda.min", type = "response")
  
  # Calculate results
  roc_obj <- roc(test_outcome, as.vector(pred))
  auc_value <- auc(roc_obj)
  
  # Print and plot results
  cat("AUC for", outcome_name, ":", auc_value, "\n")
  plot(roc_obj, main = paste("ROC Curve for", outcome_name))
  if(outcome_name=="PTB"){
  write.csv(data.frame(y_test=test_outcome,y_pred=as.numeric(pred)),file="./out2/DeepseekR1-2A.R.csv")
  }else{
    write.csv(data.frame(y_test=test_outcome,y_pred=as.numeric(pred)),file="./out2/DeepseekR1-2B.R.csv")
  }
  
}

# Analyze PTB (<32 weeks collection, <37 weeks delivery)
analyze_outcome(32, 37, "PTB")

# Analyze EarlyPTB (<28 weeks collection, <32 weeks delivery)
analyze_outcome(28, 32, "EarlyPTB")
