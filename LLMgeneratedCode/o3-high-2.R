# Load required packages
library(dplyr)
library(readr)
library(glmnet)
library(pROC)

# ---------------------------
# 1. Read in Data
# ---------------------------

# Read species abundance data
train_features <- read.csv("training_species_abundance.csv", stringsAsFactors = FALSE)
test_features  <- read.csv("validation_species_abundance.csv", stringsAsFactors = FALSE)

# Read metadata
train_metadata <- read.csv("training_metadata.csv", stringsAsFactors = FALSE)
test_metadata  <- read.csv("validation_metadata.csv", stringsAsFactors = FALSE)

# ---------------------------
# 2. Merge Species and Metadata by 'specimen'
# ---------------------------

train_data <- inner_join(train_metadata, train_features, by = "specimen")
test_data  <- inner_join(test_metadata, test_features, by = "specimen")

# ---------------------------
# 3. Identify Common Microbial Features
# ---------------------------
# Assume that the microbial features are all columns in the species abundance files except "specimen"

feature_names_train <- setdiff(names(train_features), "specimen")
feature_names_test  <- setdiff(names(test_features), "specimen")
common_features   <- intersect(feature_names_train, feature_names_test)

# ---------------------------
# Analysis 1: Predicting PTB (delivery_wk < 37) using data with collect_wk < 32
# ---------------------------

# Filter to only include specimens with collect_wk < 32 and then select the one with highest collect_wk per participant
train_ptb <- train_data %>%
  filter(collect_wk < 32) %>%
  group_by(participant_id) %>%
  filter(collect_wk == max(collect_wk)) %>%
  ungroup() %>%
  # Define PTB outcome: 1 if delivery_wk < 37, else 0
  mutate(PTB = ifelse(delivery_wk < 37, 1, 0))

test_ptb <- test_data %>%
  filter(collect_wk < 32) %>%
  group_by(participant_id) %>%
  filter(collect_wk == max(collect_wk)) %>%
  ungroup() %>%
  mutate(PTB = ifelse(delivery_wk < 37, 1, 0))

# Extract feature matrices (only common features) and outcome
X_train <- as.matrix(train_ptb[, common_features])
y_train <- train_ptb$PTB

X_test <- as.matrix(test_ptb[, common_features])
y_test <- test_ptb$PTB

# ---------------------------
# Fit Predictive Model using glmnet (LASSO Logistic Regression)
# ---------------------------
set.seed(123)  # For reproducibility
cv_fit <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 1)

# Make predictions on the test set
pred_probs <- predict(cv_fit, newx = X_test, type = "response", s = "lambda.min")

# ---------------------------
# Compute and Print AUC, and Plot the ROC Curve for PTB Model
# ---------------------------
roc_obj <- roc(y_test, as.vector(pred_probs))
auc_val <- auc(roc_obj)
cat("AUC ROC for PTB model (delivery_wk < 37):", auc_val, "\n")
plot(roc_obj, main = "ROC Curve for PTB Model (delivery_wk < 37)")
write.csv(data.frame(y_test=y_test,y_pred=as.numeric(pred_probs)),file="./out2/o3-high-2A.R.csv")
# ---------------------------
# Analysis 2: Predicting EarlyPTB (delivery_wk < 32) using data with collect_wk < 28
# ---------------------------

# Filter to only include specimens with collect_wk < 28 and then select the one with highest collect_wk per participant
train_ep <- train_data %>%
  filter(collect_wk < 28) %>%
  group_by(participant_id) %>%
  filter(collect_wk == max(collect_wk)) %>%
  ungroup() %>%
  # Define EarlyPTB outcome: 1 if delivery_wk < 32, else 0
  mutate(EarlyPTB = ifelse(delivery_wk < 32, 1, 0))

test_ep <- test_data %>%
  filter(collect_wk < 28) %>%
  group_by(participant_id) %>%
  filter(collect_wk == max(collect_wk)) %>%
  ungroup() %>%
  mutate(EarlyPTB = ifelse(delivery_wk < 32, 1, 0))

# Extract feature matrices and outcome for EarlyPTB analysis
X_train_ep <- as.matrix(train_ep[, common_features])
y_train_ep <- train_ep$EarlyPTB

X_test_ep <- as.matrix(test_ep[, common_features])
y_test_ep <- test_ep$EarlyPTB

# ---------------------------
# Fit Predictive Model for EarlyPTB
# ---------------------------
set.seed(123)  # Ensure reproducibility
cv_fit_ep <- cv.glmnet(X_train_ep, y_train_ep, family = "binomial", alpha = 1)

# Make predictions on the test set
pred_probs_ep <- predict(cv_fit_ep, newx = X_test_ep, type = "response", s = "lambda.min")

# ---------------------------
# Compute and Print AUC, and Plot the ROC Curve for EarlyPTB Model
# ---------------------------
roc_obj_ep <- roc(y_test_ep, as.vector(pred_probs_ep))
auc_val_ep <- auc(roc_obj_ep)
cat("AUC ROC for EarlyPTB model (delivery_wk < 32):", auc_val_ep, "\n")
plot(roc_obj_ep, main = "ROC Curve for EarlyPTB Model (delivery_wk < 32)")
write.csv(data.frame(y_test=y_test_ep,y_pred=as.numeric(pred_probs_ep)),file="./out2/o3-high-2B.R.csv")

