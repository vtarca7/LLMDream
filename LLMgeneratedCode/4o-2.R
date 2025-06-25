library(tidyverse)
library(glmnet)
library(pROC)

# Load data
training_features <- read.csv("training_species_abundance.csv")
validation_features <- read.csv("validation_species_abundance.csv")
training_metadata <- read.csv("training_metadata.csv")
validation_metadata <- read.csv("validation_metadata.csv")

# Ensure common features
common_features <- intersect(names(training_features), names(validation_features))
common_features <- setdiff(common_features, "specimen")

# Filter features
training_features <- training_features[, c("specimen", common_features)]
validation_features <- validation_features[, c("specimen", common_features)]

# Merge metadata
training_data <- merge(training_metadata, training_features, by = "specimen")
validation_data <- merge(validation_metadata, validation_features, by = "specimen")

# Function to select last specimen before a given collect_wk for each participant
filter_last_specimen <- function(data, wk_threshold) {
  data %>%
    filter(collect_wk < wk_threshold) %>%
    group_by(participant_id) %>%
    filter(collect_wk == max(collect_wk)) %>%
    ungroup()
}

# Prepare datasets for PTB (collect_wk < 32)
training_ptb <- filter_last_specimen(training_data, 32)
validation_ptb <- filter_last_specimen(validation_data, 32)

# Define PTB outcome (delivery_wk < 37)
training_ptb$PTB <- as.factor(training_ptb$delivery_wk < 37)
validation_ptb$PTB <- as.factor(validation_ptb$delivery_wk < 37)

# Prepare model matrices
X_train <- as.matrix(training_ptb[, common_features])
y_train <- as.numeric(training_ptb$PTB)  # Convert to numeric for glmnet
X_test <- as.matrix(validation_ptb[, common_features])
y_test <- as.numeric(validation_ptb$PTB)

set.seed(123)
# Fit LASSO logistic regression
cv_fit <- cv.glmnet(X_train, y_train, family = "binomial", alpha = 1)

# Predict on test set
pred_probs <- predict(cv_fit, newx = X_test, s = "lambda.min", type = "response")

# Compute AUC
roc_obj <- roc(y_test, pred_probs)
auc_value <- auc(roc_obj)
print(paste("AUC for PTB:", auc_value))
write.csv(data.frame(y_test=y_test,y_pred=as.numeric(pred_probs)),file="./out2/4o-2A.R.csv")

# Plot ROC curve
plot(roc_obj, col = "blue", main = "ROC Curve for PTB")

# Repeat for EarlyPTB (collect_wk < 28, delivery_wk < 32)
training_earlyptb <- filter_last_specimen(training_data, 28)
validation_earlyptb <- filter_last_specimen(validation_data, 28)

training_earlyptb$EarlyPTB <- as.factor(training_earlyptb$delivery_wk < 32)
validation_earlyptb$EarlyPTB <- as.factor(validation_earlyptb$delivery_wk < 32)

X_train_early <- as.matrix(training_earlyptb[, common_features])
y_train_early <- as.numeric(training_earlyptb$EarlyPTB)
X_test_early <- as.matrix(validation_earlyptb[, common_features])
y_test_early <- as.numeric(validation_earlyptb$EarlyPTB)

cv_fit_early <- cv.glmnet(X_train_early, y_train_early, family = "binomial", alpha = 1)

pred_probs_early <- predict(cv_fit_early, newx = X_test_early, s = "lambda.min", type = "response")

roc_obj_early <- roc(y_test_early, pred_probs_early)
auc_value_early <- auc(roc_obj_early)
print(paste("AUC for EarlyPTB:", auc_value_early))
write.csv(data.frame(y_test=y_test_early,y_pred=as.numeric(pred_probs_early)),file="./out2/4o-2B.R.csv")

plot(roc_obj_early, col = "red", main = "ROC Curve for EarlyPTB")
