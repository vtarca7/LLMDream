# Load necessary libraries
library(dplyr)
library(glmnet)
library(pROC)

# --- PTB Prediction ---

# 1. Data Loading and Preprocessing for PTB

# Load data
train_features_ptb_raw <- read.csv("training_species_abundance.csv")
validation_features_ptb_raw <- read.csv("validation_species_abundance.csv")
train_metadata_ptb <- read.csv("training_metadata.csv")
validation_metadata_ptb <- read.csv("validation_metadata.csv")

# Merge features and metadata for training set
train_data_ptb_all <- merge(train_metadata_ptb, train_features_ptb_raw, by = "specimen")
validation_data_ptb_all <- merge(validation_metadata_ptb, validation_features_ptb_raw, by = "specimen")

# Select last specimen with collect_wk < 32 for each participant in training set
train_data_ptb <- train_data_ptb_all %>%
  filter(collect_wk < 32) %>%
  group_by(participant_id) %>%
  filter(collect_wk == max(collect_wk)) %>%
  ungroup()

# Select last specimen with collect_wk < 32 for each participant in validation set
validation_data_ptb <- validation_data_ptb_all %>%
  filter(collect_wk < 32) %>%
  group_by(participant_id) %>%
  filter(collect_wk == max(collect_wk)) %>%
  ungroup()

# Define PTB outcome (delivery_wk < 37)
train_data_ptb <- train_data_ptb %>%
  mutate(PTB = ifelse(delivery_wk < 37, 1, 0))
validation_data_ptb <- validation_data_ptb %>%
  mutate(PTB = ifelse(delivery_wk < 37, 1, 0))

# Identify common species features
train_species_ptb <- train_features_ptb_raw %>% select(-specimen) %>% colnames()
validation_species_ptb <- validation_features_ptb_raw %>% select(-specimen) %>% colnames()
common_species_ptb <- intersect(train_species_ptb, validation_species_ptb)

# Prepare feature matrices for PTB prediction
train_features_matrix_ptb <- as.matrix(train_data_ptb[, common_species_ptb])
validation_features_matrix_ptb <- as.matrix(validation_data_ptb[, common_species_ptb])
train_outcome_ptb <- train_data_ptb$PTB
validation_outcome_ptb <- validation_data_ptb$PTB

# 2. Model Fitting for PTB (Logistic Regression with LASSO for feature selection)
set.seed(123) # for reproducibility
model_ptb <- cv.glmnet(train_features_matrix_ptb, train_outcome_ptb, family = "binomial", alpha = 1, nfolds = 10) # alpha=1 for LASSO

# 3. Model Evaluation for PTB
predictions_ptb <- predict(model_ptb, newx = validation_features_matrix_ptb, s = "lambda.min", type = "response")
roc_obj_ptb <- roc(validation_outcome_ptb, as.numeric(predictions_ptb))
auc_roc_ptb <- auc(roc_obj_ptb)
print(paste("AUC ROC for PTB prediction:", round(auc_roc_ptb, 3)))

write.csv(data.frame(y_test=validation_outcome_ptb,y_pred=as.numeric(predictions_ptb)),file="./out2/gemini-2A.R.csv")

# Plot ROC curve for PTB
plot(roc_obj_ptb, main = "ROC Curve for PTB Prediction", col = "blue", print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2), auc.polygon.col = "#D9F0A3")


# --- EarlyPTB Prediction ---

# 1. Data Loading and Preprocessing for EarlyPTB

# Use the same raw data, just need to re-process based on new criteria

# Select last specimen with collect_wk < 28 for each participant in training set
train_data_earlyptb <- train_data_ptb_all %>% # using _all because reloading is redundant
  filter(collect_wk < 28) %>%
  group_by(participant_id) %>%
  filter(collect_wk == max(collect_wk)) %>%
  ungroup()

# Select last specimen with collect_wk < 28 for each participant in validation set
validation_data_earlyptb <- validation_data_ptb_all %>% # using _all because reloading is redundant
  filter(collect_wk < 28) %>%
  group_by(participant_id) %>%
  filter(collect_wk == max(collect_wk)) %>%
  ungroup()

# Define EarlyPTB outcome (delivery_wk < 32)
train_data_earlyptb <- train_data_earlyptb %>%
  mutate(EarlyPTB = ifelse(delivery_wk < 32, 1, 0))
validation_data_earlyptb <- validation_data_earlyptb %>%
  mutate(EarlyPTB = ifelse(delivery_wk < 32, 1, 0))

# Ensure same common species list as for PTB is used for consistency and as per instructions (common between training and validation for PTB).
# If the instruction meant common species for *each* analysis separately, then re-calculate it.
# For now, assuming common species from PTB analysis should be used.
common_species_earlyptb <- common_species_ptb # Re-use common species from PTB analysis

# Prepare feature matrices for EarlyPTB prediction
train_features_matrix_earlyptb <- as.matrix(train_data_earlyptb[, common_species_earlyptb])
validation_features_matrix_earlyptb <- as.matrix(validation_data_earlyptb[, common_species_earlyptb])
train_outcome_earlyptb <- train_data_earlyptb$EarlyPTB
validation_outcome_earlyptb <- validation_data_earlyptb$EarlyPTB

# Remove rows with NA features, as caused by filtering and species columns not being in the filtered data frames.
# It's safer to ensure no NAs by imputation if necessary, or to filter out rows with NA values *before* model fitting.
# In this case, let's check and remove rows with NA features after subsetting.

na_train_rows_earlyptb <- apply(train_features_matrix_earlyptb, 1, function(x) any(is.na(x)))
train_features_matrix_earlyptb_clean <- train_features_matrix_earlyptb[!na_train_rows_earlyptb, ]
train_outcome_earlyptb_clean <- train_outcome_earlyptb[!na_train_rows_earlyptb]

na_validation_rows_earlyptb <- apply(validation_features_matrix_earlyptb, 1, function(x) any(is.na(x)))
validation_features_matrix_earlyptb_clean <- validation_features_matrix_earlyptb[!na_validation_rows_earlyptb, ]
validation_outcome_earlyptb_clean <- validation_outcome_earlyptb[!na_validation_rows_earlyptb]

# Check if there are any rows left after NA removal
if(nrow(train_features_matrix_earlyptb_clean) == 0 || nrow(validation_features_matrix_earlyptb_clean) == 0) {
  stop("No data left after removing rows with NA features for EarlyPTB. Check data processing steps.")
}


# 2. Model Fitting for EarlyPTB (Logistic Regression with LASSO)
set.seed(123) # for reproducibility
model_earlyptb <- cv.glmnet(train_features_matrix_earlyptb_clean, train_outcome_earlyptb_clean, family = "binomial", alpha = 1, nfolds = 10) # alpha=1 for LASSO

# 3. Model Evaluation for EarlyPTB
predictions_earlyptb <- predict(model_earlyptb, newx = validation_features_matrix_earlyptb_clean, s = "lambda.min", type = "response")
roc_obj_earlyptb <- roc(validation_outcome_earlyptb_clean, as.numeric(predictions_earlyptb))
auc_roc_earlyptb <- auc(roc_obj_earlyptb)
print(paste("AUC ROC for EarlyPTB prediction:", round(auc_roc_earlyptb, 3)))
write.csv(data.frame(y_test=validation_outcome_earlyptb_clean,y_pred=as.numeric(predictions_earlyptb)),file="./out2/gemini-2B.R.csv")

# Plot ROC curve for EarlyPTB
plot(roc_obj_earlyptb, main = "ROC Curve for EarlyPTB Prediction", col = "red", print.auc = TRUE, auc.polygon = TRUE, grid = c(0.1, 0.2), auc.polygon.col = "#FFDAB9")
