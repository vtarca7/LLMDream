# Load necessary libraries
library(tidyverse)
library(caret)       # For data manipulation and training models
library(pROC)        # For ROC analysis

# Step 1: Read the datasets
training_abundance <- read.csv("training_species_abundance.csv")
validation_abundance <- read.csv("validation_species_abundance.csv")

training_metadata <- read.csv("training_metadata.csv")
validation_metadata <- read.csv("validation_metadata.csv")

# Step 2: Filter data for each participant_id to keep only the highest collect_wk < 32
filter_data <- function(abundance, metadata) {
  # Combine abundance and metadata by specimen
  combined <- full_join(metadata, abundance, by = "specimen")
  
  # Get the maximum collect_wk < 32 for each participant_id
  max_collect_wk <- combined %>%
    filter(collect_wk < 32) %>%
    group_by(participant_id) %>%
    summarize(max_collect_wk = max(collect_wk, na.rm = TRUE), .groups = "drop") %>%
    rename(collect_wk = max_collect_wk)
  
  # Filter rows with the highest collect_wk for each participant_id
  filtered <- combined %>%
    inner_join(max_collect_wk, by = "participant_id")
  
  return(filtered)
}

training_filtered <- filter_data(training_abundance, training_metadata)
validation_filtered <- filter_data(validation_abundance, validation_metadata)

# Step 3: Align features between training and validation sets
common_features <- intersect(names(training_filtered), names(validation_filtered))
common_features <- common_features[!common_features %in% c("participant_id", "specimen", "collect_wk")]

training_features <- training_filtered %>%
  select(all_of(c("PTB", "EarlyPTB", common_features)))

validation_features <- validation_filtered %>%
  select(all_of(c("PTB", "EarlyPTB", common_features)))

# Step 4: Prepare binary outcomes
training_labels_PTB <- as.numeric(training_filtered$delivery_wk < 37)
training_labels_EarlyPTB <- as.numeric(training_filtered$delivery_wk < 32)

validation_labels_PTB <- as.numeric(validation_filtered$delivery_wk < 37)
validation_labels_EarlyPTB <- as.numeric(validation_filtered$delivery_wk < 32)

# Step 5: Fit a predictive model using the filtered data
model <- train(x = training_features[common_features], 
               y = training_labels_PTB,
               method = "glm",
               family = binomial())

# Step 6: Evaluate the model's performance on PTB outcome
prob_predictions_PTB <- predict(model, newdata = validation_features[common_features], type = "prob")[,2]
roc_curve_PTB <- roc(validation_labels_PTB, prob_predictions_PTB)
auc_value_PTB <- auc(roc_curve_PTB)

cat("AUC for PTB on test set: ", auc_value_PTB, "\n")

# Plot ROC curve
plot(roc_curve_PTB, main = "ROC Curve for PTB Prediction")
abline(a = 0, b = 1, lty = 2, col = "red") # Diagonal reference line

# Repeat the analysis for EarlyPTB outcome
training_filtered_Early <- filter_data(training_abundance, training_metadata) %>%
  filter(collect_wk < 28)

validation_filtered_Early <- filter_data(validation_abundance, validation_metadata) %>%
  filter(collect_wk < 28)

common_features_early <- intersect(names(training_filtered_Early), names(validation_filtered_Early))
common_features_early <- common_features_early[!common_features_early %in% c("participant_id", "specimen", "collect_wk")]

training_features_early <- training_filtered_Early %>%
  select(all_of(c("PTB", "EarlyPTB", common_features_early)))

validation_features_early <- validation_filtered_Early %>%
  select(all_of(c("PTB", "EarlyPTB", common_features_early)))

model_early <- train(x = training_features_early[common_features_early], 
                     y = training_labels_EarlyPTB,
                     method = "glm",
                     family = binomial())

prob_predictions_EarlyPTB <- predict(model_early, newdata = validation_features_early[common_features_early], type = "prob")[,2]
roc_curve_EarlyPTB <- roc(validation_labels_EarlyPTB, prob_predictions_EarlyPTB)
auc_value_EarlyPTB <- auc(roc_curve_EarlyPTB)

cat("AUC for EarlyPTB on test set: ", auc_value_EarlyPTB, "\n")

# Plot ROC curve
plot(roc_curve_EarlyPTB, main = "ROC Curve for EarlyPTB Prediction")
abline(a = 0, b = 1, lty = 2, col = "red") # Diagonal reference line
