# Load necessary libraries
library(dplyr)
library(tidyr)
library(caret)
library(pROC)

# Load datasets
training_species_abundance <- read.csv("training_species_abundance.csv")
validation_species_abundance <- read.csv("validation_species_abundance.csv")
training_metadata <- read.csv("training_metadata.csv")
validation_metadata <- read.csv("validation_metadata.csv")

# Merge metadata with abundance data
training_data <- merge(training_species_abundance, training_metadata, by = "specimen")
validation_data <- merge(validation_species_abundance, validation_metadata, by = "specimen")

# Function to select the latest specimen for each participant_id with collect_wk < 32
select_latest_specimen <- function(data, max_collect_wk) {
  data %>%
    group_by(participant_id) %>%
    filter(collect_wk < max_collect_wk) %>%
    arrange(desc(collect_wk)) %>%
    slice(1)
}

# Select latest specimen for each participant in training and validation sets
training_latest <- select_latest_specimen(training_data, 32)
validation_latest <- select_latest_specimen(validation_data, 32)

# Define outcome variable PTB (delivery_wk < 37)
training_latest$PTB <- ifelse(training_latest$delivery_wk < 37, 1, 0)
validation_latest$PTB <- ifelse(validation_latest$delivery_wk < 37, 1, 0)

# Define outcome variable EarlyPTB (delivery_wk < 32)
training_latest$EarlyPTB <- ifelse(training_latest$delivery_wk < 32, 1, 0)
validation_latest$EarlyPTB <- ifelse(validation_latest$delivery_wk < 32, 1, 0)

# Ensure common features between training and test sets
common_features <- intersect(names(training_latest)[-c(1, 2, 3, 4, 5)], names(validation_latest)[-c(1, 2, 3, 4, 5)])
training_subset <- training_latest[, c("specimen", "participant_id", common_features, "PTB", "EarlyPTB")]
validation_subset <- validation_latest[, c("specimen", "participant_id", common_features, "PTB", "EarlyPTB")]

# Fit model for PTB
set.seed(123)
model_PTB <- train(PTB ~ ., data = training_subset, method = "glm", family = binomial())

# Predict on test set and calculate AUC-ROC
predictions_PTB <- predict(model_PTB, newdata = validation_subset, type = "prob")[, 2]
auc_PTB <- roc(validation_subset$PTB, predictions_PTB)$auc

# Plot ROC curve for PTB
plot(roc(validation_subset$PTB, predictions_PTB), main = paste("ROC Curve (AUC =", round(auc_PTB, 3), ")"))

# Fit model for EarlyPTB
model_EarlyPTB <- train(EarlyPTB ~ ., data = training_subset, method = "glm", family = binomial())

# Predict on test set and calculate AUC-ROC
predictions_EarlyPTB <- predict(model_EarlyPTB, newdata = validation_subset, type = "prob")[, 2]
auc_EarlyPTB <- roc(validation_subset$EarlyPTB, predictions_EarlyPTB)$auc

# Plot ROC curve for EarlyPTB
plot(roc(validation_subset$EarlyPTB, predictions_EarlyPTB), main = paste("ROC Curve (AUC =", round(auc_EarlyPTB, 3), ")"))
