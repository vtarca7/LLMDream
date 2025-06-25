# Read necessary libraries
library(dplyr)
library(randomForest)

# Read data
train_species <- read.csv("training_species_abundance.csv")
val_species <- read.csv("validation_species_abundance.csv")

merged_species <- rbind(train_species, val_species)

train_metadata <- read.csv("training_metadata.csv")
val_metadata <- read.csv("validation_metadata.csv")

merged_metadata <- rbind(train_metadata, val_metadata)

# Find common species between training and validation
common_species <- intersect(unique(train_species$Species), unique(val_species$Species))

# Prepare data for PTB model
ptb_data <- merged_species %>%
  group_by(participant_id) %>% # Assuming each row is a participant's sample
  arrange(desc(collect_wk)) %>% 
  filter(collect_wk < 32) %>% 
  slice(1) %>% 
  filter(Species %in% common_species)

# Merge with metadata to get outcome variable for PTB
ptb_merged <- merge(ptb_data, merged_metadata, by = c('participant_id', 'specimen'))

ptb_merged$PTB_outcome <- ifelse(ptb_merged$delivery_wk < 37, 1, 0)

# Split data into training and test sets using stratified sampling
set.seed(42)
train_index <- sample(nrow(ptb_merged), size = floor(0.7 * nrow(ptb_merged)))

train_ptb <- ptb_merged[train_index, ]
test_ptb <- ptb_merged[-train_index, ]

# Fit a random forest model for PTB
ptb_model <- train(
  PTB_outcome ~ . - participant_id - delivery_wk,
  data = train_ptb,
  method = "rf",
  trControl = trainControl(method = "none") # Simplified validation
)

# Predict probabilities on test set
test_ptb$PTB_pred <- predict(ptb_model, newdata = test_ptb, type = "prob")[,2]

# Compute AUC-ROC
library(pROC)
roc_obj <- roc(test_ptb$PTB_outcome, test_ptb$PTB_pred)
ptb_auc <- auc(roc_obj)

cat("AUC-ROC for PTB prediction:", ptb_auc, "\n")

# Repeat similar steps for EarlyPTB model

train_species_early <- merged_species %>%
  group_by(participant_id) %>% 
  arrange(desc(collect_wk)) %>% 
  filter(collect_wk < 28) %>% 
  slice(1) %>% 
  filter(Species %in% common_species)

# Merge with metadata for EarlyPTB
train_data_early <- merge(train_species_early, merged_metadata, by = c('participant_id', 'specimen'))

train_data_early$EarlyPTB_outcome <- ifelse(train_data_early$delivery_wk < 32, 1, 0)

set.seed(42)
train_index_early <- sample(nrow(train_data_early), size = floor(0.7 * nrow(train_data_early)))

train_early <- train_data_early[train_index_early, ]
test_early <- train_data_early[-train_index_early, ]

# Fit a random forest model for EarlyPTB
early_model <- train(
  EarlyPTB_outcome ~ . - participant_id - delivery_wk,
  data = train_early,
  method = "rf",
  trControl = trainControl(method = "none")
)

# Predict probabilities on test set
test_early$EarlyPTB_pred <- predict(early_model, newdata = test_early, type = "prob")[,2]

# Compute AUC-ROC for EarlyPTB
roc_obj_early <- roc(test_early$EarlyPTB_outcome, test_early$EarlyPTB_pred)
early_auc <- auc(roc_obj_early)

cat("AUC-ROC for EarlyPTB prediction:", early_auc, "\n")
