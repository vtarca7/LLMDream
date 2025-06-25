# Load required libraries
library(dplyr)
library(caret)
library(ROCR)

# Read in data
train_data <- read.csv("training_species_abundance.csv")
val_data <- read.csv("validation_species_abundance.csv")

train_metadata <- read.csv("training_metadata.csv")
val_metadata <- read.csv("validation_metadata.csv")

# Merge data with metadata
train_data <- merge(train_data, train_metadata, by = "specimen")
val_data <- merge(val_data, val_metadata, by = "specimen")

# Filter out features not in common between training and validation sets
common_features <- intersect(train_data$feature, val_data$feature)
train_data <- train_data[train_data$feature %in% common_features, ]
val_data <- val_data[val_data$feature %in% common_features, ]

# Merge data back into metadata
train_data <- merge(train_data, train_metadata, by = "specimen")
val_data <- merge(val_data, val_metadata, by = "specimen")

# Filter out rows with collect_wk >= 32 and delivery_wk >= 37
train_data <- filter(train_data, collect_wk < 32) %>% 
  group_by(participant_id) %>%
  slice(which.max(delivery_wk < 37))
val_data <- filter(val_data, collect_wk < 32) %>% 
  group_by(participant_id) %>%
  slice(which.max(delivery_wk < 37))

# Convert data to matrix
train_X <- as.matrix(train_data[, sapply(train_data, is.numeric)])
val_X <- as.matrix(val_data[, sapply(val_data, is.numeric)])

# Create model and predict
model <- glm PTB ~ ., data = train_data, family = binomial
predict_model <- predict(model, newdata = val_X)

# Calculate AUC-ROC
auc_roc <- performance(predict_model, "ttest", metric = "auc")
print(paste("AUC-ROC for PTB:", auc_roc@y.values[2]))

# Plot ROC curve
plot roc_data(predict_model, train_data$PTB)
legend("topright", c("PTB"), lty = 1, col = "black")

# Repeat analysis for EarlyPTB outcome
model <- glm EarlyPTB ~ ., data = train_data, family = binomial
predict_model <- predict(model, newdata = val_X)

auc_roc <- performance(predict_model, "ttest", metric = "auc")
print(paste("AUC-ROC for EarlyPTB:", auc_roc@y.values[2]))

plot roc_data(predict_model, train_data$EarlyPTB)
legend("topright", c("EarlyPTB"), lty = 1, col = "black")
