# Load required libraries
library(dplyr)
library(ggplot2)
library(readr)

# Read training data
training_data <- read_csv("df_training.csv", rowNameKey = 1)

# Read test data
test_data <- read_csv("df_test.csv", rowNameKey = 1)

# Read metadata for sample IDs
ano_training <- read_csv("ano_training.csv")
ano_test <- read_csv("ano_test.csv")

# Extract sample IDs from training and test metadata
train_ids <- ano_training[sample_id != "ID"] %>% distinct(sample_id)
test_ids <- ano_test[sample_id != "ID"] %>% distinct(sample_id)

# Filter training data based on sample IDs
training_data <- training_data %>% filter(row == training_data[rowNameKey] %>% 
                                             group_by(sample_id) %>% 
                                             filter(sample_id %in% train_ids))

# Filter test data based on sample IDs
test_data <- test_data %>% filter(row =="testing_data[rowNameKey]" %>% 
                                     group_by(sample_id) %>% 
                                     filter(sample_id %in% test_ids))

# Remove the 'row' column from both datasets
training_data <- training_data %>% select(-row)
test_data <- test_data %>% select(-row)

# Handle missing values (remove rows with NAs)
training_data <- na.omit(training_data)
test_data <- na.omit(test_data)

# Select top 100 most variable features
set.seed(123)  # For reproducibility
feature_variance <- training_data %>% 
    summarise(variance = var(GA)) %>% 
    mutate(rowname = rowname,
           feature_name = set_rowname(.))
top_features <- feature_variance %>% 
    arrange(desc(variance)) %>% 
    slice(1:100) %>% 
    pull(feature_name)

# Preprocess the data (centering and scaling)
training_data_processed <- training_data %>% 
    select(-GA) %>% 
    mutate_all(funs((. - mean(.)) / sd(.)))

test_data_processed <- test_data %>% 
    select(-GA) %>% 
    mutate_all(funs((. - mean(.)) / sd(.)))

# Fit the model using elastic net regression (cv.glmnet)
library(glmnet)

# Prepare data for modeling
x_train <- as.matrix(training_data_processed)
y_train <- training_data$GA

set.seed(42)  # For reproducibility
cv_fit <- cv.glmnet(x = x_train, y = y_train,
                    alpha = 0.5, nlambda = 100)

# Get optimal lambda value
best_lambda <- cv_fit$lambda.min

# Make predictions on test set
x_test <- as.matrix(test_data_processed)
y_pred <- predict(cv_fit, s = best_lambda, newx = x_test)

# Calculate RMSE
rmse <- sqrt(mean((y_pred - test_data$GA)^2))
print(paste("Root Mean Squared Error (RMSE):", rmse))

# Create scatter plot of predicted vs actual GA
test_data <- test_data %>% 
    mutate(predicted_GA = y_pred) %>% 
    select(GA, predicted_GA)

ggplot(test_data, aes(x = predicted_GA, y = GA)) +
    geom_point(alpha = 0.6) +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(title = paste("Predicted vs Actual Gestational Age (RMSE =", round(rmse, 3), ")"),
         x = "Predicted Gestational Age",
         y = "Actual Gestational Age") +
    theme_minimal()
