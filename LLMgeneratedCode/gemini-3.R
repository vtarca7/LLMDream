# Load necessary libraries
library(readr)
library(dplyr)
library(tidyr)
library(glmnet)
library(Metrics)
library(ggplot2)

# Define file paths
feature_file_training <- "df_training.csv"
feature_file_test <- "df_test.csv"
ano_file_training <- "ano_training.csv"
ano_file_test <- "ano_test.csv"

# Load feature data
df_training_feature <- read_csv(feature_file_training)
df_test_feature <- read_csv(feature_file_test)

# Load annotation data
ano_training <- read_csv(ano_file_training)
ano_test <- read_csv(ano_file_test)

# ------------------- Data Preprocessing -------------------

# --- Training Data ---
# Transpose feature data: Samples as rows, Features as columns
df_training_feature_t <- df_training_feature %>%
  pivot_longer(cols = -Feature_Name, names_to = "Sample_ID", values_to = "Methylation") %>%
  pivot_wider(names_from = Feature_Name, values_from = Methylation)

# Extract feature names (excluding Sample_ID column for feature data)
feature_names <- colnames(df_training_feature_t)[-1]

# Merge feature data with annotation data by Sample_ID
train_data <- ano_training %>%
  inner_join(df_training_feature_t, by = "Sample_ID")

# Extract Gestational Age (GA) as target variable
train_GA <- train_data$Gestational_Age

# Select feature columns (methylation features) for training
train_features <- train_data %>% select(all_of(feature_names))

# --- Test Data ---
# Transpose feature data: Samples as rows, Features as columns
df_test_feature_t <- df_test_feature %>%
  pivot_longer(cols = -Feature_Name, names_to = "Sample_ID", values_to = "Methylation") %>%
  pivot_wider(names_from = Feature_Name, values_from = Methylation)

# Merge feature data with annotation data by Sample_ID
test_data <- ano_test %>%
  inner_join(df_test_feature_t, by = "Sample_ID")

# Extract Gestational Age (GA) as target variable
test_GA <- test_data$Gestational_Age

# Select feature columns (methylation features) for testing
test_features <- test_data %>% select(all_of(feature_names))

# --- Feature Scaling ---
# Scale features using training set statistics (important to avoid data leakage)
feature_scaling_params <- scale(train_features)
train_features_scaled <- as.matrix(feature_scaling_params)
test_features_scaled <- scale(as.matrix(test_features),
                              center = attr(feature_scaling_params, "scaled:center"),
                              scale = attr(feature_scaling_params, "scaled:scale"))

# Convert target variables to numeric vectors
train_GA <- as.numeric(train_GA)
test_GA <- as.numeric(test_GA)


# ------------------- Model Training (Ridge Regression) -------------------

# Use cross-validation to find the optimal lambda for Ridge Regression
set.seed(123) # for reproducibility
cv_model <- cv.glmnet(train_features_scaled, train_GA, alpha = 0, family = "gaussian") # alpha=0 for Ridge
best_lambda <- cv_model$lambda.min

# Train Ridge Regression model with the optimal lambda
ridge_model <- glmnet(train_features_scaled, train_GA, alpha = 0, lambda = best_lambda, family = "gaussian")

# ------------------- Model Prediction and Evaluation -------------------

# Predict on the test set
predictions <- predict(ridge_model, s = best_lambda, newx = test_features_scaled)

# Calculate RMSE
rmse_value <- rmse(test_GA, predictions)
cat("RMSE on Test Set:", rmse_value, "\n")

# ------------------- Scatter Plot -------------------

# Create a data frame for plotting
plot_data <- data.frame(
  Actual_GA = test_GA,
  Predicted_GA = as.vector(predictions) # Convert matrix to vector
)

# Generate scatter plot
scatter_plot <- ggplot(plot_data, aes(x = Actual_GA, y = Predicted_GA)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") + # Add diagonal line
  labs(
    title = "Predicted vs Actual Gestational Age (Test Set)",
    x = "Actual Gestational Age",
    y = "Predicted Gestational Age",
    caption = paste("RMSE =", round(rmse_value, 3)) # Include RMSE in caption/legend
  ) +
  theme_minimal()

# Print the scatter plot
print(scatter_plot)