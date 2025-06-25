# Load required libraries
library(glmnet)
library(ggplot2)
library(dplyr)

# Set seed for reproducibility
set.seed(123)

# Function to read and preprocess data
preprocess_data <- function(feature_file, metadata_file) {
  # Read feature data and transpose
  features <- read.csv(feature_file, row.names = 1)
  features_t <- as.data.frame(t(features)) %>% 
    mutate(Sample_ID = rownames(.))
  
  # Read metadata
  metadata <- read.csv(metadata_file)
  
  # Merge and clean data
  merged_data <- metadata %>%
    inner_join(features_t, by = "Sample_ID") %>%
    select(-Sample_ID)  # Remove Sample_ID as per requirements
  
  return(merged_data)
}

# Load and preprocess training data
train_data <- preprocess_data("df_training.csv", "ano_training.csv")
x_train <- as.matrix(train_data[, -1])  # First column is GA
y_train <- train_data$GA

# Train elastic-net model with cross-validation
cv_fit <- cv.glmnet(x_train, y_train, alpha = 0.5, nfolds = 5)

# Load and preprocess test data
test_data <- preprocess_data("df_test.csv", "ano_test.csv")
x_test <- as.matrix(test_data[, -1])
y_test <- test_data$GA

# Make predictions
predictions <- predict(cv_fit, newx = x_test, s = "lambda.min")

# Calculate RMSE
rmse <- sqrt(mean((y_test - predictions)^2))
cat(paste("Test RMSE:", round(rmse, 4), "\n"))
write.csv(data.frame(y_test=y_test,y_pred=predictions),file="./out2/DeepseekR1-3.R.csv")

# Create scatter plot
plot_data <- data.frame(Actual = y_test, Predicted = predictions)
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6, color = "blue") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  geom_smooth(method = "lm", color = "darkgreen", se = FALSE) +
  labs(title = "Predicted vs Actual Gestational Age",
       x = "Actual GA", y = "Predicted GA") +
  annotate("text", x = min(plot_data$Actual), y = max(plot_data$Predicted),
           label = paste("RMSE:", round(rmse, 2)), hjust = 0, vjust = 1, size = 5) +
  theme_minimal(base_size = 14) +
  theme(plot.title = element_text(hjust = 0.5))