# Load necessary libraries
library(caret)
library(ggplot2)

# Load the training and test datasets
df_training <- read.csv("df_training.csv", row.names = 1)
ano_training <- read.csv("ano_training.csv")
df_test <- read.csv("df_test.csv", row.names = 1)
ano_test <- read.csv("ano_test.csv")

# Merge metadata with feature data for both training and test sets
training_data <- merge(df_training, ano_training[, "Sample_ID"], by.row = FALSE, by.col = TRUE)
test_data <- merge(df_test, ano_test[, "Sample_ID"], by.row = FALSE, by.col = TRUE)

# Prepare the formula for model fitting (excluding Sample_ID from features)
formula_string <- paste(colnames(training_data)[-1], collapse = " + ")
formula_gestational_age <- as.formula(paste("Gestational_Age ~", formula_string))

# Fit a random forest model (you can choose other models like linear regression, SVM, etc.)
set.seed(123)  # For reproducibility
model_rf <- train(formula_gestational_age, data = training_data, method = "rf", verbose = FALSE)

# Predict on the test set
predictions_test <- predict(model_rf, newdata = test_data)

# Calculate RMSE
rmse_value <- sqrt(mean((predictions_test - test_data$Gestational_Age)^2))
cat("RMSE:", rmse_value, "\n")

# Generate a scatter plot of predicted vs actual gestational age values for the test set
ggplot(data = data.frame(predictions = predictions_test, actual = test_data$Gestational_Age), aes(x = actual, y = predictions)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  labs(title = "Predicted vs Actual Gestational Age",
       x = "Actual Gestational Age",
       y = "Predicted Gestational Age",
       caption = paste("RMSE:", round(rmse_value, 2))) +
  theme_minimal()
