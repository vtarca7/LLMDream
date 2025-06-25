# Load necessary libraries
library(tidyverse)
library(caret)

# Load the data
df_training <- read.csv("df_training.csv", stringsAsFactors = FALSE)
ano_training <- read.csv("ano_training.csv", stringsAsFactors = FALSE)

df_test <- read.csv("df_test.csv", stringsAsFactors = FALSE)
ano_test <- read.csv("ano_test.csv", stringsAsFactors = FALSE)

# Merge the feature data with metadata using Sample_ID
training_data <- df_training %>%
  column_to_rownames(var = "Feature") %>%
  as_tibble() %>%
  pivot_longer(-c(name), names_to = "Sample_ID", values_to = "Methylation") %>%
  left_join(ano_training, by = "Sample_ID")

test_data <- df_test %>%
  column_to_rownames(var = "Feature") %>%
  as_tibble() %>%
  pivot_longer(-c(name), names_to = "Sample_ID", values_to = "Methylation") %>%
  left_join(ano_test, by = "Sample_ID")

# Prepare the data for modeling
train_df <- training_data %>%
  spread(name, Methylation) %>%
  select(-Sample_ID)

test_df <- test_data %>%
  spread(name, Methylation) %>%
  select(-Sample_ID)

# Extract gestational age as response variable
y_train <- training_data$GA
y_test <- test_data$GA

# Fit a linear regression model
model <- lm(GA ~ ., data = train_df)

# Make predictions on the test set
predictions <- predict(model, newdata = test_df)

# Calculate RMSE
rmse_value <- sqrt(mean((predictions - y_test)^2))
cat("Root Mean Squared Error (RMSE) on Test Set:", rmse_value, "\n")

# Generate scatter plot of predicted vs actual gestational age values
plot(y_test, predictions,
     xlab = "Actual Gestational Age",
     ylab = "Predicted Gestational Age",
     main = "Predicted vs Actual Gestational Age",
     pch = 19)

abline(0, 1, col = "red", lty = 2) # Line of equality

legend("topleft", legend = paste("RMSE:", round(rmse_value, 4)), bty = "n")
