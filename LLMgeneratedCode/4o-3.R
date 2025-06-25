library(glmnet)
library(readr)
library(dplyr)
library(ggplot2)
library(caret)

# Load training data
feature_train <- read_csv("df_training.csv")
metadata_train <- read_csv("ano_training.csv")

# Load test data
feature_test <- read_csv("df_test.csv")
metadata_test <- read_csv("ano_test.csv")

# Transpose feature data so that samples are rows
feature_train <- as.data.frame(t(feature_train))
colnames(feature_train) <- feature_train[1, ]
feature_train <- feature_train[-1, ]
feature_train <- feature_train %>% mutate(Sample_ID = rownames(feature_train))

feature_test <- as.data.frame(t(feature_test))
colnames(feature_test) <- feature_test[1, ]
feature_test <- feature_test[-1, ]
feature_test <- feature_test %>% mutate(Sample_ID = rownames(feature_test))

# Convert feature data to numeric
feature_train[,-ncol(feature_train)] <- lapply(feature_train[,-ncol(feature_train)], as.numeric)
feature_test[,-ncol(feature_test)] <- lapply(feature_test[,-ncol(feature_test)], as.numeric)

# Merge with metadata
train_data <- merge(metadata_train, feature_train, by = "Sample_ID")
test_data <- merge(metadata_test, feature_test, by = "Sample_ID")

# Extract gestational age (GA) and feature matrix
GA_train <- train_data$GA
GA_test <- test_data$GA

X_train <- as.matrix(train_data[ , !(colnames(train_data) %in% c("Sample_ID", "GA"))])
X_test <- as.matrix(test_data[ , !(colnames(test_data) %in% c("Sample_ID", "GA"))])

set.seed(123)
# Fit elastic net model
cv_model <- cv.glmnet(X_train, GA_train, alpha = 0.5)

# Predict on test set
predictions <- predict(cv_model, X_test, s = "lambda.min")

# Compute RMSE
rmse <- sqrt(mean((GA_test - predictions)^2))
print(paste("RMSE on test set:", round(rmse, 2)))
write.csv(data.frame(y_test=GA_test,y_pred=predictions),file="./out2/4o-3.R.csv")

# Plot predicted vs actual GA
plot_df <- data.frame(Actual = GA_test, Predicted = predictions)
plot <- ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5) +
  geom_smooth(method = "lm", color = "blue", se = FALSE) +
  ggtitle("Predicted vs Actual Gestational Age") +
  xlab("Actual GA") +
  ylab("Predicted GA") +
  annotate("text", x = min(GA_test), y = max(predictions), label = paste("RMSE:", round(rmse, 2)), hjust = 0, size = 5, color = "red") +
  theme_minimal()

print(plot)
