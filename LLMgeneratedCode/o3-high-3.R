# Load necessary libraries
library(glmnet)
library(ggplot2)
library(dplyr)

set.seed(123)  # for reproducibility

#############################
# 1. Read and Prepare Data  #
#############################

# Read feature data (features are rows and samples are columns; first column is feature names)
df_train_features <- read.csv("df_training.csv", stringsAsFactors = FALSE)
df_test_features  <- read.csv("df_test.csv", stringsAsFactors = FALSE)

# Read metadata (assumed to have at least columns: Sample_ID and GA)
ano_train <- read.csv("ano_training.csv", stringsAsFactors = FALSE)
ano_test  <- read.csv("ano_test.csv", stringsAsFactors = FALSE)

# Transpose training feature data so that rows are samples and columns are features
# First column in df_train_features holds the feature names.
feature_names <- df_train_features[[1]]
# Remove the first column (feature names) and transpose
train_mat <- t(df_train_features[,-1])
# Convert to data.frame and set column names to the feature names
X_train <- as.data.frame(train_mat)
colnames(X_train) <- feature_names
# The row names now are the sample IDs (which originally were the column names excluding the first column)
rownames(X_train) <- colnames(df_train_features)[-1]

# Do the same for test features
feature_names_test <- df_test_features[[1]]
test_mat <- t(df_test_features[,-1])
X_test <- as.data.frame(test_mat)
colnames(X_test) <- feature_names_test
rownames(X_test) <- colnames(df_test_features)[-1]

# Merge feature data with metadata using Sample_ID (assumes Sample_ID in metadata matches the column names in feature data)
df_train <- merge(ano_train, X_train, by.x = "Sample_ID", by.y = "row.names")
df_test  <- merge(ano_test, X_test, by.x = "Sample_ID", by.y = "row.names")

# Extract the outcome (gestational age, GA) and predictors.
# We only use the methylation features for prediction.
y_train <- df_train$GA
y_test  <- df_test$GA

# Remove metadata columns (assumed to be Sample_ID and GA) from predictors.
X_train_features <- as.matrix(df_train %>% select(-Sample_ID, -GA))
X_test_features  <- as.matrix(df_test %>% select(-Sample_ID, -GA))

#############################################
# 2. Fit a Predictive Model Using glmnet     #
#############################################

# Fit LASSO regression with cross-validation to choose lambda
cv_fit <- cv.glmnet(X_train_features, y_train, alpha = 1, nfolds = 5)
best_lambda <- cv_fit$lambda.min

# Fit the final model with the best lambda
final_model <- glmnet(X_train_features, y_train, alpha = 1, lambda = best_lambda)

# Predict on the test set
y_pred <- predict(final_model, newx = X_test_features)

#############################################
# 3. Calculate RMSE and Plot Predictions      #
#############################################

# Calculate Root Mean Squared Error (RMSE)
rmse <- sqrt(mean((y_test - y_pred)^2))
print(paste("Test RMSE:", round(rmse, 2)))
write.csv(data.frame(y_test=y_test,y_pred=y_pred),file="./out2/o3-high-3.R.csv")
# Create a data frame for plotting
plot_df <- data.frame(Actual = y_test, Predicted = as.vector(y_pred))

# Generate scatter plot of Predicted vs Actual gestational age with RMSE in the legend
p <- ggplot(plot_df, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.5, color = "blue") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
  ggtitle("Predicted vs Actual Gestational Age") +
  theme_minimal() +
  labs(x = "Actual GA", y = "Predicted GA") +
  annotate("text", x = min(plot_df$Actual), y = max(plot_df$Predicted),
           label = paste("RMSE =", round(rmse, 2)),
           hjust = 0, vjust = 1, size = 5, color = "black")

# Print the plot
print(p)
