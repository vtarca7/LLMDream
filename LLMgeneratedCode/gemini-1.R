# Install necessary packages if you don't have them already
if(!requireNamespace("GEOquery", quietly = TRUE)){
  install.packages("GEOquery")
}
if(!requireNamespace("Biobase", quietly = TRUE)){
  install.packages("Biobase")
}
if(!requireNamespace("dplyr", quietly = TRUE)){
  install.packages("dplyr")
}
if(!requireNamespace("caret", quietly = TRUE)){
  install.packages("caret")
}
if(!requireNamespace("ggplot2", quietly = TRUE)){
  install.packages("ggplot2")
}

# Load necessary libraries
library(GEOquery)
library(Biobase)
library(dplyr)
library(caret)
library(ggplot2)

# 1. Download GSE149440 dataset
gse <- getGEO("GSE149440", GSEMatrix =TRUE, getGPL=FALSE)

if (length(gse) > 1) idx <- grep("GSE149440_series_matrix", names(gse))[1] else idx <- 1
gse <- gse[[idx]]

# 2. Extract expression data and metadata
expression_data <- exprs(gse)
metadata <- pData(gse)

# 3. Prepare data for training and testing

# Extract gestational age and train/test split information
metadata <- metadata %>%
  mutate(gestational_age = as.numeric(`gestational age:ch1`),
         train_set = `train:ch1`)

# Separate training and test samples based on 'train_set' variable
train_metadata <- metadata %>% filter(train_set == '1')
test_metadata <- metadata %>% filter(train_set == '0')

# Get sample names for training and testing
train_samples <- rownames(train_metadata)
test_samples <- rownames(test_metadata)

# Filter expression data for training and testing samples
train_expression <- expression_data[, train_samples, drop = FALSE]
test_expression <- expression_data[, test_samples, drop = FALSE]

# Ensure we have gestational age data and expression data for both sets
if(nrow(train_metadata) == 0 || ncol(train_expression) == 0 || nrow(test_metadata) == 0 || ncol(test_expression) == 0) {
  stop("Problem separating training and testing data. Check metadata 'train:ch1' variable.")
}

# 4. Train a prediction model (Linear Regression)

# Create training data frame with gestational age as target and gene expression as predictors
train_df <- data.frame(gestational_age = train_metadata$gestational_age, t(train_expression))

# Train a linear regression model using caret package for simplicity
# We will use all genes as predictors. In a real scenario, feature selection might be important.
model <- train(
  gestational_age ~ .,
  data = train_df,
  method = "lm" # Linear Regression
)

# 5. Apply the model on the test set

# Create test data frame in the same format as training data (without gestational_age for prediction)
test_df_predictors <- data.frame(t(test_expression))
colnames(test_df_predictors) <- colnames(train_df)[-1] # Ensure column names match training data

# Predict gestational age for the test set
predictions <- predict(model, newdata = test_df_predictors)

# Get actual gestational ages for the test set
actual_gestational_age_test <- test_metadata$gestational_age

# 6. Calculate Root Mean Squared Error (RMSE) on the test set
rmse_value <- RMSE(predictions, actual_gestational_age_test)
cat("Root Mean Squared Error (RMSE) on the test set:", rmse_value, "\n")

# 7. Generate a scatter plot of predicted vs actual gestational age for the test set

plot_data <- data.frame(actual = actual_gestational_age_test, predicted = predictions)

scatter_plot <- ggplot(plot_data, aes(x = actual, y = predicted)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") + # Add y=x line for reference
  xlab("Actual Gestational Age") +
  ylab("Predicted Gestational Age") +
  ggtitle("Predicted vs Actual Gestational Age on Test Set") +
  theme_minimal()

print(scatter_plot)

# Optional: Save the plot to a file
# ggsave("gestational_age_prediction_scatter_plot.png", plot = scatter_plot, width = 6, height = 6, units = "in")

cat("Script finished. RMSE printed and scatter plot generated.\n")
