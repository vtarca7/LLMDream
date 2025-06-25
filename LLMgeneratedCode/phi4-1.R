# Load necessary libraries
#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#BiocManager::install(c("GEOquery", "caret", "ggplot2"))

library(GEOquery)
library(caret)
library(ggplot2)

# Download the GSE149440 dataset
gse <- getGEO("GSE149440", GSEMatrix = TRUE)
if (length(gse) > 1) idx <- grep("GSM", names(gse), value = TRUE)
exprs_data <- exprs(gse[[idx]])
metadata <- pData(phenoData(gse[[idx]]))

# Extract gestational age and train/test split information
gestational_age <- metadata$`gestational age:ch1`
train_split <- metadata$`train:ch1`

# Prepare the training set
train_indices <- which(train_split == '1')
test_indices <- which(train_split == '0')

X_train <- exprs_data[train_indices, ]
y_train <- gestational_age[train_indices]
X_test <- exprs_data[test_indices, ]
y_test <- gestational_age[test_indices]

# Fit a linear model using caret package
set.seed(123)  # For reproducibility

model_fit <- train(
  X_train,
  y_train,
  method = "lm",
  trControl = trainControl(method = "none"),  # No cross-validation
  tuneLength = 0  # No tuning required for linear model
)

# Predict on the test set
predictions <- predict(model_fit, newdata = X_test)

# Calculate RMSE
rmse_value <- sqrt(mean((y_test - predictions)^2))
cat("Root Mean Squared Error (RMSE):", rmse_value, "\n")

# Plot predicted vs actual gestational age values for the test set
ggplot(data.frame(Actual = y_test, Predicted = predictions), aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "lm", col = "blue") +
  labs(title = "Predicted vs Actual Gestational Age",
       x = "Actual Gestational Age",
       y = "Predicted Gestational Age") +
  theme_minimal()
