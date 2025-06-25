# Install and load required packages
#if (!requireNamespace("BiocManager", quietly = TRUE))
#  install.packages("BiocManager")

#if (!require("GEOquery")) BiocManager::install("GEOquery")
#if (!require("glmnet")) install.packages("glmnet")

library(GEOquery)
library(glmnet)

# Download the GEO dataset (GSE149440)
# Setting GSEMatrix=TRUE returns an ExpressionSet object.
gset <- getGEO("GSE149440", GSEMatrix = TRUE)
# If multiple ExpressionSets are returned, select the first one.
if (length(gset) > 1) {
  eset <- gset[[1]]
} else {
  eset <- gset[[1]]
}

# Extract expression data and phenotype (sample metadata)
expr_data <- exprs(eset)       # rows: genes, columns: samples
pheno_data <- pData(eset)

# Check that the expected metadata columns exist
if (!all(c("gestational age:ch1", "train:ch1") %in% colnames(pheno_data))) {
  stop("The required metadata columns 'gestational age:ch1' and 'train:ch1' are not found.")
}

# Subset training and test samples based on the 'train:ch1' column
train_idx <- which(pheno_data$`train:ch1` == "1")
test_idx  <- which(pheno_data$`train:ch1` == "0")

# Prepare predictor matrices: glmnet expects rows=samples and columns=features
X_train <- t(expr_data[, train_idx])
X_test  <- t(expr_data[, test_idx])

# Prepare response vectors (convert to numeric in case they are factors/characters)
y_train <- as.numeric(as.character(pheno_data[train_idx, "gestational age:ch1"]))
y_test  <- as.numeric(as.character(pheno_data[test_idx, "gestational age:ch1"]))

# Fit a LASSO regression model (alpha = 1) using cross-validation to select lambda
set.seed(123)  # for reproducibility
cvfit <- cv.glmnet(X_train, y_train, alpha = 1)
best_lambda <- cvfit$lambda.min

# Train the final model using the best lambda
model <- glmnet(X_train, y_train, alpha = 1, lambda = best_lambda)

# Apply the model to the test set to obtain predictions
predictions <- predict(model, newx = X_test)

# Compute the Root Mean Squared Error (RMSE) on the test set
rmse <- sqrt(mean((predictions - y_test)^2))
cat("Test RMSE:", round(rmse, 2), "\n")
write.csv(data.frame(y_test=y_test,y_pred=predictions),file="./out2/o3-high-1.R.csv")

# Generate a scatter plot of predicted vs. actual gestational age values for the test set
plot(y_test, predictions,
     xlab = "Actual Gestational Age",
     ylab = "Predicted Gestational Age",
     main = paste("Predicted vs Actual Gestational Age (Test Set)\nRMSE =", round(rmse, 2)),
     pch = 19, col = "blue")
abline(0, 1, col = "red", lwd = 2)  # reference line (ideal prediction)
