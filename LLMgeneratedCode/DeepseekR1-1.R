# Install required packages if necessary
if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
if (!require("GEOquery"))
    BiocManager::install("GEOquery")
if (!require("glmnet"))
    install.packages("glmnet")
if (!require("ggplot2"))
    install.packages("ggplot2")

# Load libraries
library(GEOquery)
library(glmnet)
library(ggplot2)

# Download and load the GEO dataset
gse <- getGEO("GSE149440", GSEMatrix = TRUE)
geo_data <- gse[[1]]

# Extract phenotype data and expression matrix
pheno <- pData(geo_data)
exprs <- exprs(geo_data)

# Clean and prepare gestational age data
clean_ga <- function(x) as.numeric(gsub("[^0-9.]", "", x))
pheno$gest_age <- clean_ga(pheno[["gestational age:ch1"]])

# Identify training and test samples
train_samples <- pheno[["train:ch1"]] == "1"
test_samples <- pheno[["train:ch1"]] == "0"

# Prepare data matrices
X_train <- t(exprs[, train_samples])
y_train <- pheno$gest_age[train_samples]
X_test <- t(exprs[, test_samples])
y_test <- pheno$gest_age[test_samples]

# Train LASSO regression model with cross-validation
set.seed(123)
cv_fit <- cv.glmnet(X_train, y_train, alpha = 1)
best_lambda <- cv_fit$lambda.min

# Make predictions and calculate RMSE
preds <- predict(cv_fit, s = best_lambda, newx = X_test)
rmse <- sqrt(mean((preds - y_test)^2))
write.csv(data.frame(y_test=y_test,y_pred=preds),file="./out2/DeepseekR1-1.R.csv")
# Print RMSE
cat("Test RMSE:", rmse, "\n")

# Create scatter plot
plot_data <- data.frame(Actual = y_test, Predicted = as.numeric(preds))
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6) +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  theme_bw() +
  ggtitle("Predicted vs Actual Gestational Age (Test Set)") +
  xlab("Actual Gestational Age") +
  ylab("Predicted Gestational Age") +
  coord_fixed(ratio = 1)
