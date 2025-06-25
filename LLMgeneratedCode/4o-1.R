library(GEOquery)
library(limma)
library(caret)
library(ggplot2)
library(glmnet)

# Load dataset from GEO
geo_id <- "GSE149440"
gset <- getGEO(geo_id, GSEMatrix = TRUE, AnnotGPL = TRUE)[[1]]

# Extract expression data
gene_expr <- exprs(gset)

# Extract metadata
metadata <- pData(gset)

# Ensure metadata column names are standardized
colnames(metadata) <- make.names(colnames(metadata))

# Select gestational age and train/test indicators
gestational_age <- as.numeric(metadata$gestational.age.ch1)
train_indicator <- as.numeric(metadata$train.ch1)

# Keep only the samples with valid gestational age values
valid_samples <- !is.na(gestational_age)
gene_expr <- gene_expr[, valid_samples]
gestational_age <- gestational_age[valid_samples]
train_indicator <- train_indicator[valid_samples]

# Split into training and test sets
train_idx <- which(train_indicator == 1)
test_idx <- which(train_indicator == 0)

x_train <- t(gene_expr[, train_idx])
y_train <- gestational_age[train_idx]

x_test <- t(gene_expr[, test_idx])
y_test <- gestational_age[test_idx]

# Standardize data
preproc <- preProcess(x_train, method = c("center", "scale"))
x_train <- predict(preproc, x_train)
x_test <- predict(preproc, x_test)

# Fit a regularized regression model (LASSO)
set.seed(123)
lasso_model <- cv.glmnet(as.matrix(x_train), y_train, alpha = 1)

# Predict on test set
y_pred <- predict(lasso_model, as.matrix(x_test), s = "lambda.min")

# Compute RMSE
rmse <- sqrt(mean((y_test - y_pred)^2))
cat("RMSE on Test Set:", rmse, "\n")
write.csv(data.frame(y_test=y_test,y_pred=y_pred),file="./out2/4o-1.R.csv")

# Plot predicted vs actual gestational age
plot_data <- data.frame(Actual = y_test, Predicted = y_pred)
ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point() +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  theme_minimal() +
  labs(title = "Predicted vs Actual Gestational Age",
       x = "Actual Gestational Age",
       y = "Predicted Gestational Age")
