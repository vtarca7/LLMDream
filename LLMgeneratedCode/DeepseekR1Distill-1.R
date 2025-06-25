# Load necessary libraries
#library(RSAMArrayExpress)
library(limma)
library(RColorBrewer)

# Set working directory and load metadata
setwd("path/to/your/data")
metadata <- read.table("GSE149440 pData.txt", header = TRUE, sep = "\t")

# Read expression data (example for one chip; adjust as needed)
expressionData <- read.ExpressionSet("GSE149440 EXPRESSION")
expressionMatrix <- exprs(expressionData)

# Preprocess the expression data: filter and normalize
expressionMatrix <- log2(expressionMatrix + 1)  # Adding 1 to avoid log(0)
exprs <- apply(exprs(expressionData), 1, function(x) {
    if (medianABSDiffs(x) < 0.5) x else NA
})

# Assuming train and test data are in the metadata variable 'train:ch1'
trainSamples <- subsetRows(exprs, which(metadata$train:ch1 == "1"))

testSamples <- subsetRows(exprs, which(metadata$train:ch1 == "0"))

# Perform normalization (example using RMA for Affymetrix chips)
if (is.array(exprs)) {
    normalizedExprs <- rmaFit(exprs)
} else {
    normalizedExprs <- exprs
}

trainExpression <- normalizedExprs[match(row.names(normalizedExprs), row.names(trainSamples)), ]
testExpression <- normalizedExprs[match(row.names(normalizedExprs), row.names(testSamples)), ]

# Assuming 'gestational.age' is the column name in metadata
trainMetadata <- metadata[metadata$train:ch1 == "1", c("GESTAGE", "train:ch1")]
testMetadata <- metadata[metadata$train:ch1 == "0", c("GESTAGE", "train:ch1")]

# Prepare expression data for modeling
trainExpressionMatrix <- as.data.frame(trainExpression)
rownames(trainExpressionMatrix) <- seq_len(nrow(trainExpressionMatrix))

testExpressionMatrix <- as.data.frame(testExpression)
rownames(testExpressionMatrix) <- seq_len(nrow(testExpressionMatrix))

# Fit a regularized regression model (Lasso)
set.seed(123)
cvFit <- cv.glmnet(
    x = trainExpressionMatrix,
    y = trainMetadata$GESTAGE,
    alpha = 1,  # Lasso
    nfolds = 5,
    type.gam = "log"
)

bestModel <- glmnet(
    trainExpressionMatrix,
    trainMetadata$GESTAGE,
    s = cvFit$lambda.min,
    alpha = 1
)

# Predict on test set
predictedGest.age <- predict(bestModel, newx = testExpressionMatrix)

# Calculate RMSE
actualGest.age <- testMetadata$GESTAGE
rmse <- sqrt(mean((predictedGest.age - actualGest.age)^2))
cat("Root Mean Square Error:", rmse, "\n")

# Generate scatter plot
ggplot2::ggplot(data.frame(Predicted = predictedGest.age, Actual = actualGest.age), aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.6) +
    geom_abline(intercept = 0, slope = 1, color = "red") +
    labs(title = "Predicted vs Actual Gestational Age",
         x = "Actual Gestational Age",
         y = "Predicted Gestational Age")
