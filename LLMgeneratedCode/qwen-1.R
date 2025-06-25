#if (!requireNamespace("BiocManager", quietly = TRUE))
#    install.packages("BiocManager")

#BiocManager::install(c("GEOquery", "caret", "ggplot2"))
library(GEOquery)

# Download GSE149440
gse <- getGEO("GSE149440", GSEMatrix = TRUE, AnnotGPL = FALSE)
gse_exprs <- exprs(gse[[1]])
gse_pdata <- pData(gse[[1]])
# Convert gestational age to numeric
gse_pdata$`gestational age:ch1` <- as.numeric(as.character(gse_pdata$`gestational age:ch1`))

# Filter training and test sets
train_set <- gse_pdata[gse_pdata$`train:ch1` == '1', ]
test_set <- gse_pdata[gse_pdata$`train:ch1` == '0', ]

train_exprs <- gse_exprs[, rownames(train_set)]
test_exprs <- gse_exprs[, rownames(test_set)]

y_train <- train_set$`gestational age:ch1`
y_test <- test_set$`gestational age:ch1`
library(caret)

# Train the model using Random Forest
set.seed(123)  # for reproducibility
train_control <- trainControl(method = "cv", number = 5)
model <- train(train_exprs, y_train, method = "rf", trControl = train_control)
predictions <- predict(model, newdata = test_exprs)
# Calculate RMSE
rmse <- sqrt(mean((predictions - y_test)^2))
print(paste("RMSE:", round(rmse, 2)))

# Plot predicted vs actual gestational age
library(ggplot2)
df_plot <- data.frame(Actual = y_test, Predicted = predictions)

ggplot(df_plot, aes(x = Actual, y = Predicted)) +
  geom_point(alpha = 0.6) +
  geom_abline(slope = 1, intercept = 0, color = "blue", linetype = "dashed") +
  labs(title = "Predicted vs Actual Gestational Age",
       x = "Actual Gestational Age",
       y = "Predicted Gestational Age") +
  theme_minimal()
