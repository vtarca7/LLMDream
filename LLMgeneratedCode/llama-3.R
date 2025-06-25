# Load necessary libraries
library(dplyr)
library(reshape2)
library(ggplot2)

# Load data
df_training <- read.csv("df_training.csv")
df_test <- read.csv("df_test.csv")
ano_training <- read.csv("ano_training.csv")
ano_test <- read.csv("ano_test.csv")

# Merge training data with metadata
df_training <- merge(df_training, ano_training, by = "Sample_ID")

# Merge test data with metadata
df_test <- merge(df_test, ano_test, by = "Sample_ID")

# Fit model using Lasso regression
library(lassod)
model <- lassod(ga ~ ., data = df_training, 
                method = "L1", 
                alpha = 0.5, # adjust this value to change the level of regularization
                nboots = 100) 

# Predict on test set
predictions <- predict(model, newdata = df_test)

# Calculate RMSE on test set
rmse <- sqrt(mean((df_test$ga - predictions)^2))
print(paste("RMSE on test set:", rmse))

# Create scatter plot of predicted vs actual gestational age values for test set
ggplot(df_test, aes(x = ga, y = predictions)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "black") +
  labs(title = "Predicted vs Actual Gestational Age Values",
       subtitle = paste("RMSE:", rmse),
       x = "Actual Gestational Age (weeks)",
       y = "Predicted Gestational Age (weeks)") +
  theme_classic()
