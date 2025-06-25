# Install necessary libraries if not already installed
#install.packages(c("dplyr", "caret", "ggplot2"))

# Load required libraries
library(dplyr)
library(caret)
library(ggplot2)

# Load the data and metadata from the Gene Expression Omnibus dataset GSE149440
data <- read.csv("https://www.ncbi.nlm.nih.gov/geo/gdservice.fcgi?accession=gse149440&submit=download")
metadata <- read.table("https://www.ncbi.nlm.nih.gov/geo/gdservice.fcgi?accession=gse149440&submit=download", header = FALSE)

# Convert the metadata to a data frame
colnames(metadata) <- c("sample_id", "description", "chip_type", "protocol_version", "run",
                        "series_id", "study_name", "study_description", 
                        "design_type", "design_description", "experiment_type",
                        "experiment_description", "platform", "platform_description",
                        "platform_version", "study_date", "collection_date", "completion_date",
                        "geographic_region", "research_area", "research_concept_descriptions",
                        "sample_type", "sampleprep_method", "sequencing_library", 
                        "sequence technology", "sequence platform", "sequence version",
                        "sequencing_library_layout", "chip_name", "chip_description",
                        "array_type", "array_description", "array_feature_matrix",
                        "array_feature_types", "chip_accession")

# Create a new data frame that contains only the relevant metadata
training_data <- metadata[metadata$design_type == "RNA-seq", ]

# Create a new data frame that contains only the relevant expression data for training and testing
train_expression_data <- data[, sapply(data, function(x) which(names(x) %in% colnames(training_data)))]
test_expression_data <- train_expression_data[train_expression_data$sample_id %in% training_data$sample_id[training_data$design_type == "RNA-seq" & training_data$ch1 == 0], ]

# Create a new data frame that contains only the gestational age values
gestational_age_data <- metadata[metadata$design_description == "GestationalAge:CH1", ]
predict_gestational_age_values <- as.numeric(gestational_age_data$geographic_region)

# Fit a linear regression model to predict gestational age from expression data
model <- lm(predict_gestational_age_values ~ rowSums(train_expression_data))

# Predict the gestational age values for the test set
predictions <- predict(model, newdata = test_expression_data)

# Calculate the root mean squared error (RMSE) of the predictions on the test set
rmse <- sqrt(mean((test_expression_data$predict_gestational_age_values - predictions)^2))
print(paste("Root Mean Squared Error (RMSE):", rmse))

# Plot a scatter plot of predicted vs actual gestational age values for the test set
ggplot(test_expression_data, aes(x = predict_gestational_age_values, y = predict_gestational_age_values)) +
  geom_point() +
  geom_line(aes(y = test_expression_data$predict_gestational_age_values), color = "red") +
  labs(title = "Predicted vs Actual Gestational Age Values for the Test Set",
       subtitle = paste("RMSE:", rmse),
       x = "Predicted Gestational Age Value",
       y = "Actual Gestational Age Value")
