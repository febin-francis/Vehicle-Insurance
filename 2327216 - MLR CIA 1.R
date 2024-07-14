# Load necessary libraries
library(dplyr)  # For data manipulation
library(readr)  # For reading data
library(car)    # For diagnostic plots
library(readxl) # For reading Excel files
library(caret)  # For data splitting
library(glmnet) # For Lasso and Ridge regression

# Set working directory and load the data
setwd("/Users/febinfrancis/Desktop")
getwd()
insurance_data <- read_csv("VehicleInsuranceData.csv")
View(insurance_data)

# Check the structure of your data
str(insurance_data)

# Check for missing values
sum(is.na(insurance_data))

# Summary statistics
summary(insurance_data)

# Fit the multiple linear regression model
model <- lm(clv ~ Response + Coverage + Education + EmploymentStatus + Gender + Income + Location.Code + Marital.Status + Monthly.Premium.Auto + Months.Since.Last.Claim + Months.Since.Policy.Inception + Number.of.Open.Complaints + Number.of.Policies + Policy.Type + Policy + Renew.Offer.Type + Sales.Channel + Total.Claim.Amount + Vehicle.Class + Vehicle.Size, data = insurance_data)

# Summarize the model
summary(model)

#Splitting the data
# Split data into training (70%) and testing (30%)
set.seed(123)  # Set seed for reproducibility
train_index <- createDataPartition(insurance_data$clv, p = 0.7, list = FALSE)
train_data <- insurance_data[train_index, ]
test_data <- insurance_data[-train_index, ]

# Fit the multiple linear regression model on training data
model <- lm(clv ~ Response + Coverage + Education + EmploymentStatus + Gender + Income + Location.Code + Marital.Status + Monthly.Premium.Auto + Months.Since.Last.Claim + Months.Since.Policy.Inception + Number.of.Open.Complaints + Number.of.Policies + Policy.Type + Policy + Renew.Offer.Type + Sales.Channel + Total.Claim.Amount + Vehicle.Class + Vehicle.Size, data = train_data)

# Print summary of the model
summary(model)

# Predict using the test data
predictions <- predict(model, newdata = test_data)

# Calculate RMSE (Root Mean Squared Error)
rmse <- sqrt(mean((test_data$clv - predictions)^2))
print(paste("RMSE: ", rmse))

#Lasso Regression
# Prepare data for glmnet (matrix format required)
x <- as.matrix(train_data[, c("Response", "Coverage", "Education", "EmploymentStatus", "Gender", "Income", "Location.Code", "Marital.Status", "Monthly.Premium.Auto", "Months.Since.Last.Claim", "Months.Since.Policy.Inception", "Number.of.Open.Complaints", "Number.of.Policies", "Policy.Type", "Policy", "Renew.Offer.Type", "Sales.Channel", "Total.Claim.Amount", "Vehicle.Class", "Vehicle.Size")])
y <- train_data$clv

# Fit Lasso regression model
lasso_model <- cv.glmnet(x, y, alpha = 1)  # alpha = 1 for Lasso

# Print optimal lambda value
print(paste("Optimal Lambda: ", lasso_model$lambda.min))

# Predict using the test data
x_test <- as.matrix(test_data[, c("Response", "Coverage", "Education", "EmploymentStatus", "Gender", "Income", "Location.Code", "Marital.Status", "Monthly.Premium.Auto", "Months.Since.Last.Claim", "Months.Since.Policy.Inception", "Number.of.Open.Complaints", "Number.of.Policies", "Policy.Type", "Policy", "Renew.Offer.Type", "Sales.Channel", "Total.Claim.Amount", "Vehicle.Class", "Vehicle.Size")])
predictions <- predict(lasso_model, s = "lambda.min", newx = x_test)

# Calculate RMSE (Root Mean Squared Error)
rmse <- sqrt(mean((test_data$clv - predictions)^2))
print(paste("RMSE: ", rmse))

# Print coefficients of the model
print(coef(lasso_model, s = "lambda.min"))

# Optionally, visualize the coefficient path
plot(lasso_model)

#Ridge Regression
# Fit Ridge regression model
ridge_model <- cv.glmnet(x, y, alpha = 0)  # alpha = 0 for Ridge

# Print optimal lambda value
print(paste("Optimal Lambda: ", ridge_model$lambda.min))

# Predict using the test data
predictions <- predict(ridge_model, s = "lambda.min", newx = x_test)

# Calculate RMSE (Root Mean Squared Error)
rmse <- sqrt(mean((test_data$clv - predictions)^2))
print(paste("RMSE: ", rmse))

# Print coefficients of the model
print(coef(ridge_model, s = "lambda.min"))

# Optionally, visualize the coefficient path
plot(ridge_model)

