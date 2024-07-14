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

# Remove rows with missing values
insurance_data <- na.omit(insurance_data)

# Convert categorical variables to factors
insurance_data <- insurance_data %>%
  mutate(
    Response = as.factor(Response),
    Coverage = as.factor(Coverage),
    Education = as.factor(Education),
    EmploymentStatus = as.factor(EmploymentStatus),
    Gender = as.factor(Gender),
    Location.Code = as.factor(Location.Code),
    Marital.Status = as.factor(Marital.Status),
    Policy.Type = as.factor(Policy.Type),
    Policy = as.factor(Policy),
    Renew.Offer.Type = as.factor(Renew.Offer.Type),
    Sales.Channel = as.factor(Sales.Channel),
    Vehicle.Class = as.factor(Vehicle.Class),
    Vehicle.Size = as.factor(Vehicle.Size)
  )

# Summary statistics
summary(insurance_data)

# Fit the multiple linear regression model
model <- lm(clv ~ Response + Coverage + Education + EmploymentStatus + Gender + Income + Location.Code + Marital.Status + Monthly.Premium.Auto + Months.Since.Last.Claim + Months.Since.Policy.Inception + Number.of.Open.Complaints + Number.of.Policies + Policy.Type + Policy + Renew.Offer.Type + Sales.Channel + Total.Claim.Amount + Vehicle.Class + Vehicle.Size, data = insurance_data)

# Summarize the model
summary(model)

#Splitting the data #
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
predictions_mlr <- predict(model, newdata = test_data)

# Calculate RMSE (Root Mean Squared Error)
rmse_mlr <- sqrt(mean((test_data$clv - predictions_mlr)^2))
print(paste("RMSE MLR: ", rmse_mlr))

## Lasso Regression ##
# Prepare data for glmnet (matrix format required)
x <- model.matrix(clv ~ Response + Coverage + Education + EmploymentStatus + Gender + Income + Location.Code + Marital.Status + Monthly.Premium.Auto + Months.Since.Last.Claim + Months.Since.Policy.Inception + Number.of.Open.Complaints + Number.of.Policies + Policy.Type + Policy + Renew.Offer.Type + Sales.Channel + Total.Claim.Amount + Vehicle.Class + Vehicle.Size, data = train_data)
y <- train_data$clv

# Fit Lasso regression model
lasso_model <- cv.glmnet(x, y, alpha = 1)  # alpha = 1 for Lasso

# Print optimal lambda value
lasso_model$lambda.min

# Prepare test data for prediction
x_test <- model.matrix(clv ~ Response + Coverage + Education + EmploymentStatus + Gender + Income + Location.Code + Marital.Status + Monthly.Premium.Auto + Months.Since.Last.Claim + Months.Since.Policy.Inception + Number.of.Open.Complaints + Number.of.Policies + Policy.Type + Policy + Renew.Offer.Type + Sales.Channel + Total.Claim.Amount + Vehicle.Class + Vehicle.Size, data = test_data)

# Predict using the test data
predictions_lasso <- predict(lasso_model, s = "lambda.min", newx = x_test)

# Calculate RMSE (Root Mean Squared Error)
rmse_lasso <- sqrt(mean((test_data$clv - predictions_lasso)^2))
print(paste("RMSE Lasso: ", rmse_lasso))

# Print coefficients of the model
lasso.coef <- predict(lasso_model, type = "coefficients", s = "lambda.min")
print("Lasso Coefficients:")
print(lasso.coef)


##Ridge Regression ##
# Fit Ridge regression model
ridge_model <- cv.glmnet(x, y, alpha = 0)  # alpha = 0 for Ridge

# Print optimal lambda value
ridge_model$lambda.min

# Predict using the test data
predictions_ridge <- predict(ridge_model, s = "lambda.min", newx = x_test)

# Calculate RMSE (Root Mean Squared Error)
rmse_ridge <- sqrt(mean((test_data$clv - predictions_ridge)^2))
print(paste("RMSE Ridge: ", rmse_ridge))

# Print coefficients of the model
ridge.coef <- predict(ridge_model, type = "coefficients", s = "lambda.min")
print("Ridge Coefficients:")
print(ridge.coef)
     

