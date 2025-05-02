# Load the packages
library(rpart)
library(rpart.plot)
library(MASS)
library(caret)
library(pROC)
library(ggplot2)
library(dplyr)
library(tidyr)


credit_data <- read.csv("credit.csv")

# Examining the dataset
head(credit_data)
str(credit_data)
summary(credit_data)
dim(credit_data)

# check missing values 
sum(is.na(credit_data))

# Convert the target variable to a factor for the credibility column 
credit_data$Creditability <- as.factor(credit_data$Creditability)

# Convert other categorical variables to factors
categorical_vars <- c("Account.Balance", "Payment.Status.of.Previous.Credit", "Purpose", 
                     "Value.Savings.Stocks", "Length.of.current.employment",
                     "Sex...Marital.Status", "Guarantors", "Most.valuable.available.asset",
                     "Concurrent.Credits", "Type.of.apartment", "Occupation",
                     "No.of.dependents", "Telephone", "Foreign.Worker")

for(var in categorical_vars) {
  credit_data[[var]] <- as.factor(credit_data[[var]])
}

# splitting the data
set.seed(123) 
sample_index <- sample(1:nrow(credit_data), 0.7 * nrow(credit_data))
train_data <- credit_data[sample_index, ]
test_data <- credit_data[-sample_index, ]

cat("Training data dimensions:", dim(train_data), "\n")
cat("Testing data dimensions:", dim(test_data), "\n")

# Decision Tree 
tree_model <- rpart(Creditability ~ ., data = train_data, method = "class")
rpart.plot(tree_model, extra = 104, box.palette = "RdBu", shadow.col = "gray")
tree_pred <- predict(tree_model, test_data, type = "class")

tree_confusion <- table(Actual = test_data$Creditability, Predicted = tree_pred)
tree_accuracy <- sum(diag(tree_confusion))/sum(tree_confusion)
print(paste("Decision Tree Accuracy:", round(tree_accuracy, 4)))
print(tree_confusion)


print("Here is the Logistic summary")
# Logistic regression
logistic_model <- glm(Creditability ~ ., data = train_data, family = "binomial")
summary(logistic_model)

logit_prob <- predict(logistic_model, test_data, type = "response")
logit_pred <- ifelse(logit_prob > 0.5, "1", "0")
logit_pred <- factor(logit_pred, levels = levels(test_data$Creditability))

logit_confusion <- table(Actual = test_data$Creditability, Predicted = logit_pred)
logit_accuracy <- sum(diag(logit_confusion))/sum(logit_confusion)
print(paste("Logistic Regression Accuracy:", round(logit_accuracy, 4)))
print(logit_confusion)

#Wald test for LR
# Get coefficient summary
coef_summary <- summary(logistic_model)$coefficients

# Print coefficient summary for reference
print("Coefficient Summary:")
print(coef_summary)

# Function to perform Wald test manually
manual_wald_test <- function(model, term) {
  coefs <- coef(model)
  vcov_matrix <- vcov(model)
  
  # Find the indices for the term
  if (term == "Intercept") {
    indices <- 1
  } else {
    indices <- grep(term, names(coefs))
  }
  
  if (length(indices) == 0) {
    stop("Term not found in model")
  }
  
  # Extract coefficients and variance-covariance submatrix
  beta <- coefs[indices]
  Sigma <- vcov_matrix[indices, indices, drop = FALSE]
  
  # For a single coefficient
  if (length(indices) == 1) {
    wald_stat <- (beta / sqrt(Sigma))^2
    p_value <- 1 - pchisq(wald_stat, df = 1)
    result <- list(
      term = names(coefs)[indices],
      estimate = beta,
      standard_error = sqrt(Sigma),
      wald_statistic = wald_stat,
      df = 1,
      p_value = p_value
    )
  } else {
    # For multiple coefficients (joint test)
    wald_stat <- t(beta) %*% solve(Sigma) %*% beta
    p_value <- 1 - pchisq(wald_stat, df = length(indices))
    result <- list(
      terms = names(coefs)[indices],
      wald_statistic = wald_stat,
      df = length(indices),
      p_value = p_value
    )
  }
  
  return(result)
}

# Test for Account.Balance (all categories together)
account_balance_test <- manual_wald_test(logistic_model, "Account.Balance")
print("Wald Test for Account Balance:")
print(account_balance_test)

# Test for Duration of Credit
duration_test <- manual_wald_test(logistic_model, "Duration.of.Credit..month.")
print("Wald Test for Duration of Credit:")
print(duration_test)

# Test for Credit Amount
credit_amount_test <- manual_wald_test(logistic_model, "Credit.Amount")
print("Wald Test for Credit Amount:")
print(credit_amount_test)

# LDA 
lda_model <- lda(Creditability ~ ., data = train_data)
lda_model

lda_pred <- predict(lda_model, test_data)
lda_class <- lda_pred$class

lda_confusion <- table(Actual = test_data$Creditability, Predicted = lda_class)
lda_accuracy <- sum(diag(lda_confusion))/sum(lda_confusion)
print(paste("Discriminant Analysis Accuracy:", round(lda_accuracy, 4)))
print(lda_confusion)

# k fold cross validation 
k_folds <- 10  # 10-fold cross-validation
control <- trainControl(method = "cv", number = k_folds)

# Train LDA model with cross-validation and equal priors
# Note: The prior parameter ensures equal priors are used
lda_cv <- train(Creditability ~ ., 
               data = credit_data,
               method = "lda",
               metric = "Accuracy",
               trControl = control,
               preProcess = c("center", "scale"))

# Display cross-validation results
print("LDA Cross-Validation Results:")
print(lda_cv)
print(lda_cv$results)

# Calculate overall accuracy from CV
cv_accuracy <- lda_cv$results$Accuracy
print(paste("LDA Cross-Validation Accuracy:", round(cv_accuracy, 4)))

# Calculate number of mismatches
total_samples <- nrow(credit_data)
mismatch_count <- round(total_samples * (1 - cv_accuracy))
print(paste("Number of mismatches:", mismatch_count))

# Alternative approach using manual CV for more detailed output
# Initialize vectors to store results
cv_accuracies <- numeric(k_folds)
cv_mismatches <- numeric(k_folds)

# Create folds
folds <- createFolds(credit_data$Creditability, k = k_folds, list = TRUE, returnTrain = FALSE)

# Perform manual k-fold cross-validation
for (i in 1:k_folds) {
  # Split data into training and validation sets
  train_indices <- unlist(folds[-i])
  valid_indices <- folds[[i]]
  
  train_cv <- credit_data[train_indices, ]
  valid_cv <- credit_data[valid_indices, ]
  
  # Train LDA model with equal priors
  lda_model_cv <- lda(Creditability ~ ., data = train_cv, prior = c(0.5, 0.5))
  
  # Make predictions
  lda_pred_cv <- predict(lda_model_cv, valid_cv)
  lda_class_cv <- lda_pred_cv$class
  
  # Calculate accuracy and mismatches
  conf_matrix <- table(Actual = valid_cv$Creditability, Predicted = lda_class_cv)
  cv_accuracies[i] <- sum(diag(conf_matrix)) / sum(conf_matrix)
  cv_mismatches[i] <- sum(valid_cv$Creditability != lda_class_cv)
}

# Calculate overall CV results
mean_accuracy <- mean(cv_accuracies)
total_mismatches <- sum(cv_mismatches)

cat("\n\nManual Cross-Validation Results:\n")
cat("Fold-wise accuracies:", cv_accuracies, "\n")
cat("Mean accuracy:", round(mean_accuracy, 4), "\n")
cat("Total mismatches:", total_mismatches, "\n")
cat("Fold-wise mismatches:", cv_mismatches, "\n")

# Compare all model performances
models <- c("Decision Tree", "Logistic Regression", "Discriminant Analysis")
accuracy <- c(tree_accuracy, logit_accuracy, lda_accuracy)
results <- data.frame(Model = models, Accuracy = accuracy)
results <- results[order(results$Accuracy, decreasing = TRUE), ]
print(results)

# Create ROC curves for all models
library(pROC)
roc_tree <- roc(as.numeric(test_data$Creditability) - 1, 
              as.numeric(predict(tree_model, test_data, type = "prob")[,2]))
roc_logit <- roc(as.numeric(test_data$Creditability) - 1, logit_prob)
roc_lda <- roc(as.numeric(test_data$Creditability) - 1, lda_pred$posterior[,2])

# Plot ROC curves
plot(roc_tree, col = "red", main = "ROC Curves for Credit Risk Models")
lines(roc_logit, col = "blue")
lines(roc_lda, col = "green")
legend("bottomright", legend = c("Decision Tree", "Logistic Regression", "Discriminant Analysis"), 
      col = c("red", "blue", "green"), lwd = 2)

auc_tree <- auc(roc_tree)
auc_logit <- auc(roc_logit)
auc_lda <- auc(roc_lda)
auc_results <- data.frame(
  Model = models,
  AUC = c(auc_tree, auc_logit, auc_lda)
)
auc_results <- auc_results[order(auc_results$AUC, decreasing = TRUE), ]
print(auc_results)