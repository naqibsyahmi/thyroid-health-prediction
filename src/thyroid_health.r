# Run the following command in R to execute this script: source("src/thyroid_health.r")

## Step 1: Preliminaries

# List of required packages
packages <- c("tidyverse", "ggplot2", "dplyr", "corrplot", "randomForest",
              "Metrics", "smotefamily", "caret", "xgboost")

# Install any missing packages
installed <- packages %in% rownames(installed.packages())
if (any(!installed)) {
  install.packages(packages[!installed])
}

# Load required libraries

# Data Manipulation and Visualization
library(tidyverse)

# Plotting
library(ggplot2)

# Data Wrangling
library(dplyr)

# Correlation Plots
library(corrplot)

# Load thyroid data
thyroid_data_df <- read.csv("../data/hypothyroid.csv")

# Preview the dataset
head(thyroid_data_df)

## Step 2: Data Preprocessing

# Dimension of the dataset
cat("The thyroid dataset has", nrow(thyroid_data_df), "rows and",
    ncol(thyroid_data_df), "columns.\n")

# List all column names in the dataset
for (idx in seq_along(thyroid_data_df)) {
  cat(idx, ":", colnames(thyroid_data_df)[idx], "\n")
}

# Statistical summary for each column in the dataset
summary(thyroid_data_df)

# Structure of the dataset
str(thyroid_data_df)

# Check for duplicate rows
duplicate_values <- sum(duplicated(thyroid_data_df))

cat("\nDuplicate values in the dataset: \n")
print(duplicate_values)

# Remove duplicate rows
thyroid_data_df <- thyroid_data_df[!duplicated(thyroid_data_df), ]

cat("\nDuplicate values after removal: \n")
print(sum(duplicated(thyroid_data_df)))

thyroid_data_df

# Loop through all columns and print value counts
for (colname in names(thyroid_data_df)) {
  cat("\nColumn:", colname, "\n")
  print(table(thyroid_data_df[[colname]], useNA = "ifany"))
}

# Removing irrelevant columns
thyroid_data_df$TBG <- NULL
thyroid_data_df$referral.source <- NULL

for (idx in seq_along(thyroid_data_df)) {
  cat(idx, ":", colnames(thyroid_data_df)[idx], "\n")
}

# Replace data points '?' with NA
thyroid_data_df[thyroid_data_df == "?"] <- NA
na_counts <- data.frame(
  Column = names(thyroid_data_df),
  Missing_Values = colSums(is.na(thyroid_data_df)),
  row.names = NULL
)
print(na_counts)

# Data encoding
thyroid_data_df$binaryClass <- ifelse(thyroid_data_df$binaryClass == "P", 0,
                               ifelse(thyroid_data_df$binaryClass == "N", 1, NA))

thyroid_data_df$sex <- ifelse(thyroid_data_df$sex == "F", 1,
                       ifelse(thyroid_data_df$sex == "M", 0, NA))

thyroid_data_df[thyroid_data_df == "t"] <- 1
thyroid_data_df[thyroid_data_df == "f"] <- 0

thyroid_data_df

str(thyroid_data_df)

# Converting character columns to numeric
char_cols <- sapply(thyroid_data_df, is.character)

thyroid_data_df[char_cols] <- lapply(thyroid_data_df[char_cols], function(x) as.numeric(as.character(x)))

str(thyroid_data_df)

# Replace data points '?' with NA
thyroid_data_df[thyroid_data_df == "?"] <- NA
na_counts <- data.frame(
  Column = names(thyroid_data_df),
  Missing_Values = colSums(is.na(thyroid_data_df)),
  row.names = NULL
)
print(na_counts)

# Impute missing numeric values using the column mean
thyroid_data_df$age[is.na(thyroid_data_df$age)] <- mean(thyroid_data_df$age, na.rm = TRUE)
thyroid_data_df$sex[is.na(thyroid_data_df$sex)] <- mean(thyroid_data_df$sex, na.rm = TRUE)

# Impute missing numeric values using the column mean
cols_to_impute <- c("TSH", "T3", "TT4", "T4U", "FTI")

# Mean imputation
thyroid_data_df <- thyroid_data_df %>% 
  mutate(across(all_of(cols_to_impute), ~ ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Replace data points '?' with NA
thyroid_data_df[thyroid_data_df == "?"] <- NA
na_counts <- data.frame(
  Column = names(thyroid_data_df),
  Missing_Values = colSums(is.na(thyroid_data_df)),
  row.names = NULL
)
print(na_counts)

# Filtering out outliers
thyroid_data_df <- subset(thyroid_data_df, age < 100)

# Preview cleaned dataset
thyroid_data_df

# Preview standardised structured of the dataset
str(thyroid_data_df)

## Step 3: Exploratory Data Analysis (EDA)

# Distribution of Binary Class (Thyroid vs Normal)
counts <- table(thyroid_data_df$binaryClass)

bar_positions <- barplot(counts,
                 main = "Distribution of Binary Class (Thyroid vs Normal)",
                 xlab = "Class",
                 ylab = "Count",
                 col = "lightgreen",
                 border = "white")

text(x = bar_positions,
     y = counts - 400,
     label = counts,
     pos = 3,
     cex = 0.8)

# Scatterplot between Age and TSH levels
plot(thyroid_data_df$age, thyroid_data_df$TSH,
     main = "Scatterplot of Age vs TSH",
     xlab = "Age", ylab = "TSH",
     pch = 19, col = "steelblue")

# Add a linear regression line to observe trend
abline(lm(TSH ~ age, data = thyroid_data_df), col = "red", lwd = 2)

correlation <- cor(thyroid_data_df$age, thyroid_data_df$TSH, use = "complete.obs")
cat("The correlation between Age and TSH is: ", round(correlation, 3), "\n")

# Correlation Matrix of Thyroid-Related Features
selected_cols <- c("age", "sex", "FTI", "T3", "TT4", "TSH", "binaryClass")
subset_data <- thyroid_data_df[, selected_cols]

# Compute correlation matrix
cor_matrix <- cor(subset_data, use = "complete.obs")

library(corrplot)

corrplot(cor_matrix,
         method = "color",
         tl.col = "black",
         addCoef.col = "black",
         number.cex = 0.8,
         title = "Correlation Matrix of Thyroid-Related features",
         mar = c(0, 0, 2, 0))

## Step 4: Model Training

### Regression

# Data preparation
rf_thyroid_data <- thyroid_data_df[!is.na(thyroid_data_df$TSH), ]

# Remove all columns that contains 'measured'
rf_thyroid_data <- rf_thyroid_data[, !grepl("measured", names(rf_thyroid_data))]

# Remove the 'binaryClass' column
rf_thyroid_data <- subset(rf_thyroid_data, select = -binaryClass)

rf_thyroid_data

# Splitting the dataset into training and test sets
set.seed(42)
sample_index <- sample(1:nrow(rf_thyroid_data), 0.7 * nrow(rf_thyroid_data))

thyroid_regress_train_data <- rf_thyroid_data[sample_index, ]
thyroid_regress_test_data <- rf_thyroid_data[-sample_index, ]

# Random Forest - Training
library(randomForest)

rf_model <- randomForest(TSH ~ ., data = thyroid_regress_train_data, ntree = 100, importance = TRUE)
rf_preds_regress_train <- predict(rf_model, thyroid_regress_train_data)
rf_preds_regress_test <- predict(rf_model, thyroid_regress_test_data)

# Linear Regression - Training
lm_model <- lm(TSH ~ ., data = thyroid_regress_train_data)
lm_preds_regress_train <- predict(lm_model, thyroid_regress_train_data)
lm_preds_regress_test <- predict(lm_model, thyroid_regress_test_data)

# XGBoost - Training
library(xgboost)

train_matrix <- xgb.DMatrix(data = as.matrix(thyroid_regress_train_data
                            [, -which(names(thyroid_regress_train_data) == "TSH")]),
                            label = thyroid_regress_train_data$TSH)

test_matrix <- xgb.DMatrix(data = as.matrix(thyroid_regress_test_data
                           [, -which(names(thyroid_regress_test_data) == "TSH")]),
                           label = thyroid_regress_test_data$TSH)

xgb_model <- xgboost(data = train_matrix,
                     label = thyroid_regress_train_data$TSH,
                     objective = "reg:squarederror",
                     nrounds = 100,
                     verbose = 0)

xgb_preds_regress_train <- predict(xgb_model, train_matrix)
xgb_preds_regress_test <- predict(xgb_model, test_matrix)

### Classification

# Data preparation
thyroid_data_df$binaryClass <- as.factor(thyroid_data_df$binaryClass)

# Remove all columns that contains 'measured'
clf_thyroid_data <- thyroid_data_df[, !grepl("measured", names(thyroid_data_df))]

# Remove TSH as its also the target variable
clf_thyroid_data <- clf_thyroid_data[, setdiff(names(clf_thyroid_data), "TSH")]

clf_thyroid_data

# Splitting the dataset into training and test sets
set.seed(42)
sample_index <- sample(1:nrow(clf_thyroid_data), 0.7 * nrow(clf_thyroid_data))

thyroid_classify_train_data <- clf_thyroid_data[sample_index, ]
thyroid_classify_test_data <- clf_thyroid_data[-sample_index, ]

# Apply SMOTE to balance class distribution by oversampling class 1 (Negative Thyroid)
library(smotefamily)

smote_result <- SMOTE(X = thyroid_classify_train_data
                      [, -which(names(thyroid_classify_train_data) == "binaryClass")],
                      target = thyroid_classify_train_data$binaryClass,
                      K = 5)
smote_classify_train <- smote_result$data
smote_classify_train$binaryClass <- as.factor(smote_classify_train$class)
smote_classify_train$class <- NULL

# Random Forest - Training
library(randomForest)

rf_clf <- randomForest(binaryClass ~ ., data = smote_classify_train, 
                       ntree = 100, importance = TRUE)
rf_preds_classify_train <- predict(rf_clf, thyroid_classify_train_data)
rf_preds_classify_test <- predict(rf_clf, thyroid_classify_test_data)

## Step 5: Model Evaluation

### Regression

library(Metrics)

r2_score <- function(actual, predicted) {
  cor(actual, predicted)^2
}

results_df <- data.frame(
  Model = c("Random Forest", "Linear Regression", "XGBoost"),
  R_squared_Train = round(c(
    r2_score(thyroid_regress_train_data$TSH, rf_preds_regress_train),
    r2_score(thyroid_regress_train_data$TSH, lm_preds_regress_train),
    r2_score(thyroid_regress_train_data$TSH, xgb_preds_regress_train)
  ), 3),
  R_squared_Test = round(c(
    r2_score(thyroid_regress_test_data$TSH, rf_preds_regress_test),
    r2_score(thyroid_regress_test_data$TSH, lm_preds_regress_test),
    r2_score(thyroid_regress_test_data$TSH, xgb_preds_regress_test)
  ), 3),
  RMSE = round(c(
    rmse(thyroid_regress_test_data$TSH, rf_preds_regress_test),
    rmse(thyroid_regress_test_data$TSH, lm_preds_regress_test),
    rmse(thyroid_regress_test_data$TSH, xgb_preds_regress_test)
  ), 3)
)

library(iml)

# Remove TSH from features
X <- thyroid_regress_test_data[, setdiff(names(thyroid_regress_test_data), "TSH")]
y <- thyroid_regress_test_data$TSH

# Create iml Predictor object
predictor <- Predictor$new(
  model = rf_model,
  data = X,
  y = y
)

# Choose a single observation to interpret
shap <- Shapley$new(predictor, x.interest = X[1, ])
plot(shap)

### Classification

library(caret)

# Confusion matrix
conf_mat <- confusionMatrix(rf_preds_classify_test, thyroid_classify_test_data$binaryClass)
print(conf_mat)

cat("Test Accuracy:", round(mean(rf_preds_classify_test == thyroid_classify_test_data$binaryClass), 3), "\n")

library(iml)

# Remove TSH from features
X <- thyroid_classify_test_data[, setdiff(names(thyroid_classify_test_data), "binaryClass")]
y <- thyroid_classify_test_data$binaryClass

# Create iml Predictor object
predictor <- Predictor$new(
  model = rf_clf,
  data = X,
  y = y,
  type = "prob"
)

# Choose a single observation to interpret
shap <- Shapley$new(predictor, x.interest = X[1, ])
plot(shap)