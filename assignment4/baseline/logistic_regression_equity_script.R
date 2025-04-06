
# -------------------------------
# Logistic Regression with Group Comparison (Equity Focus)
# -------------------------------

# Load libraries
library(tidyverse)
library(caret)
library(pROC)

# STEP 1: Load the dataset (call it recidivism_ga)
recidivism_ga<- read_csv("NIJ_s_Recidivism_Challenge_Full_Dataset_20240407.csv")
# STEP 2: Explore the data
glimpse(recidivism_ga)
summary(recidivism_ga)

# STEP 3: Create binary outcome variable
recidivism_ga <- recidivism_ga %>%
  mutate(recidivism_yes = if_else(Recidivism_Within_3years == TRUE, 1, 0))

recidivism_ga <- na.omit(recidivism_ga)
# STEP 4: Partition the data into training and testing sets
set.seed(1234)
trainIndex <- createDataPartition(recidivism_ga$recidivism_yes, p = 0.7, list = FALSE)
train <- recidivism_ga[trainIndex, ]
test <- recidivism_ga[-trainIndex, ]


# STEP 5: Fit logistic regression model
model <- glm(recidivism_yes ~ Age_at_Release + Gang_Affiliated + Percent_Days_Employed,
             data = train, family = "binomial")
summary(model)

# STEP 6: Predict probabilities on test set
test_probs <- predict(model, newdata = test, type = "response")
threshold <- 0.50
test_preds <- ifelse(test_probs > threshold, 1, 0)

# STEP 7: Evaluate overall model performance
confusionMatrix(as.factor(test_preds), as.factor(test$recidivism_yes), positive = "1")
roc_obj <- roc(test$recidivism_yes, test_probs)
plot(roc_obj, col = "blue")
auc(roc_obj)

# STEP 8: Explore group-wise performance by race
group_metric_summary <- test %>%
  mutate(pred = test_preds, prob = test_probs) %>%
  group_by(Race) %>%
  summarise(
    Sensitivity = sum(pred == 1 & recidivism_yes == 1) / sum(recidivism_yes == 1),
    Specificity = sum(pred == 0 & recidivism_yes == 0) / sum(recidivism_yes == 0),
    Precision = sum(pred == 1 & recidivism_yes == 1) / sum(pred == 1),
    N = n()
  )

print(group_metric_summary)

# DISCUSSION:
# - Which group has higher sensitivity? Specificity?
# - Would a different threshold affect groups differently? (try the exercise again with 0.75 cutoff)
# - Are there equity implications of this model's performance?
