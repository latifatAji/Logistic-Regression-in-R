## Predicting the onset of Heart Failure using Machine Learning Techniques
#
#
## Data Source: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/
#
#
## Author: latifat ajibola


## Setup
require(tidyverse)
require(GGally)
require(caret)
require(DataExplorer)
rm(list = ls()
   
   ## Load Data set
   setwd("/Users/lateefah/Downloads/")
   heart_df <- read_csv("heart.csv")
   heart_df %>% head()
   heart_df %>% spec() ## check column specification
   glimpse(heart_df)  ## check data structure
   
   ## Check for missing observations
   colSums(is.na(heart_df))
   ## Change levels of some categorical variables
   heart_df <- heart_df %>% 
     mutate(HeartDisease = ifelse(HeartDisease == 1, "Heart_Disease", "Normal"))
   heart_df <- heart_df %>% 
     mutate(FastingBS = ifelse(FastingBS == 1, ">120 mg/dl", "<120 mg/dl"))
   ## Examine the distribution of the outcome to know whether they are imbalance or not
   heart_df %>% 
     group_by(HeartDisease) %>%
     summarise(size = length(HeartDisease)) %>% 
     mutate(proportion = size/sum(size)) %>% mutate_at(vars(proportion), funs(round(., 2))) 
   ## Convert all character columns to factor
   heart_df <- mutate_if(heart_df, is.character, as.factor)
   ## Convert the outcome to factor
   heart_df <- mutate_at(heart_df, vars(HeartDisease, FastingBS), as.factor)
   glimpse(heart_df)
   ## Define outcome and predictors
   y <- heart_df %>% 
     select(HeartDisease) 
   
   x <- heart_df %>% 
     select(!(HeartDisease))
   
   ## Data Spiting
   set.seed(2022)
   train_index <- caret::createDataPartition(y = heart_df$HeartDisease, times = 1, p = 0.7, list = FALSE)
   train_set <- heart_df[train_index, ]
   test_set <- heart_df[-train_index, ]
   
   plot_str(train_set)
   plot_intro(train_set)
   ## Exploring the training Sample
   
   y_train <- train_set %>% 
     select(HeartDisease)
   
   non_numeric_cols <- train_set %>% 
     select_if(is.factor) 
   
   numeric_cols <- train_set %>% 
     select_if(negate(is.factor))
   
   numeric_and_output <- bind_cols(numeric_cols, y_train)
   
   ## Visualization
   ggplot(numeric_and_output, aes(x = HeartDisease, fill = HeartDisease)) +
     geom_bar() + ggtitle("Distribution of Heart Disease") + theme_bw() +
     theme(plot.title = element_text(hjust = 0.5)) 
   
   ggpairs(numeric_and_output, columns = 1: (ncol(numeric_and_output) - 1),
           ggplot2::aes(colour = HeartDisease)) + theme_bw()
   
   ggplot(gather(numeric_cols), aes(value)) + 
     geom_histogram(bins = 10, color = 'black') + 
     facet_wrap(~key, scales = 'free_x')
   ## Correlation Plot
   corrplot::corrplot(cor(numeric_cols), order = "hclust", tl.cex = 1, addrect = 2)
   ## Bar Charts
   colms <- gather(non_numeric_cols %>% select(!(HeartDisease)))
   others <- rep(train_set$HeartDisease, 6)
   plot_data <- bind_cols(colms, others) %>% rename(HeartDisease = ...3)
   
   ggplot(plot_data, aes(value)) + 
     geom_bar(aes(fill = HeartDisease), position = "dodge") + 
     facet_wrap(~key, scales = 'free_x')
   
   ## Standardize
   scaled_numeric_train <- numeric_cols %>% scale() 
   train_data <- cbind(scaled_numeric_train, non_numeric_cols)
   
   non_numeric_cols_test <- test_set %>% 
     select_if(is.factor) 
   
   numeric_cols_test <- test_set %>% 
     select_if(negate(is.factor))
   
   centers <- apply(numeric_cols, 2, mean)
   sds <- apply(numeric_cols, 2, sd)
   
   scaled_numeric_test <- numeric_cols_test %>% scale(center = centers, scale = sds) 
   test_data <- cbind(scaled_numeric_test, non_numeric_cols_test)
   ## Predict using Logistic Regression Model
   train_data$HeartDisease <- ifelse(train_data$HeartDisease == "Heart_Disease", 1, 0)
   test_data$HeartDisease <- ifelse(test_data$HeartDisease == "Heart_Disease", 1, 0)
   logistic_reg <- glm(HeartDisease ~., data = train_data, family = 'binomial')
   summary(logistic_reg)
   logistic_pred <- predict(logistic_reg, test_data[,-ncol(test_data)], type = 'response')
   logistic_pred <- ifelse(logistic_pred > 0.5,1,0)
   
   confusionMatrix(data = factor(logistic_pred), reference = factor(test_data[,ncol(test_data)]), positive = '1')
   
   ## Trying Stepwise Regression
   stepwise_model <- MASS::stepAIC(logistic_reg, direction = 'both', trace = FALSE)
   summary(stepwise_model)
   
   stepwise_pred <- predict(stepwise_model, test_data[,-ncol(test_data)], type = 'response')
   stepwise_pred <- ifelse(stepwise_pred > 0.5, 1, 0)
   caret::confusionMatrix(data = factor(stepwise_pred), reference = factor(test_data[,ncol(test_data)]), positive = '1')
   ## Plot Variable Importance
   importance <- data.frame(stepwise_model$coefficients[-1])
   colnames(importance) <- c("Importance")
   importance %>%
     ggplot(aes(x = rownames(importance), 
                y = Importance)) +
     geom_bar(stat = "identity") +
     ylab("Importance") + xlab("Variable") +theme_bw() + coord_flip()
   