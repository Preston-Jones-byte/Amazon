
library(tidyverse)
library(tidymodels)
library(vroom)
library(ggmosaic)
library(ggplot2)
library(embed) 

setwd("//wsl.localhost/Ubuntu/home/fidgetcase/stat348/Amazon")

train <- vroom("train.csv")
test <- vroom("test.csv")

train$ACTION <- as.factor(train$ACTION)

# EDA ---------------------------------------------------------------------

dplyr::glimpse(train)

#ggplot(data= train) + geom_mosaic(aes(x=product(RESOURCE), fill=ACTION))



# Recipe ------------------------------------------------------------------


my_recipe <- recipe(ACTION ~ ., data=train) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_dummy(all_nominal_predictors())  

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)



# Logistic Regression -------------------------------------------


# logRegModel <- logistic_reg() %>% #Type of model
#   set_engine("glm")

# logReg_wf <- workflow() %>%
#   add_recipe(my_recipe) %>%
#   add_model(logRegModel) %>%
#   fit(data = train)



# Penalized regression recipe ---------------------------------------------

targetEncodeRecipe <- recipe(ACTION~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())


# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

# Penalized ---------------------------------------------------------------


my_mod <- logistic_reg(mixture= tune(), 
                       penalty= tune()) %>% #Type of model
  set_engine("glmnet")


penlegReg_wf <- workflow() %>%
add_recipe(targetEncodeRecipe) %>%
add_model(my_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 5) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
CV_results <- penlegReg_wf %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics = metric_set(roc_auc)) #Or leave metrics NULL

bestTune <- CV_results %>%
  select_best(metric="roc_auc")

final_plr_wf <- penlegReg_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)
  

plr_preds <- final_plr_wf %>%
  predict(new_data = test)


# Kaggle Submission -------------------------------------------------------

logRegPreds <-  plr_preds %>%
  bind_cols( test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

vroom_write(x=logRegPreds, file= "penlegreg.csv", delim=",") 
