
library(tidyverse)
library(tidymodels)
library(vroom)
library(ggmosaic)
library(ggplot2)
library(embed) 
library(doParallel)
library(discrim)
library(themis)

setwd("//wsl.localhost/Ubuntu/home/fidgetcase/stat348/Amazon")

train <- vroom("train.csv")
test <- vroom("test.csv")

train$ACTION <- as.factor(train$ACTION)

cl <- makePSOCKcluster(5)
registerDoParallel(cl)

# EDA ---------------------------------------------------------------------

dplyr::glimpse(train)

#ggplot(data= train) + geom_mosaic(aes(x=product(RESOURCE), fill=ACTION))



# Recipe ------------------------------------------------------------------


my_recipe <- recipe(ACTION ~ ., data=train) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_smote(all_outcomes(), neighbors= 3) %>%
  step_downsample()

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)




# Logistic Regression -------------------------------------------


logRegModel <- logistic_reg() %>% #Type of model
  set_engine("glm")

logReg_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logRegModel) %>%
  fit(data = train)

lr_preds <- logReg_wf %>%
  predict(new_data = test, type = "prob")

Preds <-  lr_preds %>%
  bind_cols(test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

vroom_write(x=Preds, file= "lr_preds.csv", delim=",")


# PR recipe ---------------------------------------------

targetEncodeRecipe <- recipe(ACTION~., data=train) %>%
  step_mutate_at(all_numeric_predictors(), fn=factor) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome=vars(ACTION))



# apply the recipe to your data
prep <- prep(targetEncodeRecipe)
baked <- bake(prep, new_data = train)

# Penalized Regression ---------------------------------------------------------------


my_mod <- logistic_reg(mixture= tune(),
                       penalty= tune()) %>% #Type of model
  set_engine("glmnet")


penlegReg_wf <- workflow() %>%
add_recipe(targetEncodeRecipe) %>%
add_model(my_mod)

## Grid of values to tune over
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 1) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats= 1)

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
  predict(new_data = test, type = "prob")

Preds <-  plr_preds %>%
  bind_cols(test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

vroom_write(x=Preds, file= "plr_preds.csv", delim=",")

# Knn ---------------------------------------------------------------------

## knn model
knn_model <- nearest_neighbor(neighbors=floor(sqrt(length(train$ACTION)))) %>% # set or tune
  set_mode("classification") %>%
  set_engine("kknn")

knn_wf <- workflow() %>%
  add_recipe(targetEncodeRecipe) %>%
  add_model(knn_model) %>%
  fit(data = train)

## Fit or Tune Model HERE
knn_preds <- predict(knn_wf, new_data=test, type="prob")

Preds <-  knn_preds %>%
  bind_cols(test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

vroom_write(x=Preds, file= "knnpreds.csv", delim=",")

# Random Forest -----------------------------------------------------------

rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees= 1000) %>%
set_engine("ranger") %>%
set_mode("classification")

randf_wf <- workflow() %>%
  add_recipe(targetEncodeRecipe) %>%
  add_model(rf_mod)

## Grid of values to tune over
grid_of_tuning_params <- grid_regular(mtry(range= c(1, ncol(baked)-1)),
                                      min_n(),
                                      levels = 10) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats= 1)



  CV_results <- randf_wf %>%
    tune_grid(resamples=folds,
              grid=grid_of_tuning_params,
              metrics= metric_set(roc_auc)) #Or leave metrics NULL


  ## Find Best Tuning Parameters
  bestTune <- CV_results %>%
    select_best(metric="roc_auc")

  ## Finalize the Workflow & fit it
  finalrf_wf <-
    randf_wf %>%
    finalize_workflow(bestTune) %>%
    fit(data=train)

  ## Predict
  rf_preds <- finalrf_wf %>%
    predict(new_data = test, type = "prob")

  Preds <-  rf_preds %>%
    bind_cols(test) %>%
    rename(ACTION=.pred_1) %>%
    select(id, ACTION)

  vroom_write(x=Preds, file= "randomforest.csv", delim=",")


# Naive bayes -------------------------------------------------------------------

## nb model
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes") # install discrim library for the naiveb

nb_wf <- workflow() %>%
add_recipe(targetEncodeRecipe) %>%
add_model(nb_model)

# Tune smoothness and Laplace here
## Grid of values to tune over
tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 3) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=2)

## Run the CV
CV_results <- nb_wf %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics = metric_set(roc_auc)) #Or leave metrics NULL

bestTune <- CV_results %>%
  select_best(metric="roc_auc")

finalnb_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)


## Predict
nb_preds <- predict(finalnb_wf, new_data=test, type="prob")

Preds <-  nb_preds %>%
  bind_cols(test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

vroom_write(x=Preds, file= "naivebayes.csv", delim=",")



# Radial Polynomial -------------------------------------------------------



svmLinear <- svm_linear(cost=tune()) %>% # set or tune
  set_mode("classification")         %>%
  set_engine("kernlab")


Linearsvm_wf <- workflow() %>%
  add_recipe(my_recipe)    %>%
  add_model(svmLinear)

# Tune smoothness and Laplace here
## Grid of values to tune over
tuning_grid <- grid_regular(cost(),
                            levels = 1) ## L^2 total tuning possibilities

## Split data for CV
folds <- vfold_cv(train, v = 5, repeats=1)

## Run the CV
CV_results <- Linearsvm_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics = NULL) #Or leave metrics NULL

bestTune <- CV_results %>%
  select_best(metric="roc_auc")

final_Linearsvm_wf <- Linearsvm_wf %>%
  finalize_workflow(bestTune)      %>%
  fit(data = train)

## Fit or Tune Model HERE
svm_predict <- predict(final_Linearsvm_wf, new_data=test, type="prob")



# Kaggle Submission -------------------------------------------------------

stopCluster(cl)

Preds <-  svm_predict %>%
  bind_cols(test) %>%
  rename(ACTION=.pred_1) %>%
  select(id, ACTION)

vroom_write(x=Preds, file= "svmL_preds.csv", delim=",")
