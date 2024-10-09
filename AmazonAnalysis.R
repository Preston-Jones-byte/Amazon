library(tidyverse)
library(tidymodels)
library(vroom)
library(ggmosaic)
library(ggplot2)
library(embed) 

setwd("//wsl.localhost/Ubuntu/home/fidgetcase/stat348/Amazon")

train <- vroom("train.csv")
test <- vroom("test.csv")


# EDA ---------------------------------------------------------------------

dplyr::glimpse(train)

ggplot(data= train) + geom_mosaic(aes(x=product(RESOURCE), fill=ACTION))



# Recipe ------------------------------------------------------------------


my_recipe <- recipe(ACTION ~ ., data=trainData) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>% 
  step_dummy(all_nominal_predictors())  %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))

# apply the recipe to your data
prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

length(baked)
