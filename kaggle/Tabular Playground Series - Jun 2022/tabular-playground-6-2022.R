library(tidyverse)
#install.packages("tidymodels")
library(tidymodels)
#install.packages("ggstatsplot")
library(ggstatsplot)
#install.packages("skimr")
library(skimr)
#install.packages("mice")
library(mice)
library(patchwork)


missing <- read_csv("data.csv",show_col_types = FALSE)
sample <- read_csv("sample_submission.csv",show_col_types = FALSE)
skim(missing)

# There are 81 columns with one of them being the ID variable. If we check the column names, we can say that these columns are divided into 4 groups :-
# - F_1 - containing 15 variables
# - F_2 - containing 25 variables
# - F_3 - containing 25 variables
# - F_4 - containing 15 variables
# Checking the column complete rate tells us that apart from group F_2 all other groups have missing values. This missing rate is approximately 1.9 percent.
# We will be splitting the dataset into 4 groups and using it as needed - for visualizations and imputations. ID variable will also be removed.

missing <- missing %>%
  select(-c("row_id"))

first <- missing %>%
  select(1:15)

second <- missing %>%
  select(16:40)

third <- missing %>%
  select(41:65)

fourth <- missing %>%
  select(66:80)

a1 <- gghistostats(first,
                   F_1_0,
                   results.subtitle = FALSE)
a2 <- gghistostats(first,
                   F_1_1,
                   results.subtitle = FALSE)
a3 <- gghistostats(first,
                   F_1_2,
                   results.subtitle = FALSE)
a4 <- gghistostats(first,
                   F_1_3,
                   results.subtitle = FALSE)
a5 <- gghistostats(first,
                   F_1_4,
                   results.subtitle = FALSE)
a6 <- gghistostats(first,
                   F_1_5,
                   results.subtitle = FALSE)
a7 <- gghistostats(first,
                   F_1_6,
                   results.subtitle = FALSE)
a8 <- gghistostats(first,
                   F_1_7,
                   results.subtitle = FALSE)
a9 <- gghistostats(first,
                   F_1_8,
                   results.subtitle = FALSE)
a10 <- gghistostats(first,
                    F_1_9,
                    results.subtitle = FALSE)
a11 <- gghistostats(first,
                    F_1_10,
                    results.subtitle = FALSE)
a12 <- gghistostats(first,
                    F_1_11,
                    results.subtitle = FALSE)
a13 <- gghistostats(first,
                    F_1_12,
                    results.subtitle = FALSE)
a14 <- gghistostats(first,
                    F_1_13,
                    results.subtitle = FALSE)
a15 <- gghistostats(first,
                    F_1_14,
                    results.subtitle = FALSE)

patch1 <- ( a1 | a2 | a3 )/
  ( a4 | a5 | a6 )/
  ( a7 | a8 | a9 )/
  ( a10 | a11 | a12 )/
  ( a13 | a14 | a15 )

patch1 + plot_annotation(
  title = "First group(F_1) variables",
  caption = "Data source: Kaggle.com, Tabular Playground Series - Jun 2022")

#As discussed before, this set of variables doesnt contain any NAs. 
# Also this is the only set with discrete values. However, 
# if you look their distributions, these are akin to normal distribution.

c1 <- gghistostats(third,
                   F_3_0,
                   results.subtitle = FALSE)
c2 <- gghistostats(third,
                   F_3_1,
                   results.subtitle = FALSE)
c3 <- gghistostats(third,
                   F_3_2,
                   results.subtitle = FALSE)
c4 <- gghistostats(third,
                   F_3_3,
                   results.subtitle = FALSE)
c5 <- gghistostats(third,
                   F_3_4,
                   results.subtitle = FALSE)
c6 <- gghistostats(third,
                   F_3_5,
                   results.subtitle = FALSE)
c7 <- gghistostats(third,
                   F_3_6,
                   results.subtitle = FALSE)
c8 <- gghistostats(third,
                   F_3_7,
                   results.subtitle = FALSE)
c9 <- gghistostats(third,
                   F_3_8,
                   results.subtitle = FALSE)
c10 <- gghistostats(third,
                    F_3_9,
                    results.subtitle = FALSE)
c11 <- gghistostats(third,
                    F_3_10,
                    results.subtitle = FALSE)
c12 <- gghistostats(third,
                    F_3_11,
                    results.subtitle = FALSE)
c13 <- gghistostats(third,
                    F_3_12,
                    results.subtitle = FALSE)
c14 <- gghistostats(third,
                    F_3_13,
                    results.subtitle = FALSE)
c15 <- gghistostats(third,
                    F_3_14,
                    results.subtitle = FALSE)
c16 <- gghistostats(third,
                    F_3_15,
                    results.subtitle = FALSE)
c17 <- gghistostats(third,
                    F_3_16,
                    results.subtitle = FALSE)
c18 <- gghistostats(third,
                    F_3_17,
                    results.subtitle = FALSE)
c19 <- gghistostats(third,
                    F_3_18,
                    results.subtitle = FALSE)
c20 <- gghistostats(third,
                    F_3_19,
                    results.subtitle = FALSE)
c21 <- gghistostats(third,
                    F_3_20,
                    results.subtitle = FALSE)
c22 <- gghistostats(third,
                    F_3_21,
                    results.subtitle = FALSE)
c23 <- gghistostats(third,
                    F_3_22,
                    results.subtitle = FALSE)
c24 <- gghistostats(third,
                    F_3_23,
                    results.subtitle = FALSE)
c25 <- gghistostats(third,
                    F_3_24,
                    results.subtitle = FALSE)

patch3 <- ( c1 | c2 | c3 )/
  ( c4 | c5 | c6 )/
  ( c7 | c8 | c9 )/
  ( c10 | c11 | c12 )/
  ( c13 | c14 | c15 )/
  ( c16 | c17 | c18 )/
  ( c19 | c20 | c21 )/
  ( c22 | c23 | c24 )/
  ( c25 )

patch3 + plot_annotation(
  title = "Third group(F_3) variables",
  caption = "Data source: Kaggle.com, Tabular Playground Series - Jun 2022")



#Similar to the first group, all the variables in this group are normally distributed without any skewness, 
#apart from F_1_19 and F_1_21 which are negatively skewed.

d1 <- gghistostats(fourth,
                   F_4_0,
                   results.subtitle = FALSE)
d2 <- gghistostats(fourth,
                   F_4_1,
                   results.subtitle = FALSE)
d3 <- gghistostats(fourth,
                   F_4_2,
                   results.subtitle = FALSE)
d4 <- gghistostats(fourth,
                   F_4_3,
                   results.subtitle = FALSE)
d5 <- gghistostats(fourth,
                   F_4_4,
                   results.subtitle = FALSE)
d6 <- gghistostats(fourth,
                   F_4_5,
                   results.subtitle = FALSE)
d7 <- gghistostats(fourth,
                   F_4_6,
                   results.subtitle = FALSE)
d8 <- gghistostats(fourth,
                   F_4_7,
                   results.subtitle = FALSE)
d9 <- gghistostats(fourth,
                   F_4_8,
                   results.subtitle = FALSE)
d10 <- gghistostats(fourth,
                    F_4_9,
                    results.subtitle = FALSE)
d11 <- gghistostats(fourth,
                    F_4_10,
                    results.subtitle = FALSE)
d12 <- gghistostats(fourth,
                    F_4_11,
                    results.subtitle = FALSE)
d13 <- gghistostats(fourth,
                    F_4_12,
                    results.subtitle = FALSE)
d14 <- gghistostats(fourth,
                    F_4_13,
                    results.subtitle = FALSE)
d15 <- gghistostats(fourth,
                    F_4_14,
                    results.subtitle = FALSE)

patch4 <- ( d1 | d2 | d3 )/
  ( d4 | d5 | d6 )/
  ( d7 | d8 | d9 )/
  ( d10 | d11 | d12 )/
  ( d13 | d14 | d15 )

patch4 + plot_annotation(
  title = "Fourth group(F_4) variables",
  caption = "Data source: Kaggle.com, Tabular Playground Series - Jun 2022")

# First group - mean - Unconditional mean imputation
# Second group - No Imputation needed
# Third group - mean - Unconditional mean imputation
# Fourth group - mean - Unconditional mean imputation


mean1 <- as.data.frame(first)
mean1 <- mice(data = mean1,
              m=1,
              seed = 1313,
              method = "mean",
              maxit = 1,
              print = FALSE)
mean1 <- complete(mean1)

mean3<- as.data.frame(third)
mean3 <- mice(data = mean3,
              m=1,
              seed = 1313,
              method = "mean",
              maxit = 1,
              print = FALSE)
mean3 <- complete(mean3)

mean4 <- as.data.frame(fourth)
mean4 <- mice(data = mean4,
              m=1,
              seed = 1313,
              method = "mean",
              maxit = 1,
              print = FALSE)
mean4 <- complete(mean4)

final <- cbind(mean1,second,mean3,mean4)

rows_sub <- as.numeric(gsub("([0-9]+)[-]([^ ]+)","\\1",sample$`row-col`))

cols_sub <- as.character(gsub("([0-9]+)[-]([^ ]+)","\\2",sample$`row-col`))

index <- nrow(final)

values <- numeric(nrow(final))
head(final)
for(i in 1:index){
  values[i] <- final[rows_sub[i]+1,cols_sub[i]]
}

submission_meanforall <- sample[1]

submission_meanforall[,2] <- data.frame(matrix(unlist(values)))

colnames(submission_meanforall)[2] <- "value"

write_csv(submission_meanforall,"submission_meanforall.csv")
head(submission_meanforall,10)
#This submission gives us a public score of 1.41613, 
# which has also been mentioned on the leaderboard as a benchmark. 
# Let us do some more exploration before we make another submission, 
# like checking correlation of variables in each group.


ggcorrmat(data = first,
          colors   = c("#B2182B", "white", "#4D4D4D"),
          title    = "Correlation among variables of the first group")



ggcorrmat(data = second,
          colors   = c("#B2182B", "white", "#4D4D4D"),
          title    = "Correlation among variables of the second group")


ggcorrmat(data = third,
          colors   = c("#B2182B", "white", "#4D4D4D"),
          title    = "Correlation among variables of the third group")


ggcorrmat(data = fourth,
          colors   = c("#B2182B", "white", "#4D4D4D"),
          title    = "Correlation among variables of the fourth group")


#Fourth group - norm.predict - Linear regression, predicted values
# Next Submission
# First group - mean - Unconditional mean imputation
# Second group - No Imputation needed
# Third group - mean - Unconditional mean imputation
# Fourth group - norm.predict - Linear regression, predicted values


norm.predict4 <- as.data.frame(fourth)
norm.predict4 <- mice(data = norm.predict4,
                      m=1,
                      seed = 1313,
                      method = "norm.predict",
                      maxit = 1,
                      print = FALSE)
norm.predict4 <- complete(norm.predict4)

final <- cbind(mean1,second,mean3,norm.predict4)

rows_sub <- as.numeric(gsub("([0-9]+)[-]([^ ]+)","\\1",sample$`row-col`))

cols_sub <- as.character(gsub("([0-9]+)[-]([^ ]+)","\\2",sample$`row-col`))

index <- nrow(final)

values <- numeric(nrow(final))

for(i in 1:index){
  values[i] <- final[rows_sub[i]+1,cols_sub[i]]
}

submission_mean13normpredict4 <- sample[1]

submission_mean13normpredict4[,2] <- data.frame(matrix(unlist(values)))

colnames(submission_mean13normpredict4)[2] <- "value"

write_csv(submission_mean13normpredict4,"submission_mean13normpredict4.csv")
head(submission_mean13normpredict4,10)
#This gives us a score of Score: 1.08780, 
# which is a significant improvement from the score where we imputed using mean.
# Rest of the methods associated with linear regression were also tried and following are the scores obtained.




#Kindly note that we ran just one iteration of this imputation. Let us retry the imputation with 100 iterations


norm.predict4 <- as.data.frame(fourth)
norm.predict4 <- mice(data = norm.predict4,
                      m=1,
                      seed = 1313,
                      method = "norm.predict",
                      maxit = 100,
                      print = FALSE)
  norm.predict4 <- complete(norm.predict4)

final <- cbind(mean1,second,mean3,norm.predict4)

rows_sub <- as.numeric(gsub("([0-9]+)[-]([^ ]+)","\\1",sample$`row-col`))

cols_sub <- as.character(gsub("([0-9]+)[-]([^ ]+)","\\2",sample$`row-col`))

index <- nrow(final)

values <- numeric(nrow(final))

for(i in 1:index)
{
  values[i] <- final[rows_sub[i]+1,cols_sub[i]]
}

submission_mean13normpredict4iter100 <- sample[1]

submission_mean13normpredict4iter100[,2] <- data.frame(matrix(unlist(values)))

colnames(submission_mean13normpredict4iter100)[2] <- "value"

write_csv(submission_mean13normpredict4iter100,"submission_mean13normpredict4iter100.csv")
head(submission_mean13normpredict4iter100,10)
#This gives us a score of 0.93567