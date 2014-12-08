install.packages('kernlab')
library(kernlab)
data(spam) # UCI spambase data

## subsampling
set.seed(3435)
trainIndicator = rbinom(4601, size = 1, prob = 0.5)
table(trainIndicator)
trainSpam = spam[trainIndicator == 1,]
testSpam = spam[trainIndicator == 0,]

## Exploratory Data Analysis
names(trainSpam)
head(trainSpam)
table(trainSpam$type)
#nonspam    spam 
#1381     906 
plot(trainSpam$capitalAve~trainSpam$type)# daya highly skewed
plot(log(trainSpam$capitalAve+1)~trainSpam$type)# spams more cpaitalAve
plot(log10(trainSpam[,1:4]+1))# some are not correlated
hCluster = hclust(dist(t(trainSpam[,1:57])))
plot(hCluster)
hClusterUpdated = hclust(dist(t(log10(trainSpam[,1:57]+1))))
plot(hClusterUpdated) #dendrogram of cluster features

## Statistical Prediction/Modeling
trainSpam$numType = as.numeric(trainSpam$type)-1
costFunction = function(x,y) sum(x !=(y>0.5))
cvError=rep(NA,55)
library(boot)
for (i in 1:55){
  lmFormula = reformulate(names(trainSpam)[i],response="numType")
  glmFit = glm(lmFormula, family = "binomial", data=trainSpam)
  cvError[i] = cv.glm(trainSpam,glmFit, costFunction,2)$delta[2]
}
## Which predictor has minimum cross-validated error?
names(trainSpam)[which.min(cvError)] #[1] "charDollar"

## Use the best model from the group
predictionModel=glm(numType~charDollar, family="binomial",data=trainSpam)

## get predictions on the test set
predictionTest=predict(predictionModel , testSpam)
predictedSpam = rep("nonspam", dim(testSpam)[1])

## classify as spam for those with prob > 0.5 (cut-off)
predictedSpam[predictionModel$fitted > 0.5] = "spam"

## classification table
table(predictedSpam, testSpam$type)
# predictedSpam nonspam spam
# nonspam    1346  458
# spam         61  449

## error rate
(61 + 458)/(1346 + 458 +61 +449) # 0.2242869

## Interpret model

# fraction of characters with dollar signs can be used
# to predict if an email is Spam
# > 6.6% dollar signs is classified as Spam
# test set error rate 22.4% and ~78% test set accuracy

## Challenge results
