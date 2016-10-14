library(tm)
library(class)
library(plyr)
library(SnowballC)
train_in <- read.csv("data/train_in.csv",stringsAsFactors = FALSE)
train_out<- read.csv("data/train_out.csv")
test_in <- read.csv("datas/test_in.csv")
table(train_out$category)
train_in<-train_in[,2]
test_in<-test_in[,2]
###no category
b<-which(train_out$category=="category")
[1] 29024 60083 82990
train_in[b]
[1] "abstract" "abstract" "abstract"

train_in<-train_in[-b]
data_in<-c(train_in,test_in)

corp <- Corpus(VectorSource(data_in))

## stem document
corp <- tm_map(corp,stemDocument)
corp <- tm_map(corp, content_transformer(removePunctuation))
corp <- tm_map(corp, content_transformer(removeWords), stopwords("english"))
corp <- tm_map(corp, content_transformer(removeNumbers))
corp <- tm_map(corp, content_transformer(tolower))
corp <- tm_map(corp, content_transformer(stripWhitespace))
#creating term matrix with TF-IDF weighting
terms <-DocumentTermMatrix(corp,control = list(weighting = function(x) weightTfIdf(x, normalize = TRUE)))
tdm<-removeSparseTerms(terms,0.99)
tdm<-as.matrix(tdm)
tdm<-t(tdm)

### distance

kdistance<-function(x,X,k,y){
  distance<-apply(X,2,function(ss){sum((ss-x)^2)})
  ind<-which(order(distance)<=(k))
  max.group<-table(y[ind])[which.max(table(y[ind]))]
  if (length(max.group==1)){
    ynew<-names(max.group)
  } else{
  ynew<-sample(names(max.group),1
               ,prob=as.numeric(1/length(max.group)))}
  return(ynew)
}

train_out<-train_out[-b,]
train_out$category<-factor(train_out$category)

###k-fold cross validation

cvk<-function(X,m,k,y){
  max<-ceiling(ncol(X)/m)
  fold<-split(sample(1:ncol(X),ncol(X),replace=FALSE),
        ceiling(seq_along(1:ncol(X))/max))
  err<-NULL
  for (i in 1:m){
    ycvpred<-NULL
  ycvpred<-apply(X[,fold[[i]]],2
                 ,function(cc){kdistance(cc,X[,-fold[[i]]],k,y[-fold[[i]]])})
  err[i]<-sum(ifelse(ycvpred==y[fold[[i]]],0,1))/length(y[fold[[i]]])}
  return(mean(err))
}

### choose the best k nearest
kpoint<-c(1,5,10,15,20)
cverr<-NULL
set.seed(7643)
for (i in 1:length(kpoint)) {
  cverr[i]<-cvk(X=tdm[,1:88633],m=3,k=kpoint[i],y=train_out$category)
}
cverr

plot(kpoint,cverr,xlab="Number of neighbor",ylab="CV error",col=3,type='b',lty=2,pch=2)
valid_res<-apply(X=tdm[,1:88633],2
                  ,function(cc){kdistance(cc,tdm[,1:88633],k=10,train_out$category)})
table(valid_res,train_out$category)
valid_res    cs  math physics  stat
cs      13633 12744    8813  8316
math     9223  8412    5834  5215
physics  3227  2877    1922  1897
stat     2045  1888    1373  1214

###missclassification rate validation
sum(ifelse(valid_res==train_out$category,0,1))/88633

## test out
test_res<-apply(X=tdm[,88634:104280],2
,function(cc){kdistance(cc,tdm[,1:88633],k=10,train_out$category)})
