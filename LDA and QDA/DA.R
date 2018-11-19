# iris<-read.csv('iris.txt',header=F) #loading and initialising the iris data so it can be used lines 1:8
# Y<-iris[,5]
# X<-iris[,-5]
# set.seed(0)
# index<-sample(1:nrow(iris),70)
# train<-iris[index,]
# test<-iris[-index,]
# t<-ncol(iris)

parkinsons<-read.csv('parkinsons.data') #loading and initialising parkinsons data so it can be used lines 10:26
status<-parkinsons[,18]
b<-parkinsons[,-18]
parkinsons<-cbind(b,status)#moving the class status to the end of the data set
parkinsons<-parkinsons[-1]# removing the non-numerical data
for (w in 1:nrow(parkinsons)) {
 if (parkinsons[w,23]==0) {#changing the class for parkinsons to -1 and 1 so it could be used with the same code
   parkinsons[w,23]<--1
 }
}
Y<-parkinsons[,23]#assigning the class and the attributes seperately
X<-parkinsons[,-23]
set.seed(0)
index<-sample(1:nrow(parkinsons),120)#random sample without replacement of the 120 points
train<-parkinsons[index,]
test<-parkinsons[-index,]
t<-ncol(parkinsons)
# calculating means of data set in order to determine the best attributes to use
mean.0<-apply(X[(Y==-1),],2,mean)
mean.1<-apply(X[(Y==1),],2,mean)
st.dev<-apply(X,2,sd)
at<-abs((mean.1-mean.0)/st.dev)
p<-2   #works with any p chosen apart from over 18 with the parkinsons data
q<-order(-at)
r<-q[1:p]
train.X<-train[,r]#using only best attributes to obtain predictions
train.Y<-train[,t]
K<-length(unique(train.Y))
test.X<-test[,r]
test.Y<-test[,t]
N<-length(test.Y)
if (p==1){
  train.X<-matrix(c(train.X),ncol=1)
test.X<-matrix(c(test.X),ncol=1)#convert a vector to matrix form so it can be used with the covariance formula
}
class0<-sum(train.Y==-1)# calculation nk
class1<-sum(train.Y==1)
n<-length(train.Y)
prior0<-class0/n # calculating priors
prior1<-class1/n
priors<-rbind(prior0,prior1)
if (p==1){
mean0<-sum(train.X[(train.Y==-1)])/class0
mean1<-sum(train.X[(train.Y==1)])/class1# apply doesnt work with vectors so I had to calculate for p=1 seperately
} else{
mean0<-(apply(train.X[(train.Y==-1),],2,sum))/class0
mean1<-(apply(train.X[(train.Y==1),],2,sum))/class1# calculating mean of each class using apply to find mean of each column
}
means<-rbind(mean0,mean1)
#covariance matrices for a specific K
cov0 <- matrix(0,p,p)
for (i in 1:n) {# using covariance formula to find mean of each class 
  if (train.Y[i]==-1){
    for (l in 1:p) {
      for (j in 1:p) {
        cov0[l,j] <- cov0[l,j]+(train.X[i,l]-means[1,l])*(train.X[i,j]-means[1,j])
      }
    }
    
  } 
}
cov1 <- matrix(0,p,p)
for (i in 1:n) {
  if (train.Y[i]==1){
    for (l in 1:p) {
      for (j in 1:p) {
        cov1[l,j] <- cov1[l,j]+(train.X[i,l]-means[2,l])*(train.X[i,j]-means[2,j])
      }
    }
    
  } 
}
ldacov<-(cov0+cov1)/(n-K) # covariance for LDA is just cov for each class divided by n-k
qdacov0<-cov0/(class0-1)# covariance for each class divided by nk-1
qdacov1<-cov1/(class1-1)

predict.Y<-c()
delta0<-c()
delta1<-c()
for (m in 1:N){
  
    delta0[m]<-sum(solve(ldacov,t(test.X[m,]))*means[1,])-1/2*sum(solve(ldacov,t(t(means[1,])))*means[1,])+log(priors[1])
    delta1[m]<-sum(solve(ldacov,t(test.X[m,]))*means[2,])-1/2*sum(solve(ldacov,t(t(means[2,])))*means[2,])+log(priors[2])
  if(delta0[m]>delta1[m]){
    predict.Y[m]<- -1
  }
    else{
      predict.Y[m]<- 1
    }
   #using formula for lda to predict the assignment of each new test point 
}
predictY <- c()
qdadelta0 <- c()
qdadelta1 <- c()

for (e in 1:N) {
  
  qdadelta0[e] <- -1/2*log(det(qdacov0))-1/2*sum(solve(qdacov0,t(test.X[e,]))*test.X[e,])+sum(solve(qdacov0,t(test.X[e,]))*means[1,])-1/2*sum(solve(qdacov0,t(t(means[1,])))*means[1,])+log(priors[1])
  qdadelta1[e] <- -1/2*log(det(qdacov1))-1/2*sum(solve(qdacov1,t(test.X[e,]))*test.X[e,])+sum(solve(qdacov1,t(test.X[e,]))*means[2,])-1/2*sum(solve(qdacov1,t(t(means[2,])))*means[2,])+log(priors[2])                         

if (qdadelta0[e]>qdadelta1[e]) {
  predictY[e] <- -1
}
else{
  predictY[e] <- 1
}#using formula for qda to predict which class each new test point belongs to 
  
}
u<-sum(predict.Y==test.Y)# shows correctly and incorrectly predicted for lda and qda 
cclda<-(u/N)*100
mclda<-((N-u)/N)*100
v<-sum(predictY==test.Y)
ccqda<-(v/N)*100
mcqda<-((N-v)/N)*100
print('correctly classified with lda')
print(cclda)
print('misclassified with lda')
print(mclda)
print('correctly classified with qda')
print(ccqda)
print('misclassified with qda')
print(mcqda)