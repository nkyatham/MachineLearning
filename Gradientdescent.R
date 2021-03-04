# Function for prediction
prediction <- function(X,w){
  return(y<-w*X)
}

# Function for predictionIntercept
predictionActual<- function(X,w,w1,w2,w3,w4,w5,c1){
  return(y<-w*X[1]+w1*X[2]+w2*X[3]+w3*X[4]+w4*X[5]+w5*X[6]+c1*X[7])
}

#  Function for gradient descent
gradient <- function(X,Y,w,w1,w2,w3,w4,w5,c1){
  c1<- (2 * X[7] * (prediction(X[7],c1) - Y))
  w <- (2 * X[1] * (prediction(X[1],w) - Y))
  w1<- (2 * X[2] * (prediction(X[2],w1) - Y))
  w2<- (2 * X[3] * (prediction(X[3],w2) - Y))
  w3<- (2 * X[4] * (prediction(X[4],w3) - Y))
  w4<- (2 * X[5] * (prediction(X[5],w4) - Y))
  w5<- (2 * X[6] * (prediction(X[6],w5) - Y))
  listOfwt<-list('weight'=mean(unlist(w)),'weight1'=mean(unlist(w1)),'weight2'=mean(unlist(w2))
                 ,'weight3'=mean(unlist(w3)),'weight4'=mean(unlist(w4)), 'weight5'=mean(unlist(w5))
                 ,'intercept'=mean(unlist(c1)))
  return(listOfwt)
}
loss<-function(X,Y,w,w1,w2,w3,w4,w5,c1){
  return(mean(unlist(Y-(predictionActual(X,w,w1,w2,w3,w4,w5,c1))^2)))
}

#  Function to train the model
training <- function(X,Y,lr,iterations){
  #w<-w1<-w2<-w3<-w4<-w5<-c1<-0
  interval<-c(-0.7,0.7)
  set.seed(8)
  wt<-runif(7,interval[1],interval[2])
  w<-wt[1];w1<-wt[2];w2<-wt[3];w3<-wt[4];w4<-wt[5];w5<-wt[6];c1<-wt[7];
  for (i in seq(iterations)) {
    # Function call for gradient
    grad<-gradient(X,Y,w,w1,w2,w3,w4,w5,c1)
    w<-w-grad$weight*lr
    w1<-w1-grad$weight1*lr
    w2<-w2-grad$weight2*lr
    w3<-w3-grad$weight3*lr
    w4<-w4-grad$weight4*lr
    w5<-w5-grad$weight5*lr
    c1<-c1-grad$intercept*lr
    print(loss(X,Y,w,w1,w2,w3,w4,w5,c1))
    finalwts<-list('weight'=w,'weight1'=w1,'weight2'=w2,'weight3'=w3,'weight4'=w4,'weight5'=w5,'intercept'=c1)
  }
  return(finalwts)
}

# Sigmoid Function
sigmoid<-function(X,finalwts){
  w<-finalwts$weight
  w1<-finalwts$weight1
  w2<-finalwts$weight2
  w3<-finalwts$weight3
  w4<-finalwts$weight4
  w5<-finalwts$weight5
  c1<-finalwts$intercept
  x<-predictionActual(X,w,w1,w2,w3,w4,w5,c1)
  return(round(unlist(1/(1+exp(-x)))))
}
