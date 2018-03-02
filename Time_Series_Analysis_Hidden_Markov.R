##########################Analysis of the apple stock data###################################
##############################Load libraries and data########################################

library(rugarch)
library(tseries)
library(fBasics)
library(zoo)
library(lmtest) 
library(depmixS4)
library(quantmod)
library(gridExtra)

setwd(".../Data")
apple <- read.table("day.csv",header=T, sep=',') 
head(apple)

applets <- zoo(apple$Adj.Close, as.Date(as.character(apple$Date), format = c("%Y-%m-%d")))
#log return time series
apple_rets <- log(applets/lag(applets,-1)) 
#strip off the dates and create numeric object
apple_ret_num <- coredata(apple_rets)

#Time series plot 
plot(applets, type='l', ylab = " adj close price", main="Plot of 2002-2017 daily apple stock prices", col = 'red')

#ACF and PACF plot
acf(coredata(applets), main="ACF plot of the 2002-2017 daily apple stock prices")
pacf(coredata(applets), main="PACF plot of the 2002-2017 daily apple stock prices")

#Compute statistics
basicStats(apple_rets)

#Histogram
hist(apple_rets, xlab="Daily return of stock prices", prob=TRUE, main="Histogram for daily return of stock prices")
xfit<-seq(min(apple_rets),max(apple_rets),length=40)
yfit<-dnorm(xfit,mean=mean(apple_rets),sd=sd(apple_rets))
lines(xfit, yfit, col="blue", lwd=2)

#QQ-plot
qqnorm(apple_rets)
qqline(apple_rets, col = 2) 

#Time plot of log return of prices
plot(apple_rets, type='l', ylab = "stock price return", main="Plot of 2002-2017 daily apple stock price return")

#Time plot of square of log return of prices
plot(apple_rets^2,type='l', ylab = "square of stock price return", main="Plot of 2002-2017 daily apple stock price squared return")

#Time plot of absolute value of log return of prices
plot(abs(apple_rets),type='l', ylab = "abs value of stock price return", main="Plot of 2002-2017 daily apple stock price abs return")

#ACF plot of log return of prices
par(mfrow=c(2,1))
acf(apple_ret_num)

#ACF plot of square of log return of prices
acf(apple_ret_num^2)

#ACF plot of absolute value of log return of prices
acf(abs(apple_ret_num))

#Test of independence
#Ljung Box test on squared values of the stock price returns
Box.test(apple_ret_num^2, lag=2, type="Ljung")
Box.test(apple_ret_num^2, lag=4, type="Ljung")
Box.test(apple_ret_num^2, lag=6, type="Ljung")

#Ljung Box test on absolute values of the stock price returns
Box.test(abs(apple_ret_num), lag=2, type="Ljung")
Box.test(abs(apple_ret_num), lag=4, type="Ljung")
Box.test(abs(apple_ret_num), lag=6, type="Ljung")

#Determine the order of the model
#PACF plot on the log return of the stock prices
pacf(apple_ret_num, lag=10, main="PACF plot of the log return of the stock prices")

#PACF plot on the squared return of the stock prices
pacf(apple_ret_num^2, lag=10, main="PACF plot of the squared log return of the stock prices")

#PACF plot on the absolute value of the return on the stock prices
pacf(abs(apple_ret_num), lag=10, main="PACF plot of the absolute value of the log return of the stock prices")

#Model 1
garch11.spec=ugarchspec(variance.model=list(garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)))
#estimate model 
garch11.fit=ugarchfit(spec=garch11.spec, data=apple_rets)
garch11.fit

#using extractors
#estimated coefficients:
coef(garch11.fit)
#unconditional mean in mean equation
uncmean(garch11.fit)
#unconditional varaince: omega/(alpha1+beta1)
uncvariance(garch11.fit)

#persistence = alpha1+beta1
persistence(garch11.fit)

#Constraints on parameters < 1

#half-life: ln(0.5)/ln(alpha1+beta1)
halflife(garch11.fit)

#create selection list of plots for garch(1,1) fit
plot(garch11.fit, which = "all")

#conditional volatility plot
plot.ts(sigma(garch11.fit), ylab="sigma(t)", col="blue")


#Compute information criteria using infocriteria() function for model selecton
infocriteria(garch11.fit)

#Model 2
garch11.t.spec=ugarchspec(variance.model=list(garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)), distribution.model = "std")

#estimate model 
garch11.t.fit=ugarchfit(spec=garch11.t.spec, data=apple_rets)
garch11.t.fit

#plot of residuals
plot(garch11.t.fit, which = "all")
persistence(garch11.t.fit)

#FORECASTS
compute h-step ahead forecasts for h=1,2,...,10
garch11.fcst=ugarchforecast(garch11.t.fit, n.ahead=12)
garch11.fcst
plot(garch11.fcst)

#rolling forecasts
garch11.t.fit=ugarchfit(spec=garch11.t.spec, data=apple_rets, out.sample=500)
garch11.fcst=ugarchforecast(garch11.t.fit, n.ahead=12, n.roll=450)
plot(garch11.fcst)

#Model 3
garch11.skt.spec=ugarchspec(variance.model=list(garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)), distribution.model = "sstd")
#estimate model 
garch11.skt.fit=ugarchfit(spec=garch11.skt.spec, data=apple_rets)
garch11.skt.fit

plot(garch11.skt.fit, which = "all")

#Model 4
egarch11.t.spec=ugarchspec(variance.model=list(model = "eGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0,0)), distribution.model = "std")
#estimate model 
egarch11.t.fit=ugarchfit(spec=egarch11.t.spec, data=apple_rets)
egarch11.t.fit

plot(egarch11.t.fit, which = "all")

#Model 5
fgarch11.t.spec=ugarchspec(variance.model=list(model = "fGARCH", garchOrder=c(1,1), submodel = "APARCH"), mean.model=list(armaOrder=c(0,0)), distribution.model = "std")
#estimate model 
fgarch11.t.fit=ugarchfit(spec=fgarch11.t.spec, data=apple_rets)
fgarch11.t.fit

plot(fgarch11.t.fit, which = "all")

#Model 6
igarch11.t.spec=ugarchspec(variance.model=list(model = "iGARCH", garchOrder=c(1,1)), mean.model=list(armaOrder=c(0 , 0 )), distribution.model = "std")

igarch11.t.fit=ugarchfit(spec=igarch11.t.spec, data=apple_rets)
igarch11.t.fit

plot(igarch11.t.fit, which = "all")

#Model 7
#Build Hidden Markov Model
set.seed(1)
hmm_apl<-depmix(list(apple_rets~1,apple_ret_num~1),data=applets,nstates=3,family=list(gaussian(),gaussian()))
#fit our model to the data set
hmm_apl_fit<-fit(hmm_apl, verbose = FALSE)
print(hmm_apl_fit)
summary(hmm_apl_fit)
#find the posterior odds for each state over our data set
hmm_post<-posterior(hmm_apl_fit)
head(hmm_apl_fit)
