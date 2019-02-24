library(midasr)
library(quantmod)
 
getSymbols("GDPC1", src = "FRED")
names(GDPC1) <- "USRealGDP"

getSymbols("UNEMPLOY", src = "FRED")
names(UNEMPLOY) <- "UNEMPLOY"

y <- as.ts(diff(log(GDPC1))[5:287]) 
x <- as.ts(diff(log(UNEMPLOY)))[2:850]  # x and y have to be at the same frequency 
trend  <- 1:length(y)

mr <- midas_r(y ~ trend + mls(y, 1:2, 1,"*") + fmls(x,2,3,nealmon), start=list(x=c(0,0,0))) # try with and without trend
agk.test(mr)

# forecast horizon, in quarters
h <- 20

xn <- rep(-0.05, 3*h) # multiply by 3 for months, a simple repetition 
# new trend values
trendn <- length(y) + 1*(1:h) # varying the multiplier will change trend 

# static forecasts combining historic and new high frequency data
fmr <- forecast(mr, list(trend=trendn, x=xn), method="dynamic")
summary(fmr)
plot(fmr)



