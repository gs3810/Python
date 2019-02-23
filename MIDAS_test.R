library(midasr)
library(quantmod)
 
getSymbols("GDPC1", src = "FRED")
names(GDPC1) <- "USRealGDP"

getSymbols("UNEMPLOY", src = "FRED")
names(UNEMPLOY) <- "UNEMPLOY"

y <- diff(log(GDPC1)) 
x <- diff(log(UNEMPLOY))

# can't get this to work 
x1 <- window(diff(UNEMPLOY), 1950) 
# t <- 1:length(y)

# mr <- midas_r(y~t+fmls(x,11,12,nealmon), start=list(x=c(0,0,0)))
#agk.test(mr)