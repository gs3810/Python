library(midasr)
library(quantmod)
 
getSymbols("GDPC1", src = "FRED")
names(GDPC1) <- "USRealGDP"

getSymbols("UNEMPLOY", src = "FRED")
names(UNEMPLOY) <- "UNEMPLOY"

y <- as.ts(diff(log(GDPC1))[5:287]) 
x <- as.ts(diff(log(UNEMPLOY)))
x <- as.ts(x[2:850])

# can't get this to work
t <- 1:length(y)
#i <- fmls(x,3,4,nealmon)

mr <- midas_r(y ~ fmls(x,2,3,nealmon), start=list(x=c(0,0,0)))
agk.test(mr)


