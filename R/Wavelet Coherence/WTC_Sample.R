library(readxl)
library(ggplot2)
library(biwavelet)

my_data <- read_excel("Machine Learning\\Wavelet_Coherence\\Stock_Prices.xlsx")
my_data.select <- my_data[,c(1,3:4)] 
my_data.select <- data.frame(my_data.select)

Date <- my_data[,c(1)]  
GM <- my_data[,c(3)]
FO <- my_data[,c(4)]
SP <- my_data[,c(2)]

# visualize 
g <- ggplot(data=my_data.select, aes(Date, FO)) + geom_line()

# set up two time series
t1 = cbind(Date, GM)
t2 = cbind(Date, SP)

# number of iterations
nrands = 100

# run WTC algoritm
wtc.AB = wtc(t1, t2, nrands = nrands)

# plotting a graph
par(oma = c(0, 0, 0, 1), mar = c(5, 4, 5, 5) + 0.1)

plot(wtc.AB, plot.phase = FALSE, lty.coi = 1, col.coi = "grey", lwd.coi = 2, 
     lwd.sig = 2, arrow.lwd = 0.03, arrow.len = 0.12, ylab = "Scale", xlab = "Period", 
     plot.cb = TRUE, main = "Wavelet Coherence: GM vs S&P")


