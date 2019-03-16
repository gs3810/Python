library(readxl)
library(ggplot2)
library(biwavelet)

my_data <- read_excel("Machine Learning\\Wavelet_Coherence\\PF_vs_Index\\Stock_Prices.xlsx")

Index <- my_data[,c(1)]
Date <- my_data[,c(4)]$DATE
stock1 <- my_data[,c(2)]
stock2 <- my_data[,c(3)]

# visualize 
ggplot(data=my_data, aes(x=Date, y=GM, by="months")) + geom_line()

# set up two time series
t1 = cbind(Index, stock1)
t2 = cbind(Index, stock2)

# number of iterations
nrands = 50

# run WTC algoritm
wtc.AB = wtc(t1, t2, nrands=nrands)
# View(wtc.AB)

# export file 
binded_rsq <- cbind(wtc.AB$rsq,wtc.AB$period)
write.csv(binded_rsq, file="Machine Learning\\Wavelet_Coherence\\PF_vs_Index\\rsq.csv")

# plotting a graph
par(oma = c(0, 0, 0, 1), mar = c(5, 4, 5, 5) + 0.1)

plot(wtc.AB, plot.phase =FALSE, plot.coi=FALSE, lty.coi = 1, col.coi = "grey", lwd.coi = 2, 
     lwd.sig = 2, arrow.lwd = 0.03, arrow.len = 0.12, ylab = "Period", xlab = "Time", 
     plot.cb = TRUE, main = "Wavelet Coherence: stock1 vs stock2")

# Adding grid lines
n = length(t1[, 1])
abline(v = seq(260, n, 260), h = 1:16, col = "brown", lty = 1, lwd = 1)

