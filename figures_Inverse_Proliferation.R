##################################################################################################
##################################################################################################
###
### Auxiliary file to generate figures presenting the results of the random walk Metropolis–Hastings 
### algorithm for the Bayesian inference of a non-local proliferation function.
###
##################################################################################################
##################################################################################################
###
### This file contains R code for making figures presented in the paper: 
###
### Bayesian inference of a non-local proliferation model
### Z. Szymańska, J.Skrzeczkowski, B.Miasojedow, P.Gwiazda
### arXiv: 2106.05955
###
### Figures are made based on simulation done using the code from file metropolis–hastings.py 
### and postrpocessing file postprocessing.py.
###
### PLEASE SEE THE README.md TO KNOW HOW TO CITE.
###
##################################################################################################
##################################################################################################

library(ggplot2)
library(reshape2)

# Files "log_mu.csv", "log_sigma_k.csv", "log_sigma_o.csv", and "log_sigma_i.csv" are output files of metropolis_hastings.py.
# If you run the simulation in tranches unite the data doing: 
# log_mu_part1.csv > log_mu.csv
# log_mu_part2.csv >> log_mu.csv
# Etc.
log_mu <- read.table("~/Destination_path/log_mu.csv", quote="\"", comment.char="")
log_sigma_k <- read.table("~/Destination_path/log_sigma_k.csv", quote="\"", comment.char="")
log_sigma_o <- read.table("~/Destination_path/log_sigma_o.csv", quote="\"", comment.char="")
log_sigma_i <- read.table("~/Destination_path/log_sigma_i.csv", quote="\"", comment.char="")
dane = data.frame(lambda = exp(unlist(log_mu)),sigma_k =exp(unlist(log_sigma_k)),sigma_o=exp(unlist(log_sigma_o)),sigma_i=exp(unlist(log_sigma_i)))

# Plot of the 1D densities.
dane$iteracja = 1:nrow(dane)
meltdane = melt(dane,id="iteracja")
meltdane$variable = factor(meltdane$variable,levels = c("lambda","sigma_k","sigma_o","sigma_i"), labels = c(expression(alpha),expression(sigma[k]),expression(sigma[o]),expression(sigma[i])))
ggplot(meltdane,aes(x=value))+stat_density(fill="steelblue")+facet_wrap(~variable,scales = "free", labeller=label_parsed)+theme_minimal()+xlab('')+ylab('')+
  theme(strip.text.x = element_text(size = 12))
ggsave("~/Destination_path/dens1db.pdf",width=8.3,height=6.25,units = "in")

# Plot of the traceplots.
meltdane = melt(dane,id="iteracja")
meltdane$variable = factor(meltdane$variable,levels = c("lambda","sigma_k","sigma_o","sigma_i"), 
  labels = c(expression(alpha),expression(sigma[k]),expression(sigma[o]),expression(sigma[i])))
ggplot(meltdane,aes(x=iteracja,y=log(value)))+geom_line()+facet_wrap(~variable,scales="free",labeller = label_parsed,ncol=2)+theme_minimal()+xlab('ITERATION NUMBER')+ylab('')+
  theme(strip.text.x = element_text(size = 12))
ggsave("~/Destination_path/traceplotb.pdf",width=8.3,height=6.25,units = "in")

# Autocorrelation plot.
test = acf(log(dane[,-5]),plot = F)
acfdane = t(apply(test$acf,1,diag))
meltdane = melt(acfdane)
meltdane$variable = factor(meltdane$Var2,levels =1:4, 
  labels = c(expression(log(alpha)),expression(log(sigma[k])),expression(log(sigma[o])),expression(log(sigma[i]))))
ggplot(meltdane,aes(x=Var1,y=value))+geom_bar(stat="identity")+facet_wrap(~variable, labeller=label_parsed,ncol=2)+theme_minimal()+xlab('')+ylab('')+
  theme(strip.text.x = element_text(size = 12))+xlab("LAG")+ylab("ACF")
ggsave("~/Destination_path/acfb.pdf",width=8.3,height=6.25,units = "in")

# Drawing the trajectories. Files "mean_nr.csv", "lower_nr.csv", and "upper_nr.csv" are output files of postprocessing.py.
# The simulation presented in the paper were run in 4 tranches of 100 000 iterations each. 
mean0 <-read.csv("~/Destination_path/mean0.csv", header=FALSE)
mean1 <-read.csv("~/Destination_path/mean1.csv", header=FALSE)
mean2 <-read.csv("~/Destination_path/mean2.csv", header=FALSE)
mean3 <-read.csv("~/Destination_path/mean3.csv", header=FALSE)
low0 <-read.csv("~/Destination_path/lower0.csv", header=FALSE)
low1 <-read.csv("~/Destination_path/lower1.csv", header=FALSE)
low2 <-read.csv("~/Destination_path/lower2.csv", header=FALSE)
low3 <-read.csv("~/Destination_path/lower3.csv", header=FALSE)
up0  <-read.csv("~/Destination_path/upper0.csv", header=FALSE)
up1  <-read.csv("~/Destination_path/upper1.csv", header=FALSE)
up2  <-read.csv("~/Destination_path/upper2.csv", header=FALSE)
up3  <-read.csv("~/Destination_path/upper3.csv", header=FALSE)
mean = (mean0 + mean1 + mean2 + mean3)/4
low = (low0 + low1 + low2 + low3)/4
up = (up0 + up1 + up2 + up3)/4

##################################################################################################
##################################################################################################
###
### Parameters for data set a.
###
##################################################################################################
##################################################################################################

tau = 8
tmax = 26 
time_obs = c(8,11.1034482759,13.1724137931,18.0689655172,22,25.1724137931)
radius_obs = c(0.2637931034,0.4655172414,0.5948275862,0.9293103448,1.3517241379,1.9155172414)
dime_obs = 2*radius_obs
bar_up = c(0.0827586207,0.0827586207,0.1275862069,0.2,0.5724137931,0.5655172414)
bar_down = c(0.0827586207,0.0827586207,0.1293103448,0.2,0.5793103448,0.5620689655)

##################################################################################################
##################################################################################################
###
### Parameters for data set b.
###
##################################################################################################
##################################################################################################

#tau = 19.3939393939
#tmax = 170 
#time_obs = c(19.3939393939,39.3939393939,59.8484848485,80,100.7575757576,120.7575757576,140.9090909091,161.6666666667)
#radius_obs = c(0.4030418251,0.4980988593,0.7053231939,0.9657794677,1.2623574144,1.4923954373,1.6254752852,2.0209125475)
#dime_obs = 2*radius_obs
#bar_up = c(0.2509505703,0.1520912548,0.247148289,0.3003802281,0.3155893536,0.711026616,1.0266159696,0.5171102662)
#bar_down = c(0.2509505703,0.1520912548,0.2623574144,0.3117870722,0.3079847909,0.7186311787,1.0228136882,0.5095057034)

##################################################################################################
##################################################################################################
###
### Parameters for data set c.
###
##################################################################################################
##################################################################################################

#tau = 13.9597315436
#tmax = 75 
#time_obs = c(13.9597315436,31.0738255034,41.1409395973,54.0939597315,67.9194630872,74.0268456376)
#radius_obs = c(0.7333333333,0.8858333333,0.9341666667,1.1041666667,1.1108333333,1.1741666667)
#dime_obs = 2*radius_obs
#bar_up = c(0.3483333333,	0.2466666667,	0.2633333333,	0.2883333333,	0.3283333333,	0.1983333333)
#bar_down = c(0.3433333333,	0.2533333333,	0.2566666667,	0.2916666667,	0.3233333333,	0.2)

pdata = data.frame(x=time_obs,y=dime_obs,d = dime_obs,down=dime_obs-bar_down,up=dime_obs+bar_up)
data_simulated = data.frame(time = seq(tau,tmax,length.out = length(unlist(mean))),means = unlist(mean),up=unlist(up),low=unlist(low))

# For data set a.
ggplot(data_simulated,aes(x=time,y=means))+geom_line(col="steelblue")+geom_ribbon(aes(ymin=up,ymax=low),alpha=0.3,fill="steelblue")+
  geom_point(data=pdata,aes(x=x,y=y),col="black",size = 2.5)+
  theme_minimal()+ylab("MAXIMUM DIAMETER (mm)")+xlab("DAYS")+xlim(7,26)+ylim(0,5.0)
ggsave("~/Destination_path/fiteda.pdf",width=8.3,height=6.25,units = "in")

# For data set b.
#ggplot(data_simulated,aes(x=time,y=means))+geom_line(col="steelblue")+geom_ribbon(aes(ymin=up,ymax=low),alpha=0.3,fill="steelblue")+
#  geom_point(data=pdata,aes(x=x,y=y),col="black",size = 2.5)+
#  theme_minimal()+ylab("MAXIMUM DIAMETER (mm)")+xlab("DAYS")+xlim(18,170)+ylim(0,5)
#ggsave("~/Destination_path/fitedb.pdf",width=8.3,height=6.25,units = "in")

# For data set c.
#ggplot(data_simulated,aes(x=time,y=means))+geom_line(col="steelblue")+geom_ribbon(aes(ymin=up,ymax=low),alpha=0.3,fill="steelblue")+
#  geom_point(data=pdata,aes(x=x,y=y),col="black",size = 2.5)+
#  theme_minimal()+ylab("MAXIMUM DIAMETER (mm)")+xlab("DAYS")+xlim(11,80)+ylim(1.0,2.7)
#ggsave("~/Destination_path/fitedc.pdf",width=8.3,height=6.25,units = "in")

##################################################################################################
##################################################################################################

# Plotting the mean trajectory together with trajectories obtained with MAP estimator. 
# To get the data for plotting trajectories for the MAP estimator make run the folkman_a_b_c_time.py 
# for appropriate data set parameters set to with MAP estimator (output of postprocessing.py code). 
# Files: "time_best_E.csv", "time_best_RK.csv", "colony_diameter_best_RK.csv", and "colony_diameter_best_RK.csv"
# Data "*_E.csv" were obtained with Euler scheme, whereas data "*_RK.csv" were obtained for 4th order Runge-Kutta scheme. 

diam_best_RK <- read.table("~/Destination_path/colony_diameter_best_RK.csv", quote="\"", comment.char="")
time_best_RK <- read.table("~/Destination_path/time_best_RK.csv", quote="\"", comment.char="")
diam_best_E <- read.table("~/Destination_path/colony_diameter_best_E.csv", quote="\"", comment.char="")
time_best_E <- read.table("~/Destination_path/time_best_E.csv", quote="\"", comment.char="")

time_best_RK = time_best_RK + tau
data_to_plot <- data.frame(time_mean = seq(tau,tmax,length.out = length(unlist(mean))),diam_mean = unlist(mean),time_best_RK = unlist(time_best_RK), diam_best_RK = unlist(diam_best_RK), unlist(time_best_E), diam_best_E = unlist(diam_best_E))

# For data set a.
ggplot(data_to_plot,aes(x=time_best_RK))+
  geom_line(aes(y=diam_best_RK),color = "darkred",size=0.7)+
  #geom_line(aes(y=diam_best_E),color = "green",size=0.7)+  
  geom_line(aes(y=diam_mean),color = "steelblue",size=0.7)+
  geom_point(data=pdata,aes(x=x,y=y),col="black",size = 2.5)+
  theme_minimal()+ylab("MAXIMUM DIAMETER (mm)")+xlab("DAYS")+xlim(7,26)+ylim(0,5.0)
ggsave("~/Destination_path/time_diam_best_a.pdf",width=8.3,height=6.25,units = "in")

# For data set b.
#ggplot(data_to_plot,aes(x=time_best))+
#  geom_line(aes(y=diam_best),color = "darkred",size=0.7)+
#  #geom_line(aes(y=diam_mean),color = "steelblue",size=0.7)+
#  geom_point(data=pdata,aes(x=x,y=y),col="black",size = 2.5)+
#  theme_minimal()+ylab("MAXIMUM DIAMETER (mm)")+xlab("DAYS")+xlim(18,170)+ylim(0,5)
#ggsave("~/Destination_path/time_diam_best_b.pdf",width=8.3,height=6.25,units = "in")

# For data set c.
#ggplot(data_to_plot,aes(x=time_best_RK))+
#  geom_line(aes(y=diam_best_RK),color = "darkred",size=0.7)+
#  #geom_line(aes(y=diam_best_E),color = "green",size=0.7)+  
#  geom_line(aes(y=diam_mean),color = "steelblue",size=0.7)+
#  geom_point(data=pdata,aes(x=x,y=y),col="black",size = 2.5)+
#  theme_minimal()+ylab("MAXIMUM DIAMETER (mm)")+xlab("DAYS")+xlim(11,80)+ylim(1.0,2.7)
#ggsave("~/Destination_path/time_diam_best_c.pdf",width=8.3,height=6.25,units = "in")
