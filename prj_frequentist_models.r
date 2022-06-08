library(ggplot2)

pines_data = read.csv("./data_pines.csv")
pines_data$ID = factor(pines_data$ID)
pines_data$species = factor(pines_data$species)
head(pines_data)

#eda
par(mfrow=c(1,1))
pairs(pines_data[c("LMA", "species", "dfromtop")])


plot(LMA ~ species, data=pines_data)
ggplot(data = pines_data,
    mapping = aes(x = ID, y = LMA)) +
    geom_boxplot(aes(color = species))

plot(LMA ~ dfromtop, data=pines_data)
ggplot(data = pines_data,
    mapping = aes(x = dfromtop, y = LMA)) +
    geom_point(aes(color = species))



#model1
median(pines_data$dfromtop)
catdfromtop = ifelse(pines_data$dfromtop > median(pines_data$dfromtop), 1, 0)

pines_data$catdfromtop = factor(catdfromtop)
lm_fit1 = lm(LMA ~ catdfromtop, data=pines_data)
summary(lm_fit1)
plot(LMA ~ catdfromtop, data=pines_data)
abline(h = lm_fit1$coeff[1], col="red")
abline(h = lm_fit1$coeff[1] + lm_fit1$coeff[2], col="blue")
par(mfrow=c(2,2))
plot(lm_fit1)


#model2
lm_fit2 = lm(LMA ~ dfromtop * species, data=pines_data)
summary(lm_fit2)
par(mfrow=c(1,1))
plot(LMA~dfromtop, data=pines_data, col=species)
abline(a = lm_fit2$coeff[1], b = lm_fit2$coeff[2], col="black")
abline(a = lm_fit2$coeff[1] + lm_fit2$coeff[3], b= lm_fit2$coeff[2] + lm_fit2$coeff[4], col="red")
par(mfrow=c(2,2))
plot(lm_fit2)


