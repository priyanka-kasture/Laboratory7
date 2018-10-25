# Principal Component Analysis and Factor Analysis in R

mydata = pca_gsp
attach(mydata)

# Define variables
X <- cbind(Ag, Mining, Constr, Manuf, Manuf_nd, Transp, Comm, Energy, TradeW, TradeR, RE, Services, Govt)

# Descriptive statistics
summary(X)
cor(X)

# Principal component analysis
pca1 <- princomp(X, scores=TRUE, cor=TRUE)
summary(pca1)

# Loadings of principal components
loadings(pca1)
#pca1$loadings

# Scree plot of eigenvalues
plot(pca1)
screeplot(pca1, type="line", main="Scree Plot")

# Biplot of score variables
biplot(pca1)

# Scores of the components
pca1$scores[1:10,]


# Factor analysis - different results from other softwares and no rotation
fa1 <- factanal(X, factor=3)
fa1

fa2 <- factanal(X, factor=3, rotation="varimax")
fa2

fa3 <- factanal(X, factors=3, rotation="varimax", scores="regression")
fa3
