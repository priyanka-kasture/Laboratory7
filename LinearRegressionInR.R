# Multiple Linear Regression in R without the built-in 'lm()'.
# The code uses Matrix Algebra.

# Author : Priyanka Kasture

attach(mtcars)

# Use 3 predictors
Y <- as.matrix(mpg) # Dependent Variable
X <- cbind(constant = 1, as.matrix(cbind(hp,cyl,wt))) # Independent Variables

#  The least squares estimates are obtained by the following matrix formula.
B <- solve(t(X)%*%X,t(X)%*%Y) # Formula : (X'X)^-1 * (X'Y)

# Computing Standard Errors
s2 <- sum((Y - X%*%B)^2)/(nrow(X) - ncol(X))
VCV <- s2*solve(t(X)%*%X)
SE <- sqrt(diag(VCV))

# Computing T-values.
t <- B/SE

# Computing P-values.
p <- 2*pt(abs(t),nrow(X) - ncol(X), lower.tail = FALSE)

# Computing Adjusted R-squared.
Y_hat <- X%*%B 
SSr <- sum((Y - Y_hat)^2)
SSt <- sum((Y - mean(Y))^2)
R2 <- 1 - (SSr/SSt)
adj.R2 <- 1 - ((1 - R2)*(nrow(X) - 1))/(nrow(X) - ncol(X[,-1]) - 1)

# This function is writtern so that you can compare the results of 'matrix algebra' method to the built-in 'lm()' method.
lm <- lm(mpg ~ hp + cyl + wt)
summary(lm)
# Compare the Adjusted R-Values, they are same.

Table <- as.data.frame(round(cbind(B,SE,t,p), digits = 3))
names(Table)[1:4] <- c("Estimate:","Standard Error:","t-value:","p-value:")

# Displaying the results.
Table

# Displaying Adjusted R-Squared.
adj.R2
