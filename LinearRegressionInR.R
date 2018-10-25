# Multiple Linear Regression in R without the built-in 'lm()'.
# The code uses Matrix Algebra.
# Chit : 1.a

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

# Residual (Error) Plot for the fitted model.
plot(resid(lm))
abline(h=0)

Table <- as.data.frame(round(cbind(B,SE,t,p), digits = 3))
names(Table)[1:4] <- c("Estimate:","Standard Error:","t-value:","p-value:")

# Displaying the results.
Table

# Displaying Adjusted R-Squared.
adj.R2

# Testing the model by subjecting it new 'wt' values.
# The model predicts the corresponding 'mpg' values.
# Compare the values with those in 'mtcars', they are almost the same.
fitted_model = lm(mpg ~ wt, mtcars)
newdata = data.frame(wt = runif(10, 1, 5))
newdata$predicted_mpg = predict(fitted_model, newdata = newdata)
newdata

# Regression Plot - Horse Power v/s Miles Per Gallon
ggplot(mtcars, aes(hp, mpg)) + geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  ylab("Miles per Gallon") +
  xlab("No. of Horsepower") +
  ggtitle("Impact of Number of Horsepower on MPG")

# The mpg is unlikely to hit zero as the hp increases, we would expect a more asymptotic line. 
# So I have applied stat_smooth to get a better fit. (Optional)
ggplot(mtcars, aes(hp, mpg)) +
  stat_smooth() + geom_point() +
  ylab("Miles per Gallon") +
  xlab ("No. of Horsepower") +
  ggtitle("Impact of Number of Horsepower on MPG")

# Regression Plot - No. of Cylinders v/s Miles Per Gallon
ggplot(mtcars, aes(cyl, mpg)) + geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  ylab("Miles per Gallon") + xlab("No. of Cylinders") +
  ggtitle("Impact of Number of Cylinders on MPG")

# Residual (Error) Diagnostics Of Final Model (Optional)
par(mfrow=c(2,2))
plot(lm)

