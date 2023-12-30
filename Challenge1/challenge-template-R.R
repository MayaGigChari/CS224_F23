
library(data.table)

X.train <- as.matrix(data.frame(fread("data/ancestry_train.data")))
y.train <- as.matrix(data.frame(fread("data/ancestry_train.solution")))

X.test <- as.matrix(data.frame(fread("data/ancestry_test.data")))



#cost function
cost = function(y.true, y.hat){
	mse = -log10(mean((y.true-y.hat)**2)+1e-5)
	return(mse)
}
	
#baseline
base = matrix(1/3, nrow(y.train), ncol(y.train))
cost(y.train, base)

y.test = matrix(1/3, nrow(X.test), ncol(y.train))


#save and zip the file
fwrite(as.data.frame(y.test), file = "predictions.csv",  
			 sep = " ", quote=FALSE, row.names = F, col.names = F)

system("zip -r predictions.zip predictions.csv")
