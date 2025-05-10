# Clear workspace
rm(list = ls())

# Load libraries
library(BGLR)
library(Metrics) # For MAPE calculation
library(data.table)
library(openxlsx)

# Load phenotype and EVD data
Y2 <- read.table(file = 'Y.csv', sep = ',', header = TRUE, stringsAsFactors = FALSE)

# Load EVDs for all matrices
EVD_G <- get(load("EVD.G.rda"))
EVD_H <- get(load("EVD.H.rda"))
EVD_V <- get(load("EVD.V.rda"))
EVD_W <- get(load("EVD.W.rda"))
EVD_GW <- get(load("EVD.GxW.rda"))

# Define ETA lists for all 18 models
ETA_models <- list(
  list(G = list(V = EVD_G$vectors, d = EVD_G$values, model = 'RKHS'), 
       E = list(V = EVD_E$vectors, d = EVD_E$values, model = 'RKHS')),
  
  list(H = list(V = EVD_H$vectors, d = EVD_H$values, model = 'RKHS'),
       E = list(V = EVD_E$vectors, d = EVD_E$values, model = 'RKHS')),
  
  list(V = list(V = EVD_V$vectors, d = EVD_V$values, model = 'RKHS'),
       E = list(V = EVD_E$vectors, d = EVD_E$values, model = 'RKHS')),
  
  list(G = list(V = EVD_G$vectors, d = EVD_G$values, model = 'RKHS'), 
       W = list(V = EVD_W$vectors, d = EVD_W$values, model = 'RKHS')),
  
  list(G = list(V = EVD_G$vectors, d = EVD_G$values, model = 'RKHS'), 
       H = list(V = EVD_H$vectors, d = EVD_H$values, model = 'RKHS'),
       E = list(V = EVD_E$vectors, d = EVD_E$values, model = 'RKHS')),
  
  list(G = list(V = EVD_G$vectors, d = EVD_G$values, model = 'RKHS'), 
       V = list(V = EVD_V$vectors, d = EVD_V$values, model = 'RKHS'),
       E = list(V = EVD_E$vectors, d = EVD_E$values, model = 'RKHS')),
  
  list(G = list(V = EVD_G$vectors, d = EVD_G$values, model = 'RKHS'),
       H = list(V = EVD_H$vectors, d = EVD_H$values, model = 'RKHS'),
       V = list(V = EVD_V$vectors, d = EVD_V$values, model = 'RKHS'),
       E = list(V = EVD_E$vectors, d = EVD_E$values, model = 'RKHS')),  
  
  list(G = list(V = EVD_G$vectors, d = EVD_G$values, model = 'RKHS'), 
       V = list(V = EVD_V$vectors, d = EVD_V$values, model = 'RKHS'), 
       W = list(V = EVD_W$vectors, d = EVD_W$values, model = 'RKHS')),
  
  list(G = list(V = EVD_G$vectors, d = EVD_G$values, model = 'RKHS'), 
       H = list(V = EVD_H$vectors, d = EVD_H$values, model = 'RKHS'), 
       W = list(V = EVD_W$vectors, d = EVD_W$values, model = 'RKHS')),
  
  list(G = list(V = EVD_G$vectors, d = EVD_G$values, model = 'RKHS'), 
       H = list(V = EVD_H$vectors, d = EVD_H$values, model = 'RKHS'),
       V = list(V = EVD_V$vectors, d = EVD_V$values, model = 'RKHS'),
       W = list(V = EVD_W$vectors, d = EVD_W$values, model = 'RKHS')), 
)

# Cross-validation parameters
folds <- 1:5
nIter <- 12000
burnIn <- 2000
colENV <- 1
colCV <- 9
CV0 <- FALSE
ESC <- FALSE

# Initialize a data.table for storing results
all_performance_metrics <- data.table(Fold = integer(), Trait = character(), Model = character(), Correlation = numeric(), MAPE = numeric())

# Set random seed for reproducibility
set.seed(1)

# Perform cross-validation for all models
cat("Starting cross-validation for all models\n")

for (colPhen in 3:7) {  # Loop over phenotype columns
  y <- Y2[, colPhen]
  gid <- Y2[, colENV]
  
  if (ESC) {
    y <- scale(y, center = TRUE, scale = TRUE)
  }
  
  for (fold in folds) {  # Loop over folds
    yNA <- y
    testing <- which(Y2[, colCV] == fold)
    
    if (CV0) {
      testing <- which(gid %in% gid[testing])
    }
    
    for (model_idx in seq_along(ETA_models)) {  # Loop over models
      model_name <- paste0("Model_", model_idx)
      yNA[testing] <- NA
      
      # Fit the model
      fm <- BGLR(y = yNA, ETA = ETA_models[[model_idx]], nIter = nIter, burnIn = burnIn, verbose = TRUE)
      fm$y <- y
      
      preds <- fm$yHat[testing]
      obs <- y[testing]
      
      # Calculate performance metrics
      correlation <- cor(obs, preds, use = "complete.obs")
      mape <- mape(obs, preds)
      
      # Store results
      all_performance_metrics <- rbind(all_performance_metrics, 
                                       data.table(Fold = fold, Trait = colPhen, Model = model_name, 
                                                  Correlation = correlation, MAPE = mape))
      
      rm(fm)
      gc()
    }
  }
}

# Save performance metrics
write.xlsx(all_performance_metrics, file = "all_performance_metrics.xlsx")

# Print summary
print(all_performance_metrics)
