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
EVD_E <- get(load("EVD.E.rda"))

# Define ETA lists for all 18 models
ETA_models <- list(
  
  list(G = list(V = EVD_G$vectors, d = EVD_G$values, model = 'RKHS'), 
       E = list(V = EVD_E$vectors, d = EVD_E$values, model = 'RKHS')),
  
  list(H = list(V = EVD_H$vectors, d = EVD_H$values, model = 'RKHS'),
       E = list(V = EVD_E$vectors, d = EVD_E$values, model = 'RKHS')),
  
  list(G = list(V = EVD_G$vectors, d = EVD_G$values, model = 'RKHS'), 
       H = list(V = EVD_H$vectors, d = EVD_H$values, model = 'RKHS'), 
       W = list(V = EVD_W$vectors, d = EVD_W$values, model = 'RKHS'))
  
)

# Parameters
nIter <- 12000
burnIn <- 2000
colENV <- 1  # Column representing the environment (Year)
ESC <- FALSE

# Years to predict
years <- unique(Y2[, colENV])

# Initialize a data.table for storing results
all_performance_metrics <- data.table(Year = integer(), Trait = character(), Model = character(), Fold = integer(), Correlation = numeric(), MAPE = numeric())

# Perform forward predictions
cat("Starting forward predictions for all years\n")

for (target_year in years) {
  cat("Predicting for year:", target_year, "\n")
  
  # Split data into training and testing based on years
  training <- which(Y2[, colENV] != target_year)
  testing <- which(Y2[, colENV] == target_year)
  
  # Generate 5 folds on the training set
  set.seed(123)  # For reproducibility
  folds <- cut(seq_along(training), breaks = 5, labels = FALSE)
  
  for (colPhen in 3:7) {  # Loop over phenotype columns
    y <- Y2[, colPhen]
    yNA <- y
    gid <- Y2[, colENV]
    
    if (ESC) {
      y <- scale(y, center = TRUE, scale = TRUE)
    }
    
    # Mask testing set
    yNA[testing] <- NA
    
    for (fold in 1:5) {  # Perform 5 iterations for cross-validation
      cat("Processing Fold:", fold, "\n")
      
      # Identify the indices for the current fold
      fold_indices <- which(folds == fold)
      training_fold <- setdiff(training, training[fold_indices])  # Use remaining 4 folds for training
      
      for (model_idx in seq_along(ETA_models)) {  # Loop over models
        model_name <- paste0("Model_", model_idx)
        
        # Prepare training data
        y_fold <- yNA
        y_fold[training_fold] <- y[training_fold]
        
        # Fit the model
        fm <- BGLR(y = y_fold, ETA = ETA_models[[model_idx]], nIter = nIter, burnIn = burnIn, verbose = TRUE)
        fm$y <- y
        
        preds <- fm$yHat[testing]
        obs <- y[testing]
        
        # Calculate performance metrics
        correlation <- cor(obs, preds, use = "complete.obs")
        mape <- mape(obs, preds)
        
        # Store results
        all_performance_metrics <- rbind(all_performance_metrics, 
                                         data.table(Year = target_year, Trait = colPhen, Model = model_name, 
                                                    Fold = fold, Correlation = correlation, MAPE = mape))
        
        rm(fm)
        gc()
      }
    }
  }
}

# Save performance metrics
write.xlsx(all_performance_metrics, file = "forward_prediction_metrics_cv5.xlsx")

# Print summary
print(all_performance_metrics)

