# Clear workspace
rm(list = ls())

# Load libraries
library(BGLR)
library(data.table)
library(openxlsx)

# Load phenotype and EVD data
Y2 <- read.xlsx('Pheno_final_train_test.xlsx')

# Load EVDs for all matrices
EVD_G <- get(load("EVD.G.rda"))
EVD_H <- get(load("EVD.H.rda"))
EVD_W <- get(load("EVD.W.rda"))

# Select only G+H+W model (Model 5)
ETA_GHW <- list(
  G = list(V = EVD_G$vectors, d = EVD_G$values, model = 'RKHS'), 
  H = list(V = EVD_H$vectors, d = EVD_H$values, model = 'RKHS'), 
  W = list(V = EVD_W$vectors, d = EVD_W$values, model = 'RKHS')
)

# Parameters
nIter <- 12000
burnIn <- 2000
colCVScheme <- 8  # Column representing the CV scheme (0 = Training, 1 = Testing)

# Identify training and testing sets
training <- which(Y2[, colCVScheme] == 0)
testing <- which(Y2[, colCVScheme] == 1)

# Initialize results tables
top_30_observed_list <- list()
top_30_predicted_list <- list()
coincidence_results <- list()

# Loop over phenotype columns (assuming columns 3 to 7 are the target traits)
for (colPhen in 3:7) {
  trait_name <- colnames(Y2)[colPhen]
  y <- Y2[, colPhen]
  
  # **Step 1: Rank Genotypes by Observed Values in Testing Population**
  observed_data <- data.table(Year = Y2[,1], Genotype = Y2[,2], Observed_Value = y, CV = Y2[,colCVScheme])
  test_observed <- observed_data[CV == 1]  # Filter only testing population
  top_30_observed <- test_observed[order(-Observed_Value)][1:30, .(Year, Genotype, Observed_Value, Trait = trait_name)]
  
  # Store results
  top_30_observed_list[[trait_name]] <- top_30_observed
  
  # **Step 2: Train Model on Training Population & Predict in Testing Population**
  yNA <- y
  yNA[testing] <- NA  # Mask testing set during training
  
  fm <- BGLR(y = yNA, ETA = ETA_GHW, nIter = nIter, burnIn = burnIn, verbose = TRUE)
  
  # Extract predictions for all genotypes
  predictions <- data.table(Year = Y2[,1], Genotype = Y2[,2], Predicted_Value = fm$yHat, CV = Y2[,colCVScheme])
  
  # Filter only testing set
  test_predictions <- predictions[CV == 1]
  
  # Rank and select top 30 based on predicted values
  top_30_predicted <- test_predictions[order(-Predicted_Value)][1:30, .(Year, Genotype, Predicted_Value, Trait = trait_name)]
  
  # Store results
  top_30_predicted_list[[trait_name]] <- top_30_predicted
  
  # **Step 3: Compute Coincidence Index (CI)**
  matching_genotypes <- intersect(top_30_observed$Genotype, top_30_predicted$Genotype)
  num_matching <- length(matching_genotypes)
  CI <- (num_matching / 30) * 100  # Coincidence Index
  
  coincidence_results[[trait_name]] <- data.table(Trait = trait_name, Matching_Genotypes = num_matching, CI = CI)
  
  # Cleanup
  rm(fm)
  gc()
}

# Combine results into final tables
final_observed <- rbindlist(top_30_observed_list)
final_predicted <- rbindlist(top_30_predicted_list)
final_coincidence <- rbindlist(coincidence_results)

# Save results to Excel
write.xlsx(final_observed, file = "Top_30_Observed.xlsx")
write.xlsx(final_predicted, file = "Top_30_Predicted.xlsx")
write.xlsx(final_coincidence, file = "Coincidence_Index.xlsx")

# Print Coincidence Index summary
print(final_coincidence)
