library(opera)
# Passing arguments
args <- commandArgs(trailingOnly = TRUE)
Y <- as.matrix(read.csv(args[1], header = FALSE))
experts_low <- as.matrix(read.csv(args[2], header = FALSE))
experts_high <- as.matrix(read.csv(args[3], header = FALSE))
alpha <- as.numeric(args[4])
first <- as.numeric(args[5])
second <- as.numeric(args[6])
third <-as.numeric(args[7])

# Reshape the experts arrays to have the correct dimensions
reshape_experts <- function(experts, n, d, k) {
  return(array(experts, dim = c(n, d, k)))
}

# Assuming n=1, d=24, k=3 for the first case
experts_low <- reshape_experts(experts_low, first, second, third)
experts_high <- reshape_experts(experts_high, first, second, third)

aggregate_boas <- function(Y, experts_low, experts_high, alpha) {
  results <- list()

  tau_low <- alpha / 2
  tau_high <- 1 - alpha / 2

  mlpol_grad_low <- mixture(Y = Y, experts = experts_low, model = "BOA", loss.gradient = TRUE,
                            loss.type = list(name = "pinball", tau = tau_low))
  mlpol_grad_high <- mixture(Y = Y, experts = experts_high, model = "BOA", loss.gradient = TRUE,
                             loss.type = list(name = "pinball", tau = tau_high))

  results$weights_low <- mlpol_grad_low$weights
  results$weights_high <- mlpol_grad_high$weights

  return(results)
}

results <- aggregate_boas(Y, experts_low, experts_high, alpha)

# Output the results
print(results$weights_low)
print(results$weights_high)