library(torch)
library(torchvision)

source("convlstm.R")

# Generate dummy data ----------------------------------------------

beams <- vector(mode = "list", length = 6)
beam <- torch_eye(6) %>% nnf_pad(c(6, 12, 12, 6)) # left, right, top, bottom
beams[[1]] <- beam

for (i in 2:6) {
  beams[[i]] <- torch_roll(beam, c(-(i-1),i-1), c(1, 2))
}

init_sequence <- torch_stack(beams, dim = 1)

sequences <- vector(mode = "list", length = 100)
sequences[[1]] <- init_sequence

for (i in 2:100) {
  sequences[[i]] <- transform_random_affine(init_sequence, degrees = 0, translate = c(0.5, 0.5))
}

input <- torch_stack(sequences, dim = 1)
# add channels dimension
input <- input$unsqueeze(3)
dim(input)


# create dataset and dataloader --------------------------------------

dummy_ds <- dataset(
  
  initialize = function(data) {
    self$data <- data
  },
  
  .getitem = function(i) {
    list(x = self$data[i, 1:5, ..], y = self$data[i, 6, ..])
  },
  
  .length = function() {
    nrow(self$data)
  }
)

ds <- dummy_ds(input)
dl <- dataloader(ds, batch_size = 100)


# train -------------------------------------------------------------------

model <- convlstm(input_dim = 1, hidden_dims = c(64, 1), kernel_sizes = c(3, 3), n_layers = 2)

optimizer <- optim_adam(model$parameters)

num_epochs <- 100

for (epoch in 1:num_epochs) {
  
  model$train()
  batch_losses <- c()
  
  for (b in enumerate(dl)) {
    
    optimizer$zero_grad()
    
    # last-timestep output from last layer
    preds <- model(b$x)[[2]][[2]][[1]]
  
    loss <- nnf_mse_loss(preds, b$y)
    batch_losses <- c(batch_losses, loss$item())
    
    loss$backward()
    optimizer$step()
  }
  
  if (epoch %% 10 == 0) cat(sprintf("\nEpoch %d, training loss:%3f\n", epoch, mean(batch_losses)))
}


# inspect predictions ------------------------------------------------------

dl <- dataloader(ds, batch_size = 1)
b <- dl$.iter()$.next()

b$y[1, 1, 6:15, 10:19]
round(as.matrix(preds[1, 1, 6:15, 10:19]), 2)

