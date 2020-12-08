
# Generate dummy data ----------------------------------------------

beams <- vector(mode = "list", length = 8)
beam <- torch_eye(6) %>% nnf_pad(rep(13, 4))
beams[[1]] <- beam

for (i in 2:8) {
  beams[[i]] <- torch_roll(beam, c(-(i-1),i-1), c(1,2))
}

init_sequence <- torch_stack(beams, dim = 1)

sequences <- vector(mode = "list", length = 100)
sequences[[1]] <- init_sequence

for (i in 2:100) {
  sequences[[i]] <- transform_random_affine(init_sequence, degrees = 0, translate = c(0.5, 0.5))
}

input <- torch_stack(sequences, dim = 1)



# create dataset and dataloader --------------------------------------

dummy_ds <- dataset(
  
  initialize = function(data) {
    self$data <- data$unsqueeze(3)
  },
  
  .getitem = function(i) {
    list(x = self$data[i, 1:4, ..], y = self$data[i, 5:8, ..])
  },
  
  .length = function() {
    nrow(self$data)
  }
)

ds <- dummy_ds(input)
ds$.getitem(1)

dl <- dataloader(ds, batch_size = 10, shuffle = TRUE)
dl$.iter()$.next()


# train -------------------------------------------------------------------

model <- convlstm(input_dim = 1, hidden_dims = 1, kernel_sizes = 3, n_layers = 1)

optimizer <- optim_adam(model$parameters)


num_epochs <- 10

for (epoch in 1:num_epochs) {
  
  model$train()
  batch_losses <- c()
  
  for (b in enumerate(dl)) {
    
    optimizer$zero_grad()
    
    # output from last layer
    preds <- model(b$x)[[1]][[1]]
    loss <- nnf_mse_loss(preds, b$y)
    batch_losses <- c(batch_losses, loss$item())
    
    loss$backward()
    optimizer$step()
  }
  
  cat(sprintf("\nEpoch %d, training loss:%3f\n", epoch, mean(batch_losses)))
}
