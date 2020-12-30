library(torch)
library(torchvision)

convlstm_cell <- nn_module(
  
  initialize = function(input_dim, hidden_dim, kernel_size, bias) {
    
    self$hidden_dim <- hidden_dim
    
    padding <- kernel_size %/% 2
    
    self$conv <- nn_conv2d(
      in_channels = input_dim + self$hidden_dim,
      # for each of input, forget, output, and cell gates
      out_channels = 4 * self$hidden_dim,
      kernel_size = kernel_size,
      padding = padding,
      bias = bias
    )
  },
  
  forward = function(x, prev_states) {

    h_prev <- prev_states[[1]]
    c_prev <- prev_states[[2]]
    
    combined <- torch_cat(list(x, h_prev), dim = 2)  # concatenate along channel axis
    combined_conv <- self$conv(combined)
    gate_convs <- torch_split(combined_conv, self$hidden_dim, dim = 2)
    cc_i <- gate_convs[[1]]
    cc_f <- gate_convs[[2]]
    cc_o <- gate_convs[[3]]
    cc_g <- gate_convs[[4]]
    
    # input, forget, output, and cell gates (corresponding to torch's LSTM)
    i <- torch_sigmoid(cc_i)
    f <- torch_sigmoid(cc_f)
    o <- torch_sigmoid(cc_o)
    g <- torch_tanh(cc_g)
    
    # cell state
    c_next <- f * c_prev + i * g
    # hidden state
    h_next <- o * torch_tanh(c_next)
    
    list(h_next, c_next)
    
  },
  
  init_hidden = function(batch_size, height, width) {
    list(torch_zeros(batch_size, self$hidden_dim, height, width, device = self$conv$weight$device),
         torch_zeros(batch_size, self$hidden_dim, height, width, device = self$conv$weight$device))
  }
)

convlstm <- nn_module(
  
  initialize = function(input_dim, hidden_dims, kernel_sizes, n_layers, bias = TRUE) {
 
    self$n_layers <- n_layers
    
    self$cell_list <- nn_module_list()
    
    for (i in 1:n_layers) {
      cur_input_dim <- if (i == 1) input_dim else hidden_dims[i - 1]
      self$cell_list$append(convlstm_cell(cur_input_dim, hidden_dims[i], kernel_sizes[i], bias))
    }
  },
  
  # we always assume batch-first
  forward = function(x) {
    
    batch_size <- x$size()[1]
    seq_len <- x$size()[2]
    height <- x$size()[4]
    width <- x$size()[5]
   
    # initialize hidden states
    init_hidden <- vector(mode = "list", length = self$n_layers)
    for (i in 1:self$n_layers) {
      init_hidden[[i]] <- self$cell_list[[i]]$init_hidden(batch_size, height, width)
    }
    
    # list containing the outputs, of length seq_len, for each layer
    # this is the same as h, at each step in the sequence
    layer_output_list <- vector(mode = "list", length = self$n_layers)
    
    # list containing the last states (h, c) for each layer
    layer_state_list <- vector(mode = "list", length = self$n_layers)

    cur_layer_input <- x
    hidden_states <- init_hidden
    
    # loop over layers
    for (i in 1:self$n_layers) {
      
      # every layer's hidden state starts from 0 (non-stateful)
      h_c <- hidden_states[[i]]
      h <- h_c[[1]]
      c <- h_c[[2]]
      # outputs, of length seq_len, for this layer
      # equivalently, list of h states for each time step
      output_sequence <- vector(mode = "list", length = seq_len)
      
      # loop over timesteps
      for (t in 1:seq_len) {
        h_c <- self$cell_list[[i]](cur_layer_input[ , t, , , ], list(h, c))
        h <- h_c[[1]]
        c <- h_c[[2]]
        # keep track of output (h) for every timestep
        # h has dim (batch_size, hidden_size, height, width)
        output_sequence[[t]] <- h
      }

      # stack hs for all timesteps over seq_len dimension
      # stacked_outputs has dim (batch_size, seq_len, hidden_size, height, width)
      # same as input to forward (x)
      stacked_outputs <- torch_stack(output_sequence, dim = 2)
      
      # pass the list of outputs (hs) to next layer
      cur_layer_input <- stacked_outputs
      
      # keep track of list of outputs or this layer
      layer_output_list[[i]] <- stacked_outputs
      # keep track of last state for this layer
      layer_state_list[[i]] <- list(h, c)
    }
 
    list(layer_output_list, layer_state_list)
      
  }
    
)


# convlstm output ---------------------------------------------------------

# batch_size, seq_len, channels, height, width
x <- torch_rand(c(2, 4, 3, 16, 16))


# single-layer ------------------------------------------------------------

model <- convlstm(input_dim = 3, hidden_dims = 5, kernel_sizes = 3, n_layers = 1)

ret <- model(x)
layer_outputs <- ret[[1]]
layer_last_states <- ret[[2]]

# for each layer, tensor of size (batch_size, seq_len, hidden_size, height, width)
layer_outputs[[1]]

# list of 2 tensors for each layer
layer_last_states[[1]]
# h, of size (batch_size, hidden_size, height, width)
layer_last_states[[1]][[1]]
# c, of size (batch_size, hidden_size, height, width)
layer_last_states[[1]][[2]]


# multiple-layer ----------------------------------------------------------

model <- convlstm(input_dim = 3, hidden_dims = c(5, 5, 1), kernel_sizes = c(3, 3, 3), n_layers = 3)

ret <- model(x)
layer_outputs <- ret[[1]]
layer_last_states <- ret[[2]]

# for each layer, tensor of size (batch_size, seq_len, hidden_size, height, width)
dim(layer_outputs[[1]])
dim(layer_outputs[[2]])
dim(layer_outputs[[3]])

# list of 2 tensors for each layer
str(layer_last_states)
# h, of size (batch_size, hidden_size, height, width)
dim(layer_last_states[[3]][[1]])
# c, of size (batch_size, hidden_size, height, width)
dim(layer_last_states[[3]][[2]])


