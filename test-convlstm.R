
# Generate dummy dataset ----------------------------------------------

beams <- vector(mode = "list", length = 5)
beam <- torch_eye(6) %>% nnf_pad(rep(13, 4))
beams[[1]] <- beam

for (i in 2:5) {
  beams[[i]] <- torch_roll(beam, c(-(i-1),i-1), c(1,2))
}

init_sequence <- torch_stack(beams, dim = 1)

sequences <- vector(mode = "list", length = 5)
sequences[[1]] <- init_sequence

for (i in 2:100) {
  sequences[[i]] <- transform_random_affine(init_sequence, degrees = 0, translate = c(0.5, 0.5))
}

input <- torch_stack(sequences, dim = 1)
