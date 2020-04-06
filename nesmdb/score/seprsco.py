import numpy as np

def seprsco_to_exprsco(seprsco):
  rate, nsamps, score = seprsco

  score_len = score.shape[0]

  exprsco = np.zeros((score_len, 4, 3), dtype=np.uint8)

  exprsco[:, :, 0] = score
  
  exprsco[:, :, 1] = 15
  exprsco[:, 2, 1] = 0

  return (rate, nsamps, exprsco)