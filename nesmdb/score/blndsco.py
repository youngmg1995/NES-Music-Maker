import numpy as np

def blndsco_to_exprsco(blndsco):
  rate, nsamps, score = blndsco

  score_len = len(score)

  exprsco = np.zeros((score_len, 4, 3), dtype=np.uint8)

  for i, frame in enumerate(score):
    for j, note in enumerate(frame[:3]):
      exprsco[i, j, 0] = note
      exprsco[i, j, 1] = 0 if j == 2 else 15

  return (rate, nsamps, exprsco)