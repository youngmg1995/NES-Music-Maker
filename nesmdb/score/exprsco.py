import numpy as np

def exprsco_to_rawsco(exprsco, clock=1789773.):
  rate, nsamps, exprsco = exprsco

  m = exprsco[:, :3, 0]
  m_zero = np.where(m == 0)

  m = m.astype(np.float32)
  f = 440 * np.power(2, ((m - 69) / 12))

  f_p, f_tr = f[:, :2], f[:, 2:]

  t_p = np.round((clock / (16 * f_p)) - 1)
  t_tr = np.round((clock / (32 * f_tr)) - 1)
  t = np.concatenate([t_p, t_tr], axis=1)

  t = t.astype(np.uint16)
  t[m_zero] = 0
  th = np.right_shift(np.bitwise_and(t, 0b11100000000), 8)
  tl = np.bitwise_and(t, 0b00011111111)

  rawsco = np.zeros((exprsco.shape[0], 4, 4), dtype=np.uint8)
  rawsco[:, :, 2:] = exprsco[:, :, 1:]
  rawsco[:, :3, 0] = th
  rawsco[:, :3, 1] = tl
  rawsco[:, 3, 1:] = exprsco[:, 3, :]

  return (clock, rate, nsamps, rawsco)