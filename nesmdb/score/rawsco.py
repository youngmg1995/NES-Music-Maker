

fs = 44100.
dt = 1. / fs

def rawsco_to_ndf(rawsco):
  clock, rate, nsamps, score = rawsco

  if rate == 44100:
    ar = True
  else:
    ar = False

  max_i = score.shape[0]

  samp = 0
  t = 0.
  # ('apu', ch, func, func_val, natoms, offset)
  ndf = [
      ('clock', int(clock)),
      ('apu', 'ch', 'p1', 0, 0, 0),
      ('apu', 'ch', 'p2', 0, 0, 0),
      ('apu', 'ch', 'tr', 0, 0, 0),
      ('apu', 'ch', 'no', 0, 0, 0),
      ('apu', 'p1', 'du', 0, 1, 0),
      ('apu', 'p1', 'lh', 1, 1, 0),
      ('apu', 'p1', 'cv', 1, 1, 0),
      ('apu', 'p1', 'vo', 0, 1, 0),
      ('apu', 'p1', 'ss', 7, 2, 1), # This is necessary to prevent channel silence for low notes
      ('apu', 'p2', 'du', 0, 3, 0),
      ('apu', 'p2', 'lh', 1, 3, 0),
      ('apu', 'p2', 'cv', 1, 3, 0),
      ('apu', 'p2', 'vo', 0, 3, 0),
      ('apu', 'p2', 'ss', 7, 4, 1), # This is necessary to prevent channel silence for low notes
      ('apu', 'tr', 'lh', 1, 5, 0),
      ('apu', 'tr', 'lr', 127, 5, 0),
      ('apu', 'no', 'lh', 1, 6, 0),
      ('apu', 'no', 'cv', 1, 6, 0),
      ('apu', 'no', 'vo', 0, 6, 0),
  ]
  ch_to_last_tl = {ch:0 for ch in ['p1', 'p2']}
  ch_to_last_th = {ch:0 for ch in ['p1', 'p2']}
  ch_to_last_timer = {ch:0 for ch in ['p1', 'p2', 'tr']}
  ch_to_last_du = {ch:0 for ch in ['p1', 'p2']}
  ch_to_last_volume = {ch:0 for ch in ['p1', 'p2', 'no']}
  last_no_np = 0
  last_no_nl = 0

  for i in range(max_i):
    for j, ch in enumerate(['p1', 'p2']):
      th, tl, volume, du = score[i, j]
      timer = (th << 8) + tl
      last_timer = ch_to_last_timer[ch]

      # NOTE: This will never be perfect reconstruction because phase is not incremented when the channel is off
      retrigger = False
      if last_timer == 0 and timer != 0:
        ndf.append(('apu', 'ch', ch, 1, 0, 0))
        retrigger = True
      elif last_timer != 0 and timer == 0:
        ndf.append(('apu', 'ch', ch, 0, 0, 0))

      if du != ch_to_last_du[ch]:
        ndf.append(('apu', ch, 'du', du, 0, 0))
        ch_to_last_du[ch] = du

      if volume > 0 and volume != ch_to_last_volume[ch]:
        ndf.append(('apu', ch, 'vo', volume, 0, 0))
      ch_to_last_volume[ch] = volume

      if tl != ch_to_last_tl[ch]:
        ndf.append(('apu', ch, 'tl', tl, 0, 2))
        ch_to_last_tl[ch] = tl
      if retrigger or th != ch_to_last_th[ch]:
        ndf.append(('apu', ch, 'th', th, 0, 3))
        ch_to_last_th[ch] = th

      ch_to_last_timer[ch] = timer

    j = 2
    ch = 'tr'
    th, tl, _, _ = score[i, j]
    timer = (th << 8) + tl
    last_timer = ch_to_last_timer[ch]
    if last_timer == 0 and timer != 0:
      ndf.append(('apu', 'ch', ch, 1, 0, 0))
    elif last_timer != 0 and timer == 0:
      ndf.append(('apu', 'ch', ch, 0, 0, 0))
    if timer != last_timer:
      ndf.append(('apu', ch, 'tl', tl, 0, 2))
      ndf.append(('apu', ch, 'th', th, 0, 3))
    ch_to_last_timer[ch] = timer

    j = 3
    ch = 'no'
    _, np, volume, nl = score[i, j]
    if last_no_np == 0 and np != 0:
      ndf.append(('apu', 'ch', ch, 1, 0, 0))
    elif last_no_np != 0 and np == 0:
      ndf.append(('apu', 'ch', ch, 0, 0, 0))
    if volume > 0 and volume != ch_to_last_volume[ch]:
      ndf.append(('apu', ch, 'vo', volume, 0, 0))
    ch_to_last_volume[ch] = volume
    if nl != last_no_nl:
      ndf.append(('apu', ch, 'nl', nl, 0, 2))
      last_no_nl = nl
    if np > 0 and np != last_no_np:
      ndf.append(('apu', ch, 'np', 16 - np, 0, 2))
      ndf.append(('apu', ch, 'll', 0, 0, 3))
    last_no_np = np

    if ar:
      wait_amt = 1
    else:
      t += 1. / rate
      wait_amt = min(int(fs * t) - samp, nsamps - samp)

    ndf.append(('wait', wait_amt))
    samp += wait_amt

  remaining = nsamps - samp
  assert remaining >= 0
  if remaining > 0:
    ndf.append(('wait', remaining))

  return ndf
    