from nesmdb.apu import *
from nesmdb.vgm.bintypes import *

def ndr_to_vgm(ndr):
  # Old python 2 implementations are commented out
  assert ndr[0][0] == 'clock'
  clock = ndr[0][1]

  ndr = ndr[1:]

  EMPTYBYTE = i2lub(0)
  #flatten = lambda vgm: list(''.join(vgm))
  flatten = lambda vgm: [c2b(c) for c in b''.join(vgm)]
  byte_list = lambda b: [c2b(c) for c in b]
  vgm = flatten([EMPTYBYTE] * 48)

  # VGM identifier
  vgm[:0x04] = [c2b(c) for c in [0x56, 0x67, 0x6d, 0x20]]
  # Version
  #vgm[0x08:0x0c] = i2lub(0x161)
  vgm[0x08:0x0c] = byte_list(i2lub(0x161))
  # Clock rate
  #vgm[0x84:0x88] = i2lub(clock)
  vgm[0x84:0x88] = byte_list(i2lub(clock))
  # Data offset
  #vgm[0x34:0x38] = i2lub(0xc0 - 0x34)
  vgm[0x34:0x38] = byte_list(i2lub(0xc0 - 0x34))

  wait_sum = 0
  for comm in ndr:
    itype = comm[0]
    if itype == 'wait':
      amt = comm[1]
      wait_sum += amt

      while amt > 65535:
        vgm.append(c2b(0x61))
        vgm.append(i2lusb(65535))
        amt -= 65535

      vgm.append(c2b(0x61))
      vgm.append(i2lusb(amt))
    elif itype == 'apu':
      arg1 = h2b(comm[1])
      arg2 = h2b(comm[2])
      vgm.append(c2b(0xb4))
      vgm.append(arg1)
      vgm.append(arg2)
    elif itype == 'ram':
      raise NotImplementedError()
    else:
      raise NotImplementedError()

  # Halt
  vgm.append(c2b(0x66))
  vgm = flatten(vgm)

  # Total samples
  #vgm[0x18:0x1c] = i2lub(wait_sum)
  vgm[0x18:0x1c] = byte_list(i2lub(wait_sum))
  # EoF offset
  #vgm[0x04:0x08] = i2lub(len(vgm) - 0x04)
  vgm[0x04:0x08] = byte_list(i2lub(len(vgm) - 0x04))

  #vgm = ''.join(vgm)
  vgm = b''.join(vgm)
  return vgm