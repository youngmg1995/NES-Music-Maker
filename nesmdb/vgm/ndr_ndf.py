from collections import Counter, OrderedDict

from nesmdb.apu import *
from nesmdb.vgm.bintypes import *

def ndf_to_ndr(ndf):
  ndr = ndf[:1]
  ndf = ndf[1:]

  registers = {
      'p1': [0x00] * 4,
      'p2': [0x00] * 4,
      'tr': [0x00] * 4,
      'no': [0x00] * 4,
      'dm': [0x00] * 4,
      'ch': [0x00],
      'fc': [0x00]
  }

  # Convert commands to VGM
  regn_to_val = OrderedDict()
  for comm in ndf:
    itype = comm[0]
    if itype == 'wait':
      for _, (arg1, arg2) in regn_to_val.items():
        ndr.append(('apu', b2h(c2b(arg1)), b2h(c2b(arg2))))
      regn_to_val = OrderedDict()

      amt = comm[1]

      ndr.append(('wait', amt))
    elif itype == 'apu':
      dest = comm[1]
      param = comm[2]
      val = comm[3]
      natoms = comm[4]
      param_offset = comm[5]

      # Find offset/bitmask
      reg = registers[dest]
      param_bitmask = func_to_bitmask(dest, param)

      # Apply mask
      mask_bin = '{:08b}'.format(param_bitmask)
      nbits = mask_bin.count('1')
      if val < 0 or val >= (2 ** nbits):
        raise ValueError('{}, {} (0, {}]: invalid value specified {}'.format(comm[1], comm[2], (2 ** nbits), val))
      assert val >= 0 and val < (2 ** nbits)
      shift = max(0, 7 - mask_bin.rfind('1')) % 8
      val_old = reg[param_offset]
      reg[param_offset] &= (255 - param_bitmask)
      reg[param_offset] |= val << shift
      assert reg[param_offset] < 256
      val_new = reg[param_offset]

      arg1 = register_memory_offsets[dest] + param_offset
      arg2 = reg[param_offset]

      regn_to_val[(dest, param_offset, natoms)] = (arg1, arg2)
    elif itype == 'ram':
      # TODO
      continue
    else:
      raise NotImplementedError()

  for _, (arg1, arg2) in regn_to_val.items():
    ndr.append(('apu', b2h(c2b(arg1)), b2h(c2b(arg2))))

  return ndr