import nesmdb.vgm
import nesmdb.score

def seprsco_to_vgm(seprsco):
  exprsco = nesmdb.score.seprsco_to_exprsco(seprsco)
  rawsco = nesmdb.score.exprsco_to_rawsco(exprsco)
  ndf = nesmdb.score.rawsco_to_ndf(rawsco)
  ndr = nesmdb.vgm.ndf_to_ndr(ndf)
  vgm = nesmdb.vgm.ndr_to_vgm(ndr)
  return vgm

def seprsco_to_wav(seprsco):
  exprsco = nesmdb.score.seprsco_to_exprsco(seprsco)
  rawsco = nesmdb.score.exprsco_to_rawsco(exprsco)
  ndf = nesmdb.score.rawsco_to_ndf(rawsco)
  ndr = nesmdb.vgm.ndf_to_ndr(ndf)
  vgm = nesmdb.vgm.ndr_to_vgm(ndr)
  wav = nesmdb.vgm.vgm_to_wav(vgm)
  return wav