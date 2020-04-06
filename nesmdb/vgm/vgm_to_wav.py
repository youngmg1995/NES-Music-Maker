import distutils.spawn
import os
import subprocess
import pkg_resources
import tempfile

import numpy as np
from scipy.io.wavfile import read as wavread, write as wavwrite


def load_vgmwav(wav_fp):
  fs, wav = wavread(wav_fp)
  assert fs == 44100
  if wav.ndim == 2:
    wav = wav[:, 0]
  wav = wav.astype(np.float32)
  wav /= 32767.
  return wav


def save_vgmwav(wav_fp, wav):
  wave = wav.copy()
  wave *= 32767.
  wave = np.clip(wave, -32768., 32767.)
  wave = wave.astype(np.int16)
  wavwrite(wav_fp, 44100, wave)
  

def vgm_to_wav(vgm):
  # Try to get binary fp from build dir
  bin_fp = None
  try:
    import nesmdb
    import inspect
    bin_dir = os.path.dirname(inspect.getfile(nesmdb))
    bin_fp = os.path.join(bin_dir, 'vgm2wav.exe')
  except:
    pass

  # Try to get binary fp at ${VGMTOWAV}
  try:
    env_var = os.environ['VGMTOWAV']
    bin_fp = env_var
  except:
    pass

  # Make sure it is accessible
  if bin_fp is not None:
    if not (os.path.isfile(bin_fp) and os.access(bin_fp, os.X_OK)):
      raise Exception('vgm2wav should be at \'{}\' but it does not exist or is not executable'.format(bin_fp))

  # Try finding it on global path otherwise
  if bin_fp is None:
    bin_fp = distutils.spawn.find_executable('vgm2wav')

  # Ensure vgm2wav was found
  if bin_fp is None:
    raise Exception('Could not find vgm2wav executable. Please set $VGMTOWAV environment variable')

  tmpdir = tempfile.mkdtemp()
  vf_filename = 'temp_vf'
  wf_filename = 'temp_wf'
    
  # Ensure the file is read/write by the creator only
  saved_umask = os.umask(77)
    
  vf_path = os.path.join(tmpdir, vf_filename)
  wf_path = os.path.join(tmpdir, wf_filename)

  try:
    vf = open(vf_path, 'wb')
    vf.write(vgm)
    vf.seek(0)
    vf.close()
  
    res = subprocess.call('{} --loop-count 1 {} {}'.format(bin_fp, vf_path, wf_path).split())
    if res > 0:
      pass
      os.remove(vf_path)
      os.remove(wf_path)
      os.umask(saved_umask)
      os.rmdir(tmpdir)
      raise Exception('Invalid return code {} from vgm2wav'.format(res))

    wav = load_vgmwav(wf_path)
    return wav

  except IOError:
    print('IOError')
  finally:
    try:
      os.remove(vf_path)
    except FileNotFoundError:
      pass
    try:
      os.remove(wf_path)
    except FileNotFoundError:
      pass
    os.umask(saved_umask)
    os.rmdir(tmpdir)
    
    
    
    