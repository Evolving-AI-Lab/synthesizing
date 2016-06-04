import numpy as np

def normalize(img, out_range=(0.,1.), in_range=None):
  if not in_range:
    min_val = np.min(img)
    max_val = np.max(img)
  else:
    min_val = in_range[0]
    max_val = in_range[1]
  result = np.copy(img)
  result[result > max_val] = max_val
  result[result < min_val] = min_val
  result = (result - min_val) / (max_val - min_val) * (out_range[1] - out_range[0]) + out_range[0]
  return result

def patchShow(images, out_range=(0.,1.), in_range=None, rows=0, cols=0):
  num = images.shape[0]
  ih = images.shape[2]
  iw = images.shape[3]
  
  if rows == 0 and cols == 0:
    rows = np.ceil(np.sqrt(num*iw/ih))
  if cols == 0:
    cols = np.ceil(num / float(rows))
  if rows == 0:
    rows = np.ceil(num / float(cols))
    
  result = np.zeros((rows*(ih+1)+1, cols*(iw+1)+1, 3))
  for ind in range(num):
    r,c = divmod(ind, cols)
    result[r*(ih+1)+1:(r+1)*(ih+1), c*(iw+1)+1:(c+1)*(iw+1), :] = images[ind].transpose((1,2,0))    
  result = normalize(result, out_range, in_range)
  return result


def patchShow_single(images, out_range=(0.,1.), in_range=None):
  num = images.shape[0]
  c = images.shape[1]
  ih = images.shape[2]
  iw = images.shape[3]
    
  result = np.zeros((ih, iw, 3))

  # Normalize before saving
  result[:] = images[0].copy().transpose((1,2,0))
  result = normalize(result, out_range, in_range)
  return result

