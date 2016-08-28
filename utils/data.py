import numpy as np
import Image

def reshapeDataset(dataset, newWidth, newHeight):
  """Reshape a given datset to a new one with predefined size: (newWidth, newHeight).

  Args:
    dataset (numpy array): dataset array
    newWidth (int)       : width of the new image
    newHeight (int)      : height of the new image

  Returns:
    numpy array : reshaped dataset

  """
  new_dataset = []
  for data in dataset:
    size = (newWidth, newHeight)
    img = Image.fromarray(data)
    img = img.resize(size)
    img = np.array(img)
    new_dataset.append(img[np.newaxis, :, :])

  return np.array(new_dataset)
