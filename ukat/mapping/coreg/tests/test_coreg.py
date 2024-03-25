import numpy as np
import matplotlib.pyplot as plt
from ukat.mapping.coreg.elastix_parameters import (DWI_BSplines, T1_BSplines,ASL_BSplines,Custom_Parameter_Map)
from ukat.mapping.coreg.coregistration import _coregister, default_elastix_parameters

#Fixed/Target
vol1 = np.zeros((50, 50))
for x in range(vol1.shape[0]):
    for y in range(vol1.shape[1]):
        vol1[x, y] = np.linalg.norm(np.subtract([x, y], [20, 20])) < 4
#Moving/Source
vol2 = np.zeros((50, 50))
for x in range(vol2.shape[0]):
    for y in range(vol2.shape[1]):
        vol2[x, y] = np.linalg.norm(np.subtract([x, y], [22, 22])) < 4

plt.imshow(vol1)
plt.imshow(vol2)

#Vol1 source Vol2 target/moving
spacing = 0.1
target = vol1
source = vol2

coregistered, deformation_field = _coregister(target,source, default_elastix_parameters(), spacing)
plt.imshow(coregistered)
