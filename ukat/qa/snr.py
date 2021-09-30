import numpy as np


class Tsnr:
    """
    Attributes
    ----------
    tsnr_map : np.ndarray
        Map of temporal signal to noise ratio.
    """

    def __init__(self, pixel_array, affine, mask=None):
        """Initialise a temporal signal to noise ratio (tSNR) class instance.

        Parameters
        ----------
        pixel_array : np.ndarray
            A array containing the signal from each voxel at each inversion
            time with the last dimension being repeated dynamics i.e. the
            array needed to generate a tSNR map would have dimensions [x,
            y, z, d].
        affine : np.ndarray
            A matrix giving the relationship between voxel coordinates and
            world coordinates.
        mask : np.ndarray, optional
            A boolean mask of the voxels to fit. Should be the shape of the
            desired tSNR map rather than the raw data i.e. omit the dynamics
            dimension.
        """

        self.pixel_array = pixel_array
        self.shape = pixel_array.shape[:-1]
        self.dimensions = len(pixel_array.shape)
        self.n_d = pixel_array.shape[-1]
        self.affine = affine
        # Generate a mask if there isn't one specified
        if mask is None:
            self.mask = np.ones(self.shape, dtype=bool)
        else:
            self.mask = mask
        # Don't process any nan values
        self.mask[np.isnan(np.sum(pixel_array, axis=-1))] = False

        # Initialise output attributes
        self.tsnr_map = np.zeros(self.shape)
        self.tsnr_map = self.__tsnr__()

    def __tsnr__(self):
        # Regress out linear and quadratic temporal drifts associated with
        # hardware using a GLM of the form Y = X * beta + error
        # as in Hutton C et al. The impact of physiological noise correction
        # on fMRI at 7T. NeuroImage 2011;57:101–112 doi:
        # 10.1016/j.neuroimage.2011.04.018.

        # Vectorise image
        pixel_array_vector = np.reshape(self.pixel_array,
                                        (np.prod(self.shape), self.n_d))
        x = np.vstack([np.ones(self.n_d),
                       np.arange(1, self.n_d + 1),
                       np.arange(1, self.n_d + 1) ** 2]).T
        beta = np.linalg.pinv(x).dot(pixel_array_vector.T)
        pixel_array_vector_detrended = pixel_array_vector.T - x[:, 1:].dot(beta[1:])
        pixel_array_detrended = pixel_array_vector_detrended.T.reshape((
            *self.shape, self.n_d))
        tsnr_map = pixel_array_detrended.mean( axis=-1) / \
                   pixel_array_detrended.std(axis=-1) # Might want to try
        # ddof=1 as per Kevins code...
        tsnr_map[tsnr_map > 1000] = 0
        return tsnr_map

