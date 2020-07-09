import numpy as np


class T2Star(object):
    """Package containing algorithms that calculate parameter maps
    of the MRI scans acquired during the UKRIN-MAPS project.

    Attributes
    ----------
    See parameters of __init__ class

    """

    def __init__(self, pixel_array, echo_list):
        """Initialise a T2StarCode class instance.

        Parameters
        ----------
        pixel_array : 4D/3D array
            A 4D/3D array containing the signal from each voxel at each
            echo time i.e. the dimensions of the array are [x, y, z, TE].
        echo_list : list()
            An array of the echo times used for the last dimension of the
            raw data.
        """

        self.pixel_array = pixel_array
        self.echo_list = echo_list

    # Consider splitting these methods into SubClasses at some point.
    # Could create a Diffusion Toolbox where BValues are an attribute.
    # Or Fitting, where Inversion Time and Echo Time are attributes.
    # https://www.youtube.com/watch?v=RSl87lqOXDE

    def t2star_nottingham(self):
        """
        Generates a T2* map from a series of volumes collected with different
        echo times.

        Parameters
        ----------
        See class attributes in __init__

        Returns
        -------
        t2star : 3D array
            An array containing the T2* map generated by the function with T2*
            measured in milliseconds.
        r2star : 3D array
            An array containing the R2* map generated by the function with R2*
            measured in milliseconds.
        m0 : 3D array
            An array containing the M0 map generated by the function.
        """
        self.pixel_array[self.pixel_array == 0] = 1E-10
        # If raw data is 2D (3D inc echo times) then add a dimension so it
        # can be processed in the same way as 3D data
        if len(self.pixel_array.shape) == 3:
            self.pixel_array = np.expand_dims(self.pixel_array, 2)
        t2star = np.zeros(self.pixel_array.shape[0:3])
        r2star = np.zeros(self.pixel_array.shape[0:3])
        m0 = np.zeros(self.pixel_array.shape[0:3])
        with np.errstate(invalid='ignore', over='ignore'):
            for s in range(self.pixel_array.shape[2]):
                for x in range(np.shape(self.pixel_array)[0]):
                    for y in range(np.shape(self.pixel_array)[1]):
                        noise = 0.0
                        sd = 0.0
                        s_w = 0.0
                        s_wx = 0.0
                        s_wx2 = 0.0
                        s_wy = 0.0
                        s_wxy = 0.0
                        for d in range(self.pixel_array.shape[3]):
                            noise = noise + self.pixel_array[x, y, s, d]
                            sd = sd + self.pixel_array[x, y, s, d] * \
                                self.pixel_array[x, y, s, d]
                        noise = noise / self.pixel_array.shape[3]
                        sd = sd / self.pixel_array.shape[3] - noise ** 2
                        sd = sd ** 2
                        sd = np.sqrt(sd)
                        for d in range(self.pixel_array.shape[3]):
                            te_tmp = self.echo_list[d]
                            if self.pixel_array[x, y, s, d] > sd:
                                sigma = np.log(
                                    self.pixel_array[x, y, s, d] /
                                    (self.pixel_array[x, y, s, d] - sd))
                                sig = self.pixel_array[x, y, s, d]
                                weight = 1 / (sigma ** 2)
                            else:
                                sigma = np.log(
                                    self.pixel_array[x, y, s, d] / 0.0001)
                                sig = np.log(self.pixel_array[x, y, s, d])
                                weight = 1 / (sigma ** 2)
                            weight = 1 / (sigma ** 2)
                            s_w = s_w + weight
                            s_wx = s_wx + weight * te_tmp
                            s_wx2 = s_wx2 + weight * te_tmp ** 2
                            s_wy = s_wy + weight * sig
                            s_wxy = s_wxy + weight * te_tmp * sig
                        delta = (s_w * s_wx2) - (s_wx ** 2)
                        if ((delta == 0.0) or (np.isinf(delta))
                           or (np.isnan(delta))):
                            t2star[x, y, s] = 0
                            r2star[x, y, s] = 0
                            m0[x, y, s] = 0
                        else:
                            a = (1 / delta) * (s_wx2 * s_wy - s_wx * s_wxy)
                            b = (1 / delta) * (s_w * s_wxy - s_wx * s_wy)
                            t2stars_temp = np.real(-1 / b)
                            r2stars_temp = np.real(-b)
                            m0_temp = np.real(np.exp(a))
                            if (t2stars_temp < 0) or (t2stars_temp > 5000):
                                t2star[x, y, s] = 0
                                r2star[x, y, s] = 0
                                m0[x, y, s] = 0
                            else:
                                t2star[x, y, s] = t2stars_temp
                                r2star[x, y, s] = r2stars_temp
                                m0[x, y, s] = m0_temp
        del t2stars_temp, r2stars_temp, m0_temp, delta
        return t2star, r2star, m0

    def t2star_joao(self):
        """
        Generates a T2* map from a series of volumes
        collected with different echo times.
        It's a rewritten version of t2star_nottingham with a few modifications.

        Parameters
        ----------
        See class attributes in __init__

        Returns
        -------
        t2star : 3D array
            An array containing the T2* map generated by
            the function with T2* measured in milliseconds.
        r2star : 3D array
        """
        self.pixel_array[self.pixel_array == 0] = 1E-10
        # If raw data is 2D (3D inc echo times) then add a dimension
        # so it can be processed in the same way as 3D data
        if len(self.pixel_array.shape) == 3:
            self.pixel_array = np.expand_dims(self.pixel_array, 2)
        number_echoes = len(self.echo_list)
        matrix_ones = np.ones(np.shape(
            np.squeeze(self.pixel_array[..., 0])))
        with np.errstate(invalid='ignore', over='ignore'):
            noise = (np.sum(self.pixel_array, axis=3) /
                     (number_echoes * matrix_ones))
            sd = (np.absolute(np.sum(np.square(self.pixel_array), axis=3)
                  / (number_echoes * matrix_ones) - np.square(noise)))
            s_w = s_wx = s_wx2 = np.zeros(np.shape(matrix_ones))
            s_wy = s_wxy = s_w
            for echo in range(number_echoes):
                te = self.echo_list[echo] * 0.001 * matrix_ones
                # Conversion from ms to seconds
                sigma = sig = np.zeros(np.shape(matrix_ones))
                matrix_iterator = np.nditer(sd, flags=['multi_index'])
                while not matrix_iterator.finished:
                    ind = matrix_iterator.multi_index
                    if self.pixel_array[ind][echo] > sd[ind]:
                        sigma[ind] = (np.log(self.pixel_array[ind][echo]
                                             / (self.pixel_array[ind][echo]
                                             - sd[ind])))
                        sig[ind] = self.pixel_array[ind][echo]
                    else:
                        sigma[ind] = (np.log(self.pixel_array[ind][echo]
                                             / 0.0001))
                        sig[ind] = np.log(self.pixel_array[ind][echo])
                    matrix_iterator.iternext()
                weight = matrix_ones / np.square(sigma)
                s_w = s_w + weight
                s_wx = s_wx + (weight * te)
                s_wx2 = s_wx2 + weight * np.square(te)
                s_wy = s_wy + weight * sig
                s_wxy = s_wxy + weight * te * sig
            delta = (s_w * s_wx2) - (np.square(s_wx))
            b = (matrix_ones / delta) * (s_w * s_wxy - s_wx * s_wy)
            t2star = np.real(-matrix_ones / b)
            conditions = ((np.isinf(t2star)) | (np.isnan(t2star)) |
                          (t2star < 0.0) | (t2star > 500.0))
            t2star = np.where(conditions, 0.0, t2star)
        return t2star

    def r2star(self):
        """
        Generates a R2* map from a series of volumes collected
        with different echo times. It calls t2star_joao().

        Parameters
        ----------
        See class attributes in __init__

        Returns
        -------
        r2Star : 3D array
            An array containing the R2* map generated
            by the function with R2* measured in seconds.
        """ 
        return np.ones(np.shape(self.t2star_joao()))/self.t2star_joao()
