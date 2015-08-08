#!/usr/bin/python
# -*- coding: latin-1 -*-
"""Implementation of the oriented difference of gaussians (ODOG) brightness
model by Blakeslee and McCourt. Requires some helper functions found in the
utils.py module of TUBvision/stimuli, so be sure to have it available."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from model_utils import degrees_to_pixels, pad_array

class OdogModel(object):
    """
    Represents an oriented difference of gaussians model for lightness
    perception. The implementation follows publications by Blakeslee and
    McCourt. Default values are chosen to match these publications.
    The adaptation implementation is described in Betz et al. (2015)

    Basic usage:
    >>> om = OdogModel()
    >>> result = om.evaluate(stimulus)

    Parameters
    ----------
    img_size : tupel of 2 ints, optional
               the default size of images to be evaluated with the model.
               Smaller images will be padded, larger images will raise an
               error. Powers of 2 are ideal for speed. Default is (1024, 1024).
    spatial_scales : list of numbers, optional
                     the spatial frequencies of the filters. Each number
                     corresponds to one DOG filter and specifies the
                     center size, i.e. the distance from zero-crossing to
                     zero-crossing, in degrees of visual angle. For an octave
                     range from 12 to 3/16, use
                     3. / 2. ** np.arange(-2,5),
                     For settings corresponding to Robinson et al.'s
                     interpretation of the ODOG parameters, use
                     2 ** np.arange(-2, 5) * 1.5 * (log(2) / 6) ** .5
                     Default are Robinson et al.'s values.
    orientations : list of numbers, optional
                   the orientations of the filters in degrees. 0 degrees is
                   a vertical filter, 90 degrees is a horizontal one.
                   Default is 30deg steps from 0 to 150.
    pixels_per_degree : number, optional
                        specifies how many pixels fit within one degree of
                        visual angle in the experimental setup. Default is
                        30.
    weights_slope : number, optional
                    the slope of the log-log function that relates a filter
                    spatial frequency to its weight in the summation.
                    Default is 0.1.
    UNODOG : bool, optional
             set to true to turn orientation normalization off. Default is
             False.

    References
    ----------
    [1] Blakeslee, B., & McCourt, M. E. (1997). Similar mechanisms underlie
    simultaneous brightness contrast and grating induction. Vision research,
    37(20), 2849–69.
    [2] Blakeslee, B., & McCourt, M. E. (1999). A multiscale spatial filtering
    account of the White effect, simultaneous brightness contrast and grating
    induction. Vision research, 39(26), 4361–77.
    [3] Blakeslee, B., & McCourt, M. E. (2003). A multiscale spatial filtering
    account of brightness phenomena. In L. Harris & M. Jenkin (Eds.), Levels of
    perception (pp. 45–70). New York, New York, USA: Springer.
    [4] Betz, T., Shapley, R., Wichmann, F.A., & Maertens, M. (2015) Testing
    the role of luminance edges in White's illusion with contour adaptation.
    Journal of Vision
    """
    def __init__(self,
                 img_size = (1024, 1024),
                 spatial_scales=2.**np.arange(-2,5) * 1.5 * (np.log(2) / 6)**.5,
                 orientations=np.arange(0,180,30),
                 pixels_per_degree=30,
                 weights_slope=.1,
                 UNODOG = False):
        """
        Create an ODOG model instance.
        """
        # determine standard deviation from zero_crossing distance,
        # assuming that the surround sigma is 2 times center sigma.
        center_sigmas = np.array(spatial_scales)
        center_sigmas = center_sigmas * .125 * np.sqrt(3. / np.log(2))
        center_sigmas = degrees_to_pixels(center_sigmas, pixels_per_degree)

        # create filter bank. filtering will be done in frequency space,
        # therefore we directly save the Fourier transforms of the filters.
        # Because our stimuli are real, we only have to save half of the
        # symmetric matrix.
        self.multiscale_filters = np.empty((len(orientations),
                                            len(spatial_scales),
                                            img_size[0],
                                            int(img_size[1] / 2 + 1)),
                                            dtype='complex128')
        for i, angle in enumerate(orientations):
            for j, center_sigma in enumerate(center_sigmas):
                self.multiscale_filters[i, j, :, :] = np.fft.rfft2(
                    difference_of_gaussians((center_sigma, 2 * center_sigma),
                                            center_sigma, angle, img_size))

        self.img_size = img_size
        self.spatial_scales = np.array(spatial_scales)
        spatial_frequency = 1. / self.spatial_scales
        # The weights are applied according to the spatial frequency, which is
        # proportional to the reciprocal of the space constant. The exact
        # conversion factor is irrelevant, since only the relative weights
        # matter, and those are unaffected by a multiplicative factor applied
        # to all frequencies.
        self.scale_weights = spatial_frequency ** weights_slope
        self.orientations = orientations
        self.UNODOG = UNODOG

    def evaluate(self, image, return_detailed=False, pad_val=None, adapt=None,
            max_attenuation=.5, adapt_mu=.08, adapt_sigma=.005):
        """
        Apply the model to an input image, represented as a 2D numpy array.
        Optionally returns the responses of all individual filters, and the
        vector of normalization weights given to the orientations.

        Parameters
        ----------
        image : 2D numpy array
                the grayscale image for which to compute the model response.
                Must be smaller or equal in size to the img_size parameter of
                the model. Smaller images will be padded with the image's
                border value if it is the same everywhere, otherwise with the
                image mean.
        return_detailed : boolean (optional)
                          flag that specifies whether the method only returns
                          the model response, or additionally returns all
                          filter responses and the normalization weights.
                          Default is False.
        pad_val : scalar
                  allows explicit setting of the desired pad value
        adapt : 2D numpy array (optional)
                a stimulus that the model filters will be adapted to.
                Adaptation will lead to reduced filter response to the actual
                input image, proportional to the filter response to the
                adapting stimulus. Should have the same shape as image. If no
                adapting stimulus is passed, no adaptation occurs.
        max_attenuation : scalar in [0, 1] (optional)
                          the factor that filter responses that reach
                          adapt_saturation level will be multiplied with to
                          simulate adaptation. Smaller values imply stronger
                          adaptation. Ignored if adapt == None.
        adapt_mu : number (optional)
                   the value of the adaptor response for which the adaptation
                   level reaches 50% of the maxmial attenuation.
        adapt_sigma : number (optional)
                      the standard deviation of the cummulative Gaussian
                      relating the response to the adaptor to the adaptation
                      strength.

        Returns
        -------
        result : 2D numpy array
                 the model prediction.
        responses : 4D numpy array (optional)
                    the responses of the individual ODOG filters. First
                    dimension gives the orientations, second dimension the
                    spatial scales, third and fourth dimensions are the shape
                    of the input image. Only returned if return_detailed is
                    True. In that case, the output is a tuple of (result,
                    responses, weights).
        weights : 1D numpy array (optional)
                  the normalization weights of the orientations.
        """

        # compute response to adaptor if there is one
        if adapt is not None:
            assert adapt.shape == image.shape
            [_, adapt_weights, _] = self.evaluate(adapt, True)
            # normalize adaptation weights to max of 1 for all values above
            # cutoff
            adapt_weights = 1 - (1 - max_attenuation) * norm.cdf(
                    np.abs(adapt_weights), loc=adapt_mu, scale=adapt_sigma)
        else:
            adapt_weights = 1

        # use padding to make the stimulus match the size of the filters.
        # if there is a constant border value, pad with that value, otherwise
        # pad with image mean.
        input_size = image.shape
        pad_amount = None
        if input_size != self.img_size:
            pad_amount = np.round((np.array(((-.1, -.1), (.1, .1))) +
                        self.img_size - input_size).T / 2)
            if pad_amount.min() < 0:
                raise RuntimeError('input image is larger than model filters')
            if pad_val is None:
                if not border_is_constant(image):
                    pad_val = image.mean()
                else:
                    pad_val = image[0,0]
            image = pad_array(image, pad_amount, pad_val)
        image = np.fft.rfft2(image)
        # compute filter output as the inverse Fourier transform of the product
        # of filters and image in frequency space
        responses = np.fft.fftshift(np.fft.irfft2(
                        self.multiscale_filters * image, s = self.img_size),
                        axes=(2,3))
        if pad_amount is not None:
            responses = responses[:, :,
                    pad_amount[0, 0] : self.img_size[0] -pad_amount[0, 1],
                    pad_amount[1, 0] : self.img_size[1] -pad_amount[1, 1]]
        # apply adaptation to filter outputs
        responses = responses * adapt_weights
        # compute the weighted sum over different spatial scales
        orientation_output = np.tensordot(responses, self.scale_weights, (1,0))
        # normalize filter response within each orientation with its RMS
        normalization = 1. / (orientation_output ** 2).mean(-1).mean(-1) ** .5
        if self.UNODOG:
            normalization[:] = 1
        # set filters with practically no signal to 0 (rather arbitrary)
        normalization[normalization > 1e10] = 0
        # sum orientation outputs according to the normalization weights
        result = np.tensordot(orientation_output, normalization, (0,0))
        if return_detailed:
            return (result, responses, normalization)
        return result

    def evaluate_file(self, filename):
        """
        convenvience function that allows directly evaluating an image that is
        available as a grayscale image file.
        """
        image = plt.imread(filename)
        return self.evaluate(image)

def difference_of_gaussians(sigma_y, sigma_x, angle=0, size=None):
    """
    Compute a 2D difference of Gaussians kernel.

    Parameters
    ----------
    sigma_y : number or tuple of 2 numbers
              The standard deviations of the two Gaussians along the vertical
              if angle is 0, or along a line angle degrees from the vertical in
              the clockwise direction. If only a single number is given, it is
              used for both Gaussians.
    sigma_x : number or tuple of 2 numbers
              Same as sigma_y, only for the orthogonal direction.
    angle : number or tuple of two numbers, optional
            The rotation angles of the two Gaussians in degrees. If only a
            single number is given, it is used for both Gaussians. Default is
            0.
    size : tuple of two ints, optional
           the shape of the output. The default is chosen such that the output
           array is large enough to contain the kernel up to 5 standard
           deviations of the larger Gaussian in each direction.

    Returns
    -------
    output : 2D numpy array
    """
    if not np.iterable(sigma_y):
        sigma_y = (sigma_y, sigma_y)
    if not np.iterable(sigma_x):
        sigma_x = (sigma_x, sigma_x)
    if not np.iterable(angle):
        angle = (angle, angle)

    outer_gaussian = gaussian(sigma_y[1], sigma_x[1], angle[1], size)
    inner_gaussian = gaussian(sigma_y[0], sigma_x[0], angle[0],
                                outer_gaussian.shape)
    return inner_gaussian - outer_gaussian

def gaussian(sigma_y, sigma_x, angle=0, size=None):
    """
    compute a two-dimensional Gaussian kernel.

    Parameters
    ----------
    sigma_y : number
              the standard deviation of an unrotated kernel along the vertical
    sigma_x : number
              standard deviation of an unrotated kernel along the horizontal
    angle : number, optional
            the rotation angle of the kernel in the clockwise direction in
            degrees. Default is 0.
    size : tupel of 2 ints, optional
           the shape of the output. The default is chosen such that the output
           array is large enough to contain the kernel up to 7.5 standard
           deviations in each direction.

    Returns
    -------
    output : 2D numpy array
    """

    # compute the covariance matrix of the rotated multivariate gaussian
    theta = np.radians(angle)
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    sigma = np.dot(np.dot(R, np.array([[sigma_x ** 2, 0], [0, sigma_y ** 2]])),
                    R.T)

    if size is None:
        size = (np.ceil(7.5 * sigma[1,1] ** .5), np.ceil(7.5 * sigma[0,0] ** .5))

    # create a grid on which to evaluate the multivariate gaussian pdf formula
    (Y,X) = np.ogrid[-(size[0] - 1) / 2. : size[0] / 2.,
                     -(size[1] - 1) / 2. : size[1] / 2.]
    # apply the gaussian pdf formula to every point in the grid
    gauss = 1 / (np.linalg.det(sigma) ** .5 * 2 * np.pi) * \
            np.exp(-.5 / (1 - np.prod(sigma[::-1,:].diagonal()) /
                              np.prod(sigma.diagonal()))  *
                    (X ** 2 / sigma[0,0] + Y ** 2 / sigma[1,1] -
                    2 * sigma[1,0] * X * Y / sigma[0,0] / sigma[1,1]))
    return gauss / gauss.sum()

def border_is_constant(arr):
    """
    Helper function to check if the border of an array has a constant value.
    """
    if len(arr.shape) != 2:
        raise ValueError('function only works for 2D arrays')
    border =  np.concatenate((arr[[0,-1],:].ravel(), arr[:,[0,-1]].ravel()))
    return len(np.unique(border)) == 1
