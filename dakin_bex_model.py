#!/usr/bin/python
# -*- coding: latin-1 -*-
"""Implementation of the brightness model by Dakin and Bex. Requires some
helper functions found in the utils.py module of TUBvision/stimuli, so be
sure to have it available."""

from __future__ import division

import numpy as np
import matplotlib.pyplot as plt

from model_utils import pad_array

class DBmodel(object):
    """
    Represents the model for lightness perception by Dakin and Bex (2003).

    Basic usage:
    >>> model = DBModel()
    >>> result = model.evaluate(stimulus)

    Parameters
    ----------
    img_size : tupel of 2 ints, optional
               the default size of images to be evaluated with the model.
               Smaller images will be padded, larger images will raise an
               error. Powers of 2 are ideal for speed. Default is (1024, 1024).
    bandwidth : scalar
                the frequency bandwidth (full width at half maximum) of the
                log-Gabor filters used by the model, in octaves.
                Default is 2, although the value given in the article is 1.
                We found that a value of 2 is required to reproduce the
                results by Dakin and Bex, so maybe the number given in the
                article is wrong, or they use a different definition of
                bandwidth.
    References
    ----------
    Dakin, S., and Bex, P. (2003). Natural image statistics mediate brightness
    'filling in'. Proceedings of the Royal Society 270(1531).
    """

    def __init__(self, img_size = (1024, 1024), bandwidth=2):
        self.img_size = np.array(img_size)
        nyquist = int(min(img_size) / 2)
        log_nyquist = int(np.log2(nyquist))
        frequencies = np.logspace(0, log_nyquist, 2 * log_nyquist + 1, base=2)
        self.multiscale_filters = np.empty((2 * log_nyquist + 1, img_size[0],
                img_size[1]))
        for i, freq in enumerate(frequencies):
            self.multiscale_filters[i, ...] = log_gabor(freq, img_size,
                                                            bandwidth)
        self.frequencies = frequencies

    def filter_responses(self, image):
        """
        Compute the responses of the model's bank of log Gabor filters to an
        input image.

        Parameters
        ----------
        image : 2D numpy array
                the image in grayscale.

        Returns
        -------
        responses : nxpxq numpy array
                    p and q are the image dimensions, n is the number of
                    filters in the model, and depends on image size. Responses
                    are ordered from lowest to highest spatial frequency.
        """
        # remove the DC component from the image
        image = image - image.mean()
        input_size = np.array(image.shape)
        if any(input_size != self.img_size):
            pad_amount = np.round((np.array(((-.1, -.1), (.1, .1))) +
                        self.img_size - input_size).T / 2)
            if pad_amount.min() < 0:
                raise RuntimeError('input image is larger than model filters')
            pad_value = 0
            image = pad_array(image, pad_amount, pad_value)
        image = np.fft.fftshift(np.fft.fft2(image))
        responses = np.real(np.fft.ifft2(np.fft.ifftshift(
                        self.multiscale_filters * image, axes=(1,2))))
        if all(input_size == self.img_size):
            return responses
        low_freq_cut = np.log2(self.img_size / input_size).max()
        return responses[2*low_freq_cut:, pad_amount[0, 0] : -pad_amount[0, 1],
                            pad_amount[1, 0] : -pad_amount[1, 1]]

    def filter_energy(self, image=None, responses=None):
        """
        Return the energy in the different frequency bands, defined as the
        variance of the filter response. Takes either an image or previously
        computed filter responses as input. Energy is normalized to have mean 1
        across all frequency bands.

        Parameters
        ----------
        image : 2D numpy array
                the image for which the energy should be computed.
        responses : list
                    the output of DBmodel.filter_responses(). This is an
                    alternative input that saves recomputing the filter
                    responses.
        Returns
        -------
        energies : 1D numpy array
                   the energy of the filter responses, ordered from lowest to
                   highest frequency.
        """
        if responses is None:
            if image is None:
                raise ValueError("Either image or responses need to be given")
            responses = self.filter_responses(image)
        energy = np.var(np.reshape(responses, (responses.shape[0], -1)), 1)
        return (energy / energy.mean())

    def evaluate(self, image):
        """
        Apply the model to an input image, represented as a 2D numpy array.

        Parameters
        ----------
        image : 2D numpy array
                the image in grayscale.

        Returns
        -------
        2D numpy array
        the model's prediction for the input stimulus.
        """
        alpha = 0
        epsilon = .001
        responses = self.filter_responses(image)
        energy = self.filter_energy(responses=responses)
        # equalize energy of different frequency responses
        responses /= (energy ** .5)[:, np.newaxis, np.newaxis]
        frequencies = self.frequencies[:len(energy)]
        while True:
            weights = 1. / (frequencies ** alpha)
            candidate = np.tensordot(responses, weights, (0,0))
            cand_energy = self.filter_energy(image=candidate)
            slope = np.polyfit(np.log(frequencies), np.log(cand_energy), 1)[0]
            slope_error = slope + .04
            alpha += (.5 * slope_error)
            if abs(slope_error) < epsilon:
                break
        return candidate

    def evaluate_file(self, filename):
        """
        convenvience function that allows directly evaluating an image that is
        available as a grayscale image file.
        """
        image = plt.imread(filename)
        return self.evaluate(image)

def log_gabor(f_peak, size, bandwidth=1):
    """
    Return a frequency space representation of a log-Gabor filter.

    Parameters
    ----------
    f_peak : scalar
             the peak frequency of the filter
    size : (y,x) tuple
           the shape of the filter
    bandwidth : scalar
                the frequency bandwidth of the filter in octaves.
    """
    Y, X = np.ogrid[-(size[0] // 2) : (size[0] + 1) // 2,
                    -(size[1] // 2) : (size[1] + 1) // 2]
    radius = (X ** 2 + Y ** 2) ** .5
    # set radius at center to 1 so it does not cause problems with log
    radius[size[0] // 2, size[1] // 2] = 1

    # determine kappa from bandwidth according to Boukerroui, Noble and Brady
    # (2004)
    kappa = np.exp(-1/4 * np.sqrt(2 * np.log(2)) * bandwidth)
    G = np.exp((-(np.log(radius / f_peak) ** 2) /
                 (2 * np.log(kappa) ** 2)))
    # the filters can be normalized to equal energy or equal max.
    # it appears that Daking and Bex used equal maximum. the
    # normalization to equal energy is given here for convenience:
    #normalization = np.exp(-1/8 * np.log(kappa) ** 2) * np.sqrt(
    #                    -2 * np.pi ** .5 / f_peak / np.log(kappa))
    # equal max:
    normalization = 1
    G *= normalization
    # set value at center to 0
    G[size[0] // 2, size[1] // 2] = 0
    return G
