Implementations of the oriented difference of gaussians model for brightness
perception by Blakeslee and McCourt, and Dakin and Bex's model of brightness
perception. Basic usage is

>>> import odog_model
>>> om = odog_model.OdogModel()
>>> result = om.evaluate(stimulus)

or

>>> import dakin_bex_model as dbm
>>> model = dbm.DBmodel()
>>> result = model.evaluate(stimulus)

where stimulus is the image you want to analyze as a 2D numpy array.
An example stimulus (White's illusion) that can be used with the default
parameters of the model can be loaded with

>>> import matplotlib.pyplot as plt
>>> stimulus = plt.imread('example_stimulus.png')

See docstrings for further details and model parameters.

Get in touch with `Torsten
<http://www.cognition.tu-berlin.de/menue/tubvision/people/torsten_betz/>`_
in case you have questions.
