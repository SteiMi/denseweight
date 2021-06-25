# DenseWeight

This package implements the method for imbalanced regression *DenseWeight*. The corresponding paper "Density-based weighting for imbalanced regression" is accepted for publication and will soon be available.

The goal of DenseWeight is to allow training machine learning models for regression tasks that emphasize performance for data points with rare target values in comparison to data points with more common target values. This can be useful when rare samples are of particular interest e.g. when estimating precipitation and you are interested in estimating rare, extreme precipitation events as well as possible. The parameter alpha controls the intensity of the density-based weighting scheme (alpha = 0.0 -> uniform weighting; larger alpha -> more emphasis on rare samples).

DenseWeight judges the rarity of a target value based on its density, which is obtained through Kernel Density Estimation (KDE). This package uses the fast convolutional-based KDE implementation FFTKDE from [KDEpy](https://github.com/tommyod/KDEpy) to allow the application of DenseWeight for large datasets.

## Installation

DenseWeight is available at [PyPI](https://pypi.org/project/denseweight/) and can be installed via pip:

```
pip install denseweight
```

## Usage

```
import numpy as np
from denseweight import DenseWeight

# Create toy target variable with 1000 samples
y = np.random.normal(size=1000)

# Define DenseWeight
dw = DenseWeight(alpha=1.0)

# Fit DenseWeight and get the weights for the 1000 samples
weights = dw.fit(y)

# Calculate the weight for an arbitrary target value
weights = dw([0.1206])
```

These weights can be used as sample weights for machine learning algorithms which support them. They can also be easily integrated into loss function for models like Neural Networks to create a cost-sensitive learning solution to data imbalance in regression tasks which we call *DenseLoss* (more details on this in the paper).