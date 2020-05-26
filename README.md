# py_gait: Gait pattern analysis python library

library for anlaysis of the walk pattern from relavent signals (3D accelerometer, Mobile phone and etc.)

## Installation

Download or clone this repository

```{bash}
> git clone git@github.com:sungcheolkim78/py_auc.git
```

Install libary locally

```{bash}
> pip3 install -e .
```

## Usage

```{python}
import py_gait

g = py_gait.GAIT('CO001')

g.fields
```

