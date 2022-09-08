# Paris Real Estate Price Model

This repository implements a model to estimate the real estate prices of old apartments in Paris. The model takes a geographical coordination and returns an estimate of the price per square meter.

### External sources and data:

* [List of real estate transactions carried out on the entire French territory since 2014](https://www.data.gouv.fr/en/datasets/r/90a98de0-f562-4328-aa16-fe0dd1dca60f)
  * [Documentation of the data (french)](https://www.data.gouv.fr/en/datasets/r/d573456c-76eb-4276-b91c-e6b9c89d6656)
  * [Translation of documentation (english with Google Translate)](https://drive.google.com/file/d/12miiSujVTzmdvp0ErIHHlc7DVKbJ6AEL/view?usp=sharing)
* [List of cadastral parcels in Paris](https://cadastre.data.gouv.fr/data/etalab-cadastre/2021-04-01/shp/departements/75/cadastre-75-parcelles-shp.zip)
  * [Documentation of the data (french)](https://cadastre.data.gouv.fr/)

### How to use
All source files are to be found in `src/`

1. Download data with `data_downloader.py`. This creates the `data/` folder
2. Execute the filtering and training with `price_estimator.py`

A more thorough documentation of the code can be found in the `documentation.ipynb` jupyter notebook.

### Dependencies
* [pyshp](https://pypi.org/project/pyshp/#overview)
* [sklearn](https://scikit-learn.org/stable/)
