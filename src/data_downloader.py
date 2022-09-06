import requests
import os
import zipfile

list_transactions_url = 'https://www.data.gouv.fr/en/datasets/r/90a98de0-f562-4328-aa16-fe0dd1dca60f'
cadastral_parcels_url = 'https://cadastre.data.gouv.fr/data/etalab-cadastre/2021-04-01/shp/departements/75/cadastre-75-parcelles-shp.zip'

transactions_file = 'valeursfoncieres-2020.txt'
parcels_file = 'cadastre-75-parcelles-shp.zip'

if os.path.exists('../data'):
    exit()

os.makedirs('../data')

print('Downloading', transactions_file, '...')
response = requests.get(list_transactions_url)
size = open('../data/valeursfoncieres-2020.txt', 'wb').write(response.content)
print(transactions_file, 'downloaded.\t\t{} bytes'.format(size))

print('Downloading', parcels_file)
response = requests.get(cadastral_parcels_url)
size = open('../data/cadastre-75-parcelles-shp.zip', 'wb').write(response.content)
print(parcels_file, 'downloaded.\t{} bytes'.format(size))

if not os.path.exists('../data/cadastre-75-parcelles-shp'):
    os.makedirs('../data/cadastre-75-parcelles-shp')

with zipfile.ZipFile('../data/cadastre-75-parcelles-shp.zip', 'r') as zip_ref:
    zip_ref.extractall('../data/cadastre-75-parcelles-shp')
    
print(parcels_file, 'extracted.')
print('All done.')