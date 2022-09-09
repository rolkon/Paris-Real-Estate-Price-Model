import os
import pandas as pd
import numpy as np
import shapefile
from tqdm import tqdm

def pre_filter_data(transactions_df, locations_sf):
    # Remove all datapoints where there is no financial information
    transactions_df = transactions_df[~np.isnan(transactions_df['Valeur fonciere'])]

    # Create sets of entries in location_sf file to pre-filter the transactions_df dataset
    sf_departm_set = set()
    sf_commune_set = set()
    sf_section_set = set()
    sf_no_plan_set = set()

    for record in locations_sf.records():
        sf_departm_set.add(str(int(record[1]) // 1000))
        sf_commune_set.add(int(record[1]) %  1000)
        sf_section_set.add(record[3])
        sf_no_plan_set.add(int(record[4]))

    # pre-filter transactions_df
    pre_filtered_df = transactions_df[(transactions_df['Code departement'].isin(sf_departm_set)) & \
                                      (transactions_df['Code commune'].isin(sf_commune_set)) & \
                                      (transactions_df['Section'].isin(sf_section_set)) & \
                                      (transactions_df['No plan'].isin(sf_no_plan_set)) & \
                                      (transactions_df['Code type local'].isin({1.0, 2.0})) \
                                     ]

    return pre_filtered_df

def filter_and_join_data(pre_filtered_df, locations_sf):

    # define DataFrame for filtered and joined data
    filtered_df = pd.DataFrame(columns=['LocationID', 'Cost', 'Area', 'CostPerArea', 'PosX', 'PosY'])

    # Create set of entries in the pre_filtered_df to skip iterations later
    df_section_set = set()
    df_no_plan_set = set()

    for departm, commune, section, no_plan in zip(pre_filtered_df['Code departement'],
                                                  pre_filtered_df['Code commune'],
                                                  pre_filtered_df['Section'],
                                                  pre_filtered_df['No plan']):
        df_section_set.add(section)
        df_no_plan_set.add(no_plan)

    for rec in tqdm(locations_sf.records()):
        if int(rec['numero']) in df_no_plan_set: # plan number entries have greatest difference between the two datasets, best filter
            if rec['section'] in df_section_set: # section has second greatest difference
                # department code and commune code are consolidated in single dataframe, separated by division and modulo
                filtered_transaction = pre_filtered_df[(pre_filtered_df['Code departement'] == str(int(rec[1]) // 1000)) & \
                                                       (pre_filtered_df['Code commune'] == int(rec[1]) % 1000) & \
                                                       (pre_filtered_df['Section'] == rec[3]) & \
                                                       (pre_filtered_df['No plan'] == int(rec[4]))]

                if len(filtered_transaction) > 0:

                    location_id = rec.oid
                    # if multiple transactions are found under the same code, I used the most expensive one
                    total_cost_all_transactions = np.max(filtered_transaction['Valeur fonciere'].to_numpy())
                    area = rec['contenance']
                    cost_per_area = total_cost_all_transactions / area

                    #calculate average (center) of shape to get a geo location
                    points = np.array([[point[0], point[1]] for point in locations_sf.shape(location_id).points])

                    avg_x = np.sum(points[:,0]) / points.shape[0]
                    avg_y = np.sum(points[:,1]) / points.shape[0]

                    data_entry = pd.DataFrame({'LocationID':[location_id],
                                               'Cost':[total_cost_all_transactions],
                                               'Area':[area],
                                               'CostPerArea':[cost_per_area],
                                               'PosX':[avg_x],
                                               'PosY':[avg_y]})

                    filtered_df = pd.concat([filtered_df, data_entry])

    return filtered_df

def main():
    #download datasets
    if not os.path.exists('../data/valeursfoncieres-2020.csv') or \
        not os.path.exists('../data/cadastre-75-parcelles-shp.zip'):
        print("Files not downloaded. Execute the data_downloader script.")
        exit()

    print("Loading datasets...")
    transactions_df = pd.read_csv('../data/valeursfoncieres-2020.csv', sep=',', low_memory=False)
    locations_sf = shapefile.Reader('../data/cadastre-75-parcelles-shp.zip')

    print("Pre-filtering data...")
    pre_filtered_df = pre_filter_data(transactions_df, locations_sf)

    print("Filtering and joining data...")
    filtered_df = filter_and_join_data(pre_filtered_df, locations_sf)
    
    #save data
    filtered_df.to_csv('../data/cost_location_data.csv')
    
if __name__ == '__main__':
    main()