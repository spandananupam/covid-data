#!/opt/anaconda3/bin/python
from sklearn.neighbors import BallTree, KNeighborsClassifier, kneighbors_graph, radius_neighbors_graph
from geopy.geocoders import Bing
from more_itertools import unique_everseen
import pandas as pd
import numpy as np
import pickle
import json
import apis
import os
import ast


def randomDistrict(state):

    # Takes care of faulty data and travel. Needs to be fixed later in time
    dist = {
        'KL': 'Thiruvananthapuram',
        'DL': 'New Delhi',
        'TG': 'Hyderabad',
        'RJ': 'Jaipur',
        'HR': 'Gurgaon',
        'UP': 'Allahabad',
        'LA': 'Lakshadweep',
        'TN': 'Chennai',
        'JK': 'Jammu',
        'KA': 'Bangalore',
        'MH': 'Mumbai',
        'PB': 'Amritsar',
        'AP': 'Visakhapatnam',
        'HP': 'Shimla',
        'UT': 'Dehradun',
        'OR': 'Khordha',
        'PY': 'Mahe',
        'WB': 'Kolkata',
        'CH': 'Chandigarh',
        'CT': 'Raipur',
        'GJ': 'Surat',
        'MP': 'Bhopal',
        'BR': 'Patna',
        'MN': 'Imphal West',
        'GA': 'North Goa',
        'MZ': 'Aizawl',
        'AN': 'Nicobars',
        'AS': 'Dhubri',
        'JH': 'Ranchi',
        'AR': 'Tirap',
        'NL': 'Mon',
        'TR': 'Dhalai',
        'DN': 'Daman',
        'ML': 'Ribhoi',
        'SK': 'North  District',
        'UN': 'Dadra AND Nagar Haveli',

    }
    return dist[state]


def prepareDistrictModel():

    # Prepare the census data district wise
    districtStats = pd.read_csv('data/census_final.csv')
    districtStats['Coordinates'] = districtStats['Coordinates'].apply(
        ast.literal_eval)
    distCoord = np.asarray(list(districtStats['Coordinates']))
    distCoordRad = np.deg2rad(distCoord)
    model = BallTree(distCoordRad, metric='haversine')
    return model, districtStats


def getNearestDistrictData(model, districtStats, point):

    point = np.asarray(point).reshape(1, -1)
    pointRad = np.deg2rad(point)
    index = np.squeeze(model.query(pointRad, return_distance=False))
    row = districtStats.iloc[int(index), :]
    return row['Population'], row['Literacy rate'], row['Coordinates'], index


def makeGraph(dataset, model, districtStats, R, sigma):

    RADIUS_OF_EARTH = 6378

    dataFile = json.load(open(dataset))
    dates = [date for date in dataFile]

    # Saving locations from dictionary
    placesList = []
    for date in dates:
        for state in list(dataFile[date]):
            if state == 'TT':
                pass
            try:
                for district in list(dataFile[date][state]['districts']):
                    if district == 'Unknown' or district == 'Other State':
                        district = randomDistrict(state)
                    place = district + ',' + state + ',' + 'India'
                    if not place in placesList:
                        placesList.append(place)

            except KeyError:
                place = state + ',' + 'India'
                if not place in placesList:
                    placesList.append(place)
    print('Updated places')

    # Geolocator, we save stuff to geoUP.p
    geolocator = Bing(
        api_key=apis.bing())

    uniquePlacesList = list(unique_everseen(placesList))
    geocodedDistrictList = list(districtStats['Coordinates'])
    geocodedUniqueNearestDistrictList = list(
        np.zeros_like(uniquePlacesList).astype(str))

    # Initialize if not present
    if not os.path.exists('data/geoUP.p'):
        geocodedUniquePlacesList = list(
            np.zeros_like(uniquePlacesList).astype(str))
        with open('data/geoUP.p', 'wb') as f:
            pickle.dump(geocodedUniquePlacesList, f)

    # Add new locations if any
    with open('data/geoUP.p', 'rb') as f:
        geocodedUniquePlacesList = pickle.load(f)
        for i in range(len(uniquePlacesList)):
            if geocodedUniquePlacesList[i] == '':
                geocodedUniquePlacesList[i] = ((
                    geolocator.geocode(uniquePlacesList[i]).latitude), (geolocator.geocode(uniquePlacesList[i]).longitude))
    print('Geo mapping stuff done')

    # Save to pickle
    with open('data/geoUP.p', 'wb') as f:
        pickle.dump(geocodedUniquePlacesList, f)

    for i in range(len(uniquePlacesList)):
        _, _, coordinate, _ = getNearestDistrictData(
            model, districtStats, geocodedUniquePlacesList[i])
        geocodedUniqueNearestDistrictList[i] = coordinate

    # Map stuff to different lists this got error
    numberOfDistricts = len(geocodedDistrictList)
    numberOfDates = len(dates)
    arrayFinal = np.zeros((numberOfDates, numberOfDistricts)).astype(str)
    print('Making final time resolved array')

    for dateIndex in range(numberOfDates):
        for districtIndex in range(numberOfDistricts):
            date = dates[dateIndex]
            district = list(districtStats['Coordinates'])[districtIndex]
            try:
                place = uniquePlacesList[geocodedUniqueNearestDistrictList.index(
                    district)]

            # If that district is not enlisted in corona affected places
            except ValueError:
                pass

            if place:
                dump = place.split(',')
                number = 0

                # Check to see if district or state only data
                if len(dump) == 2:
                    try:
                        number = dataFile[date][dump[0]]['total']['confirmed']
                    # If that state does not exist on that date
                    except KeyError:
                        pass
                else:
                    try:
                        number = dataFile[date][dump[1]
                                                ]['districts'][dump[0]]['total']['confirmed']
                    # If that district does not exist in this state on the date
                    except KeyError:
                        pass

                arrayFinal[dateIndex, districtIndex] = ','.join((str(number), str(list(districtStats['Literacy rate'])[
                    districtIndex]), str(list(districtStats['Population'])[districtIndex])))

            else:
                pass

    print('Array made')

    E = radius_neighbors_graph(
        model, R/RADIUS_OF_EARTH, mode='distance', metric='haversine').toarray()
    W = 1 - np.exp(-(E*E)/sigma)

    return arrayFinal, W


def load_data(DATASET, R, SIGMA, TEST_NUMBER):
    print('Running')
    dataset = 'data/'+DATASET
    os.chdir(os.path.split(os.path.dirname(os.path.realpath(__file__)))[0])
    model, districtStats = prepareDistrictModel()
    print('District ball-tree made')
    X, W = makeGraph(dataset, model, districtStats, R, SIGMA)
    X_train = X[:TEST_NUMBER, :]
    X_test = X[TEST_NUMBER:, :]
    print('Done executing')
    return X_train, X_test, W


if __name__ == '__main__':
    X_train, X_test, W = load_data(
        DATASET='data-all.json', R=300, SIGMA=1, TEST_NUMBER=100)
