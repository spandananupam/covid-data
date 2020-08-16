from data_utils import load_data

nodeFeatures_train, nodeFeatures_test, weightedAdjacency = load_data(
    DATASET='data-all.json', R=300, SIGMA=1, TEST_NUMBER=150)
