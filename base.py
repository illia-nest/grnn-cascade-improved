# Requires data.xlsx; Contains GRNN class; Contains iterative procedure (n_iter=50); Streams output to Experiments.xlsx;
import time
import math
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, median_absolute_error, r2_score, root_mean_squared_error
from sklearn.base import BaseEstimator, ClassifierMixin

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.10f' % x)


# Creating GRNN class
class GRNN(BaseEstimator, ClassifierMixin):
    def __init__(self, name = "GRNN", sigma = 0.1):
        self.name = name
        self.sigma = 2 * np.power(sigma, 2)
        
    def predict(self, instance_X, X_train_scaled, Y_train):
        gausian_distances = np.exp(-np.power(np.sqrt((np.square(X_train_scaled-instance_X).sum(axis=1))),2) / self.sigma)
        gausian_distances_sum = gausian_distances.sum()
        
        if gausian_distances_sum < math.pow(10, -7):
            gausian_distances_sum = math.pow(10, -7)
            
        return np.multiply(gausian_distances, Y_train).sum() / gausian_distances_sum


def f(params, X_train_scaled, Y_train, X_test_scaled, Y_test):
    s, = params  # Unpack the parameters
    grnn = GRNN(sigma=s)
    predictions = np.array([grnn.predict(i, X_train_scaled, Y_train) for i in X_test_scaled])
    return mean_squared_error(Y_test, predictions)


def step1():
    # Data prep for step 1
    data = pd.read_excel('data.xlsx')
    train = data.iloc[:-5, :]
    test = data.iloc[-5:, :]

    X_train = train.iloc[:, :4]
    X_test = test.iloc[:, :4]

    Y_train = train.iloc[:, 4:]
    Y_test = test.iloc[:, 4:]

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    results = pd.DataFrame(columns=['zone',
                                    'iteration',
                                    'time', 
                                    'sigma', 
                                    'y_true', 
                                    'y_pred' , 
                                    'R2', 
                                    'MSE', 
                                    'RMSE', 
                                    'MAE', 
                                    'MAPE', 
                                    'MedError', 
                                    'MaxError'])


    start_time = time.time()

    for i in range(Y_train.shape[1]):
        res = differential_evolution(f, seed=42, bounds=[(0.001, 10)], args=(X_train_scaled, Y_train.iloc[:, i], X_test_scaled, Y_test.iloc[:, i]))
        s = res["x"][0]

        grnn = GRNN(sigma=s)
        predictions = np.apply_along_axis(lambda x: grnn.predict(x, X_train_scaled, Y_train.iloc[:, i]), axis=1, arr=X_test_scaled)
        
        exp_time = time.time() - start_time


        MaxError = max_error(                       Y_test.iloc[:,i], predictions)
        MAE = mean_absolute_error(                  Y_test.iloc[:,i], predictions)
        MSE = mean_squared_error(                   Y_test.iloc[:,i], predictions)
        MedError = median_absolute_error(           Y_test.iloc[:,i], predictions)
        RMSE = root_mean_squared_error(             Y_test.iloc[:,i], predictions)
        MAPE = mean_absolute_percentage_error(      Y_test.iloc[:,i], predictions)
        R2 = r2_score(                              Y_test.iloc[:,i], predictions)

        results.loc[i] = [
            f'zone_{i+1}',
            0, # iteration
            exp_time,
            s,
            Y_test.iloc[:,i].to_list(),
            predictions,
            R2,
            MSE,
            RMSE,
            MAE,
            MAPE,
            MedError,
            MaxError,
        ]

    return results, X_train, X_test, Y_train, Y_test


def step2(results: pd.DataFrame,
        Y_train: pd.DataFrame,
        Y_test: pd.DataFrame,
        n_iter: int = 50,
        add_X: bool = False,
        X_train: pd.DataFrame = None,
        X_test: pd.DataFrame = None
        ):
    
    results = results.copy()
    
    Y_train_new = Y_train
    Y_train_new.columns = [x + ' predicted' for x in Y_train.columns]
    
    # Step 2 and 3
    iter_range = [x*6 for x in range(1, n_iter + 1)]
    for iter_number, iter_for_sum in enumerate(iter_range):
        Y_test_new = results[results['iteration'] == iter_number].y_pred.apply(pd.Series).T
        Y_test_new.columns = Y_train_new.columns
        Y_test_new.index = Y_test.index

        Y_test_new = pd.DataFrame(Y_test_new, columns=Y_train_new.columns, index=Y_test.index)
        
            
        # Creating two datasets (unscaled)
        Y_train_new_1 = Y_train_new.iloc[:, :3]
        Y_train_new_2 = Y_train_new.iloc[:, -3:]

        Y_test_new_1 = Y_test_new.iloc[:, :3]
        Y_test_new_2 = Y_test_new.iloc[:, -3:]
        
        # Adding X to train and test and scaling
        if add_X:
            united_train_1 = pd.concat([X_train, Y_train_new_1], axis=1)
            united_train_2 = pd.concat([X_train, Y_train_new_2], axis=1)

            united_test_1 = pd.concat([X_test, Y_test_new_1], axis=1)
            united_test_2 = pd.concat([X_test, Y_test_new_2], axis=1)

            scaler_1 = MinMaxScaler()
            scaler_1.fit(united_train_1)
            Y_train_new_1_scaled = scaler_1.transform(united_train_1)
            Y_test_new_1_scaled = scaler_1.transform(united_test_1)

            scaler_2 = MinMaxScaler()
            scaler_2.fit(united_train_2)
            Y_train_new_2_scaled = scaler_2.transform(united_train_2)
            Y_test_new_2_scaled = scaler_2.transform(united_test_2)
        else:
            # Scaling sets
            scaler_1 = MinMaxScaler()
            scaler_1.fit(Y_train_new_1)
            Y_train_new_1_scaled = scaler_1.transform(Y_train_new_1)
            Y_test_new_1_scaled = scaler_1.transform(Y_test_new_1)

            scaler_2 = MinMaxScaler()
            scaler_2.fit(Y_train_new_2)
            Y_train_new_2_scaled = scaler_2.transform(Y_train_new_2)
            Y_test_new_2_scaled = scaler_2.transform(Y_test_new_2)

        # Polyfeatures go here

        Y_test_1 = Y_test.iloc[:, :3]
        Y_test_2 = Y_test.iloc[:, -3:]

        new_datasets = [(Y_train_new_1_scaled, Y_test_new_1_scaled, Y_train_new_1.to_numpy(), Y_test_1),
                        (Y_train_new_2_scaled, Y_test_new_2_scaled, Y_train_new_2.to_numpy(), Y_test_2)]
        cur_zone = 0
        for dataset in new_datasets:
            X_train_scaled_, X_test_scaled_, Y_train_, Y_test_ = dataset

            for i in range(Y_test_.shape[1]):
                start_time = time.time()

                res = differential_evolution(f, seed=42, bounds=[(0.001, 10)], args=(X_train_scaled_, Y_train_[:, i], X_test_scaled_, Y_test_.iloc[:, i]))
                s = res["x"][0]

                grnn = GRNN(sigma=s)
                predictions_2 = np.apply_along_axis(lambda x: grnn.predict(x, X_train_scaled_, Y_train_[:, i]), axis=1, arr=X_test_scaled_)

                exp_time = time.time() - start_time


                MaxError = max_error(                   Y_test_.iloc[:,i], predictions_2)
                MAE = mean_absolute_error(              Y_test_.iloc[:,i], predictions_2)
                MSE = mean_squared_error(               Y_test_.iloc[:,i], predictions_2)
                MedError = median_absolute_error(       Y_test_.iloc[:,i], predictions_2)
                RMSE = root_mean_squared_error(         Y_test_.iloc[:,i], predictions_2)
                MAPE = mean_absolute_percentage_error(  Y_test_.iloc[:,i], predictions_2)
                R2 = r2_score(                          Y_test_.iloc[:,i], predictions_2)


                results.loc[iter_for_sum + cur_zone] = [
                    f'zone_{cur_zone+1}',
                    iter_number + 1,
                    exp_time,
                    s,
                    Y_test_.iloc[:,i].to_list(),
                    predictions_2,
                    R2,
                    MSE,
                    RMSE,
                    MAE,
                    MAPE,
                    MedError,
                    MaxError,
                ]
                
                cur_zone += 1
                
    results = results.sort_values(by=['zone', 'iteration'])
    
    return results


def record_results(results):
    with pd.ExcelWriter('Experiments.xlsx', engine='openpyxl', mode='a') as writer:
        results.to_excel(writer, index=False, sheet_name=f"Exp. No. {pd.Timestamp.today().strftime('%Y-%m-%d-%H-%M-%S')}")


def main():
    results, X_train, X_test, Y_train, Y_test = step1()
    results_2 = step2(results, Y_train, Y_test, 50, True, X_train, X_test)
    record_results(results_2)


if __name__ == '__main__':
    main()

