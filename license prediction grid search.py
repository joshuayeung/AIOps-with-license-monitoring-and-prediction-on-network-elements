import pandas as pd
import sqlalchemy
import datetime
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
    # prepare training dataset
    train_size = int(len(X) * 0.66)
    train, test = X[0:train_size], X[train_size:]
    history = [x for x in train]

    # make predictions
    predictions = list()
    for t in range(len(test)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit(disp=0)
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])

    # calculate out of sample error
    error = mean_squared_error(test, predictions)
    return error

# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values, ne, resource):
    dataset = dataset.astype('float32')
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                if parameter[(parameter.object_name==ne) & (parameter.resource_name==resource) & (parameter.p==p) & (parameter.d==d) & (parameter.q==q)].values.size > 0:
                    print(order, ' already grid searched')
                    continue
                try:
                    mse = evaluate_arima_model(dataset, order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.3f' % (order,mse))
                    data = [[ne, resource, p, d, q, mse, np.nan]]
                    para = pd.DataFrame(data, columns = ['object_name','resource_name','p','d','q','mse','error'])
                    para.to_sql(name="license_prediction_parameter",if_exists='append',con = yourdb, index=False)
                except Exception as inst:
                    print(order, 'error: ', inst)
                    str(inst).replace("'", "")
                    try:
                        data = [[ne, resource, p, d, q, np.nan, str(inst)]]
                        para = pd.DataFrame(data, columns = ['object_name','resource_name','p','d','q','mse','error'])
                        para.to_sql(name="license_prediction_parameter",if_exists='append',con = yourdb, index=False)
                    except:
                        data = [[ne, resource, p, d, q, np.nan, 'some error']]
                        para = pd.DataFrame(data, columns = ['object_name','resource_name','p','d','q','mse','error'])
                        para.to_sql(name="license_prediction_parameter",if_exists='append',con = yourdb, index=False)
                    continue
    print('Best ARIMA%s MSE=%.3f' % (best_cfg, best_score))


yourdb = sqlalchemy.create_engine('mssql+pyodbc://user:password@DATABASE_IP\\SQLINSTANCE/database?driver=SQL+Server+Native+Client+11.0')

sql = "SELECT * FROM daily_license_summary"

df = pd.read_sql(sql, yourdb)

objects = df['object_name'].unique()

sql_para = "SELECT * FROM license_prediction_parameter"

parameter = pd.read_sql(sql_para, yourdb)

# evaluate parameters
p_values = [0, 1, 2, 4, 6, 8, 10]

d_values = range(0, 3)
q_values = range(0, 3)
warnings.filterwarnings("ignore")

for ne in objects:
    for resource in df[df.object_name==ne].resource_name.unique():
        print("Now predicting: ", ne, resource)
        
        forecast = pd.DataFrame({}, columns=df.columns.tolist())

        # Since license capacity may change, we need to get the latest license capacity to predict.
        latest_total_resource = df[(df.result_time==df[(df.object_name==ne) & (df.resource_name==resource)].result_time.max()) &
                            (df.object_name==ne) & (df.resource_name==resource)].total_resource.values[0]
        total_resource = df[(df.object_name==ne) & (df.resource_name==resource)].total_resource.values
        act_num = df[(df.object_name==ne) & (df.resource_name==resource)].used_resource.values
        history = act_num
        if latest_total_resource > 0: # Some license items maybe listed in output but the value is 0 (means that we didn't buy it)
            if sum(act_num)/sum(total_resource) > 0.01: # It is time consuming if we predict low usage license items.
                evaluate_models(act_num, p_values, d_values, q_values, ne, resource)
            else:
                print('Usage too little, skipped')
                data = [[ne, resource, np.nan, np.nan, np.nan, np.nan, 'Usage too little, skipped']]
                para = pd.DataFrame(data, columns = ['object_name','resource_name','p','d','q','mse','error'])
                para.to_sql(name="license_prediction_parameter",if_exists='append',con = yourdb, index=False)
        else:
            print('No license resource, skipped')
            data = [[ne, resource, np.nan, np.nan, np.nan, np.nan, 'No license resource, skipped']]
            para = pd.DataFrame(data, columns = ['object_name','resource_name','p','d','q','mse','error'])
            para.to_sql(name="license_prediction_parameter",if_exists='append',con = yourdb, index=False)
