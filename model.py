from pyexpat import model
import pandas as pd
import numpy as np
import pandas as pd
import pickle

MODEL=model
PERIOD=14

def forecast(sales: dict, item_info: dict, store_info: dict):
    
    """
    Функция для предсказания продаж:
    params sales:  исторические данные по продажам (pr_sales_in_units - целевой признак)
    params item_info: характеристики товара (pr_sku_id - конкретный товар)
    params store_info: характеристики магазина (st_id - конкретный магазин)

    """
    with open('app/pickle_model/model_cbr.pcl', 'rb') as fid:
            model = pickle.load(fid)
    #загружаем датасет с календарем
    with open('app/pickle_model/holidays_covid_calendar.csv', 'rb') as fid:
            calendar = pickle.load(fid)
    
    # Создаем единый датасет data
    data = {}
    # Добавляем данные из словаря sales
    for key in sales:
        data[key] = item_info[key]
    # Добавляем данные из словаря item_info
    for key in item_info:
        data[key] = item_info[key]
     # Добавляем данные из словаря store_info
    for key in store_info:
        data[key] = store_info[key]


    def predict_sale(sales, store_info, item_info):
        # Создание массива с параметрами квартиры
        features = [[sales, store_info, item_info]]
        # Выполнение предсказания с использованием обученной модели gs
        try:
            predicted_sale = model.predict(features)
        except:
            print('по данному товару прогноз невозможен, проверьте данные по товару или магазину')
    return predicted_sale[0]

    #формирование списка предсказанной цены

    predicted_sales = data.apply(lambda row: predict_sale(row['pr_sales_in_units'], row['st_id'],row['pr_sku_id']), axis=1).tolist()