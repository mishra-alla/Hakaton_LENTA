from pyexpat import model
import pandas as pd
import numpy as np
import pandas as pd
import pickle

MODEL=model
PERIOD=14


def forecast(sales: dict, item_info: dict, store_info: dict):
    
    """
    Функция для предсказания продажЖ
    :params sales = pr_sales_in_units: исторические данные по продажам
    :params item_info = pr_sku_id: характеристики товара
    :params store_info = st_id: характеристики магазина

    """
    with open('app/pickle_model/model.pcl', 'rb') as fid:
            model = pickle.load(fid)
    with open('app/pickle_model/predict_model', 'rb') as fid:
            predict_model = pickle.load(fid)
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
    predicted_sale = model.predict(features)

    return predicted_sale[0]
   
    #Выбираем один из наиболее популярных магазинов и продуктов
    st_id = 'c81e728d9d4c2f636f067f89cc14862c'
    pr_sku_id = '6b1344097385a42484abd4746371e416'
    period=14
    # функция предсказания продаж 1 товара 1 магазина на 14 дней
    #def predict_future(data, model, predict_model, period):

    # выбираем определенный магазин и товар
    sku_st = data[(data['st_id'] == st_id) & (data['pr_sku_id'] == pr_sku_id)]
    data = sku_st.reset_index()
    future = model.make_future_df(periods=period) # датафрейм, для которого будем делать прогноз
    predict_model = predict_model.predict(future)

    # добавим предсказания в датафрейм прогноза
    future['price_units_stand'] = predict_model.prediction

    # добавим переменную "holiday"
    #future['date'] = pd.to_datetime(future['date'], format='%d.%m.%Y')
    future.set_index('date', inplace=True)

    # добавим данные из календаря о праздниках
    data_future = future.merge(calendar[['holiday']], left_index=True, right_index=True, how='left')

    # вернем 'date' в колонку
    data_future.reset_index(inplace=True)

    pred_model = model.predict(data=data_future)

    pred_model[['prediction']] = (pred_model[['prediction']] \
                         .apply(lambda x: (x * pr_sales_in_units_sd) + pr_sales_in_units_mean))

    sales_submission = pred_model.rename(columns={'prediction': 'target'}).round()
    sales_submission = sales_submission.merge(data[['st_id', 'pr_sku_id']], left_index=True, right_index=True)
    sales_submission = sales_submission.assign(st_id=st_id, pr_sku_id=pr_sku_id)
    sales_submission = sales_submission[['st_id', 'pr_sku_id', 'date', 'target']]
    return sales_submission
    
    """
    Функция для предсказания продаж
    :params sales = pr_sales_in_units: исторические данные по продажам
    :params item_info = pr_sku_id: характеристики товара
    :params store_info = st_id: характеристики магазина

    """
    # Создаем пустой датафрейм для хранения всех результатов
    result_df_future = pd.DataFrame()

    st_ids = ['c81e728d9d4c2f636f067f89cc14862c', '42a0e188f5033bc65bf8d78622277c4e']
    pr_sku_ids = ['4ce0eb956648ab3ff6bb0afa3158cc42']

# Считаем прогнозы для нужных магазинов и товаров
    for st_id in st_ids:
        for pr_sku_id in pr_sku_ids: # store_dict[st_id]:
            try:
                df_1 = make_predict_future(data, st_id, pr_sku_id)
                result_df_future = result_df_future.append(df_1, ignore_index=True)
            except Exception as e:
                print(f"Ошибка при обработке st_id:{st_id}, pr_sku_id:{pr_sku_id}: {e}")




def forecast(sales: dict, item_info: dict, store_info: dict) -> list:

    sales = [el["sales_units"] for el in sales]
    mean_sale = sum(sales) / len(sales)
    return [mean_sale] * 5
