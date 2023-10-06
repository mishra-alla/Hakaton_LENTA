import pandas as pd
import numpy as np
import pandas as pd
import pickle

def forecast(sales: dict, item_info: dict, store_info: dict):
    
    # функция предсказания продаж 1 товара 1 магазина на 14 дней
def predict_future(df, model, predict_model, period):
    
    # Загрузка модели
    try:
        with open('app/pickle_model/model_lgbm_v4.pcl', 'rb') as fid:
            model = pickle.load(fid)
    except:
        with open('/code/app/pickle_model/model_lgbm_v4.pcl', 'rb') as fid:
            model = pickle.load(fid)
    
    # выбираем определенный магазин и отвар
    sku_st = df[(df['st_id'] == st_id) & (df['pr_sku_id'] == pr_sku_id)]
    df = sku_st.reset_index()
    future = model.make_future_df(periods=period) # датафрейм, для которого будем делать прогноз
    predict_model = predict_model.predict(future)

    # добавим предсказания в датафрейм прогноза
    future['price_units_stand'] = predict_model.prediction

    # добавим переменную "holiday"
    #future['date'] = pd.to_datetime(future['date'], format='%d.%m.%Y')
    future.set_index('date', inplace=True)

    # добавим данные из календаря о праздниках
    future_df = future.merge(calendar[['holiday']], left_index=True, right_index=True, how='left')

    # вернем 'date' в колонку
    future_df.reset_index(inplace=True)

    pred_model = model.predict(df=future_df)

    pred_model[['prediction_5', 'prediction', 'prediction_95']] = (pred_model[['prediction_5', 'prediction', 'prediction_95']] \
                         .apply(lambda x: (x * pr_sales_in_units_sd) + pr_sales_in_units_mean))

    sales_submission = pred_model.rename(columns={'prediction': 'target'}).drop(['prediction_5', 'prediction_95'], axis=1).round()
    sales_submission = sales_submission.merge(df[['st_id', 'pr_sku_id']], left_index=True, right_index=True)
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
                df_1 = make_predict_future(df, st_id, pr_sku_id)
                result_df_future = result_df_future.append(df_1, ignore_index=True)
            except Exception as e:
                print(f"Ошибка при обработке st_id:{st_id}, pr_sku_id:{pr_sku_id}: {e}")




def forecast(sales: dict, item_info: dict, store_info: dict) -> list:
    """
    Функция для предсказания продажЖ
    :params sales = pr_sales_in_units: исторические данные по продажам
    :params item_info = pr_sku_id: характеристики товара
    :params store_info = st_id: характеристики магазина

    """
    sales = [el["sales_units"] for el in sales]
    mean_sale = sum(sales) / len(sales)
    return [mean_sale] * 5
