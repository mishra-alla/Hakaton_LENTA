from pyexpat import model
import pandas as pd
import numpy as np
import pandas as pd
import pickle

MODEL=model
PERIOD=14

def prepare_data(data, calendar):

    #сделаем дату индексом и переведем в объект datetime
    data.set_index('date', inplace=True)
    data.index = pd.to_datetime(data.index)
    calendar.set_index('date', inplace=True)
    calendar.index = pd.to_datetime(calendar.index)
    # добавим календарь праздников к данным
    data = pd.merge(calendar, merged_df, left_index=True, right_index=True)
    # изменим порядок столбцов на удобный
    new_order = ['is_active', 'store', 'sku', 'sales_type', 'sales_units', 'sales_units_promo',
             'sales_rub', 'sales_rub_promo','group', 'category', 'subcategory', 'uom', 'division',
             'city', 'type_format', 'loc', 'size', 'year', 'month', 'day', 'weekday', 'holiday']
    data = data[new_order]

    #Удалим магазины с малой активностью
    # список магазинов
    list_store = data['store'].unique().tolist()
    # список для хранения данных о магазинах и их количестве дней работы
    activ_store_data = []
    store_data = []

    for st in list_store:
        min_date = data.loc[data['store'] == st].index.min().date()
        max_date = data.loc[data['store'] == st].index.max().date()
        delta = max_date - min_date
        days = delta.days
        if days > 180:
            activ_store_data.append({'stores': st, 'days_worked': days})
        else:
            store_data.append({'stores': st, 'days_worked': days})
    # получим список идентификаторов магазинов из списка store_data
    store_ids = [entry['stores'] for entry in store_data]
    mask = data['store'].isin(store_ids)# cоздаем маску для строк на удаления
    activ_store = data[~mask]
    activ_store = activ_store.drop('is_active', axis=1)# удалим столбцец 'is_active'

    # Удалим отрицательные значения
    mask_negativ = ((activ_store['sales_units'] < 0) | (activ_store['sales_units_promo'] < 0) | (activ_store['sales_rub'] < 0)\
                | (activ_store['sales_rub_promo'] < 0))
    # удаляем строки, соответствующие маске, из activ_store
    activ_store = activ_store[~mask_negativ]

    # Обработка нулевых значений (заполним нули в столбцах sales_units и sales_rub)
    activ_store['sales_units'] = activ_store['sales_units'].replace(0, np.nan) # меням 0 на пропуски
    activ_store['sales_rub'] = activ_store['sales_rub'].replace(0, np.nan)
    activ_store['sales_units'] = activ_store['sales_units'].fillna(method='ffill') # заменим пропуски на значение
    activ_store['sales_rub'] = activ_store['sales_rub'].fillna(method='ffill')


    # первую стороку заполним средним значением
    activ_store['sales_units'] = activ_store['sales_units'].fillna(activ_store['sales_units'].mean())
    activ_store['sales_rub'] = activ_store['sales_rub'].fillna(activ_store['sales_rub'].mean())
    # приведем к целым значениям
    activ_store['sales_units'] = activ_store['sales_units'].astype(int)
    activ_store['sales_units_promo'] = activ_store['sales_units_promo'].astype(int)
    activ_store.reset_index()


    # Добавим долю продаж промо 'promo_part' в таблицу
    # сделаем группировку и агрегацию по товарам, сорртировака по сумме
    ales_type_sorted = activ_store.groupby('sku')[['sales_units', 'sales_units_promo']]\
                                  .agg(['sum']).sort_values(by=('sales_units', 'sum'), ascending=False)
    # добавим столбец с долей товаров с промо в процентах
    activ_store['promo_part'] = activ_store['sku'].map(
    activ_store.groupby('sku')[['sales_units', 'sales_units_promo']]
    .agg({'sales_units': 'sum', 'sales_units_promo': 'sum'})
    .eval('sales_units_promo / sales_units').round(3))

    # удалим те строки с промо, у которых есть продажи без промо
    activ_store = activ_store[~(activ_store['sales_type'] == 1) & (activ_store['sku']\
             .isin(activ_store.loc[activ_store['sales_type'] == 0, 'sku']))]
    # удалим столбцы 'sales_type', 'sales_units_promo', 'sales_rub_promo'
    activ_store = activ_store.drop(['sales_type', 'sales_units_promo', 'sales_rub_promo'], axis=1)

    # Добавим признак цена за единицу товара 'price_units'
    activ_store['price_units'] = (activ_store['sales_rub'] / activ_store['sales_units']).round(2)

    # Проведем НОРМАЛИЗАЦИЮ числовых признаков: sales_units, price_units и sales_rub
    # нормализация продажи в штуках
    sales_units_mean = activ_store.sales_units.mean()
    sales_units_sd = activ_store.sales_units.std()
    #Сформируем целевой признак 'sales_units_stand'
    activ_store['sales_units_stand'] = (activ_store.sales_units - sales_units_mean) / sales_units_sd

    # нормализация продажи в рублях
    sales_rub_mean = activ_store.sales_rub.mean()
    sales_rub_sd = activ_store.sales_rub.std()
    #Сформируем признак 'sales_rub_stand'
    activ_store['sales_rub_stand'] = (activ_store.sales_rub - sales_rub_mean) / sales_rub_sd

    # нормализация доли промо продажи в рублях за штуку
    price_units_mean = activ_store.price_units.mean()
    price_units_sd = activ_store.price_units.std()
    #Сформируем признак 'price_units_stand'
    activ_store['price_units_stand'] = (activ_store.price_units - price_units_mean) / price_units_sd

    #Разделим признаки на числовые и категориальные, создадим списки:
    numeric = activ_store[activ_store.select_dtypes(include='number').columns]
    categorical = activ_store[activ_store.select_dtypes(include='object').columns]
    numeric_columns = numeric.columns.tolist()
    categorical_columns = categorical.columns.tolist()

    # Добавим признаки: доли продаж товаров по категориям и по магазинам (в шт и в руб)
    def sales_share(data, name_col, name, columns_to_process):
        for column in columns_to_process:
            unique_counts = data.groupby(column)['sku'].nunique()# количество уникальных товаров в каждой категории/типа магазина
            sales_sum = data.groupby(column)[name_col].sum() # сумма продаж для каждой категории/типа магазина
            sku_sales = unique_counts / sales_sum # доля продаж уникальных товаров в каждой категории/типа магазина
            # доля продаж уникальных товаров в таблицу
            data[f'sales_share_{column}_{name}'] = data[column].map(sku_sales)
        return data
    columns_to_process = ['group', 'category', 'subcategory']                 # перебор по категориям
    columns_to_process_sku = ['store', 'sku','year', 'weekday', 'month',
                              'division', 'city', 'type_format', 'loc','size'] # перебор по магазинам
    #Добавим доли продаж товаров по категориям
    activ_store = sales_share(activ_store, 'sales_units', 'unit', columns_to_process) #в шт
    activ_store = sales_share(activ_store, 'sales_rub', 'rub', columns_to_process)   #в rub
    #Добавим доли продаж товаров по магазинам
    activ_store = sales_share(activ_store, 'sales_units', 'sku_unit', columns_to_process_sku) #в шт
    activ_store = sales_share(activ_store, 'sales_rub', 'sku_rub', columns_to_process_sku) #в rub

    #КЛАСТЕРИЗАЦИЯ
    clast_activ_store=activ_store.copy()# создадим копию данных
    # переведем столбцы в категориальные признаки
    clast_activ_store['type_format'] = clast_activ_store['type_format'].astype('object')
    clast_activ_store['loc'] = clast_activ_store['loc'].astype('object')
    clast_activ_store['size'] = clast_activ_store['size'].astype('object')
    # список числовых столбцов
    numeric_columns = clast_activ_store.select_dtypes(include=['number']).columns
    # переберем столбцы и преобразуем их в 'category', если они не являются числовыми
    for column in clast_activ_store.columns:
        if column not in numeric_columns:
            clast_activ_store[column] = clast_activ_store[column].astype('category')


    #Новые признаки
    def clast_futures(data, name_col, name, columns_to_process):
        for column in columns_to_process:
            data[f'avg_sales_by_{column}_{name}'] = data.groupby(column)[name_col].transform('mean')
        # средние продажи по выходным и праздникам в рублях/шт
        data[f'avg_holiday_sales_{name}'] = data[name_col] * data['holiday']
        return data
    columns_to_process_cl = ['store', 'sku','year', 'weekday', 'month', 'category', 'subcategory',
                              'division', 'city', 'type_format', 'loc','size']
    # средние продажи
    clast_activ_store = clast_futures(clast_activ_store, 'sales_rub', 'rub', columns_to_process_cl)   #в rub
    clast_activ_store = clast_futures(clast_activ_store, 'sales_units', 'units', columns_to_process_cl) #в шт

    # группируем данные по 'sku' и 'store' и считаем сумму продаж
    sku_sales_sum = clast_activ_store.groupby(['sku', 'store'])['sales_units'].sum().reset_index()
    n_clusters = 11  # количество кластеров
    X = sku_sales_sum[['sales_units']] # таблица с количеством продаж 'sales_units'
    kmeans = KMeans(n_clusters=n_clusters)# KMeans с количеством кластеров

    # кластеризация
    sku_sales_sum['cluster'] = kmeans.fit_predict(X)
    # признаки для кластеризации
    cluster_features = ['sales_units', 'avg_sales_by_store_units', 'avg_sales_by_sku_units',
                      'avg_sales_by_weekday_units', 'avg_sales_by_month_units', 'avg_holiday_sales_units']
    # масштабирование признаков
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clast_activ_store[cluster_features])
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clast_activ_store['cluster'] = kmeans.fit_predict(X_scaled)

    # соединим с исходной таблицей activ_store
    clast_activ_store = clast_activ_store.reset_index().merge(sku_sales_sum[['sku', 'store', 'cluster']],\
                                                          on=['sku', 'store', 'cluster'], how='left').set_index('date')
    return clast_activ_store

def forecast(sales: dict, item_info: dict, store_info: dict):
    
    """
    Функция для предсказания продаж:
    params sales:  исторические данные по продажам (pr_sales_in_units - целевой признак)
    params item_info: характеристики товара (pr_sku_id - конкретный товар)
    params store_info: характеристики магазина (st_id - конкретный магазин)

    """
    with open('app/pickle_model/model_cbr.pcl', 'rb') as fid:
            model_cbr = pickle.load(fid)
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