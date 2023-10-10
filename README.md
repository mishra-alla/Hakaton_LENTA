# Прогноз спроса на товары собственного производства крупной сети продуктовых магазинов Лента - Hakaton_LENTA.
Решение задачи по предсказательной модели и его интерфейса по прогнозированию спроса на товары заказчика собственного производства Лента (скропортящиеся товары).

### **Заказчик**: ООО “Лента”

##### **Бизнес цель:** Разработать прогностический инструмент в помощь планирования приготовления товаров собственного производства (кухня, пекарня и пр.).

##### **Побочная бизнес цель:** При получении точных прогнозов ожидается снижение объема упущенных продаж, уменьшение числа списаний по сроку годности.

## **Постановка задачи**:
**Задача:** Необходимо создать алгоритм прогноза спроса на 14 дней для товаров собственного производства. Гранулярность ТК-SKU-День.

**Финальный результат:** отчёт/таблица, в которой строим прогноз спроса по товарам, в каком количестве необходимо их приготовить в каждом ТК:
- для повышения доступности товаров на полке,
- для увеличения продаж, без увеличения списаний.

**Источник данных:** Хакатон Лента.

**Целевой признак:** **`sales_units`** – общее число проданных товаров (шт.)

**Метрика качества:** **MAPE**, посчитанный на уровне товар, магазин, день.
> Если есть пропущенные значения и по каким-то товарам не предоставлен прогноз, прогноз считается равным нулю.
```
def wape(y_true: np.array, y_pred: np.array):
return np.sum(np.abs(y_true-y_pred))/np.sum(np.abs(y_true))
```
**Стек:** `Временные ряды, pandas, numpy,seaborn, matplotlib, phik, CatBoost, ORBIT, ARIMAX, pickle, sklearn (pipeline, preprocessing, cluster)`

## **Описание проекта:**

### **Данные**
- **sales_df_train.csv** –данные по продажам за скользящий год для обучения.
- **pr_df.csv** – данные по товарной иерархии.
- **pr_st.csv** – данные по магазинам.
- **sales_submission.csv** – пример файла с результатом работы модели прогноза спроса.

### **Описание данных:**
1. **sales_df_train.csv** –данные по продажам за скользящий год для обучения.

Столбцы:
- `st_id` перевод в `store` – захэшированное id магазина;
- `pr_sku_id` перевод в `sku` – захэшированное id товара;
- `date` – дата;
- `pr_sales_type_id` перевод в `sales_type` – флаг наличия промо;
- `pr_sales_in_units` перевод в `sales_units` – общее число проданных товаров (шт.);
- `pr_promo_sales_in_units` перевод в `sales_units_promo` – число проданных товаров с признаком промо;
- `pr_sales_in_rub` перевод в `sale_rub` – продажи в РУБ всего (промо и без);
- `pr_promo_sales_in_rub` перевод в `sale_rub_promo` – продажи с признаком промо в РУБ;

2. **pr_df.csv** – данные по товарной иерархии.
От большего к меньшему `pr_group_id` - `pr_cat_id` - `pr_subcat_id` - `pr_sku_id`.

Столбцы:
- `pr_group_id` перевод в `group` – захэшированная группа товара;
- `pr_cat_id` перевод в `category` – захэшированная категория товара;
- `pr_subcat_id` перевод в `subcategory` – захэшированная подкатегория товара;
- `pr_sku_id` перевод в `sku` – захэшированное id товара;
- `pr_uom_id` перевод в `uom` - (маркер, обозначающий продаётся товар на вес или в ШТ).

3. **pr_st.csv** – данные по магазинам.

Столбцы:
- `st_id` перевод в `store` – захэшированное id магазина;
- `st_city_id` перевод в `city` – захэшированное id города;
- `st_division_code id` перевод в `division` – захэшированное id дивизиона;
- `st_type_format_id` перевод в `type_format` – id формата магазина;
- `st_type_loc_id` перевод в `loc` – id тип локации/окружения магазина;
- `st_type_size_id` перевод в `size` – id типа размера магазина;
- `st_is_active` перевод в `is_active` – флаг активного магазина на данный момент.

4. **sales_submission.csv** – пример файла с результатом работы модели прогноза спроса.
> Необходимо подготовить файл в таком же формате, заполнив колонку `target` предсказаниями (по умолчанию колонка заполнена нулями).

Столбцы:
- `st_id` – захэшированное id магазина;
- `pr_sku_id` – захэшированное id товара;
- `date` – дата (день);
- `target` – спрос в шт.

### **Описание проекта:**
- Модель строилась на прогноз в 14 дней;
- Метрика качества на л2 наиболее активных по продажам магазинах и 2х самых продаваемых товарах составляет **MAPE = 19%**
- Для реализации задачи обучена модель **CatBoostRegressor**.
- данные подаются в продакшн.

# Команда DS:
- Алла Мишра
- Михаил Грибанов
- Ирина Балычева 
