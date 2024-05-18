# Импорт необходимых библиотек
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import xgboost
import seaborn as sns

# загрузка данных
index_names = ['unit_nr', 'time_cycles']
setting_names = ['setting_1', 'setting_2', 'setting_3']
sensor_names = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi',
                'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
col_names = index_names + setting_names + sensor_names

train = pd.read_csv(('train_FD001.txt'), sep='\s+', header=None, names=col_names)
test = pd.read_csv(('test_FD001.txt'), sep='\s+', header=None, names=col_names)
y_test = pd.read_csv(('RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

st.title("Прогнозное Техническое Обслуживание (ТО)")

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Данные', 'Задача', 'EDA', 'Выбор модели', 'Демо модель'])

with tab1:
    st.write("""

Данные взяты на сайте NASA по ссылке:

https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data

В настоящем проекте использован только датасет FD001. Набор данных включает симуляции нескольких турбодвигателей со временем, каждая строка содержит следующую информацию:
1. Номер двигателя
2. Время в циклах
3. Три рабочих настройки
4. 21 показании сенсоров
""")

    st.image("turbofan-engine.png", use_column_width=True)
    st.write("""
**Описания сенсоров на английском:**


| Name      |Description                      |Unit     |    
|-----------|---------------------------------|---------|    
| T2        | Total temperature at fan inlet  | °R      |    
| T24       | Total temperature at LPC outlet | °R      |    
| T30       | Total temperature at HPC outlet | °R      |    
| T50       | Total temperature at LPT outlet | °R      |    
| P2        | Pressure at fan inlet           | psia    |    
| P15       | Total pressure in bypass-duct   | psia    |    
| P30       | Total pressure at HPC outlet    | psia    |    
| Nf        | Physical fan speed              | rpm     |    
| Nc        | Physical core speed             | rpm     |    
| epr       | Engine pressure ratio (P50/P2)  | --      |    
| Ps30      | Static pressure at HPC outlet   | psia    |    
| phi       | Ratio of fuel flow to Ps30      | pps/psi |    
| NRf       | Corrected fan speed             | rpm     |    
| NRc       | Corrected core speed            | rpm     |    
| BPR       | Bypass Ratio                    | --      |    
| farB      | Burner fuel-air ratio           | --      |    
| htBleed   | Bleed Enthalpy                  | --      |    
| Nf_dmd    | Demanded fan speed              | rpm     |    
| PCNfR_dmd | Demanded corrected fan speed    | rpm     |    
| W31       | HPT coolant bleed               | lbm/s   |    
| W32       | LPT coolant bleed               | lbm/s   |    
""")
    spinner = st.empty()
    form = st.form("pulse")

with tab2:
    st.write('**Своевременное ТО**')
    st.image("PM.png", use_column_width=True)
    st.write("""
    **Преимущества Своевременного Обслуживания**

    1. Снижение затрат:
        - Минимизация затрат на запасы запчастей и материалов.
        - Снижение расходов на внеплановые ремонты и аварийное обслуживание.
    2. Повышение надежности оборудования:
        - Предотвращение неожиданных поломок и связанных с ними простоев.
        - Увеличение времени бесперебойной работы оборудования.
    3. Улучшение эффективности:
        - Оптимизация графиков технического обслуживания.
        - Снижение времени простоя за счет точного планирования и выполнения работ.
    4. Улучшение управления ресурсами:
        - Более эффективное использование технического персонала.
        - Оптимизация процессов закупки и хранения запасных частей.
    
    В нашем случае своевременное ТО будет осуществляться за счет предсказания **RUL (Remaining Useful Life)** - оставшегося времени до отказа оборудования. 
    
    Данная модель также может быть применена в различных отраслях, где надежность оборудования имеет важное значение.
    """)

with tab3:
    st.write('Посмотрим, что мы можем узнать о количестве циклов, которые двигатели проходили в среднем перед поломкой.')
    st.dataframe(train[index_names].describe())
    st.write('Данные по уникальным двигателям')
    st.dataframe(train[index_names].groupby('unit_nr').max().describe())
    st.write('Набор данных содержит в общей сложности 20631 строк, номера двигателей начинаются с 1 и заканчиваются на 100. Двигатель, который сломался первым, сделал это после 128 циклов, в то время как двигатель, который работал дольше всего, сломался после 362 циклов. Средний двигатель ломается между 199 и 206 циклами, однако стандартное отклонение 46 циклов довольно велико.')
    st.write('**Вычисляем оставшееся время до отказа оборудования - RUL**')
    def add_remaining_useful_life(df):
        # Получить общее количество циклов для каждого устройства
        grouped_by_unit = df.groupby(by="unit_nr")
        max_cycle = grouped_by_unit["time_cycles"].max()

        # Объединить максимальное количество циклов обратно в оригинальный DataFrame
        result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

        # Рассчитать оставшийся полезный срок службы для каждой строки
        remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
        result_frame["RUL"] = remaining_useful_life

        # Удалить max_cycle, так как он больше не нужен
        result_frame = result_frame.drop("max_cycle", axis=1)
        return result_frame

    train = add_remaining_useful_life(train)

    df_max_rul = train[['unit_nr', 'RUL']].groupby('unit_nr').max().reset_index()
    plt.figure()
    df_max_rul['RUL'].hist(bins=15, figsize=(15,7))
    plt.xlabel('RUL')
    plt.ylabel('frequency')
    st.pyplot(plt)
    st.write('Гистограмма показывает, что большинство двигателей выходят из строя примерно через 200 циклов. Кроме того, распределение смещено вправо, с небольшим количеством двигателей, прослуживших более 300 циклов.')

    st.write('**Построим графики зависимости RUL от показании каждого сенсора**')
    def plot_sensor(ax, sensor_name):
        for i in train['unit_nr'].unique():
            if (i % 10 == 0):  # отображать только каждое 10-е устройство
                ax.plot('RUL', sensor_name, data=train[train['unit_nr'] == i])
        ax.set_xlim(250, 0)  # инвертировать ось x, чтобы RUL уменьшался до нуля
        ax.set_xticks(np.arange(0, 275, 25))
        ax.set_ylabel(sensor_name)
        ax.set_xlabel('Remaining Useful Life')

    # Создание фигуры и сетки подграфиков размером 3x7
    fig, axes = plt.subplots(nrows=7, ncols=3, figsize=(15, 20))
    axes = axes.flatten()  # Преобразовать 2D массив осей в 1D для удобства итерации

    # Построение графика для каждого сенсора на соответствующем подграфике
    for i, sensor_name in enumerate(sensor_names):
        plot_sensor(axes[i], sensor_name)

    # Настройка расположения подграфиков
    plt.tight_layout()
    st.pyplot(fig)

    st.write('Графики сенсоров позволяют нам понять их взаимосвязь с убывающим RUL. Мы видим, что некоторые сенсоры имеют четкие тренды относительно RUL, в то время как другие остаются стабильными или имеют непредсказуемое поведение. На основе нашего анализа данных мы определяем, что значения некоторых сенсоров остаются постоянными во времени и они мало влияют на RUL.')

    st.write('**Извлечение значимых признаков, которые сильно влияют на RUL**')

    plt.figure(figsize=(25,18))
    sns.heatmap(train.corr(),annot=True ,cmap='Reds')
    st.pyplot(plt)

    st.write('Отберем признаки, которые имеют абсолютное значение корреляции с RUL >= 0.3')

    # Вычисление корреляционной матрицы для всех признаков
    cor = train.corr()

    # Выбор признаков, которые имеют абсолютное значение корреляции с RUL >= 0.3
    train_relevant_features = cor.loc[abs(cor['RUL']) >= 0.3, 'RUL']

    list_relevant_features=train_relevant_features.index
    list_relevant_features=list_relevant_features[1:]
    st.dataframe(list_relevant_features)
    st.write('В приведённом выше списке содержатся значимые признаки, которые имеют корреляцию по модулю больше или равную 0.3 с нашей целевой переменной RUL. Мы оставим только эти признаки в качестве предикторов.')

with tab4:
    X_train = train[list_relevant_features]
    y_train = X_train.pop('RUL')

    # Поскольку истинные значения RUL для тестового набора данных предоставляются только для последнего цикла работы каждого двигателя,
    # тестовый набор данных также разбивается для представления только последнего цикла работы.
    X_test = test.groupby('unit_nr').last().reset_index()[X_train.columns]
    
    # Функция для оценки модели
    def evaluate(y_true, y_pred, label='test'):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        variance = r2_score(y_true, y_pred)
        print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))
        return rmse, variance;

    st.write('**Базовая модель**')
    ## Линейная регрессия
    # Обучаем модель
    lm = LinearRegression()
    lm.fit(X_train, y_train)

    # Прогноз и оценка
    y_pred_train = lm.predict(X_train)
    RMSE_train, R2_train = evaluate(y_train, y_pred_train, 'train')

    y_pred_test = lm.predict(X_test)
    RMSE_test, R2_test = evaluate(y_test, y_pred_test)

    # Dataframe, который будет содержать результаты всех примененных моделей
    Results=pd.DataFrame({'Model':['Linear Regression'],'RMSE-Train':[RMSE_train],'R2-Train':[R2_train],'RMSE-Test':[RMSE_test],'R2-Test':[R2_test]})
    st.dataframe(pd.DataFrame({'Model':['Linear Regression'],'RMSE-Train':[44.668192],'R2-Train':[0.579449],'RMSE-Test':[31.952633],'R2-Test':[0.408774]}), hide_index=True)

    def plot_results(y_true, y_pred):
        width = 0.8

        actuals = [int(x) for x in y_true.values]
        predictions = list(y_pred)

        indices = np.arange(len(y_pred))

        plt.figure(figsize=(60,20))

        plt.bar(indices, actuals, width=width,
                color='b', label='Actual RUL')
        plt.bar([i for i in indices], predictions,
                width=0.5*width, color='r', alpha=0.7, label='Predicted RUL')

        plt.legend(prop={'size': 30})
        plt.tick_params(labelsize=30)

        st.pyplot(plt)

    plot_results(y_test,y_pred_test)

    st.write('**Пересмотр RUL**')

    clipped_rul = train.loc[train['unit_nr']==20].copy()
    clipped_rul['RUL'].clip(upper=125, inplace=True)

    # Строим график для пересмотра RUL
    fig, ax1 = plt.subplots(figsize=(13, 5))
    ax2 = ax1.twinx()

    # График данных от сенсора
    train_unit_20 = train.loc[train['unit_nr'] == 20]
    signal = ax1.plot('RUL', 'phi', 'b', data=train_unit_20)
    rul_line = ax2.plot('RUL', 'RUL', 'k', linewidth=4, data=train_unit_20)
    rul = train_unit_20['RUL']
    rul_line2 = ax2.plot(rul, np.clip(rul, None, 125), '--g', linewidth=4, label='clipped_rul')

    # Метки и пределы
    ax1.set_xlim(250, 0)
    ax1.set_xticks(np.arange(0, 275, 25))
    ax1.set_ylabel('phi', labelpad=20)
    ax1.set_xlabel('RUL', labelpad=20)
    ax2.set_ylabel('RUL', labelpad=20)
    ax2.set_ylim(0, 250)
    ax2.set_yticks(np.linspace(ax2.get_ybound()[0], ax2.get_ybound()[1], 6))
    ax1.set_yticks(np.linspace(ax1.get_ybound()[0], ax1.get_ybound()[1], 6))

    # Легенда
    lines = signal + rul_line + rul_line2
    labels = ['phi', 'RUL', 'clipped_RUL']
    ax1.legend(lines, labels, loc=0)

    st.pyplot(fig)

    st.write(""" Можно предположить, что RUL вначале остается постоянным и начинает линейно убывать только после некоторого времени. Нашей целью будет получение аналогичного "изгиба" на кривой.
    
    Для этого обрежем RUL на уровне примерно 125 циклов.
    """)

    st.dataframe(pd.DataFrame({'Model':['Linear Regression', 'Linear regression Clipped'],'RMSE-Train':[44.668192, 21.491019],'R2-Train':[0.579449, 0.734043],'RMSE-Test':[31.952633, 21.900213],'R2-Test':[0.408774, 0.722261]}), hide_index=True)

    st.write("""
    Производительность модели сильно улучшилась, это информирует нас о том, что обновленная гипотеза полезна для прогнозирования RUL. 
    
    В моделях далее мы будем использовать обрезанную RUL.
    """)

    st.write('**Сравнение метрик остальных рассмотренных моделей**')

    st.dataframe(pd.DataFrame({'Model':['Linear Regression', 'Linear regression Clipped', 'Support Vector Regression', 'Random Forest', 'Random Forest Tuned', 'XGBoost', 'XGBoost Tuned'],
                                'RMSE-Train':[44.6682, 21.491, 18.598, 6.764, 15.4269, 16.9904, 16.2047],
                                'R2-Train':[0.5794, 0.734, 0.8008, 0.9737, 0.863, 0.8338, 0.8488],
                                'RMSE-Test':[31.9526, 21.9002, 19.2536, 17.8823, 17.8469, 18.4177, 17.9422],
                                'R2-Test':[0.4088, 0.7223, 0.7853, 0.8148, 0.8156, 0.8036, 0.8136]}), hide_index=True)

    st.write("""
    На основе оценочных метрик, лучшая модель для выбора Random Forest после оптимизации гиперпараметров.

    Если важны вычислительные ресурсы и простота, Support Vector Regression является ещё одним хорошим вариантом благодаря своей высокой производительности и более простой настройке по сравнению с ансамблевыми методами, такими как Random Forest и XGBoost.
    """)

    y_train_clipped = y_train.clip(upper=125)
    sc = StandardScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_test_scaled = sc.transform(X_test)

    ## Random Forest tuned
    # Обучаем модель
    rf_tuned = RandomForestRegressor(n_estimators=100, max_depth=13, min_samples_leaf=9, min_samples_split=2, max_features="sqrt", random_state=42)
    rf_tuned.fit(X_train_scaled, y_train_clipped)

    # Прогноз
    y_pred_train = rf_tuned.predict(X_train_scaled)
    y_pred_test = rf_tuned.predict(X_test_scaled)

    st.write('**Фактические RUL против предсказанных выбранной моделью**')
    plot_results(y_test,y_pred_test)

with tab5:
    # Загрузка TXT-файла с данными сенсоров
    st.subheader("Загрузите TXT файл с данными сенсоров")
    uploaded_file = st.file_uploader("Загрузить TXT", type=["txt"])

    if uploaded_file is not None:
        # Чтение TXT-файла
        sensor_df = pd.read_csv(uploaded_file, sep='\t')

        # Отображение загруженных данных
        st.write("Загруженные данные сенсоров:")
        st.write(sensor_df)

        # Получение названий сенсоров из столбцов DataFrame
        sensor_names = sensor_df.columns.tolist()

        # Кнопка для прогнозирования оставшегося срока службы
        if st.button("Предсказать RUL"):
            # Выполнение прогноза
            sensor_values = sensor_df.iloc[0].values  # Предполагается, что загружена только одна строка
            transformed = sc.transform(np.array(sensor_values).reshape(1, -1))
            # Предполагается, что у вас есть обученная модель с соответствующими признаками и метками
            predicted_rul = rf_tuned.predict(transformed)

            # Отображение предсказанного RUL
            int_pred_rul = int(predicted_rul[0])
            
            # Отображение предупреждений на основе предсказанного RUL
            if int_pred_rul <= 20:
                st.error(f"Прогнозируемый RUL: {int_pred_rul} циклов")
                st.error("Компрессор высокого давления должен быть срочно заменен!")
            elif int_pred_rul <= 50:
                st.warning(f"Прогнозируемый RUL: {int_pred_rul} циклов")
                st.warning("Необходимо провести тщательный осмотр компрессора высокого давления")
            else:
                st.success(f"Прогнозируемый RUL: {int_pred_rul} циклов")
                st.success("Необходимости в ТО нет")

st.markdown("---")
st.write("Проект подготовлен Бакытжаном Казиевым на Demo Day @ Outpeer.kz")
