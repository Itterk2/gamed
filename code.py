import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Путь к файлу с данными Avito
DATA_PATH = r'C:\Users\Admin\Desktop\cdoo\avito.xlsx'

@st.cache_data
def load_data(path):
    # Загружаем данные
    df = pd.read_excel(path)
    
    # Парсинг координат (lat;lon) для отображения на карте
    if 'Координаты' in df.columns:
        coords = df['Координаты'].astype(str).str.split(';', expand=True)
        if coords.shape[1] == 2:
            df['lat'] = pd.to_numeric(coords[0], errors='coerce')
            df['lon'] = pd.to_numeric(coords[1], errors='coerce')
    
    # Очистка колонки "Цена"
    if 'Цена' in df.columns:
        df['Цена'] = pd.to_numeric(df['Цена'], errors='coerce')
        
    # Парсинг названия, например: "1-к. квартира, 35 м², 9/10 эт."
    if 'Название' in df.columns:
        # Извлекаем количество комнат
        rooms = df['Название'].str.extract(r'(?i)(\d+)-к|Студия', expand=False)
        df['Комнаты'] = rooms.fillna('Студия/Другое')
        
        # Извлекаем площадь
        area = df['Название'].str.extract(r'(?i)(\d+[.,]?\d*)\s*м²', expand=False)
        df['Площадь'] = pd.to_numeric(area.str.replace(',', '.'), errors='coerce')
        
        # Извлекаем этаж
        floor = df['Название'].str.extract(r'(?i)(\d+)/\d+\s*эт', expand=False)
        df['Этаж'] = pd.to_numeric(floor, errors='coerce')
        
    return df

# Настройка страницы

def load_excel_data(path: str) -> pd.DataFrame:
    """Load supplementary Excel data containing price per square meter by district/quarter."""
    df_extra = pd.read_excel(path)
    return df_extra

def load_pdf_data(path: str) -> pd.DataFrame:
    """Extract tables from the PDF with housing market price indices.
    Requires `tabula-py` and Java runtime.
    """
    try:
        import tabula
    except ImportError:
        raise ImportError("tabula-py is required to read PDF files. Install via 'pip install tabula-py'.")
    tables = tabula.read_pdf(path, pages='all', multiple_tables=True)
    if isinstance(tables, list) and tables:
        df_pdf = pd.concat(tables, ignore_index=True)
    else:
        df_pdf = pd.DataFrame()
    return df_pdf
st.set_page_config(layout="wide", page_title="Аналитика Недвижимости Avito")

st.title("🏡 Анализ цен на недвижимость (Avito)")

st.markdown("""
Это приложение отображает актуальные объявления о недвижимости.
Слева вы можете настроить фильтры для поиска подходящих объектов.
""")

with st.spinner('Загрузка данных...'):
    try:
        df = load_data(DATA_PATH)
    except Exception as e:
        st.error(f"Ошибка при загрузке данных: {e}")
        st.stop()

# --- БОКОВАЯ ПАНЕЛЬ (ФИЛЬТРЫ) ---
st.sidebar.header("Фильтры")

# Фильтр по цене
if 'Цена' in df.columns and not df['Цена'].isna().all():
    min_price = int(df['Цена'].min(skipna=True))
    max_price = int(df['Цена'].max(skipna=True))
    
    if min_price == max_price:
        max_price = min_price + 1000
        
    price_range = st.sidebar.slider(
        "Стоимость (руб.)",
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price),
        step=1000,
        format="%d ₽"
    )
    df_filtered = df[(df['Цена'] >= price_range[0]) & (df['Цена'] <= price_range[1])]
else:
    df_filtered = df

# Фильтр по комнатам
if 'Комнаты' in df.columns:
    rooms_options = sorted(df['Комнаты'].astype(str).unique())
    selected_rooms = st.sidebar.multiselect(
        "Количество комнат", 
        options=rooms_options, 
        default=rooms_options
    )
    if selected_rooms:
        df_filtered = df_filtered[df_filtered['Комнаты'].astype(str).isin(selected_rooms)]

# Фильтр по площади
if 'Площадь' in df.columns and not df['Площадь'].isna().all():
    min_area = float(df['Площадь'].min(skipna=True))
    max_area = float(df['Площадь'].max(skipna=True))
    if pd.notna(min_area) and pd.notna(max_area) and min_area < max_area:
        area_range = st.sidebar.slider(
            "Площадь (м²)",
            min_value=min_area,
            max_value=max_area,
            value=(min_area, max_area)
        )
        df_filtered = df_filtered[(df_filtered['Площадь'] >= area_range[0]) & (df_filtered['Площадь'] <= area_range[1])]

st.sidebar.markdown(f"**Найдено объектов:** {len(df_filtered)}")

# --- ОСНОВНОЙ КОНТЕНТ ---

# Основные метрики
st.subheader("📊 Общая статистика")
col1, col2, col3 = st.columns(3)
col1.metric("Средняя цена", f"{df_filtered['Цена'].mean():,.0f} ₽" if 'Цена' in df_filtered and not df_filtered.empty else "Н/Д")
col2.metric("Медианная цена", f"{df_filtered['Цена'].median():,.0f} ₽" if 'Цена' in df_filtered and not df_filtered.empty else "Н/Д")
col3.metric("Средняя площадь", f"{df_filtered['Площадь'].mean():.1f} м²" if 'Площадь' in df_filtered and not df_filtered.empty else "Н/Д")

# Карта
st.subheader("📍 Объекты на карте")
if 'lat' in df_filtered.columns and 'lon' in df_filtered.columns:
    map_data = df_filtered.dropna(subset=['lat', 'lon'])
    if not map_data.empty:
        st.map(map_data[['lat', 'lon']])
    else:
        st.info("Нет данных с координатами для отображения на карте.")

# Таблица данных
st.subheader("📋 Список недвижимости")
display_cols = ['Название', 'Цена', 'Адрес пользователя', 'Площадь', 'Этаж', 'Описание']
actual_cols = [c for c in display_cols if c in df_filtered.columns]

if not df_filtered.empty:
    st.dataframe(
        df_filtered[actual_cols].style.format({'Цена': '{:,.0f} ₽', 'Площадь': '{:.1f}'}),
        use_container_width=True,
        height=300
    )
else:
    st.info("По заданным фильтрам ничего не найдено.")

# Галерея объектов
st.subheader("🖼️ Галерея объектов (первые 10)")
for idx, row in df_filtered.head(10).iterrows():
    with st.container():
        col_img, col_info = st.columns([1, 3])
        
        with col_img:
            if 'Изображения' in row and pd.notna(row['Изображения']):
                images = str(row['Изображения']).split(';')
                valid_img = next((img for img in images if img.startswith('http')), None)
                if valid_img:
                    st.image(valid_img, use_container_width=True)
                else:
                    st.write("Нет фото")
            else:
                st.write("Нет фото")
                
        with col_info:
            title = row.get('Название', 'Без названия')
            st.markdown(f"### {title}")
            
            price = row.get('Цена', 0)
            st.markdown(f"**Цена:** {price:,.0f} ₽")
            
            address = row.get('Адрес пользователя', '')
            st.markdown(f"**Адрес:** {address}")
            
            desc = str(row.get('Описание', ''))
            st.write(desc[:250] + ("..." if len(desc) > 250 else ""))
            
            if 'URL' in row and pd.notna(row['URL']):
                st.markdown(f"[Ссылка на Avito]({row['URL']})")
                
        st.divider()

# Visualize price per square meter from Excel data
excel_path = r'C:\Users\Admin\Desktop\cdoo\data.xls'
try:
    df_excel = load_excel_data(excel_path)
    if not df_excel.empty and 'Квартал' in df_excel.columns and 'Цена_м2' in df_excel.columns:
        st.subheader('📈 Цена за м² по кварталам (Excel)')
        fig_excel, ax_excel = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Квартал', y='Цена_м2', data=df_excel, ax=ax_excel)
        ax_excel.set_xlabel('Квартал')
        ax_excel.set_ylabel('Цена за м²')
        ax_excel.set_title('Средняя цена за квадратный метр по кварталам')
        plt.xticks(rotation=45)
        st.pyplot(fig_excel)
    else:
        st.info('В Excel файле нет нужных колонок "Квартал" и/или "Цена_м2".')
except Exception as e:
    st.error(f"Ошибка загрузки Excel данных: {e}")

# Machine Learning model
st.sidebar.subheader("ML модель")
if st.sidebar.button("Запустить модель ML"):
    # Load supplementary datasets
    excel_path = r'C:\Users\Admin\Desktop\cdoo\data.xls'
    pdf_path = r'C:\Users\Admin\Desktop\cdoo\05.rosstat.gov.ru.pdf'
    try:
        df_excel = load_excel_data(excel_path)
    except Exception as e:
        st.error(f"Не удалось загрузить Excel файл: {e}")
        df_excel = pd.DataFrame()
    try:
        df_pdf = load_pdf_data(pdf_path)
    except Exception as e:
        st.error(f"Не удалось загрузить PDF файл: {e}")
        df_pdf = pd.DataFrame()
    # Prepare main dataset for modeling
    model_df = df.copy()
    if 'Площадь' in model_df.columns and 'Цена' in model_df.columns:
        model_df['Цена_м2'] = model_df['Цена'] / model_df['Площадь']
    # Merge extra data
    for extra in [df_excel, df_pdf]:
        if not extra.empty:
            common_cols = set(model_df.columns).intersection(set(extra.columns))
            if common_cols:
                join_col = list(common_cols)[0]
                model_df = model_df.merge(extra, on=join_col, how='left')
    model_df = model_df.dropna(subset=['Цена_м2'])
    numeric_cols = model_df.select_dtypes(include=['number']).columns.tolist()
    numeric_cols.remove('Цена_м2')
    X = model_df[numeric_cols]
    y = model_df['Цена_м2']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)
    st.subheader("🧪 Результаты модели")
    st.write(f"RMSE: {rmse:,.2f} ₽/м²")
    st.write(f"R²: {r2:.3f}")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
    ax.set_xlabel('Настоящая цена за м²')
    ax.set_ylabel('Предсказанная цена за м²')
    ax.set_title('Сравнение фактической и предсказанной цены за м²')
    st.pyplot(fig)
