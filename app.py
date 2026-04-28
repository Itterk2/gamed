"""
Real Estate Price Analysis - Streamlit App
ML-based analysis of apartment prices in Dagestan (Republic of Dagestan).
Data sources:
  - data.xls  : average price per sq.m. by quarter (Rosstat, secondary market)
  - avito.xlsx : current apartment listings scraped from Avito
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PolynomialFeatures

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths (relative to this script)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_XLS_PATH = os.path.join(BASE_DIR, "data.xls")
AVITO_PATH = os.path.join(BASE_DIR, "avito.xlsx")

# ---------------------------------------------------------------------------
# Rosstat price indices (%) for the Republic of Dagestan, secondary market.
# Source: 05.rosstat.gov.ru — "Индексы цен на рынке жилья по РД"
# quarter-over-quarter price index (previous quarter = 100).
# ---------------------------------------------------------------------------
ROSSTAT_INDICES = {
    "2021 Q1": 105.8, "2021 Q2": 105.8, "2021 Q3": 105.2, "2021 Q4": 103.9,
    "2022 Q1": 109.8, "2022 Q2": 102.1, "2022 Q3": 104.4, "2022 Q4": 103.7,
    "2023 Q1": 105.8, "2023 Q2": 102.3, "2023 Q3": 104.6, "2023 Q4": 104.6,
    "2024 Q1": 112.0, "2024 Q2": 99.9,  "2024 Q3": 100.5, "2024 Q4": 100.6,
    "2025 Q1": 114.0, "2025 Q2": 100.0, "2025 Q3": 99.9,  "2025 Q4": 97.8,
}


# ===== DATA LOADING =========================================================

@st.cache_data
def load_quarterly_prices(path: str) -> pd.DataFrame:
    """Parse data.xls into a tidy DataFrame with columns: Year, Quarter, Label, Price_m2."""
    raw = pd.read_excel(path, header=None)
    # Row 2 contains years, row 3 contains quarter names, row 4 has values
    years_row = raw.iloc[2].dropna().values  # [2021, 2022, 2023, 2024, 2025]
    quarters_row = raw.iloc[3].dropna().values  # 20 quarter labels
    values_row = raw.iloc[4].dropna().values  # first 4 are text labels, rest are prices

    # Extract numeric values (skip the 4 text labels at the start)
    prices = [float(v) for v in values_row if _is_number(v)]
    quarter_labels = [str(q).strip() for q in quarters_row]

    # Build year-quarter pairs
    records = []
    qi = 0
    for yr in years_row:
        yr_int = int(yr)
        for _ in range(4):
            if qi < len(prices) and qi < len(quarter_labels):
                records.append({
                    "Year": yr_int,
                    "Quarter": quarter_labels[qi],
                    "Label": f"{yr_int} {quarter_labels[qi]}",
                    "Price_m2": prices[qi],
                })
            qi += 1
    df = pd.DataFrame(records)
    df["QuarterNum"] = range(1, len(df) + 1)
    return df


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False


@st.cache_data
def load_avito(path: str) -> pd.DataFrame:
    """Load and enrich Avito apartment listings."""
    df = pd.read_excel(path)

    # Parse coordinates
    if "Координаты" in df.columns:
        coords = df["Координаты"].astype(str).str.split(";", expand=True)
        if coords.shape[1] >= 2:
            df["lat"] = pd.to_numeric(coords[0], errors="coerce")
            df["lon"] = pd.to_numeric(coords[1], errors="coerce")

    # Clean price
    if "Цена" in df.columns:
        df["Цена"] = pd.to_numeric(df["Цена"], errors="coerce")

    # Parse apartment features from title  e.g. "1-к. квартира, 35 м², 9/10 эт."
    if "Название" in df.columns:
        rooms = df["Название"].str.extract(r"(\d+)-к", expand=False)
        df["Комнаты"] = pd.to_numeric(rooms, errors="coerce")

        area = df["Название"].str.extract(r"(\d+[.,]?\d*)\s*м", expand=False)
        df["Площадь"] = pd.to_numeric(area.str.replace(",", ".", regex=False), errors="coerce")

        floor = df["Название"].str.extract(r"(\d+)/\d+\s*эт", expand=False)
        df["Этаж"] = pd.to_numeric(floor, errors="coerce")

        total_floors = df["Название"].str.extract(r"\d+/(\d+)\s*эт", expand=False)
        df["Всего_этажей"] = pd.to_numeric(total_floors, errors="coerce")

    # Price per sq.m.
    if "Площадь" in df.columns and "Цена" in df.columns:
        df["Цена_м2"] = df["Цена"] / df["Площадь"]

    return df


@st.cache_data
def build_rosstat_df() -> pd.DataFrame:
    """Return Rosstat price index data as a DataFrame."""
    records = []
    for label, idx in ROSSTAT_INDICES.items():
        parts = label.split()
        records.append({"Year": int(parts[0]), "QLabel": parts[1], "Label": label, "Index": idx})
    return pd.DataFrame(records)


# ===== ML MODELS =============================================================

def train_price_forecast(df_q: pd.DataFrame):
    """Train several models to forecast price per sq.m. over quarters."""
    X = df_q[["QuarterNum"]].values
    y = df_q["Price_m2"].values

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)

    # 2. Polynomial (degree=2)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    lr_poly = LinearRegression()
    lr_poly.fit(X_poly, y)

    # 3. Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
    gb.fit(X, y)

    models = {
        "Линейная регрессия": (lr, None),
        "Полиномиальная (степень 2)": (lr_poly, poly),
        "Gradient Boosting": (gb, None),
    }

    # Cross-validation scores
    scores = {}
    for name, (m, p) in models.items():
        if p is not None:
            cv = cross_val_score(LinearRegression(), X_poly, y, cv=min(5, len(y)), scoring="r2")
        else:
            cv = cross_val_score(m.__class__(**m.get_params()), X, y, cv=min(5, len(y)), scoring="r2")
        scores[name] = cv.mean()

    return models, scores


def train_avito_model(df: pd.DataFrame):
    """Train RandomForest to predict apartment price from features."""
    features = ["Комнаты", "Площадь", "Этаж", "Всего_этажей"]
    available = [f for f in features if f in df.columns]
    df_model = df.dropna(subset=available + ["Цена"]).copy()
    if len(df_model) < 5:
        return None, None, None, None, None

    X = df_model[available]
    y = df_model["Цена"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "R2": r2_score(y_test, y_pred),
    }
    return rf, X_test, y_test, y_pred, metrics


# ===== STREAMLIT APP =========================================================

st.set_page_config(layout="wide", page_title="Анализ цен на недвижимость — Дагестан")

st.title("Анализ цен на недвижимость — Республика Дагестан")
st.markdown(
    "Машинное обучение и визуализация цен на жилье на основе данных **Росстат** и **Avito**."
)

# --- Load all data ---
try:
    df_quarters = load_quarterly_prices(DATA_XLS_PATH)
except Exception as e:
    st.error(f"Не удалось загрузить data.xls: {e}")
    df_quarters = pd.DataFrame()

try:
    df_avito = load_avito(AVITO_PATH)
except Exception as e:
    st.error(f"Не удалось загрузить avito.xlsx: {e}")
    df_avito = pd.DataFrame()

df_rosstat = build_rosstat_df()

# -------------------------------------------------------------------------
# TAB 1 — Quarterly Price per sq.m. + ML Forecast
# -------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Цена за м² по кварталам",
    "Индексы цен (Росстат)",
    "Avito — анализ объявлений",
    "ML-модель (Avito)",
])

with tab1:
    st.subheader("Средняя цена за 1 м² (вторичный рынок, Махачкала)")

    if not df_quarters.empty:
        # --- Raw data chart ---
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(df_quarters["Label"], df_quarters["Price_m2"], "o-", color="#1f77b4", linewidth=2, label="Факт")
        ax1.fill_between(range(len(df_quarters)), df_quarters["Price_m2"], alpha=0.15, color="#1f77b4")
        ax1.set_xlabel("Квартал")
        ax1.set_ylabel("Цена за м² (руб.)")
        ax1.set_title("Динамика цен за 1 м² по кварталам (2021–2025)")
        plt.xticks(rotation=45, ha="right")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig1)

        # --- ML Forecast ---
        st.subheader("Прогноз цен (машинное обучение)")
        models, scores = train_price_forecast(df_quarters)

        # Forecast 4 quarters ahead
        last_q = df_quarters["QuarterNum"].max()
        future_q = np.arange(last_q + 1, last_q + 5).reshape(-1, 1)
        future_labels = [f"2026 Q{i}" for i in range(1, 5)]

        fig2, ax2 = plt.subplots(figsize=(12, 5))
        ax2.plot(df_quarters["QuarterNum"], df_quarters["Price_m2"], "ko-", label="Факт", markersize=6)

        colors = {"Линейная регрессия": "#2ca02c", "Полиномиальная (степень 2)": "#ff7f0e", "Gradient Boosting": "#d62728"}
        all_x = np.arange(1, last_q + 5).reshape(-1, 1)
        for name, (m, poly_t) in models.items():
            if poly_t is not None:
                y_all = m.predict(poly_t.transform(all_x))
            else:
                y_all = m.predict(all_x)
            ax2.plot(all_x, y_all, "--", label=f"{name} (R²={scores[name]:.3f})", color=colors[name], linewidth=1.5)

        ax2.axvline(x=last_q + 0.5, color="gray", linestyle=":", alpha=0.7)
        ax2.text(last_q + 1, df_quarters["Price_m2"].min(), "Прогноз", fontsize=10, color="gray")

        xtick_labels = df_quarters["Label"].tolist() + future_labels
        ax2.set_xticks(range(1, last_q + 5))
        ax2.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=7)
        ax2.set_xlabel("Квартал")
        ax2.set_ylabel("Цена за м² (руб.)")
        ax2.set_title("Факт + Прогноз на 2026 год")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

        # Scores table
        st.markdown("**Качество моделей (R² по кросс-валидации):**")
        score_df = pd.DataFrame({"Модель": scores.keys(), "R²": [f"{v:.4f}" for v in scores.values()]})
        st.dataframe(score_df, hide_index=True)

        # Forecast values table
        st.markdown("**Прогнозные значения на 2026 год:**")
        forecast_records = []
        for name, (m, poly_t) in models.items():
            if poly_t is not None:
                preds = m.predict(poly_t.transform(future_q))
            else:
                preds = m.predict(future_q)
            for i, lbl in enumerate(future_labels):
                forecast_records.append({"Модель": name, "Квартал": lbl, "Прогноз (руб./м²)": f"{preds[i]:,.0f}"})
        st.dataframe(pd.DataFrame(forecast_records), hide_index=True)

        # Raw data
        with st.expander("Исходные данные (data.xls)"):
            st.dataframe(df_quarters[["Label", "Price_m2"]].rename(columns={"Label": "Квартал", "Price_m2": "Цена за м² (руб.)"}), hide_index=True)
    else:
        st.warning("Файл data.xls не найден или пуст.")

# -------------------------------------------------------------------------
# TAB 2 — Rosstat Price Indices
# -------------------------------------------------------------------------
with tab2:
    st.subheader("Индексы цен на рынке жилья (Росстат, РД)")
    st.markdown("Поквартальные индексы цен (предыдущий квартал = 100%).")

    if not df_rosstat.empty:
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        bars = ax3.bar(df_rosstat["Label"], df_rosstat["Index"], color=np.where(df_rosstat["Index"] >= 100, "#2ca02c", "#d62728"))
        ax3.axhline(y=100, color="black", linestyle="-", linewidth=0.8)
        ax3.set_xlabel("Квартал")
        ax3.set_ylabel("Индекс цен (%)")
        ax3.set_title("Индексы цен на вторичном рынке жилья по кварталам")
        plt.xticks(rotation=45, ha="right")
        ax3.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        st.pyplot(fig3)

        # Cumulative growth
        cumulative = (df_rosstat["Index"] / 100).cumprod() * 100
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        ax4.plot(df_rosstat["Label"], cumulative, "s-", color="#9467bd", linewidth=2)
        ax4.fill_between(range(len(cumulative)), cumulative, 100, alpha=0.15, color="#9467bd")
        ax4.set_xlabel("Квартал")
        ax4.set_ylabel("Кумулятивный индекс (%)")
        ax4.set_title("Накопленный рост цен с 2021 Q1")
        plt.xticks(rotation=45, ha="right")
        ax4.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig4)

        with st.expander("Данные индексов"):
            st.dataframe(df_rosstat, hide_index=True)

# -------------------------------------------------------------------------
# TAB 3 — Avito Listings Analysis
# -------------------------------------------------------------------------
with tab3:
    st.subheader("Анализ объявлений Avito (Махачкала)")

    if not df_avito.empty:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Всего объявлений", len(df_avito))
        if "Цена" in df_avito.columns:
            col2.metric("Средняя цена", f"{df_avito['Цена'].mean():,.0f} руб.")
            col3.metric("Медианная цена", f"{df_avito['Цена'].median():,.0f} руб.")
        if "Площадь" in df_avito.columns:
            col4.metric("Средняя площадь", f"{df_avito['Площадь'].mean():.1f} м²")

        c1, c2 = st.columns(2)

        # Price distribution
        with c1:
            if "Цена" in df_avito.columns:
                fig5, ax5 = plt.subplots(figsize=(6, 4))
                sns.histplot(df_avito["Цена"].dropna(), bins=15, kde=True, ax=ax5, color="#1f77b4")
                ax5.set_xlabel("Цена (руб.)")
                ax5.set_ylabel("Количество")
                ax5.set_title("Распределение цен")
                plt.tight_layout()
                st.pyplot(fig5)

        # Price vs Area
        with c2:
            if "Площадь" in df_avito.columns and "Цена" in df_avito.columns:
                fig6, ax6 = plt.subplots(figsize=(6, 4))
                sns.scatterplot(data=df_avito.dropna(subset=["Площадь", "Цена"]), x="Площадь", y="Цена", hue="Комнаты", palette="Set2", ax=ax6, s=80)
                ax6.set_xlabel("Площадь (м²)")
                ax6.set_ylabel("Цена (руб.)")
                ax6.set_title("Цена vs Площадь")
                plt.tight_layout()
                st.pyplot(fig6)

        c3, c4 = st.columns(2)

        # Price per sq.m. by rooms
        with c3:
            if "Цена_м2" in df_avito.columns and "Комнаты" in df_avito.columns:
                fig7, ax7 = plt.subplots(figsize=(6, 4))
                avito_rooms = df_avito.dropna(subset=["Цена_м2", "Комнаты"])
                sns.boxplot(data=avito_rooms, x="Комнаты", y="Цена_м2", ax=ax7, palette="pastel")
                ax7.set_xlabel("Количество комнат")
                ax7.set_ylabel("Цена за м² (руб.)")
                ax7.set_title("Цена за м² по количеству комнат")
                plt.tight_layout()
                st.pyplot(fig7)

        # Price per sq.m. distribution
        with c4:
            if "Цена_м2" in df_avito.columns:
                fig8, ax8 = plt.subplots(figsize=(6, 4))
                sns.histplot(df_avito["Цена_м2"].dropna(), bins=15, kde=True, ax=ax8, color="#ff7f0e")
                ax8.set_xlabel("Цена за м² (руб.)")
                ax8.set_ylabel("Количество")
                ax8.set_title("Распределение цены за м²")
                plt.tight_layout()
                st.pyplot(fig8)

        # Map
        if "lat" in df_avito.columns and "lon" in df_avito.columns:
            map_data = df_avito.dropna(subset=["lat", "lon"])
            if not map_data.empty:
                st.subheader("Объекты на карте")
                st.map(map_data[["lat", "lon"]])

        # Correlation heatmap
        numeric_cols = df_avito.select_dtypes(include=[np.number]).columns.tolist()
        useful_cols = [c for c in ["Цена", "Площадь", "Комнаты", "Этаж", "Всего_этажей", "Цена_м2"] if c in numeric_cols]
        if len(useful_cols) >= 2:
            st.subheader("Корреляционная матрица")
            fig9, ax9 = plt.subplots(figsize=(8, 6))
            corr = df_avito[useful_cols].corr()
            sns.heatmap(corr, annot=True, cmap="RdYlGn", center=0, ax=ax9, fmt=".2f")
            ax9.set_title("Корреляция числовых признаков")
            plt.tight_layout()
            st.pyplot(fig9)

        with st.expander("Таблица объявлений"):
            display_cols = [c for c in ["Название", "Цена", "Площадь", "Комнаты", "Этаж", "Цена_м2", "Адрес пользователя"] if c in df_avito.columns]
            st.dataframe(df_avito[display_cols], hide_index=True)
    else:
        st.warning("Файл avito.xlsx не найден или пуст.")

# -------------------------------------------------------------------------
# TAB 4 — ML Model on Avito data
# -------------------------------------------------------------------------
with tab4:
    st.subheader("ML-модель: прогноз цены квартиры (RandomForest)")

    if not df_avito.empty:
        rf, X_test, y_test, y_pred, metrics = train_avito_model(df_avito)

        if rf is not None:
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("RMSE", f"{metrics['RMSE']:,.0f} руб.")
            mc2.metric("MAE", f"{metrics['MAE']:,.0f} руб.")
            mc3.metric("R²", f"{metrics['R2']:.3f}")

            c5, c6 = st.columns(2)

            # Actual vs Predicted
            with c5:
                fig10, ax10 = plt.subplots(figsize=(6, 5))
                ax10.scatter(y_test, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5, s=80)
                mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                ax10.plot([mn, mx], [mn, mx], "--r", linewidth=2, label="Идеальное совпадение")
                ax10.set_xlabel("Фактическая цена (руб.)")
                ax10.set_ylabel("Предсказанная цена (руб.)")
                ax10.set_title("Факт vs Прогноз")
                ax10.legend()
                ax10.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig10)

            # Feature importance
            with c6:
                importances = rf.feature_importances_
                feat_names = X_test.columns.tolist()
                fig11, ax11 = plt.subplots(figsize=(6, 5))
                idx_sorted = np.argsort(importances)
                ax11.barh([feat_names[i] for i in idx_sorted], importances[idx_sorted], color="#2ca02c")
                ax11.set_xlabel("Важность признака")
                ax11.set_title("Важность признаков (RandomForest)")
                plt.tight_layout()
                st.pyplot(fig11)

            # Residuals
            residuals = y_test.values - y_pred
            fig12, ax12 = plt.subplots(figsize=(10, 4))
            ax12.bar(range(len(residuals)), residuals, color=np.where(residuals >= 0, "#2ca02c", "#d62728"))
            ax12.axhline(y=0, color="black", linewidth=0.8)
            ax12.set_xlabel("Наблюдение")
            ax12.set_ylabel("Ошибка (руб.)")
            ax12.set_title("Остатки модели (Факт - Прогноз)")
            ax12.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            st.pyplot(fig12)

            # Interactive predictor
            st.subheader("Калькулятор цены квартиры")
            pc1, pc2, pc3, pc4 = st.columns(4)
            with pc1:
                inp_rooms = st.number_input("Комнаты", min_value=1, max_value=10, value=2)
            with pc2:
                inp_area = st.number_input("Площадь (м²)", min_value=10.0, max_value=500.0, value=50.0, step=5.0)
            with pc3:
                inp_floor = st.number_input("Этаж", min_value=1, max_value=30, value=5)
            with pc4:
                inp_total = st.number_input("Всего этажей", min_value=1, max_value=50, value=10)

            input_features = ["Комнаты", "Площадь", "Этаж", "Всего_этажей"]
            available_features = [f for f in input_features if f in X_test.columns]
            input_vals = {"Комнаты": inp_rooms, "Площадь": inp_area, "Этаж": inp_floor, "Всего_этажей": inp_total}
            inp_df = pd.DataFrame([{f: input_vals[f] for f in available_features}])

            predicted = rf.predict(inp_df)[0]
            st.success(f"Прогнозная цена: **{predicted:,.0f} руб.** ({predicted / inp_area:,.0f} руб./м²)")
        else:
            st.warning("Недостаточно данных для обучения модели (нужно >= 5 объявлений с полными данными).")
    else:
        st.warning("Файл avito.xlsx не найден или пуст.")

# --- Footer ---
st.markdown("---")
st.caption("Источники: Росстат (05.rosstat.gov.ru), Avito.ru | Машинное обучение: scikit-learn")
