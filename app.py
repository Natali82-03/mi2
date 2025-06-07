import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
import streamlit as st
import time
import joblib  # Для загрузки препроцессора

# --- ФУНКЦИЯ ОБУЧЕНИЯ МОДЕЛИ ---
@st.cache_data
def train_model():
    # Загрузка предобработанных данных
    df = pd.read_csv("processed_L_Score.csv")
    
    X = df.drop(columns=['L_Status'])
    y = df['L_Status']
    
    # Разделение выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2, stratify=y)
    
    # Обучение модели
    model = XGBClassifier(random_state=42, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Оценка
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return model, X.columns.tolist(), accuracy, f1


# --- ОСНОВНАЯ ФУНКЦИЯ ПРИЛОЖЕНИЯ ---
def main():
    # Настройки страницы
    st.set_page_config(
        page_title="CreditScore PRO",
        page_icon="💰",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    # Стилизация
    st.markdown("""
    <style>
        .header-style {
            font-size: 24px;
            font-weight: bold;
            color: #2e86c1;
        }
        .metric-value {
            font-size: 28px !important;
            font-weight: bold !important;
        }
        .stSpinner > div {
            text-align: center;
            margin-top: 10px;
        }
        .risk-indicator {
            transition: all 0.5s ease;
        }
        .pulse-animation {
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        .approved {
            color: #2ecc71;
            font-weight: bold;
        }
        .rejected {
            color: #e74c3c;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True)

    # --- ЗАГРУЗКА МОДЕЛИ И ПРЕПРОЦЕССОРА ---
    with st.spinner('Загрузка модели... Это займет несколько секунд'):
        model, feature_columns, accuracy, f1 = train_model()

    try:
        preprocessor = joblib.load('preprocessor.pkl')
    except FileNotFoundError:
        st.error("Файл 'preprocessor.pkl' не найден. Убедитесь, что он загружен.")
        return

    # --- ИНТЕРФЕЙС ПРИЛОЖЕНИЯ ---
    st.markdown('<p class="header-style">CreditScore PRO</p>', unsafe_allow_html=True)
    st.caption("Система кредитного скоринга для оценки заявок")

    # --- ФОРМА ВВОДА ДАННЫХ КЛИЕНТА ---
    with st.form("credit_form"):
        st.subheader("Данные клиента")
        
        input_values = {}
        cols = st.columns(2)
        
        with cols[0]:
            # Категориальные признаки
            input_values['P_Gender'] = st.selectbox("Пол", ['male', 'female'], key='P_Gender')
            input_values['P_Education'] = st.selectbox(
                "Образование", 
                ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'],
                key='P_Education'
            )
            input_values['P_Home'] = st.selectbox(
                "Тип жилья", 
                ['OWN', 'MORTGAGE', 'RENT', 'OTHER'],
                key='P_Home'
            )
            input_values['L_Intent'] = st.selectbox(
                "Цель кредита", 
                ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
                key='L_Intent'
            )
            
        with cols[1]:
            # Числовые признаки
            input_values['P_Age'] = st.number_input("Возраст", min_value=18, max_value=100, value=30, key='P_Age')
            input_values['P_Income'] = st.number_input("Доход (годовой)", min_value=0, value=50000, key='P_Income')
            input_values['P_Emp_Exp'] = st.number_input("Опыт работы (лет)", min_value=0, max_value=50, value=3, key='P_Emp_Exp')
            input_values['L_Amount'] = st.number_input("Сумма кредита", min_value=0, value=10000, key='L_Amount')
            input_values['L_Rate'] = st.number_input("Процентная ставка", min_value=0.0, max_value=30.0, value=10.0, key='L_Rate')
            input_values['L_Pers_Income'] = st.number_input(
                "Отношение кредита к доходу", 
                min_value=0.0, max_value=1.0, value=0.3, key='L_Pers_Income'
            )
        
        # Дополнительные категориальные признаки
        input_values['Credit_History'] = st.selectbox(
            "Кредитная история (рейтинг)", 
            [1.0, 2.0, 3.0, 4.0],
            format_func=lambda x: f"{x:.0f}",
            key='Credit_History'
        )
        input_values['L_Defaults'] = st.selectbox(
            "Были ли дефолты", 
            ['No', 'Yes'],
            key='L_Defaults'
        )
        
        submitted = st.form_submit_button("Оценить заявку", type="primary")

    # --- ОБРАБОТКА РЕЗУЛЬТАТОВ ---
    if submitted:
        try:
            # Преобразование введенных данных в DataFrame
            input_df = pd.DataFrame([input_values])

            # Применяем препроцессор
            input_processed = preprocessor.transform(input_df)

            # Предсказание
            prediction = model.predict(input_processed)[0]
            proba = model.predict_proba(input_processed)[0][1]

            # Отображение результата
            st.markdown("---")
            st.subheader("Результат оценки кредитной заявки")
            
            if prediction == 1:
                st.markdown(f'<p class="approved">✅ Кредит одобрен!</p>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f'<p class="rejected">❌ Кредит не одобрен</p>', unsafe_allow_html=True)
            
            # Прогресс-бар с вероятностью
            st.write(f"Вероятность одобрения: {proba*100:.1f}%")
            progress_bar = st.progress(0)
            for percent_complete in range(int(proba*100)):
                time.sleep(0.01)
                progress_bar.progress(percent_complete + 1)
            progress_bar.progress(float(proba))
            
            # Рекомендации
            with st.expander("Рекомендации"):
                if prediction == 0:
                    st.write("""
                    **Возможные причины отказа:**
                    - Высокое отношение кредита к доходу
                    - Недостаточный кредитный рейтинг
                    - Наличие дефолтов в истории
                    - Недостаточный опыт работы
                    """)
                    st.write("**Что можно сделать:**")
                    st.write("- Уменьшите запрашиваемую сумму кредита")
                    st.write("- Улучшите кредитную историю")
                    st.write("- Рассмотрите кредит с более высокой процентной ставкой")
                else:
                    st.write("""
                    **Рекомендации:**
                    - Убедитесь, что вы можете комфортно обслуживать кредит
                    - Рассмотрите возможность досрочного погашения
                    - Проверьте все условия кредитного договора
                    """)

        except Exception as e:
            st.error(f"Ошибка при оценке заявки: {str(e)}")

    # --- БОКОВАЯ ПАНЕЛЬ С ИНФОРМАЦИЕЙ ---
    with st.sidebar:
        st.header("ℹ️ О системе")
        st.info("CreditScore PRO использует машинное обучение для оценки кредитных заявок на основе исторических данных.")
        
        st.markdown("---")
        st.write("📊 **Метрики модели:**")
        st.metric("Точность", f"{accuracy:.2%}")
        st.metric("F1-score", f"{f1:.2%}")
        
        st.markdown("---")
        st.write("**📌 Как использовать:**")
        st.write("1. Введите данные клиента")
        st.write("2. Нажмите 'Оценить заявку'")
        st.write("3. Просмотрите результат и рекомендации")
        
        st.markdown("---")
        st.write("**🔍 Важные факторы для одобрения:**")
        st.write("- Хорошая кредитная история")
        st.write("- Низкое отношение кредита к доходу")
        st.write("- Отсутствие дефолтов")
        st.write("- Достаточный опыт работы")
        st.write("- Адекватная процентная ставка")


# --- ТОЧКА ВХОДА ---
if __name__ == "__main__":
    main()
