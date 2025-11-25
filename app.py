import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(page_title="Data Consultant AI", page_icon="📊", layout="wide")

# 타이틀
st.title("📊 Data Consultant AI")
st.write("CSV 또는 Excel 파일을 업로드하면 EDA, 데이터 진단, 전처리 방식, 분석방법론을 추천합니다.")

# 파일 업로드
uploaded_file = st.file_uploader("CSV 또는 Excel 파일 업로드", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # 파일 확장자 확인
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    df = None
    
    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV 파일 업로드 완료!")
        
    elif file_extension in ['xlsx', 'xls']:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        
        st.info(f"📋 사용 가능한 시트: {', '.join(sheet_names)}")
        selected_sheet = st.selectbox("분석할 시트를 선택하세요", sheet_names)
        
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        st.success(f"✅ Excel 파일 업로드 완료! (시트: {selected_sheet})")
    
    if df is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        st.sidebar.title("📋 분석 메뉴")
        st.sidebar.markdown("---")
        st.sidebar.info("💡 원하는 분석 항목을 여러 개 선택할 수 있습니다.")
        
        show_preview = st.sidebar.checkbox("1️⃣ 데이터 미리보기", value=True)
        show_data_info = st.sidebar.checkbox("2️⃣ 데이터 전체 정보", value=False)
        show_variable_types = st.sidebar.checkbox("3️⃣ 변수 타입 구분", value=False)
        show_variable_analysis = st.sidebar.checkbox("4️⃣ 변수 분석 (수치형/명목형)", value=False)
        show_correlation = st.sidebar.checkbox("5️⃣ 상관관계 분석 (히트맵)", value=False)
        show_outlier_analysis = st.sidebar.checkbox("6️⃣ 이상치 탐지 및 분석", value=False)
        show_missing_analysis = st.sidebar.checkbox("7️⃣ 결측치 상세 분석", value=False)
        show_quality_report = st.sidebar.checkbox("8️⃣ 데이터 품질 리포트", value=False)
        show_target_analysis = st.sidebar.checkbox("9️⃣ 타겟 변수 분석 & 모델링", value=False)
        
        st.sidebar.markdown("---")
        
        # 1️⃣ ~ 5️⃣는 이전 코드 그대로 유지 (생략)
        # ... 이전 코드 ...
        
        # ============================================
        # 6️⃣ 이상치 탐지 및 분석
        # ============================================
        if show_outlier_analysis:
            st.header("6️⃣ 이상치 탐지 및 분석")
            
            if numeric_cols:
                st.info("💡 수치형 변수의 이상치를 IQR과 Z-score 방법으로 탐지합니다.")
                
                outlier_method = st.radio(
                    "이상치 탐지 방법 선택",
                    ["📊 IQR 방법", "📈 Z-score 방법", "🔍 두 방법 모두"],
                    horizontal=True
                )
                
                st.markdown("---")
                
                # IQR 방법
                if outlier_method in ["📊 IQR 방법", "🔍 두 방법 모두"]:
                    with st.expander("📊 IQR 방법 (사분위수 범위)", expanded=True):
                        st.markdown("### IQR 방법으로 이상치 탐지")
                        st.caption("💡 IQR = Q3 - Q1, 이상치 = Q1 - 1.5*IQR 미만 또는 Q3 + 1.5*IQR 초과")
                        
                        iqr_outliers = []
                        for col in numeric_cols:
                            Q1 = df[col].quantile(0.25)
                            Q3 = df[col].quantile(0.75)
                            IQR = Q3 - Q1
                            lower_bound = Q1 - 1.5 * IQR
                            upper_bound = Q3 + 1.5 * IQR
                            
                            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                            outlier_count = len(outliers)
                            outlier_ratio = (outlier_count / len(df)) * 100
                            
                            if outlier_count > 0:
                                iqr_outliers.append({
                                    '변수명': col,
                                    '이상치 개수': outlier_count,
                                    '이상치 비율(%)': f"{outlier_ratio:.2f}",
                                    '하한선': f"{lower_bound:.2f}",
                                    '상한선': f"{upper_bound:.2f}",
                                    'Q1': f"{Q1:.2f}",
                                    'Q3': f"{Q3:.2f}",
                                    'IQR': f"{IQR:.2f}"
                                })
                        
                        if iqr_outliers:
                            iqr_df = pd.DataFrame(iqr_outliers)
                            st.dataframe(iqr_df, use_container_width=True)
                            
                            # 시각화
                            st.markdown("#### 📊 변수별 이상치 비율")
                            
                            plt.rcParams['font.family'] = 'Malgun Gothic'
                            plt.rcParams['axes.unicode_minus'] = False
                            
                            fig, ax = plt.subplots(figsize=(10, max(5, len(iqr_outliers) * 0.4)))
                            iqr_df['이상치 비율'] = iqr_df['이상치 비율(%)'].astype(float)
                            iqr_df_sorted = iqr_df.sort_values('이상치 비율', ascending=True)
                            
                            ax.barh(iqr_df_sorted['변수명'], iqr_df_sorted['이상치 비율'], color='coral')
                            ax.set_xlabel('이상치 비율 (%)')
                            ax.set_ylabel('변수명')
                            ax.set_title('IQR 방법: 변수별 이상치 비율', fontweight='bold', fontsize=14)
                            ax.grid(axis='x', alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.success("✅ IQR 방법으로 탐지된 이상치가 없습니다!")
                
                # Z-score 방법
                if outlier_method in ["📈 Z-score 방법", "🔍 두 방법 모두"]:
                    with st.expander("📈 Z-score 방법 (표준화 점수)", expanded=True):
                        st.markdown("### Z-score 방법으로 이상치 탐지")
                        st.caption("💡 Z-score = (X - 평균) / 표준편차, 이상치 = |Z-score| > 임계값")
                        
                        z_threshold = st.slider("Z-score 임계값 입력", min_value=2.0, max_value=5.0, value=3.0, step=0.1)
                        
                        z_outliers = []
                        for col in numeric_cols:
                            mean = df[col].mean()
                            std = df[col].std()
                            
                            if std > 0:
                                z_scores = np.abs((df[col] - mean) / std)
                                outliers = df[z_scores > z_threshold]
                                outlier_count = len(outliers)
                                outlier_ratio = (outlier_count / len(df)) * 100
                                
                                if outlier_count > 0:
                                    z_outliers.append({
                                        '변수명': col,
                                        '이상치 개수': outlier_count,
                                        '이상치 비율(%)': f"{outlier_ratio:.2f}",
                                        '평균': f"{mean:.2f}",
                                        '표준편차': f"{std:.2f}",
                                        'Z-score 임계값': z_threshold
                                    })
                        
                        if z_outliers:
                            z_df = pd.DataFrame(z_outliers)
                            st.dataframe(z_df, use_container_width=True)
                            
                            st.markdown("#### 📈 변수별 이상치 비율")
                            
                            plt.rcParams['font.family'] = 'Malgun Gothic'
                            plt.rcParams['axes.unicode_minus'] = False
                            
                            fig, ax = plt.subplots(figsize=(10, max(5, len(z_outliers) * 0.4)))
                            z_df['이상치 비율'] = z_df['이상치 비율(%)'].astype(float)
                            z_df_sorted = z_df.sort_values('이상치 비율', ascending=True)
                            
                            ax.barh(z_df_sorted['변수명'], z_df_sorted['이상치 비율'], color='steelblue')
                            ax.set_xlabel('이상치 비율 (%)')
                            ax.set_ylabel('변수명')
                            ax.set_title(f'Z-score 방법: 변수별 이상치 비율 (임계값: {z_threshold})', fontweight='bold', fontsize=14)
                            ax.grid(axis='x', alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.success(f"✅ Z-score 방법 (임계값: {z_threshold})으로 탐지된 이상치가 없습니다!")
                
                # 이상치 상세 시각화
                with st.expander("🔍 이상치 상세 시각화", expanded=False):
                    selected_outlier_var = st.selectbox("시각화할 변수 선택", numeric_cols, key="outlier_viz")
                    
                    plt.rcParams['font.family'] = 'Malgun Gothic'
                    plt.rcParams['axes.unicode_minus'] = False
                    
                    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                    
                    axes[0].boxplot(df[selected_outlier_var].dropna())
                    axes[0].set_title(f'{selected_outlier_var} - 박스플롯', fontweight='bold')
                    axes[0].set_ylabel(selected_outlier_var)
                    axes[0].grid(axis='y', alpha=0.3)
                    
                    axes[1].hist(df[selected_outlier_var].dropna(), bins=30, edgecolor='black', alpha=0.7)
                    axes[1].set_title(f'{selected_outlier_var} - 히스토그램', fontweight='bold')
                    axes[1].set_xlabel(selected_outlier_var)
                    axes[1].set_ylabel('빈도')
                    axes[1].grid(axis='y', alpha=0.3)
                    
                    axes[2].scatter(df.index, df[selected_outlier_var], alpha=0.5, s=10)
                    axes[2].set_title(f'{selected_outlier_var} - 산점도', fontweight='bold')
                    axes[2].set_xlabel('인덱스')
                    axes[2].set_ylabel(selected_outlier_var)
                    axes[2].grid(alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
            else:
                st.warning("⚠️ 수치형 변수가 없어 이상치 분석을 수행할 수 없습니다.")
            
            st.markdown("---")
        
        # ============================================
        # 7️⃣ 결측치 상세 분석
        # ============================================
        if show_missing_analysis:
            st.header("7️⃣ 결측치 상세 분석")
            
            def detect_missing_values(df):
                missing_info = []
                
                for col in df.columns:
                    null_count = df[col].isnull().sum()
                    
                    if df[col].dtype == 'object':
                        empty_string = (df[col] == '').sum()
                        whitespace = df[col].apply(lambda x: isinstance(x, str) and x.strip() == '').sum()
                        na_strings = df[col].isin(['NA', 'N/A', 'na', 'n/a', 'null', 'NULL', 'None', 'none', '-', '?']).sum()
                        
                        total_missing = null_count + empty_string + whitespace + na_strings
                        
                        if total_missing > 0:
                            missing_info.append({
                                '변수명': col,
                                '결측치(NaN)': null_count,
                                '빈 문자열': empty_string,
                                '공백 문자열': whitespace,
                                'NA 문자열': na_strings,
                                '총 결측치': total_missing,
                                '결측 비율(%)': f"{(total_missing / len(df)) * 100:.2f}"
                            })
                    else:
                        if null_count > 0:
                            missing_info.append({
                                '변수명': col,
                                '결측치(NaN)': null_count,
                                '빈 문자열': 0,
                                '공백 문자열': 0,
                                'NA 문자열': 0,
                                '총 결측치': null_count,
                                '결측 비율(%)': f"{(null_count / len(df)) * 100:.2f}"
                            })
                
                return pd.DataFrame(missing_info)
            
            missing_df = detect_missing_values(df)
            
            if len(missing_df) > 0:
                st.warning(f"⚠️ {len(missing_df)}개 변수에서 결측치가 발견되었습니다.")
                
                st.markdown("### 📊 결측치 요약 테이블")
                st.dataframe(missing_df, use_container_width=True)
                
                st.markdown("### 📈 결측치 시각화")
                
                plt.rcParams['font.family'] = 'Malgun Gothic'
                plt.rcParams['axes.unicode_minus'] = False
                
                fig, ax = plt.subplots(figsize=(12, max(6, len(missing_df) * 0.4)))
                missing_df_sorted = missing_df.sort_values('총 결측치', ascending=True)
                
                bars = ax.barh(missing_df_sorted['변수명'], missing_df_sorted['총 결측치'], color='salmon')
                ax.set_xlabel('결측치 개수')
                ax.set_ylabel('변수명')
                ax.set_title('변수별 결측치 분포', fontweight='bold', fontsize=14)
                ax.grid(axis='x', alpha=0.3)
                
                for bar, value in zip(bars, missing_df_sorted['총 결측치']):
                    ax.text(value + 0.5, bar.get_y() + bar.get_height()/2,
                            str(value),
                            va='center', fontsize=10, color='black')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            else:
                st.success("✅ 결측치가 없습니다!")
            
            st.markdown("---")
        
        # 8️⃣는 이전 코드 그대로 (생략)
        # ...
        
        # ============================================
        # 9️⃣ 타겟 변수 분석 & 모델링
        # ============================================
        if show_target_analysis:
            st.header("9️⃣ 타겟 변수 분석 & 모델링")
            
            st.info("💡 분석할 타겟 변수와 독립 변수를 선택하여 회귀 또는 분류 모델을 학습합니다.")
            
            # 변수 목록 표시
            st.markdown("### 📋 전체 변수 목록")
            variable_info = []
            for col in df.columns:
                variable_info.append({
                    '변수명': col,
                    '데이터 타입': str(df[col].dtype),
                    '고유값 개수': df[col].nunique(),
                    '결측치': df[col].isnull().sum(),
                    '결측 비율(%)': f"{(df[col].isnull().sum() / len(df)) * 100:.2f}"
                })
            
            var_df = pd.DataFrame(variable_info)
            st.dataframe(var_df, use_container_width=True, height=300)
            
            st.markdown("---")
            
            # 타겟 변수 선택
            st.markdown("### 🎯 1단계: 타겟 변수 선택")
            target_var = st.selectbox("예측하고자 하는 타겟 변수를 선택하세요", df.columns.tolist())
            
            if target_var:
                # 타겟 변수 타입 판단
                is_numeric = df[target_var].dtype in ['int64', 'float64']
                unique_count = df[target_var].nunique()
                
                # 문제 유형 결정
                if is_numeric and unique_count <= 20:
                    problem_type = st.radio(
                        "문제 유형을 선택하세요",
                        ["회귀 (Regression)", "분류 (Classification)"],
                        horizontal=True,
                        help="고유값이 20개 이하인 수치형 변수는 분류로 간주할 수도 있습니다."
                    )
                    is_classification = (problem_type == "분류 (Classification)")
                else:
                    is_classification = not is_numeric
                    if is_classification:
                        st.success("🏷️ **분류(Classification) 문제**로 자동 판단되었습니다.")
                    else:
                        st.success("📊 **회귀(Regression) 문제**로 자동 판단되었습니다.")
                
                st.markdown("---")
                
                # 독립 변수 선택
                st.markdown("### 📊 2단계: 독립 변수 선택")
                
                available_features = [col for col in df.columns if col != target_var]
                
                # 수치형만 사용할지, 전체 사용할지 선택
                feature_option = st.radio(
                    "독립 변수 선택 옵션",
                    ["수치형 변수만 사용", "전체 변수 사용 (명목형 자동 인코딩)"],
                    horizontal=True
                )
                
                if feature_option == "수치형 변수만 사용":
                    selectable_features = [col for col in available_features if col in numeric_cols]
                else:
                    selectable_features = available_features
                
                if not selectable_features:
                    st.error("⚠️ 선택 가능한 독립 변수가 없습니다.")
                else:
                    # 변수 선택 데이터프레임
                    feature_selection_df = pd.DataFrame({
                        '선택': [False] * len(selectable_features),
                        '변수명': selectable_features,
                        '데이터 타입': [str(df[col].dtype) for col in selectable_features],
                        '고유값': [df[col].nunique() for col in selectable_features]
                    })
                    
                    col_btn1, col_btn2 = st.columns(2)
                    with col_btn1:
                        select_all = st.button("✅ 전체 선택")
                    with col_btn2:
                        deselect_all = st.button("❌ 전체 해제")
                    
                    if select_all:
                        feature_selection_df['선택'] = True
                    elif deselect_all:
                        feature_selection_df['선택'] = False
                    
                    edited_features = st.data_editor(
                        feature_selection_df,
                        column_config={
                            "선택": st.column_config.CheckboxColumn("선택", default=False)
                        },
                        disabled=["변수명", "데이터 타입", "고유값"],
                        hide_index=True,
                        use_container_width=True,
                        height=300
                    )
                    
                    selected_features = edited_features[edited_features['선택'] == True]['변수명'].tolist()
                    
                    if len(selected_features) == 0:
                        st.warning("⚠️ 최소 1개 이상의 독립 변수를 선택해주세요.")
                    else:
                        st.success(f"✅ {len(selected_features)}개 변수 선택됨: {', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}")
                        
                        st.markdown("---")
                        
                        # 모델 학습 옵션
                        st.markdown("### ⚙️ 3단계: 모델 학습 설정")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            test_size = st.slider("테스트 데이터 비율", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
                        with col2:
                            random_state = st.number_input("랜덤 시드", min_value=0, max_value=999, value=42)
                        
                        # 모델 학습 버튼
                        if st.button("🚀 모델 학습 시작", type="primary"):
                            with st.spinner("모델을 학습하고 있습니다..."):
                                try:
                                    # 데이터 준비
                                    df_model = df[[target_var] + selected_features].copy()
                                    df_model = df_model.dropna()
                                    
                                    if len(df_model) < 10:
                                        st.error("⚠️ 결측치 제거 후 데이터가 너무 적습니다 (최소 10개 필요).")
                                    else:
                                        X = df_model[selected_features].copy()
                                        y = df_model[target_var].copy()
                                        
                                        # 명목형 변수 인코딩
                                        categorical_features = [col for col in selected_features if col in categorical_cols]
                                        if categorical_features:
                                            st.info(f"🔄 명목형 변수 인코딩 중: {', '.join(categorical_features)}")
                                            for col in categorical_features:
                                                le = LabelEncoder()
                                                X[col] = le.fit_transform(X[col].astype(str))
                                        
                                        # 분류 문제: 타겟도 인코딩
                                        if is_classification and df[target_var].dtype == 'object':
                                            le_target = LabelEncoder()
                                            y = le_target.fit_transform(y)
                                        
                                        # 데이터 분할
                                        X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, test_size=test_size, random_state=random_state
                                        )
                                        
                                        st.success(f"✅ 학습 데이터: {len(X_train)}개, 테스트 데이터: {len(X_test)}개")
                                        
                                        # 회귀 모델
                                        if not is_classification:
                                            st.markdown("---")
                                            st.markdown("## 📊 회귀 모델 결과")
                                            
                                            tab1, tab2 = st.tabs(["선형 회귀", "랜덤 포레스트 회귀"])
                                            
                                            with tab1:
                                                st.markdown("### 🔹 선형 회귀 (Linear Regression)")
                                                
                                                # 스케일링
                                                scaler = StandardScaler()
                                                X_train_scaled = scaler.fit_transform(X_train)
                                                X_test_scaled = scaler.transform(X_test)
                                                
                                                # 모델 학습
                                                lr = LinearRegression()
                                                lr.fit(X_train_scaled, y_train)
                                                
                                                # 예측
                                                y_pred_train = lr.predict(X_train_scaled)
                                                y_pred_test = lr.predict(X_test_scaled)
                                                
                                                # 성능 지표
                                                col1, col2, col3, col4 = st.columns(4)
                                                with col1:
                                                    st.metric("Train R²", f"{r2_score(y_train, y_pred_train):.4f}")
                                                with col2:
                                                    st.metric("Test R²", f"{r2_score(y_test, y_pred_test):.4f}")
                                                with col3:
                                                    st.metric("Train RMSE", f"{np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
                                                with col4:
                                                    st.metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
                                                
                                                # 계수 시각화
                                                st.markdown("#### 📈 변수 중요도 (회귀 계수)")
                                                coef_df = pd.DataFrame({
                                                    '변수명': selected_features,
                                                    '계수': lr.coef_
                                                }).sort_values('계수', key=abs, ascending=False)
                                                
                                                plt.rcParams['font.family'] = 'Malgun Gothic'
                                                plt.rcParams['axes.unicode_minus'] = False
                                                
                                                fig, ax = plt.subplots(figsize=(10, max(5, len(coef_df) * 0.3)))
                                                colors = ['green' if x > 0 else 'red' for x in coef_df['계수']]
                                                ax.barh(coef_df['변수명'], coef_df['계수'], color=colors, alpha=0.7)
                                                ax.set_xlabel('계수')
                                                ax.set_title('선형 회귀 계수', fontweight='bold')
                                                ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
                                                ax.grid(axis='x', alpha=0.3)
                                                plt.tight_layout()
                                                st.pyplot(fig)
                                                plt.close()
                                                
                                                # 예측 vs 실제
                                                st.markdown("#### 🎯 예측값 vs 실제값")
                                                fig, ax = plt.subplots(figsize=(8, 6))
                                                ax.scatter(y_test, y_pred_test, alpha=0.5)
                                                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                                                ax.set_xlabel('실제값')
                                                ax.set_ylabel('예측값')
                                                ax.set_title('선형 회귀: 예측 vs 실제', fontweight='bold')
                                                ax.grid(alpha=0.3)
                                                plt.tight_layout()
                                                st.pyplot(fig)
                                                plt.close()
                                            
                                            with tab2:
                                                st.markdown("### 🌲 랜덤 포레스트 회귀 (Random Forest)")
                                                
                                                # 모델 학습
                                                rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
                                                rf.fit(X_train, y_train)
                                                
                                                # 예측
                                                y_pred_train_rf = rf.predict(X_train)
                                                y_pred_test_rf = rf.predict(X_test)
                                                
                                                # 성능 지표
                                                col1, col2, col3, col4 = st.columns(4)
                                                with col1:
                                                    st.metric("Train R²", f"{r2_score(y_train, y_pred_train_rf):.4f}")
                                                with col2:
                                                    st.metric("Test R²", f"{r2_score(y_test, y_pred_test_rf):.4f}")
                                                with col3:
                                                    st.metric("Train RMSE", f"{np.sqrt(mean_squared_error(y_train, y_pred_train_rf)):.4f}")
                                                with col4:
                                                    st.metric("Test RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred_test_rf)):.4f}")
                                                
                                                # 특성 중요도
                                                st.markdown("#### 📈 변수 중요도 (Feature Importance)")
                                                importance_df = pd.DataFrame({
                                                    '변수명': selected_features,
                                                    '중요도': rf.feature_importances_
                                                }).sort_values('중요도', ascending=False)
                                                
                                                fig, ax = plt.subplots(figsize=(10, max(5, len(importance_df) * 0.3)))
                                                ax.barh(importance_df['변수명'], importance_df['중요도'], color='forestgreen', alpha=0.7)
                                                ax.set_xlabel('중요도')
                                                ax.set_title('랜덤 포레스트 변수 중요도', fontweight='bold')
                                                ax.grid(axis='x', alpha=0.3)
                                                plt.tight_layout()
                                                st.pyplot(fig)
                                                plt.close()
                                                
                                                # 예측 vs 실제
                                                st.markdown("#### 🎯 예측값 vs 실제값")
                                                fig, ax = plt.subplots(figsize=(8, 6))
                                                ax.scatter(y_test, y_pred_test_rf, alpha=0.5, color='green')
                                                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                                                ax.set_xlabel('실제값')
                                                ax.set_ylabel('예측값')
                                                ax.set_title('랜덤 포레스트: 예측 vs 실제', fontweight='bold')
                                                ax.grid(alpha=0.3)
                                                plt.tight_layout()
                                                st.pyplot(fig)
                                                plt.close()
                                        
                                        # 분류 모델
                                        else:
                                            st.markdown("---")
                                            st.markdown("## 🏷️ 분류 모델 결과")
                                            
                                            tab1, tab2 = st.tabs(["로지스틱 회귀", "랜덤 포레스트 분류"])
                                            
                                            with tab1:
                                                st.markdown("### 🔹 로지스틱 회귀 (Logistic Regression)")
                                                
                                                # 스케일링
                                                scaler = StandardScaler()
                                                X_train_scaled = scaler.fit_transform(X_train)
                                                X_test_scaled = scaler.transform(X_test)
                                                
                                                # 모델 학습
                                                logit = LogisticRegression(max_iter=1000, random_state=random_state)
                                                logit.fit(X_train_scaled, y_train)
                                                
                                                # 예측
                                                y_pred_train = logit.predict(X_train_scaled)
                                                y_pred_test = logit.predict(X_test_scaled)
                                                
                                                # 성능 지표
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.metric("Train Accuracy", f"{accuracy_score(y_train, y_pred_train):.4f}")
                                                with col2:
                                                    st.metric("Test Accuracy", f"{accuracy_score(y_test, y_pred_test):.4f}")
                                                
                                                # 분류 리포트
                                                st.markdown("#### 📊 분류 성능 리포트")
                                                st.text(classification_report(y_test, y_pred_test))
                                                
                                                # 혼동 행렬
                                                st.markdown("#### 🎯 혼동 행렬 (Confusion Matrix)")
                                                cm = confusion_matrix(y_test, y_pred_test)
                                                
                                                fig, ax = plt.subplots(figsize=(8, 6))
                                                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                                                ax.set_xlabel('예측 클래스')
                                                ax.set_ylabel('실제 클래스')
                                                ax.set_title('로지스틱 회귀: 혼동 행렬', fontweight='bold')
                                                plt.tight_layout()
                                                st.pyplot(fig)
                                                plt.close()
                                            
                                            with tab2:
                                                st.markdown("### 🌲 랜덤 포레스트 분류 (Random Forest)")
                                                
                                                # 모델 학습
                                                rf_clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
                                                rf_clf.fit(X_train, y_train)
                                                
                                                # 예측
                                                y_pred_train_rf = rf_clf.predict(X_train)
                                                y_pred_test_rf = rf_clf.predict(X_test)
                                                
                                                # 성능 지표
                                                col1, col2 = st.columns(2)
                                                with col1:
                                                    st.metric("Train Accuracy", f"{accuracy_score(y_train, y_pred_train_rf):.4f}")
                                                with col2:
                                                    st.metric("Test Accuracy", f"{accuracy_score(y_test, y_pred_test_rf):.4f}")
                                                
                                                # 분류 리포트
                                                st.markdown("#### 📊 분류 성능 리포트")
                                                st.text(classification_report(y_test, y_pred_test_rf))
                                                
                                                # 특성 중요도
                                                st.markdown("#### 📈 변수 중요도 (Feature Importance)")
                                                importance_df = pd.DataFrame({
                                                    '변수명': selected_features,
                                                    '중요도': rf_clf.feature_importances_
                                                }).sort_values('중요도', ascending=False)
                                                
                                                fig, ax = plt.subplots(figsize=(10, max(5, len(importance_df) * 0.3)))
                                                ax.barh(importance_df['변수명'], importance_df['중요도'], color='forestgreen', alpha=0.7)
                                                ax.set_xlabel('중요도')
                                                ax.set_title('랜덤 포레스트 변수 중요도', fontweight='bold')
                                                ax.grid(axis='x', alpha=0.3)
                                                plt.tight_layout()
                                                st.pyplot(fig)
                                                plt.close()
                                                
                                                # 혼동 행렬
                                                st.markdown("#### 🎯 혼동 행렬 (Confusion Matrix)")
                                                cm = confusion_matrix(y_test, y_pred_test_rf)
                                                
                                                fig, ax = plt.subplots(figsize=(8, 6))
                                                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
                                                ax.set_xlabel('예측 클래스')
                                                ax.set_ylabel('실제 클래스')
                                                ax.set_title('랜덤 포레스트: 혼동 행렬', fontweight='bold')
                                                plt.tight_layout()
                                                st.pyplot(fig)
                                                plt.close()
                                        
                                except Exception as e:
                                    st.error(f"⚠️ 모델 학습 중 오류 발생: {str(e)}")
                                    st.exception(e)
            
            st.markdown("---")

else:
    st.info("👆 CSV 또는 Excel 파일을 업로드하면 분석이 시작됩니다.")