import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


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
        # CSV 파일 읽기
        df = pd.read_csv(uploaded_file)
        st.success("✅ CSV 파일 업로드 완료!")
        
    elif file_extension in ['xlsx', 'xls']:
        # Excel 파일의 시트명 가져오기
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        
        st.info(f"📋 사용 가능한 시트: {', '.join(sheet_names)}")
        
        # 시트명 선택
        selected_sheet = st.selectbox("분석할 시트를 선택하세요", sheet_names)
        
        # 선택한 시트 읽기
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
        st.success(f"✅ Excel 파일 업로드 완료! (시트: {selected_sheet})")
    
    # 데이터가 로드되었을 때만 분석 표시
    if df is not None:
        # 변수 타입 구분 (전역적으로 사용)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 사이드바 네비게이션 (체크박스로 다중 선택)
        st.sidebar.title("📋 분석 메뉴")
        st.sidebar.markdown("---")
        st.sidebar.info("💡 원하는 분석 항목을 여러 개 선택할 수 있습니다.")
        
        # 체크박스로 다중 선택
        show_preview = st.sidebar.checkbox("1️⃣ 데이터 미리보기", value=True)
        show_data_info = st.sidebar.checkbox("2️⃣ 데이터 전체 정보", value=False)
        show_variable_types = st.sidebar.checkbox("3️⃣ 변수 타입 구분", value=False)
        show_variable_analysis = st.sidebar.checkbox("4️⃣ 변수 분석 (수치형/명목형)", value=False)
        show_correlation = st.sidebar.checkbox("5️⃣ 상관관계 분석 (히트맵)", value=False)
        show_outlier_analysis = st.sidebar.checkbox("6️⃣ 이상치 탐지 및 분석", value=False)
        show_missing_analysis = st.sidebar.checkbox("7️⃣ 결측치 상세 분석", value=False)
        show_quality_report = st.sidebar.checkbox("8️⃣ 데이터 품질 리포트", value=False)
        show_target_analysis = st.sidebar.checkbox("9️⃣ 타겟 변수 분석", value=False)
        
        st.sidebar.markdown("---")
        
        # ============================================
        # 1️⃣ 데이터 미리보기
        # ============================================
        if show_preview:
            st.header("1️⃣ 데이터 미리보기")
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("### 📊 기본 정보")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("전체 행 수", f"{df.shape[0]:,}")
            with col2:
                st.metric("전체 열 수", df.shape[1])
            with col3:
                memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                st.metric("메모리 사용량", f"{memory_usage:.2f} MB")
            
            st.markdown("---")
        
        # ============================================
        # 2️⃣ 데이터 전체 정보
        # ============================================
        if show_data_info:
            st.header("2️⃣ 데이터 전체 정보")
            
            # 기본 정보
            st.markdown("### 📊 기본 통계")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("행 개수", f"{df.shape[0]:,}")
            with col2:
                st.metric("열 개수", df.shape[1])
            with col3:
                st.metric("중복 행", df.duplicated().sum())
            with col4:
                memory_usage = df.memory_usage(deep=True).sum() / 1024**2
                st.metric("메모리 사용량", f"{memory_usage:.2f} MB")
            
            # 데이터 타입 정보
            st.markdown("### 📋 데이터 타입 정보")
            dtype_df = pd.DataFrame({
                '변수명': df.columns,
                '데이터 타입': df.dtypes.values,
                '고유값 개수': [df[col].nunique() for col in df.columns],
                '결측치': [df[col].isnull().sum() for col in df.columns],
                '샘플 데이터': [str(df[col].dropna().iloc[0])[:50] if len(df[col].dropna()) > 0 else 'N/A' for col in df.columns]
            })
            st.dataframe(dtype_df, use_container_width=True, height=400)
            
            st.markdown("---")
        
        # ============================================
        # 3️⃣ 변수 타입 구분
        # ============================================
        if show_variable_types:
            st.header("3️⃣ 변수 타입 구분")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📊 수치형 변수")
                st.info(f"**총 {len(numeric_cols)}개**")
                if numeric_cols:
                    with st.expander("📋 수치형 변수 목록 보기", expanded=True):
                        numeric_df = pd.DataFrame({
                            '번호': range(1, len(numeric_cols) + 1),
                            '변수명': numeric_cols
                        })
                        st.dataframe(numeric_df, use_container_width=True, height=400)
                else:
                    st.write("수치형 변수가 없습니다.")
            
            with col2:
                st.markdown("### 🏷️ 명목형 변수")
                st.info(f"**총 {len(categorical_cols)}개**")
                if categorical_cols:
                    with st.expander("📋 명목형 변수 목록 보기", expanded=True):
                        categorical_df = pd.DataFrame({
                            '번호': range(1, len(categorical_cols) + 1),
                            '변수명': categorical_cols
                        })
                        st.dataframe(categorical_df, use_container_width=True, height=400)
                else:
                    st.write("명목형 변수가 없습니다.")
            
            st.markdown("---")
        
        # ============================================
        # 4️⃣ 변수 분석 (수치형/명목형)
        # ============================================
        if show_variable_analysis:
            st.header("4️⃣ 변수 분석 (수치형/명목형)")
            
            # 하위 메뉴 선택
            analysis_type = st.radio(
                "분석할 변수 유형을 선택하세요",
                ["📊 수치형 변수 분석", "🏷️ 명목형 변수 분석", "📈 전체 변수 분석"],
                horizontal=True
            )
            
            st.markdown("---")
            
            # 수치형 변수 분석
            if analysis_type in ["📊 수치형 변수 분석", "📈 전체 변수 분석"]:
                st.markdown("## 📊 수치형 변수 분석")
                
                if numeric_cols:
                    # 기본 통계
                    with st.expander("📈 기본 통계량", expanded=True):
                        st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                    
                    # 왜도와 첨도 (개선됨)
                    with st.expander("📐 왜도(Skewness)와 첨도(Kurtosis)", expanded=False):
                        skew_kurt_df = pd.DataFrame({
                            '변수명': numeric_cols,
                            '왜도': [df[col].skew() for col in numeric_cols],
                            '첨도': [df[col].kurtosis() for col in numeric_cols]
                        })
                        st.dataframe(skew_kurt_df, use_container_width=True)
                        
                        st.caption("💡 왜도: 0에 가까우면 대칭, 양수면 오른쪽 꼬리, 음수면 왼쪽 꼬리")
                        st.caption("💡 첨도: 3에 가까우면 정규분포, 3보다 크면 뾰족, 작으면 평평")
                        
                        # 왜도 또는 첨도가 3 이상인 변수 찾기
                        skewed_vars = skew_kurt_df[(abs(skew_kurt_df['왜도']) >= 3) | (abs(skew_kurt_df['첨도']) >= 3)]
                        
                        if len(skewed_vars) > 0:
                            st.markdown("---")
                            st.markdown("### ⚠️ 비정규성이 의심되는 변수 (왜도 또는 첨도 ≥ 3)")
                            st.dataframe(skewed_vars, use_container_width=True)
                            
                            # Shapiro-Wilk Test
                            st.markdown("### 🔬 정규성 검정 (Shapiro-Wilk Test)")
                            
                            shapiro_results = []
                            for var in skewed_vars['변수명']:
                                if len(df[var].dropna()) <= 5000:
                                    stat, p_value = stats.shapiro(df[var].dropna())
                                    shapiro_results.append({
                                        '변수명': var,
                                        'W 통계량': f"{stat:.4f}",
                                        'p-value': f"{p_value:.6f}",
                                        '정규성': '정규분포 ✅' if p_value > 0.05 else '비정규분포 ❌'
                                    })
                                else:
                                    shapiro_results.append({
                                        '변수명': var,
                                        'W 통계량': 'N/A',
                                        'p-value': 'N/A',
                                        '정규성': '데이터 > 5000개'
                                    })
                            
                            shapiro_df = pd.DataFrame(shapiro_results)
                            st.dataframe(shapiro_df, use_container_width=True)
                            st.caption("💡 p-value > 0.05: 정규분포를 따른다고 볼 수 있음")
                            
                            # 변환 분석
                            st.markdown("### 🔄 데이터 변환 후 왜도/첨도 변화")
                            
                            transformation_results = []
                            
                            for var in skewed_vars['변수명']:
                                # 원본
                                original_skew = df[var].skew()
                                original_kurt = df[var].kurtosis()
                                
                                # 로그 변환 (양수 값만)
                                if (df[var] > 0).all():
                                    log_data = np.log(df[var])
                                    log_skew = log_data.skew()
                                    log_kurt = log_data.kurtosis()
                                else:
                                    log_skew = np.nan
                                    log_kurt = np.nan
                                
                                # 제곱근 변환 (음수 아닌 값만)
                                if (df[var] >= 0).all():
                                    sqrt_data = np.sqrt(df[var])
                                    sqrt_skew = sqrt_data.skew()
                                    sqrt_kurt = sqrt_data.kurtosis()
                                else:
                                    sqrt_skew = np.nan
                                    sqrt_kurt = np.nan
                                
                                # Box-Cox 변환 (양수 값만)
                                if (df[var] > 0).all():
                                    try:
                                        boxcox_data, _ = stats.boxcox(df[var].dropna())
                                        boxcox_skew = pd.Series(boxcox_data).skew()
                                        boxcox_kurt = pd.Series(boxcox_data).kurtosis()
                                    except:
                                        boxcox_skew = np.nan
                                        boxcox_kurt = np.nan
                                else:
                                    boxcox_skew = np.nan
                                    boxcox_kurt = np.nan
                                
                                transformation_results.append({
                                    '변수명': var,
                                    '원본_왜도': f"{original_skew:.3f}",
                                    '원본_첨도': f"{original_kurt:.3f}",
                                    '로그_왜도': f"{log_skew:.3f}" if not np.isnan(log_skew) else 'N/A',
                                    '로그_첨도': f"{log_kurt:.3f}" if not np.isnan(log_kurt) else 'N/A',
                                    '제곱근_왜도': f"{sqrt_skew:.3f}" if not np.isnan(sqrt_skew) else 'N/A',
                                    '제곱근_첨도': f"{sqrt_kurt:.3f}" if not np.isnan(sqrt_kurt) else 'N/A',
                                    'BoxCox_왜도': f"{boxcox_skew:.3f}" if not np.isnan(boxcox_skew) else 'N/A',
                                    'BoxCox_첨도': f"{boxcox_kurt:.3f}" if not np.isnan(boxcox_kurt) else 'N/A'
                                })
                            
                            transform_df = pd.DataFrame(transformation_results)
                            st.dataframe(transform_df, use_container_width=True, height=400)
                            
                            st.caption("💡 왜도/첨도가 0에 가까울수록 정규분포에 가까움")
                            st.caption("💡 N/A: 해당 변환을 적용할 수 없음 (음수/0 값 포함)")
                            
                            # 권장 변환 방법
                            st.markdown("### 💡 권장 변환 방법")
                            for _, row in transform_df.iterrows():
                                var_name = row['변수명']
                                
                                # 각 변환의 왜도 절대값 계산
                                scores = {}
                                if row['로그_왜도'] != 'N/A':
                                    scores['로그 변환'] = abs(float(row['로그_왜도']))
                                if row['제곱근_왜도'] != 'N/A':
                                    scores['제곱근 변환'] = abs(float(row['제곱근_왜도']))
                                if row['BoxCox_왜도'] != 'N/A':
                                    scores['Box-Cox 변환'] = abs(float(row['BoxCox_왜도']))
                                
                                if scores:
                                    best_transform = min(scores, key=scores.get)
                                    st.success(f"**{var_name}**: {best_transform} 권장 (왜도: {scores[best_transform]:.3f})")
                                else:
                                    st.warning(f"**{var_name}**: 변환 불가 (음수/0 값 포함)")
                        else:
                            st.success("✅ 모든 수치형 변수의 왜도와 첨도가 정상 범위입니다!")
                    
                    # 분포 시각화
                    with st.expander("📊 분포 시각화", expanded=False):
                        selected_num_var = st.selectbox("시각화할 수치형 변수 선택", numeric_cols)
                        
                        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
                        
                        # 히스토그램
                        axes[0].hist(df[selected_num_var].dropna(), bins=30, edgecolor='black', alpha=0.7)
                        axes[0].set_title(f'{selected_num_var} - 히스토그램', fontsize=14, fontweight='bold')
                        axes[0].set_xlabel(selected_num_var)
                        axes[0].set_ylabel('빈도')
                        axes[0].grid(axis='y', alpha=0.3)
                        
                        # 박스플롯
                        axes[1].boxplot(df[selected_num_var].dropna(), vert=True)
                        axes[1].set_title(f'{selected_num_var} - 박스플롯', fontsize=14, fontweight='bold')
                        axes[1].set_ylabel(selected_num_var)
                        axes[1].grid(axis='y', alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                else:
                    st.warning("⚠️ 수치형 변수가 없습니다.")
                
                if analysis_type == "📈 전체 변수 분석":
                    st.markdown("---")
            
            # 명목형 변수 분석
            if analysis_type in ["🏷️ 명목형 변수 분석", "📈 전체 변수 분석"]:
                st.markdown("## 🏷️ 명목형 변수 분석")
                
                if categorical_cols:
                    st.info(f"총 {len(categorical_cols)}개의 명목형 변수가 있습니다. 변수를 선택하여 상세 정보를 확인하세요.")
                    
                    for col in categorical_cols:
                        with st.expander(f"📌 {col}", expanded=False):
                            unique_values = df[col].unique()
                            value_counts = df[col].value_counts()
                            
                            col_info1, col_info2 = st.columns(2)
                            
                            with col_info1:
                                st.metric("고유값 개수", len(unique_values))
                                st.metric("결측치", df[col].isnull().sum())
                            
                            with col_info2:
                                st.markdown("**고유값 목록:**")
                                with st.expander("고유값 전체 보기"):
                                    unique_df = pd.DataFrame({
                                        '번호': range(1, len(unique_values) + 1),
                                        '고유값': unique_values
                                    })
                                    st.dataframe(unique_df, use_container_width=True, height=200)
                            
                            st.markdown("**빈도표:**")
                            st.dataframe(value_counts.reset_index().rename(columns={col: '빈도', 'index': col}))
                            
                            # 빈도 시각화
                            if len(value_counts) <= 20:
                                fig, ax = plt.subplots(figsize=(10, max(5, len(value_counts) * 0.3)))
                                value_counts.plot(kind='barh', ax=ax, color='skyblue', edgecolor='black')
                                ax.set_xlabel('빈도')
                                ax.set_ylabel(col)
                                ax.set_title(f'{col} - 빈도 분포', fontweight='bold', fontsize=12)
                                ax.grid(axis='x', alpha=0.3)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            else:
                                st.caption("💡 고유값이 20개를 초과하여 시각화를 생략합니다.")
                else:
                    st.warning("⚠️ 명목형 변수가 없습니다.")
            
            st.markdown("---")
        
        # ============================================
        # 5️⃣ 상관관계 분석 (히트맵)
        # ============================================
        if show_correlation:
            st.header("5️⃣ 상관관계 분석 (히트맵)")
            
            if numeric_cols and len(numeric_cols) >= 2:
                st.info("💡 분석할 수치형 변수를 선택하세요. 최소 2개 이상 선택해야 합니다.")
                
                # 변수 선택 (데이터프레임 형식)
                st.markdown("### 📋 변수 선택")
                
                # 전체 선택/해제 버튼
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button("✅ 전체 선택"):
                        st.session_state.selected_corr_vars = numeric_cols.copy()
                with col_btn2:
                    if st.button("❌ 전체 해제"):
                        st.session_state.selected_corr_vars = []
                
                # 세션 스테이트 초기화
                if 'selected_corr_vars' not in st.session_state:
                    st.session_state.selected_corr_vars = numeric_cols.copy() if len(numeric_cols) <= 10 else numeric_cols[:10]
                
                # 변수 선택 데이터프레임
                var_selection_df = pd.DataFrame({
                    '선택': [var in st.session_state.selected_corr_vars for var in numeric_cols],
                    '변수명': numeric_cols,
                    '평균': [df[col].mean() for col in numeric_cols],
                    '표준편차': [df[col].std() for col in numeric_cols]
                })
                
                # 데이터프레임 편집 (체크박스)
                edited_df = st.data_editor(
                    var_selection_df,
                    column_config={
                        "선택": st.column_config.CheckboxColumn(
                            "선택",
                            help="분석에 포함할 변수를 선택하세요",
                            default=False,
                        )
                    },
                    disabled=["변수명", "평균", "표준편차"],
                    hide_index=True,
                    use_container_width=True,
                    height=300
                )
                
                # 선택된 변수 업데이트
                selected_vars = edited_df[edited_df['선택'] == True]['변수명'].tolist()
                
                if len(selected_vars) >= 2:
                    st.success(f"✅ {len(selected_vars)}개 변수 선택됨")
                    
                    # 상관계수 계산
                    correlation_matrix = df[selected_vars].corr()
                    
                    # 히트맵 시각화
                    st.markdown("### 📊 상관관계 히트맵")
                    
                    fig, ax = plt.subplots(figsize=(max(10, len(selected_vars) * 0.8), max(8, len(selected_vars) * 0.7)))
                    
                    # 히트맵 그리기
                    sns.heatmap(
                        correlation_matrix,
                        annot=True,
                        fmt='.2f',
                        cmap='coolwarm',
                        center=0,
                        square=True,
                        linewidths=0.5,
                        cbar_kws={"shrink": 0.8},
                        vmin=-1, vmax=1,
                        ax=ax
                    )
                    
                    ax.set_title('변수 간 상관관계 히트맵', fontsize=16, fontweight='bold', pad=20)
                    plt.xticks(rotation=45, ha='right')
                    plt.yticks(rotation=0)
                    plt.tight_layout()
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # 상관계수 매트릭스 테이블
                    st.markdown("### 📋 상관계수 매트릭스")
                    st.dataframe(
                        correlation_matrix.style.background_gradient(cmap='coolwarm', vmin=-1, vmax=1).format("{:.3f}"),
                        use_container_width=True
                    )
                    
                    # 강한 상관관계 찾기
                    st.markdown("### 🔍 주요 상관관계")
                    
                    corr_pairs = []
                    for i in range(len(correlation_matrix.columns)):
                        for j in range(i+1, len(correlation_matrix.columns)):
                            corr_pairs.append({
                                '변수1': correlation_matrix.columns[i],
                                '변수2': correlation_matrix.columns[j],
                                '상관계수': correlation_matrix.iloc[i, j]
                            })
                    
                    corr_pairs_df = pd.DataFrame(corr_pairs)
                    corr_pairs_df['상관계수_abs'] = corr_pairs_df['상관계수'].abs()
                    corr_pairs_df = corr_pairs_df.sort_values('상관계수_abs', ascending=False)
                    
                    top_corr = corr_pairs_df.head(min(10, len(corr_pairs_df)))
                    
                    st.dataframe(
                        top_corr[['변수1', '변수2', '상관계수']].style.background_gradient(
                            subset=['상관계수'], cmap='coolwarm', vmin=-1, vmax=1
                        ).format({'상관계수': '{:.3f}'}),
                        use_container_width=True
                    )
                    
                    st.caption("💡 상관계수 해석: 1에 가까우면 강한 양의 상관관계, -1에 가까우면 강한 음의 상관관계, 0에 가까우면 상관관계 없음")
                    
                elif len(selected_vars) == 1:
                    st.warning("⚠️ 최소 2개 이상의 변수를 선택해주세요.")
                else:
                    st.info("📌 변수를 선택해주세요.")
                    
            else:
                st.warning("⚠️ 상관관계 분석을 위해서는 최소 2개 이상의 수치형 변수가 필요합니다.")
            
            st.markdown("---")
        
        # ... (6️⃣, 7️⃣는 이전과 동일) ...
        
        # ============================================
        # 8️⃣ 데이터 품질 리포트 (개선됨)
        # ============================================
        if show_quality_report:
            st.header("8️⃣ 데이터 품질 리포트")
            
            # 계산 방법 설명
            with st.expander("📖 데이터 품질 점수 계산 방법", expanded=False):
                st.markdown("""
                ### 📊 품질 점수 계산 기준
                
                #### **수치형 변수 (int64, float64)**
                
                **총점 = 결측치 점수(40%) + 이상치 점수(30%) + 고유값 점수(30%)**
                
                | 항목 | 가중치 | 계산 방법 |
                |-----|--------|----------|
                | **결측치 점수** | 40% | `100 - (결측치 비율 × 100)` |
                | **이상치 점수** | 30% | `100 - (이상치 비율 × 100)` |
                | **고유값 점수** | 30% | 고유값 비율에 따라 차등 부여 |
                
                **고유값 점수 세부:**
                - 고유값 < 1%: **50점** (상수 변수 의심)
                - 고유값 > 95%: **80점** (ID 변수 의심)
                - 그 외: **100점** (정상)
                
                **이상치 탐지:** IQR 방법 (Q1 - 1.5×IQR ~ Q3 + 1.5×IQR 범위 벗어난 값)
                
                ---
                
                #### **명목형 변수 (object, category)**
                
                **총점 = 결측치 점수(50%) + 공백 문자열 점수(30%) + 고유값 점수(20%)**
                
                | 항목 | 가중치 | 계산 방법 |
                |-----|--------|----------|
                | **결측치 점수** | 50% | `100 - (결측치 비율 × 100)` |
                | **공백 문자열 점수** | 30% | `100 - (빈 문자열 + 공백 비율 × 100)` |
                | **고유값 점수** | 20% | 고유값 비율에 따라 차등 부여 |
                
                **고유값 점수 세부:**
                - 고유값 < 1%: **60점** (카테고리가 너무 적음)
                - 고유값 > 50%: **70점** (카테고리가 너무 많음)
                - 그 외: **100점** (정상)
                
                ---
                
                #### **등급 분류**
                
                | 등급 | 점수 범위 | 의미 |
                |-----|----------|------|
                | **우수** | 90점 이상 | 품질이 매우 좋음 |
                | **양호** | 70~89점 | 품질이 양호함 |
                | **보통** | 50~69점 | 일부 개선 필요 |
                | **개선 필요** | 50점 미만 | 심각한 품질 문제 |
                """)
            
            # 품질 점수 계산 함수
            def calculate_quality_score(df):
                scores = []
                
                for col in df.columns:
                    # 결측치 비율
                    missing_ratio = df[col].isnull().sum() / len(df)
                    missing_score = max(0, 100 - missing_ratio * 100)
                    
                    # 고유값 비율
                    unique_ratio = df[col].nunique() / len(df)
                    
                    # 수치형 변수
                    if df[col].dtype in ['int64', 'float64']:
                        # 이상치 비율 (IQR)
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        outlier_ratio = len(df[(df[col] < lower_bound) | (df[col] > upper_bound)]) / len(df)
                        outlier_score = max(0, 100 - outlier_ratio * 100)
                        
                        # 고유값 점수 (너무 적거나 많으면 감점)
                        if unique_ratio < 0.01:  # 고유값이 1% 미만
                            unique_score = 50
                        elif unique_ratio > 0.95:  # 고유값이 95% 초과 (거의 모두 다른 값)
                            unique_score = 80
                        else:
                            unique_score = 100
                        
                        total_score = (missing_score * 0.4 + outlier_score * 0.3 + unique_score * 0.3)
                        
                    # 명목형 변수
                    else:
                        # 공백/특수문자 비율
                        empty_ratio = ((df[col] == '').sum() + 
                                     df[col].apply(lambda x: isinstance(x, str) and x.strip() == '').sum()) / len(df)
                        empty_score = max(0, 100 - empty_ratio * 100)
                        
                        # 고유값 점수
                        if unique_ratio < 0.01:
                            unique_score = 60
                        elif unique_ratio > 0.5:
                            unique_score = 70
                        else:
                            unique_score = 100
                        
                        total_score = (missing_score * 0.5 + empty_score * 0.3 + unique_score * 0.2)
                    
                    scores.append({
                        '변수명': col,
                        '데이터 타입': str(df[col].dtype),
                        '결측치 비율(%)': f"{missing_ratio * 100:.2f}",
                        '고유값 비율(%)': f"{unique_ratio * 100:.2f}",
                        '품질 점수': round(total_score, 1),
                        '등급': '우수' if total_score >= 90 else '양호' if total_score >= 70 else '보통' if total_score >= 50 else '개선 필요'
                    })
                
                return pd.DataFrame(scores)
            
            quality_df = calculate_quality_score(df)
            
            # 전체 품질 점수
            overall_score = quality_df['품질 점수'].mean()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("전체 품질 점수", f"{overall_score:.1f}점")
            with col2:
                excellent_count = len(quality_df[quality_df['등급'] == '우수'])
                st.metric("우수 등급 변수", f"{excellent_count}개")
            with col3:
                needs_improvement = len(quality_df[quality_df['등급'] == '개선 필요'])
                st.metric("개선 필요 변수", f"{needs_improvement}개")
            with col4:
                avg_missing = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
                st.metric("평균 결측 비율", f"{avg_missing:.2f}%")
            
            # 품질 리포트 테이블
            st.markdown("### 📊 변수별 품질 평가")
            
            # 색상 함수
            def color_grade(val):
                if val == '우수':
                    return 'background-color: #d4edda'
                elif val == '양호':
                    return 'background-color: #d1ecf1'
                elif val == '보통':
                    return 'background-color: #fff3cd'
                else:
                    return 'background-color: #f8d7da'
            
            styled_df = quality_df.style.applymap(color_grade, subset=['등급'])
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # 등급별 분포
            st.markdown("### 📈 품질 등급 분포")
            grade_counts = quality_df['등급'].value_counts()
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = {'우수': '#28a745', '양호': '#17a2b8', '보통': '#ffc107', '개선 필요': '#dc3545'}
            grade_counts.plot(kind='bar', ax=ax, color=[colors.get(x, 'gray') for x in grade_counts.index])
            ax.set_title('데이터 품질 등급 분포', fontweight='bold', fontsize=14)
            ax.set_xlabel('등급')
            ax.set_ylabel('변수 개수')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
            
            # 개선 권장사항
            if needs_improvement > 0:
                st.markdown("### 💡 개선 권장사항")
                improvement_df = quality_df[quality_df['등급'] == '개선 필요']
                
                for _, row in improvement_df.iterrows():
                    with st.expander(f"⚠️ {row['변수명']} (점수: {row['품질 점수']}점)"):
                        st.write(f"**데이터 타입:** {row['데이터 타입']}")
                        st.write(f"**결측치 비율:** {row['결측치 비율(%)']}%")
                        st.write(f"**고유값 비율:** {row['고유값 비율(%)']}%")
                        st.write("**권장 조치:**")
                        
                        missing_ratio = float(row['결측치 비율(%)'])
                        if missing_ratio > 30:
                            st.write("- 결측치가 30%를 초과합니다. 결측치 처리 또는 변수 제거를 고려하세요.")
                        elif missing_ratio > 10:
                            st.write("- 결측치가 10%를 초과합니다. 적절한 대체 방법을 검토하세요.")
                        
                        unique_ratio = float(row['고유값 비율(%)'])
                        if unique_ratio < 1:
                            st.write("- 고유값이 매우 적습니다. 상수 변수일 가능성이 있으니 확인하세요.")
                        elif unique_ratio > 95:
                            st.write("- 고유값이 너무 많습니다. ID 변수거나 분석에 적합하지 않을 수 있습니다.")
            
            st.markdown("---")
        
        # ... (9️⃣는 이전과 동일) ...
            
else:
    st.info("👆 CSV 또는 Excel 파일을 업로드하면 분석이 시작됩니다.")