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
        # 데이터 미리보기
        st.subheader("1️⃣ 데이터 미리보기")
        st.dataframe(df.head())
        
        # 변수 타입 구분
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # 변수 구분 표시
        st.subheader("2️⃣ 변수 타입 구분")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 수치형 변수")
            st.info(f"**총 {len(numeric_cols)}개**")
            if numeric_cols:
                with st.expander("📋 수치형 변수 목록 보기"):
                    # 데이터프레임 형식으로 표시
                    numeric_df = pd.DataFrame({
                        '변수명': numeric_cols
                    })
                    st.dataframe(numeric_df, use_container_width=True, height=300)
            else:
                st.write("수치형 변수가 없습니다.")
        
        with col2:
            st.markdown("### 🏷️ 명목형 변수")
            st.info(f"**총 {len(categorical_cols)}개**")
            if categorical_cols:
                with st.expander("📋 명목형 변수 목록 보기"):
                    # 데이터프레임 형식으로 표시
                    categorical_df = pd.DataFrame({
                        '변수명': categorical_cols
                    })
                    st.dataframe(categorical_df, use_container_width=True, height=300)
            else:
                st.write("명목형 변수가 없습니다.")
        
        # 수치형 변수 분석
        if numeric_cols:
            st.subheader("3️⃣ 수치형 변수 분석")
            
            # 기본 통계
            st.markdown("#### 📈 기본 통계량")
            st.write(df[numeric_cols].describe())
            
            # 왜도와 첨도
            st.markdown("#### 📐 왜도(Skewness)와 첨도(Kurtosis)")
            skew_kurt_df = pd.DataFrame({
                '변수명': numeric_cols,
                '왜도': [df[col].skew() for col in numeric_cols],
                '첨도': [df[col].kurtosis() for col in numeric_cols]
            })
            st.dataframe(skew_kurt_df)
            
            st.caption("💡 왜도: 0에 가까우면 대칭, 양수면 오른쪽 꼬리, 음수면 왼쪽 꼬리")
            st.caption("💡 첨도: 3에 가까우면 정규분포, 3보다 크면 뾰족, 작으면 평평")
            
            # 분포 시각화
            st.markdown("#### 3️⃣ 분포 시각화")
            
            # 변수 선택
            selected_num_var = st.selectbox("시각화할 수치형 변수 선택", numeric_cols)
            
            # 히스토그램과 박스플롯
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
        
        # 명목형 변수 분석
        if categorical_cols:
            st.subheader("4️⃣ 명목형 변수 분석")
            st.info(f"총 {len(categorical_cols)}개의 명목형 변수가 있습니다. 변수를 선택하여 상세 정보를 확인하세요.")
            
            # 1단계: 명목형 변수 목록만 표시
            with st.expander("🏷️ 명목형 변수 선택", expanded=False):
                # 각 변수를 개별 expander로 표시 (2단계)
                for col in categorical_cols:
                    with st.expander(f"📌 {col}"):
                        unique_values = df[col].unique()
                        value_counts = df[col].value_counts()
                        
                        col_info1, col_info2 = st.columns(2)
                        
                        with col_info1:
                            st.metric("고유값 개수", len(unique_values))
                            st.metric("결측치", df[col].isnull().sum())
                        
                        with col_info2:
                            st.markdown("**고유값 목록:**")
                            # 고유값을 expander로 감싸기
                            with st.expander("고유값 전체 보기"):
                                # 고유값도 데이터프레임 형식으로 표시
                                unique_df = pd.DataFrame({
                                    '번호': range(1, len(unique_values) + 1),
                                    '고유값': unique_values
                                })
                                st.dataframe(unique_df, use_container_width=True, height=200)
                        
                        st.markdown("**빈도표:**")
                        st.dataframe(value_counts.reset_index().rename(columns={col: '빈도', 'index': col}))
                        
                        # 빈도 시각화
                        if len(value_counts) <= 20:  # 고유값이 20개 이하일 때만 시각화
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
        
        # 데이터 정보
        st.subheader("5️⃣ 데이터 전체 정보")
        
        # 기본 정보
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("행 개수", f"{df.shape[0]:,}")
        with col2:
            st.metric("열 개수", df.shape[1])
        with col3:
            st.metric("중복 행", df.duplicated().sum())
        with col4:
            # 데이터 메모리 사용량
            memory_usage = df.memory_usage(deep=True).sum() / 1024**2  # MB 단위
            st.metric("메모리 사용량", f"{memory_usage:.2f} MB")
        
        # 결측치 상세 분석
        st.markdown("#### 🔍 결측치 상세 분석")
        
        # 공백 및 특수 결측치 탐지 함수
        def detect_missing_values(df):
            """다양한 형태의 결측치 탐지"""
            missing_info = []
            
            for col in df.columns:
                # 기본 결측치 (NaN, None)
                null_count = df[col].isnull().sum()
                
                # 문자열 컬럼인 경우 추가 검사
                if df[col].dtype == 'object':
                    # 빈 문자열 ''
                    empty_string = (df[col] == '').sum()
                    # 공백만 있는 문자열 (스페이스, 탭 등)
                    whitespace = df[col].apply(lambda x: isinstance(x, str) and x.strip() == '').sum()
                    # 'NA', 'N/A', 'null', 'NULL', 'None' 등
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
            st.dataframe(missing_df, use_container_width=True)
            
            # 결측치 시각화
            with st.expander("📊 결측치 시각화"):
                fig, ax = plt.subplots(figsize=(12, max(6, len(missing_df) * 0.4)))
                missing_df_sorted = missing_df.sort_values('총 결측치', ascending=True)
                
                ax.barh(missing_df_sorted['변수명'], missing_df_sorted['총 결측치'])
                ax.set_xlabel('결측치 개수')
                ax.set_ylabel('변수명')
                ax.set_title('변수별 결측치 분포', fontweight='bold', fontsize=14)
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.success("✅ 결측치가 없습니다!")
        
        # 데이터 타입 정보
        st.markdown("#### 📋 데이터 타입 정보")
        with st.expander("데이터 타입 상세 보기"):
            dtype_df = pd.DataFrame({
                '변수명': df.columns,
                '데이터 타입': df.dtypes.values,
                '고유값 개수': [df[col].nunique() for col in df.columns],
                '샘플 데이터': [str(df[col].dropna().iloc[0])[:50] if len(df[col].dropna()) > 0 else 'N/A' for col in df.columns]
            })
            st.dataframe(dtype_df, use_container_width=True)
            
else:
    st.info("👆 CSV 또는 Excel 파일을 업로드하면 분석이 시작됩니다.")