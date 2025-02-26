import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Konfigurasi halaman
st.set_page_config(
    page_title="Analisis E-commerce LGCM",
    page_icon="📊",
    layout="wide"
)

# Fungsi untuk menangani missing values
def handle_missing_values(df):
    initial_missing = df.isnull().sum()
    has_missing = initial_missing.any()
    
    if has_missing:
        # Simpan informasi tentang missing values
        missing_info = pd.DataFrame({
            'Kolom': initial_missing.index,
            'Jumlah Missing': initial_missing.values,
            'Persentase': (initial_missing / len(df) * 100).round(2)
        })
        
        # Isi missing values dengan mean
        df_cleaned = df.fillna(df.mean())
        
        return True, df_cleaned, missing_info
    
    return False, df, None

# Fungsi untuk mendeteksi dan menangani outliers menggunakan Z-score
def handle_outliers(df, handle=False, threshold=3):
    outliers_info = {}
    df_cleaned = df.copy()
    
    for column in df.columns:
        # Hitung z-scores
        z_scores = stats.zscore(df[column])
        outliers_mask = abs(z_scores) > threshold
        outliers = df[column][outliers_mask]
        
        if not outliers.empty:
            mean_val = df[column].mean()
            std_val = df[column].std()
            lower_bound = mean_val - threshold * std_val
            upper_bound = mean_val + threshold * std_val
            
            outliers_info[column] = {
                'jumlah': len(outliers),
                'nilai': outliers.values,
                'batas_bawah': lower_bound,
                'batas_atas': upper_bound
            }
            
            if handle:
                # Ganti outliers dengan mean
                df_cleaned.loc[outliers_mask, column] = mean_val
    
    has_outliers = len(outliers_info) > 0
    return has_outliers, df_cleaned, outliers_info

# Fungsi untuk mengecek normalitas dan melakukan normalisasi
def check_and_transform_distribution(df):
    normality_results = {}
    df_normalized = df.copy()
    scaler = MinMaxScaler()
    
    for column in df.columns:
        # Uji Shapiro-Wilk untuk normalitas
        statistic, p_value = stats.shapiro(df[column])
        is_normal = p_value > 0.05
        
        normality_results[column] = {
            'is_normal': is_normal,
            'p_value': p_value,
            'statistic': statistic
        }
        
        # Jika tidak normal, lakukan normalisasi
        if not is_normal:
            df_normalized[column] = scaler.fit_transform(df[column].values.reshape(-1, 1)).flatten()
    
    needs_transform = any(not result['is_normal'] for result in normality_results.values())
    return needs_transform, df_normalized, normality_results

# LGCM Model Definition
LGCM_MODEL = """
# Measurement model untuk variabel manifest dan laten
Harga =~ Harga1 + Harga2 + Harga3
Kualitas =~ Kualitas1 + Kualitas2 + Kualitas3
Loyalitas =~ Loyalitas1 + Loyalitas2 + Loyalitas3
# Definisi variabel intercept dan slope
Intercept_Harga =~ 1*Harga1 + 1*Harga2 + 1*Harga3
Slope_Harga =~ 0*Harga1 + 1*Harga2 + 2*Harga3
Intercept_Kualitas =~ 1*Kualitas1 + 1*Kualitas2 + 1*Kualitas3
Slope_Kualitas =~ 0*Kualitas1 + 1*Kualitas2 + 2*Kualitas3
Intercept_Loyalitas =~ 1*Loyalitas1 + 1*Loyalitas2 + 1*Loyalitas3
Slope_Loyalitas =~ 0*Loyalitas1 + 1*Loyalitas2 + 2*Loyalitas3
# Pengaruh inovasi terhadap intercept dan slope
Intercept_Harga ~ Inovasi
Slope_Harga ~ Inovasi
Intercept_Kualitas ~ Inovasi
Slope_Kualitas ~ Inovasi
Intercept_Loyalitas ~ Inovasi
Slope_Loyalitas ~ Inovasi
# Menambahkan kovarians antar intercept dan slope
Intercept_Harga ~~ Slope_Harga
Intercept_Kualitas ~~ Slope_Kualitas
Intercept_Loyalitas ~~ Slope_Loyalitas
# Menambahkan kovarians antar error term yang relevan
Harga1 ~~ Harga2
Kualitas1 ~~ Kualitas2
Loyalitas1 ~~ Loyalitas2
# Kovarians tambahan berdasarkan modification indices
Harga2 ~~ Kualitas2
Kualitas2 ~~ Loyalitas2
Harga3 ~~ Kualitas3
Kualitas3 ~~ Loyalitas3
"""

# Fungsi untuk memvalidasi dan memproses data CSV
def process_csv_data(df):
    required_columns = [
        'Harga1', 'Harga2', 'Harga3',
        'Kualitas1', 'Kualitas2', 'Kualitas3',
        'Loyalitas1', 'Loyalitas2', 'Loyalitas3',
        'Inovasi'
    ]
    
    # Validasi kolom
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        return False, f"Kolom yang diperlukan tidak ditemukan: {', '.join(missing_cols)}", None
    
    # Pre-processing data
    
    # 1. Handling Missing Values
    has_missing, df_no_missing, missing_info = handle_missing_values(df)
    
    # 2. Initial outlier detection (without handling)
    has_outliers, df_with_outliers, outliers_info = handle_outliers(df_no_missing, handle=False)
    
    # Store original data with outliers for later use if needed
    st.session_state['data_with_outliers'] = df_no_missing
    st.session_state['outliers_info'] = outliers_info
    st.session_state['has_outliers'] = has_outliers
    
    # Will be updated later based on user choice
    df_after_outliers = df_no_missing
    
    # 3. Check and Transform Distribution
    needs_transform, df_normalized, normality_results = check_and_transform_distribution(df_after_outliers)
    
    # Prepare preprocessing report
    preprocessing_report = {
        'missing_values': {'has_missing': has_missing, 'info': missing_info},
        'outliers': {'has_outliers': has_outliers, 'info': outliers_info, 'handled': False},
        'normality': {'needs_transform': needs_transform, 'info': normality_results, 'df_normalized': df_normalized}
    }
    
    # Reshape final processed data
    processed_data = pd.DataFrame({
        'Harga': [df_normalized['Harga1'].iloc[0], df_normalized['Harga2'].iloc[0], df_normalized['Harga3'].iloc[0]],
        'Kualitas': [df_normalized['Kualitas1'].iloc[0], df_normalized['Kualitas2'].iloc[0], df_normalized['Kualitas3'].iloc[0]],
        'Loyalitas': [df_normalized['Loyalitas1'].iloc[0], df_normalized['Loyalitas2'].iloc[0], df_normalized['Loyalitas3'].iloc[0]]
    }, index=['Bulan Pertama', 'Bulan Kedua', 'Bulan Ketiga'])
    
    inovasi_value = df_normalized['Inovasi'].iloc[0]
    
    return True, "Data valid", (processed_data, inovasi_value, preprocessing_report)

# Fungsi untuk menghitung statistik deskriptif
def calculate_descriptive_stats(data):
    stats_df = pd.DataFrame({
        'Mean': data.mean(),
        'Std Dev': data.std(),
        'Min': data.min(),
        'Max': data.max(),
        'Median': data.median()
    })
    return stats_df

# Fungsi untuk membuat visualisasi trajektori
def plot_trajectory(data, variable):
    fig = px.line(
        data, 
        x=['Bulan Pertama', 'Bulan Kedua', 'Bulan Ketiga'],
        y=variable,
        markers=True,
        title=f'Trajektori Pertumbuhan: {variable}'
    )
    fig.update_layout(
        xaxis_title="Waktu Pengukuran",
        yaxis_title="Nilai",
        showlegend=True
    )
    return fig

# Fungsi untuk estimasi parameter LGCM
def estimate_lgcm_parameters(data, covariate):
    time_points = len(data)
    mean_trajectory = data.mean()
    slope = np.polyfit(range(time_points), data, 1)[0]
    intercept = np.polyfit(range(time_points), data, 1)[1]
    
    covariate_correlation = np.corrcoef(data, [covariate] * time_points)[0, 1]
    
    predicted_values = intercept + slope * np.array(range(time_points))
    residuals = data - predicted_values
    error_variance = np.var(residuals)
    
    return {
        'intercept': intercept,
        'slope': slope,
        'error_variance': error_variance,
        'mean_trajectory': mean_trajectory,
        'covariate_correlation': covariate_correlation
    }

# Inisialisasi state untuk halaman
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Tampilkan homepage atau halaman analisis
if st.session_state.page == 'home':
    # Custom CSS untuk background gradient
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #EEF2FF 0%, #E0E7FF 100%);
        }
        .small-font {
            font-size: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Header
    st.markdown("""
        <h1 style='text-align: center; color: #312E81; margin-bottom: 2rem;'>
            🛍️ Analisis Faktor yang Berpengaruh Terhadap E-commerce menggunakan LGCM
        </h1>
    """, unsafe_allow_html=True)
    
    # Penjelasan Singkat
    st.markdown("""
        <p style='text-align: center; font-size: 1.2em; color: #1F2937; margin-bottom: 3rem;'>
            Platform analisis komprehensif untuk memahami faktor-faktor yang mempengaruhi
            keputusan pemilihan e-commerce dengan pendekatan Latent Growth Curve Modeling
        </p>
    """, unsafe_allow_html=True)
    
    # Fitur Utama dalam 3 kolom
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; height: 200px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                <h3 style='color: #4F46E5;'>📊 Pre-processing Data</h3>
                <p class='small-font'>Penanganan missing values, outliers, dan normalisasi data untuk analisis yang akurat</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; height: 200px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                <h3 style='color: #4F46E5;'>📈 Analisis Trajektori</h3>
                <p class='small-font'>Visualisasi dan analisis pola pertumbuhan dari waktu ke waktu</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div style='background-color: white; padding: 1.5rem; border-radius: 10px; height: 200px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
                <h3 style='color: #4F46E5;'>🧠 Model LGCM</h3>
                <p class='small-font'>Pemodelan pertumbuhan laten dengan kovariat inovasi</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Panduan Penggunaan
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
        <div style='background-color: white; padding: 2rem; border-radius: 10px; margin-top: 2rem; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);'>
            <h2 style='color: #312E81; margin-bottom: 1rem;'>📝 1. Format Data yang Dibutuhkan</h2>
            <p>File CSV Anda harus memiliki kolom-kolom berikut:</p>
            <ul>
                <li><strong>Harga:</strong> Harga1, Harga2, Harga3 (nilai 0-10)</li>
                <li><strong>Kualitas:</strong> Kualitas1, Kualitas2, Kualitas3 (nilai 0-10)</li>
                <li><strong>Loyalitas:</strong> Loyalitas1, Loyalitas2, Loyalitas3 (nilai 0-10)</li>
                <li><strong>Kovariat:</strong> Inovasi (nilai 0-10)</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

else:
    st.title("Analisis Faktor Pemilihan E-commerce menggunakan LGCM")
    st.markdown("""
    Aplikasi ini menganalisis faktor-faktor yang mempengaruhi pemilihan e-commerce 
    menggunakan Latent Growth Curve Modeling (LGCM) dengan Inovasi sebagai kovariat.
    """)

    # Upload data
    st.header("1. Upload Data")
    st.markdown("""
    Unggah file CSV dengan kolom berikut:
    - Harga1, Harga2, Harga3 (nilai 0-10)
    - Kualitas1, Kualitas2, Kualitas3 (nilai 0-10)
    - Loyalitas1, Loyalitas2, Loyalitas3 (nilai 0-10)
    - Inovasi (nilai 0-10)
    """)

# File uploader
uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])

if uploaded_file is not None:
    try:
        # Baca file CSV
        df = pd.read_csv(uploaded_file)
        
        # Proses dan validasi data
        is_valid, message, processed_data = process_csv_data(df)
        
        # preprocessing data
        if not is_valid:
            st.error(message)
            st.stop()
        else:
            df, inovasi_value, preprocessing_report = processed_data
            
            # Ambil df_normalized dari preprocessing_report
            df_normalized = preprocessing_report['normality']['df_normalized']
            
            # Definisikan normalization_columns
            normalization_columns = [
                'Harga1', 'Harga2', 'Harga3', 
                'Kualitas1', 'Kualitas2', 'Kualitas3', 
                'Loyalitas1', 'Loyalitas2', 'Loyalitas3', 
                'Inovasi'
            ]
            
            st.success("✅ Data berhasil diunggah dan divalidasi")
            
            # Menampilkan hasil pre-processing
            st.markdown("""
                <div style='background-color: #f8fafc; padding: 2rem; border-radius: 10px; margin: 1rem 0;'>
                    <h2 style='color: #1e40af; font-size: 1.8rem; margin-bottom: 1.5rem;'>
                        2️⃣ Hasil Pre-processing Data
                    </h2>
                </div>
            """, unsafe_allow_html=True)
            
            # 2.1 Missing Values
            st.markdown("""
                <div style='background-color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3 style='color: #3b82f6; margin-bottom: 1rem;'>2.1 Penanganan Missing Values</h3>
            """, unsafe_allow_html=True)
            
            if preprocessing_report['missing_values']['has_missing']:
                st.markdown("🔍 **Ditemukan missing values:**")
                st.dataframe(preprocessing_report['missing_values']['info'])
                st.info("💡 Missing values telah diisi dengan nilai mean")
            else:
                st.markdown("""
                    <div style='background-color: #ecfdf5; padding: 1rem; border-radius: 8px; border-left: 4px solid #059669;'>
                        ✅ Tidak ditemukan missing values
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # 2.2 Outliers
            st.markdown("""
                <div style='background-color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3 style='color: #3b82f6; margin-bottom: 1rem;'>2.2 Deteksi Outliers</h3>
            """, unsafe_allow_html=True)
            
            if preprocessing_report['outliers']['has_outliers']:
                st.markdown("🔍 **Ditemukan outliers:**")
                for col, info in preprocessing_report['outliers']['info'].items():
                    st.markdown(f"""
                        <div style='background-color: #fef2f2; padding: 1rem; border-radius: 8px; margin: 0.5rem 0;'>
                            <strong style='color: #dc2626;'>{col}:</strong>
                            <br>Jumlah outliers: {info['jumlah']}
                        </div>
                    """, unsafe_allow_html=True)
                
                handle_outliers_option = st.checkbox("🔧 Tangani outliers menggunakan metode Z-score", value=False)
                
                if handle_outliers_option:
                    _, df_no_outliers, _ = handle_outliers(st.session_state['data_with_outliers'], handle=True)
                    needs_transform, df_normalized, normality_results = check_and_transform_distribution(df_no_outliers)
                    
                    # Update processed data
                    df = pd.DataFrame({
                        'Harga': [df_normalized['Harga1'].iloc[0], df_normalized['Harga2'].iloc[0], df_normalized['Harga3'].iloc[0]],
                        'Kualitas': [df_normalized['Kualitas1'].iloc[0], df_normalized['Kualitas2'].iloc[0], df_normalized['Kualitas3'].iloc[0]],
                        'Loyalitas': [df_normalized['Loyalitas1'].iloc[0], df_normalized['Loyalitas2'].iloc[0], df_normalized['Loyalitas3'].iloc[0]]
                    }, index=['Bulan Pertama', 'Bulan Kedua', 'Bulan Ketiga'])
                    
                    st.markdown("""
                        <div style='background-color: #eff6ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6;'>
                            💡 Outliers telah ditangani dengan metode Z-score
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='background-color: #fff7ed; padding: 1rem; border-radius: 8px; border-left: 4px solid #ea580c;'>
                            ⚠️ Outliers terdeteksi tetapi tidak ditangani (sesuai pilihan user)
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div style='background-color: #ecfdf5; padding: 1rem; border-radius: 8px; border-left: 4px solid #059669;'>
                        ✅ Tidak ditemukan outliers
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # 2.3 Normalitas
            st.markdown("""
                <div style='background-color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3 style='color: #3b82f6; margin-bottom: 1rem;'>2.3 Uji Normalitas</h3>
            """, unsafe_allow_html=True)
            
            normality_df = pd.DataFrame({
                col: {
                    'Normal': 'Ya' if info['is_normal'] else 'Tidak',
                    'P-value': f"{info['p_value']:.4f}",
                    'Statistic': f"{info['statistic']:.4f}"
                }
                for col, info in preprocessing_report['normality']['info'].items()
            }).T
            
            st.dataframe(normality_df)
            
            if preprocessing_report['normality']['needs_transform']:
                st.markdown("""
                    <div style='background-color: #eff6ff; padding: 1rem; border-radius: 8px; border-left: 4px solid #3b82f6;'>
                        💡 Data yang tidak normal telah dinormalisasi menggunakan MinMaxScaler
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                    <h5 style='color: #1e40af; margin-top: 1rem;'>Data setelah normalisasi:</h5>
                """, unsafe_allow_html=True)
                
                df_normalized_full = df_normalized[normalization_columns]
                st.dataframe(df_normalized_full.head().round(6))
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Menampilkan data yang akan dianalisis
            st.markdown("""
                <div style='background-color: #f8fafc; padding: 2rem; border-radius: 10px; margin: 1rem 0;'>
                    <h2 style='color: #1e40af; font-size: 1.8rem; margin-bottom: 1.5rem;'>
                        3️⃣ Data yang Akan Dianalisis
                    </h2>
                </div>
            """, unsafe_allow_html=True)

            # Analisis Deskriptif
            st.markdown("""
                <div style='background-color: white; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>
                    <h3 style='color: #3b82f6; margin-bottom: 1rem;'>📊 Analisis Deskriptif</h3>
                    <p style='color: #4b5563; margin-bottom: 1rem;'>Berikut adalah ringkasan statistik untuk setiap variabel:</p>
            """, unsafe_allow_html=True)

            # Tentukan kolom yang akan dianalisis
            columns_to_analyze = [
                'Harga1', 'Harga2', 'Harga3', 
                'Kualitas1', 'Kualitas2', 'Kualitas3', 
                'Loyalitas1', 'Loyalitas2', 'Loyalitas3', 
                'Inovasi'
            ]

            # Buat DataFrame statistik deskriptif
            descriptive_df = pd.DataFrame({
                col: {
                    'Mean': df_normalized[col].mean(),
                    'Std Dev': df_normalized[col].std(),
                    'Min': df_normalized[col].min(),
                    'Median': df_normalized[col].median(),
                    'Max': df_normalized[col].max(),
                    'Skewness': df_normalized[col].skew(),
                    'Kurtosis': df_normalized[col].kurtosis()
                } for col in columns_to_analyze
            }).T.round(6)

            # Tampilkan DataFrame
            st.dataframe(descriptive_df)

            # Tambahkan penjelasan statistik
            st.markdown("""
                <div style='background-color: #f8fafc; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                    <h4 style='color: #1e40af; margin-bottom: 0.5rem;'>📝 Keterangan Statistik:</h4>
                    <ul style='color: #4b5563; margin-left: 1.5rem;'>
                        <li><strong>Mean:</strong> Rata-rata nilai dari setiap variabel</li>
                        <li><strong>Std Dev:</strong> Standar deviasi yang menunjukkan sebaran data</li>
                        <li><strong>Min/Max:</strong> Nilai minimum dan maksimum</li>
                        <li><strong>Median:</strong> Nilai tengah data</li>
                        <li><strong>Skewness:</strong> Ukuran kemiringan distribusi data</li>
                        <li><strong>Kurtosis:</strong> Ukuran keruncingan distribusi data</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Tombol Analisis
            analyze_button = st.button("🔍 Analisis Data")

            # Analisis Keputusan
            if analyze_button:
                # Create a custom container with gradient background
                st.markdown("""
                    <div style='background: linear-gradient(120deg, #E0F7FA 0%, #B2EBF2 100%); 
                                padding: 2rem; 
                                border-radius: 15px; 
                                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                                margin: 2rem 0;'>
                        <h1 style='color: #006064; text-align: center; font-size: 2.2rem; margin-bottom: 1.5rem;'>
                            📊 4. Analisis Keputusan
                        </h1>
                    </div>
                """, unsafe_allow_html=True)

                # 4.1 Identifikasi Variabel Pertumbuhan
                st.markdown("""
                    <div style='background: white; 
                                padding: 2rem; 
                                border-radius: 12px; 
                                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                                border-left: 5px solid #00BCD4;
                                margin: 1.5rem 0;'>
                        <h2 style='color: #00838F; margin-bottom: 1.5rem;'>
                            🎯 4.1 Identifikasi Variabel Pertumbuhan
                        </h2>
                    </div>
                """, unsafe_allow_html=True)

                # Calculate growth variables
                growth_variables = ['Harga', 'Kualitas', 'Loyalitas']
                growth_results = {}

                # Inovasi section with custom styling
                st.markdown("""
                    <div style='background: #E1F5FE; 
                                padding: 1.5rem; 
                                border-radius: 8px; 
                                margin: 1rem 0;'>
                        <p style='color: #0277BD; font-style: italic;'>
                            ℹ️ Catatan: Inovasi merupakan kovariat yang tidak memiliki pengukuran berulang
                        </p>
                    </div>
                """, unsafe_allow_html=True)

                # Divider with custom styling
                st.markdown("""
                    <div style='height: 3px; 
                                background: linear-gradient(90deg, #B2EBF2 0%, #80DEEA 50%, #B2EBF2 100%); 
                                margin: 2rem 0;'>
                    </div>
                """, unsafe_allow_html=True)

                # Variables with repeated measurements section
                st.markdown("""
                    <div style='background: #FFFFFF; 
                                padding: 1.5rem; 
                                border-radius: 10px; 
                                border: 1px solid #E0F7FA;
                                margin: 1rem 0;'>
                        <h3 style='color: #00838F; margin-bottom: 1rem;'>
                            📈 Variabel dengan Pengukuran Berulang:
                        </h3>
                """, unsafe_allow_html=True)

                # Calculate and display growth variables
                for var in growth_variables:
                    columns = [col for col in df_normalized.columns if col.startswith(var)]
                    timepoints = np.array([1, 2, 3]).reshape(-1, 1)
                    
                    intercepts = []
                    slopes = []
                    
                    for _, row in df_normalized[columns].iterrows():
                        model = LinearRegression()
                        model.fit(timepoints, row.values)
                        intercepts.append(model.intercept_)
                        slopes.append(model.coef_[0])
                    
                    # Display results with custom styling
                    st.markdown(f"""
                        <div style='background: #F5FCFD; 
                                    padding: 1.2rem; 
                                    border-radius: 8px; 
                                    margin: 0.8rem 0;
                                    border-left: 4px solid {
                                        "#00BCD4" if var == "Harga" else 
                                        "#4CAF50" if var == "Kualitas" else "#FF5722"
                                    };'>
                            <h4 style='color: #00695C; margin-bottom: 0.5rem;'>{var}:</h4>
                            <p style='color: #00838F; margin: 0.3rem 0;'>
                                <strong>Intercept:</strong> {np.mean(intercepts):.4f}
                            </p>
                            <p style='color: #00838F; margin: 0.3rem 0;'>
                                <strong>Slope:</strong> {np.mean(slopes):.4f}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)

                # 4.2 Visualisasi Trajektori Pertumbuhan yang Lebih Menarik
                st.markdown("""
                    <div style='
                        background: linear-gradient(135deg, #4CAF50 0%, #2E7D32 100%);
                        color: white;
                        padding: 2rem; 
                        border-radius: 15px; 
                        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
                        margin: 2rem 0;
                        display: flex;
                        align-items: center;
                    '>
                        <div style='
                            background: rgba(255,255,255,0.2); 
                            border-radius: 50%; 
                            width: 70px; 
                            height: 70px; 
                            display: flex; 
                            align-items: center; 
                            justify-content: center; 
                            margin-right: 20px;
                        '>
                            <span style='font-size: 2.5rem;'>📈</span>
                        </div>
                        <h2 style='
                            margin: 0; 
                            font-size: 2rem; 
                            text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
                        '>
                            4.2 Visualisasi Trajektori Pertumbuhan E-commerce
                        </h2>
                    </div>
                """, unsafe_allow_html=True)

                # Definisikan variabel sebelum digunakan
                variables = ['Harga', 'Kualitas', 'Loyalitas']

                try:
                    # Impor semopy dengan penanganan kesalahan yang lebih baik
                    try:
                        from semopy import Model
                    except ImportError:
                        st.error("""
                        <div style='
                            background-color: #ffebee; 
                            border-left: 5px solid #f44336; 
                            padding: 15px; 
                            border-radius: 5px;
                        '>
                            🚨 Paket 'semopy' tidak terinstall. 
                            Silakan install dengan menjalankan:
                            <code>pip install semopy</code>
                        </div>
                        """, unsafe_allow_html=True)
                        st.stop()

                    # Fit LGCM model and get estimates
                    model = Model(LGCM_MODEL)
                    model.fit(df_normalized)
                    estimates = pd.DataFrame(model.inspect())
                    
                    # Waktu yang digunakan (titik waktu)
                    timepoints = np.array([1, 2, 3])
                    
                    # Estimasi nilai intercept dan slope dari model
                    intercept_harga = estimates.loc[estimates['lval'] == 'Intercept_Harga', 'Estimate'].values[0]
                    slope_harga = estimates.loc[estimates['lval'] == 'Harga', 'Estimate'].values[0]
                    intercept_kualitas = estimates.loc[estimates['lval'] == 'Intercept_Kualitas', 'Estimate'].values[0]
                    slope_kualitas = estimates.loc[estimates['lval'] == 'Kualitas', 'Estimate'].values[0]
                    intercept_loyalitas = estimates.loc[estimates['lval'] == 'Intercept_Loyalitas', 'Estimate'].values[0]
                    slope_loyalitas = estimates.loc[estimates['lval'] == 'Loyalitas', 'Estimate'].values[0]
                    
                    # Menghitung trajektori untuk masing-masing variabel
                    traj_harga = intercept_harga + slope_harga * timepoints
                    traj_kualitas = intercept_kualitas + slope_kualitas * timepoints
                    traj_loyalitas = intercept_loyalitas + slope_loyalitas * timepoints
                    
                    # Definisikan trajectories untuk digunakan dalam interpretasi
                    trajectories = [traj_harga, traj_kualitas, traj_loyalitas]
                    
                    # Membuat plot menggunakan Plotly dengan desain yang lebih menarik
                    fig = go.Figure()
                    
                    # Palet warna yang lebih estetis
                    color_palette = {
                        'Harga': '#2196F3',     # Vibrant Blue
                        'Kualitas': '#4CAF50',  # Vivid Green
                        'Loyalitas': '#FF5722'  # Deep Orange
                    }
                    
                    # Plot untuk variabel Harga, Kualitas, dan Loyalitas
                    variables_data = [
                        ('Harga', traj_harga, color_palette['Harga']),
                        ('Kualitas', traj_kualitas, color_palette['Kualitas']),
                        ('Loyalitas', traj_loyalitas, color_palette['Loyalitas'])
                    ]
                    
                    for var, trajectory, color in variables_data:
                        fig.add_trace(go.Scatter(
                            x=timepoints,
                            y=trajectory,
                            mode='lines+markers+text',
                            name=var,
                            line=dict(color=color, width=4, dash='solid'),
                            marker=dict(
                                size=12,
                                color=color,
                                symbol='circle',
                                line=dict(color='white', width=2)
                            ),
                            text=[f'{val:.2f}' for val in trajectory],
                            textposition='top center',
                            hovertemplate=f'<b>{var}</b><br>Waktu: %{{x}}<br>Nilai: %{{y:.2f}}<extra></extra>'
                        ))
                    
                    # Update layout dengan desain modern
                    fig.update_layout(
                        title={
                            'text': "Trajektori Pertumbuhan Variabel E-commerce",
                            'y': 0.95,
                            'x': 0.5,
                            'xanchor': 'center',
                            'yanchor': 'top',
                            'font': dict(size=24, color='#2c3e50')
                        },
                        xaxis_title="Periode Waktu",
                        yaxis_title="Nilai Variabel",
                        xaxis=dict(
                            ticktext=['Bulan 1', 'Bulan 2', 'Bulan 3'],
                            tickvals=[1, 2, 3],
                            showgrid=True,
                            gridcolor='lightgray',
                            linecolor='gray',
                            linewidth=2,
                            tickcolor='gray'
                        ),
                        yaxis=dict(
                            showgrid=True,
                            gridcolor='lightgray',
                            linecolor='gray',
                            linewidth=2,
                            tickcolor='gray'
                        ),
                        plot_bgcolor='rgba(247, 249, 249, 0.9)',
                        paper_bgcolor='white',
                        height=600,
                        width=1000,
                        font=dict(family="Arial, sans-serif", size=14),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                            bgcolor='rgba(255,255,255,0.9)',
                            bordercolor='lightgray',
                            borderwidth=1
                        ),
                        hovermode='x unified'
                    )
                    
                    # Tampilkan plot
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabel Trajektori dengan Styling Gradient
                    st.markdown("### 📊 Rincian Trajektori")
                    trajectory_df = pd.DataFrame({
                        'Periode': ['Bulan 1', 'Bulan 2', 'Bulan 3'],
                        '💰 Harga': traj_harga,
                        '⭐ Kualitas': traj_kualitas,
                        '❤️ Loyalitas': traj_loyalitas
                    })
                    
                    # Styling tabel dengan gradient biru
                    st.dataframe(
                        trajectory_df.style.background_gradient(cmap='Blues', subset=['💰 Harga', '⭐ Kualitas', '❤️ Loyalitas'])
                    )
                                        
                except Exception as e:
                    # Penanganan kesalahan dengan desain yang lebih menarik
                    st.error(f"""
                    <div style='
                        background-color: #ffebee; 
                        border-left: 5px solid #f44336; 
                        padding: 15px; 
                        border-radius: 5px;
                    '>
                        🚨 Kesalahan dalam Visualisasi Trajektori: 
                        {str(e)}
                        
                        Pastikan:
                        - Model LGCM berhasil dijalankan
                        - Estimasi valid
                        - Paket semopy terinstall
                    </div>
                    """, unsafe_allow_html=True)

                # 4.3 Interpretasi Hasil dengan Desain Modern
                st.markdown("""
                    <div style='
                        background: linear-gradient(135deg, #4A90E2 0%, #50C878 100%);
                        color: white;
                        padding: 2rem; 
                        border-radius: 15px; 
                        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
                        margin: 2rem 0;
                        display: flex;
                        align-items: center;
                    '>
                        <div style='font-size: 3rem; margin-right: 1rem;'>🔍</div>
                        <div>
                            <h2 style='margin: 0; font-size: 2rem;'>4.3 Interpretasi Hasil</h2>
                            <p style='margin: 0; opacity: 0.8;'>Analisis Mendalam Model LGCM</p>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                try:
                    # Mengambil hasil estimasi dari model LGCM
                    estimates = model.inspect()
                    
                    # Definisi warna yang konsisten
                    color_palette = {
                        'Harga': '#2196F3',     # Vibrant Blue
                        'Kualitas': '#4CAF50',  # Vivid Green
                        'Loyalitas': '#FF5722'  # Deep Orange
                    }
                    
                    # Menampilkan hasil estimasi untuk intercept dan slope
                    intercept_harga = estimates.loc[estimates['lval'] == 'Intercept_Harga', 'Estimate'].values[0]
                    slope_harga = estimates.loc[estimates['lval'] == 'Harga', 'Estimate'].values[0]
                    intercept_kualitas = estimates.loc[estimates['lval'] == 'Intercept_Kualitas', 'Estimate'].values[0]
                    slope_kualitas = estimates.loc[estimates['lval'] == 'Kualitas', 'Estimate'].values[0]
                    intercept_loyalitas = estimates.loc[estimates['lval'] == 'Intercept_Loyalitas', 'Estimate'].values[0]
                    slope_loyalitas = estimates.loc[estimates['lval'] == 'Loyalitas', 'Estimate'].values[0]
                    
                    # Pengaruh Inovasi
                    inovasi_impact_harga = estimates.loc[estimates['lval'] == 'Intercept_Harga', 'Estimate'].values[0]
                    inovasi_impact_kualitas = estimates.loc[estimates['lval'] == 'Intercept_Kualitas', 'Estimate'].values[0]
                    inovasi_impact_loyalitas = estimates.loc[estimates['lval'] == 'Intercept_Loyalitas', 'Estimate'].values[0]
                    
                    # Membuat DataFrame untuk menampilkan hasil estimasi
                    results_df = pd.DataFrame({
                        'Variabel': ['Harga', 'Kualitas', 'Loyalitas'],
                        'Intercept': [intercept_harga, intercept_kualitas, intercept_loyalitas],
                        'Slope': [slope_harga, slope_kualitas, slope_loyalitas],
                        'Pengaruh Inovasi': [inovasi_impact_harga, inovasi_impact_kualitas, inovasi_impact_loyalitas]
                    })
                    
                    # Menampilkan tabel hasil estimasi dengan styling
                    st.markdown("### 📊 Hasil Estimasi Model LGCM")
                    st.dataframe(
                        results_df.round(4).style.set_properties(**{
                            'background-color': 'rgba(41, 128, 185, 0.1)',  # Light blue with low opacity
                            'color': 'black'
                        })
                    )
                    
                    # Interpretasi detail untuk setiap variabel
                    st.markdown("### Interpretasi Detail")
                    
                    # Interpretasi untuk Harga
                    st.markdown("#### 1. Harga")
                    st.markdown(f"""
                    - **Nilai Awal (Intercept)**: {intercept_harga:.3f}
                        - Menunjukkan persepsi awal konsumen terhadap harga
                        - {'' if intercept_harga > 0 else 'Negatif, '}Mengindikasikan {'positif' if intercept_harga > 0 else 'kurang positif'} terhadap harga awal
                    
                    - **Perubahan (Slope)**: {slope_harga:.3f}
                        - Menunjukkan perubahan persepsi harga seiring waktu
                        - {'Meningkat' if slope_harga > 0 else 'Menurun'} dengan laju {abs(slope_harga):.3f} per unit waktu
                    
                    - **Pengaruh Inovasi**: {inovasi_impact_harga:.3f}
                        - {'Positif' if inovasi_impact_harga > 0 else 'Negatif'}, menunjukkan inovasi {'meningkatkan' if inovasi_impact_harga > 0 else 'menurunkan'} persepsi harga
                    """)
                    
                    # Interpretasi untuk Kualitas
                    st.markdown("#### 2. Kualitas")
                    st.markdown(f"""
                    - **Nilai Awal (Intercept)**: {intercept_kualitas:.3f}
                        - Menunjukkan persepsi awal konsumen terhadap kualitas
                        - {'' if intercept_kualitas > 0 else 'Negatif, '}Mengindikasikan {'positif' if intercept_kualitas > 0 else 'kurang positif'} terhadap kualitas awal
                    
                    - **Perubahan (Slope)**: {slope_kualitas:.3f}
                        - Menunjukkan perubahan persepsi kualitas seiring waktu
                        - {'Meningkat' if slope_kualitas > 0 else 'Menurun'} dengan laju {abs(slope_kualitas):.3f} per unit waktu
                    
                    - **Pengaruh Inovasi**: {inovasi_impact_kualitas:.3f}
                        - {'Positif' if inovasi_impact_kualitas > 0 else 'Negatif'}, menunjukkan inovasi {'meningkatkan' if inovasi_impact_kualitas > 0 else 'menurunkan'} persepsi kualitas
                    """)
                    
                    # Interpretasi untuk Loyalitas
                    st.markdown("#### 3. Loyalitas")
                    st.markdown(f"""
                    - **Nilai Awal (Intercept)**: {intercept_loyalitas:.3f}
                        - Menunjukkan tingkat loyalitas awal konsumen
                        - {'' if intercept_loyalitas > 0 else 'Negatif, '}Mengindikasikan {'positif' if intercept_loyalitas > 0 else 'kurang positif'} terhadap loyalitas awal
                    
                    - **Perubahan (Slope)**: {slope_loyalitas:.3f}
                        - Menunjukkan perubahan loyalitas seiring waktu
                        - {'Meningkat' if slope_loyalitas > 0 else 'Menurun'} dengan laju {abs(slope_loyalitas):.3f} per unit waktu
                    
                    - **Pengaruh Inovasi**: {inovasi_impact_loyalitas:.3f}
                        - {'Positif' if inovasi_impact_loyalitas > 0 else 'Negatif'}, menunjukkan inovasi {'meningkatkan' if inovasi_impact_loyalitas > 0 else 'menurunkan'} loyalitas
                    """)

                except Exception as e:
                    st.error(f"Terjadi kesalahan dalam interpretasi hasil: {str(e)}")
                    st.markdown("Pastikan model LGCM telah berhasil dijalankan dan menghasilkan estimasi yang valid.")
                
                # Analisis LGCM untuk setiap variabel dengan kovariat
                lgcm_results = {}
                for var in df.columns:
                    lgcm_results[var] = estimate_lgcm_parameters(df[var], inovasi_value)
                             
                # 5. Output dan Rekomendasi
                st.markdown("""
                    <div style="
                        background-color: #1E88E5;
                        padding: 20px;
                        border-radius: 10px;
                        margin: 20px 0;
                        text-align: center;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                    ">
                        <h1 style="
                            color: white;
                            margin: 0;
                            font-size: 32px;
                            font-weight: bold;
                        ">5. Output dan Rekomendasi 📊</h1>
                    </div>
                """, unsafe_allow_html=True)

                try:
                    # Custom CSS untuk styling
                    st.markdown("""
                        <style>
                        .main-header {
                            color: #1E88E5;
                            font-size: 28px;
                            font-weight: bold;
                            padding: 10px;
                            border-radius: 5px;
                            margin-bottom: 20px;
                        }
                        .sub-header {
                            color: #43A047;
                            font-size: 24px;
                            padding: 8px;
                            border-left: 4px solid #43A047;
                            margin: 20px 0;
                        }
                        .effect-box {
                            background-color: #E3F2FD;
                            padding: 15px;
                            border-radius: 10px;
                            margin: 10px 0;
                        }
                        .recommendation-box {
                            background-color: #F1F8E9;
                            padding: 15px;
                            border-radius: 10px;
                            margin: 10px 0;
                            border-left: 4px solid #689F38;
                        }
                        .innovation-box {
                            background-color: #FFF3E0;
                            padding: 15px;
                            border-radius: 10px;
                            margin: 10px 0;
                        }
                        </style>
                    """, unsafe_allow_html=True)

                    # Ambil hasil estimasi dari model
                    estimates = model.inspect()
                    
                    # 5.1 Kesimpulan Keseluruhan
                    st.markdown('<p class="main-header">5.1 Kesimpulan Keseluruhan 📈</p>', unsafe_allow_html=True)
                    
                    # Hitung efek total untuk setiap variabel
                    variables = ['Harga', 'Kualitas', 'Loyalitas']
                    total_effects = []
                    
                    for var in variables:
                        intercept = estimates.loc[estimates['lval'] == f'Intercept_{var}', 'Estimate'].values[0]
                        slope = estimates.loc[estimates['lval'] == var, 'Estimate'].values[0]
                        inovasi_impact = estimates.loc[estimates['lval'] == f'Intercept_{var}', 'Estimate'].values[0]
                        
                        total_effect = abs(intercept) + abs(slope) + abs(inovasi_impact)
                        total_effects.append({'variable': var, 'effect': total_effect})
                    
                    # Urutkan faktor berdasarkan pengaruh
                    sorted_effects = sorted(total_effects, key=lambda x: x['effect'], reverse=True)
                    
                    # Tampilkan urutan faktor dalam box berwarna
                    st.markdown('<p class="sub-header">Urutan Faktor Berdasarkan Pengaruh:</p>', unsafe_allow_html=True)
                    
                    cols = st.columns(len(sorted_effects))
                    for i, (effect, col) in enumerate(zip(sorted_effects, cols), 1):
                        with col:
                            st.markdown(f"""
                                <div class="effect-box">
                                    <h3 style="color: #1E88E5; text-align: center;">{i}</h3>
                                    <h4 style="text-align: center;">{effect['variable']}</h4>
                                    <p style="text-align: center; color: #424242;">Efek: {effect['effect']:.4f}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # 5.2 Peran Inovasi sebagai Kovariat
                    st.markdown('<p class="main-header">5.2 Peran Inovasi sebagai Kovariat 🚀</p>', unsafe_allow_html=True)
                    
                    # Analisis pengaruh inovasi
                    inovasi_effects = []
                    for var in variables:
                        effect = estimates.loc[estimates['lval'] == f'Intercept_{var}', 'Estimate'].values[0]
                        inovasi_effects.append({'variable': var, 'effect': effect})
                    
                    # Urutkan pengaruh inovasi
                    sorted_inovasi = sorted(inovasi_effects, key=lambda x: abs(x['effect']), reverse=True)
                    
                    # Tampilkan dalam box berwarna
                    for effect in sorted_inovasi:
                        impact = "positif" if effect['effect'] > 0 else "negatif"
                        magnitude = abs(effect['effect'])
                        if magnitude < 0.2:
                            strength = "lemah"
                            color = "#FFA726"
                        elif magnitude < 0.5:
                            strength = "moderat"
                            color = "#FF7043"
                        else:
                            strength = "kuat"
                            color = "#E64A19"
                            
                        st.markdown(f"""
                            <div class="innovation-box">
                                <h4 style="color: {color};">{effect['variable']}</h4>
                                <p>Pengaruh {impact} dengan kekuatan {strength}</p>
                                <p style="font-weight: bold;">Nilai: {effect['effect']:.4f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # 5.3 Rekomendasi untuk E-commerce
                    st.markdown('<p class="main-header">5.3 Rekomendasi untuk E-commerce 💡</p>', unsafe_allow_html=True)
                    
                    # Rekomendasi untuk setiap variabel
                    for var in variables:
                        st.markdown(f'<p class="sub-header">{var}</p>', unsafe_allow_html=True)
                        intercept = estimates.loc[estimates['lval'] == f'Intercept_{var}', 'Estimate'].values[0]
                        slope = estimates.loc[estimates['lval'] == var, 'Estimate'].values[0]
                        
                        recommendations = []
                        if var == 'Harga':
                            if slope < 0:
                                recommendations = [
                                    "🏷️ Perlu evaluasi strategi penetapan harga",
                                    "🎯 Pertimbangkan program diskon atau reward yang lebih menarik",
                                    "📊 Tingkatkan transparansi dalam penetapan harga"
                                ]
                            else:
                                recommendations = [
                                    "💰 Pertahankan strategi harga yang kompetitif",
                                    "💎 Fokus pada value proposition untuk mempertahankan persepsi harga"
                                ]
                        elif var == 'Kualitas':
                            if slope < 0:
                                recommendations = [
                                    "⭐ Tingkatkan quality control untuk produk dan layanan",
                                    "📝 Perbaiki sistem review dan rating produk",
                                    "✅ Implementasikan program jaminan kualitas"
                                ]
                            else:
                                recommendations = [
                                    "🏆 Pertahankan standar kualitas yang baik",
                                    "📈 Lakukan monitoring berkelanjutan terhadap kepuasan pelanggan"
                                ]
                        elif var == 'Loyalitas':
                            if slope < 0:
                                recommendations = [
                                    "🎁 Kembangkan program loyalitas yang lebih menarik",
                                    "👤 Tingkatkan personalisasi layanan",
                                    "🛠️ Perbaiki sistem penanganan keluhan pelanggan"
                                ]
                            else:
                                recommendations = [
                                    "🌟 Pertahankan dan tingkatkan program loyalitas yang ada",
                                    "👥 Fokus pada pengembangan komunitas pelanggan"
                                ]
                        
                        for rec in recommendations:
                            st.markdown(f"""
                                <div class="recommendation-box">
                                    <p style="font-size: 16px; margin: 0;">{rec}</p>
                                </div>
                            """, unsafe_allow_html=True)
                    
                    # Rekomendasi Inovasi
                    st.markdown('<p class="main-header">Rekomendasi Inovasi 🎯</p>', unsafe_allow_html=True)
                    
                    max_inovasi_effect = max(inovasi_effects, key=lambda x: abs(x['effect']))
                    
                    recommendations = []
                    if max_inovasi_effect['variable'] == 'Harga':
                        recommendations = [
                            "💻 Implementasikan inovasi dalam sistem penetapan harga dinamis",
                            "📦 Kembangkan sistem bundling produk yang inovatif",
                            "💳 Integrasikan sistem pembayaran dengan teknologi terbaru"
                        ]
                    elif max_inovasi_effect['variable'] == 'Kualitas':
                        recommendations = [
                            "🤖 Implementasikan AI untuk quality control",
                            "📍 Kembangkan sistem tracking produk real-time",
                            "🔐 Tingkatkan teknologi untuk verifikasi keaslian produk"
                        ]
                    else:  # Loyalitas
                        recommendations = [
                            "⛓️ Kembangkan program loyalitas berbasis teknologi blockchain",
                            "🎯 Implementasikan sistem rekomendasi AI",
                            "👤 Tingkatkan personalisasi menggunakan machine learning"
                        ]
                    
                    for i, rec in enumerate(recommendations, 1):
                        st.markdown(f"""
                            <div class="recommendation-box" style="background-color: #FFF8E1;">
                                <h4 style="color: #FB8C00; margin: 0;">Rekomendasi {i}</h4>
                                <p style="font-size: 16px; margin: 10px 0 0 0;">{rec}</p>
                            </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Terjadi kesalahan dalam membuat output dan rekomendasi: {str(e)}")
                    st.markdown("Pastikan semua analisis sebelumnya telah berhasil dijalankan.")
                    
    except Exception as e:
        st.error(f"Terjadi kesalahan saat membaca file: {str(e)}")
        st.stop()

# Footer
st.markdown("---")
st.markdown("""
    ### Tentang Aplikasi
    Aplikasi ini menggunakan metode Latent Growth Curve Modeling (LGCM) untuk menganalisis 
    faktor-faktor yang mempengaruhi pemilihan e-commerce, dengan Inovasi sebagai kovariat. 
    Analisis mencakup:
    - Pre-processing data (missing values, outliers, normalitas)
    - Statistik deskriptif
    - Identifikasi Variabel Pertumbuhan
    - Visualisasi Trajektori Pertumbuhan
    - Interpretasi Hasil
    - Rekomendasi
""")

# Pembatas dengan garis horizontal
st.markdown("<hr style='border: 2px solid #E0E7FF; margin: 20px 0;'>", unsafe_allow_html=True)

# Informasi created by dengan styling
st.markdown("""
    ### 👩‍💻 Created by
    **Fadlila Agustina**  
    📧 fadlilagustina@icloud.com
    ©️ 2025
""")