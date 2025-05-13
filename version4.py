import streamlit as st
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import plotly.express as px
import graphviz

# Streamlit sayfa ayarları
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
page_selection = st.sidebar.radio("Sayfa seçimi", ["Yakıt Analizi", "Yeni Jeneratör Kombinasyonları"])

# SFoC verileri (load-% : g/kWh) - Sabit veri olduğu için global kalabilir
sfoc_data_global = {
        25: 205,
        50: 186,
        75: 178,
        85: 175,
        100: 178
    }

# Yük paylaşımı kurallarına göre çalışan jeneratör sayısını belirle
def determine_generator_usage(total_power, unit_power):
    if unit_power <= 0:
        return None, None 
    if total_power <= 0:
        return 0, 0.0 

    for n in range(1, 4):
        load_per_gen = total_power / (n * unit_power) * 100
        if 40 <= load_per_gen <= 92: 
            return n, load_per_gen
                
    return None, None

# Polinom interpolasyon fonksiyonu (sfoc_data'yı argüman olarak alır)
def interpolate_sfoc_non_linear(load_percentage, sfoc_data_input):
    loads = list(sfoc_data_input.keys())
    sfocs = list(sfoc_data_input.values())
    sorted_indices = np.argsort(loads)
    sorted_loads = np.array(loads)[sorted_indices]
    sorted_sfocs = np.array(sfocs)[sorted_indices]
        
    # İnterpolasyon için en az 2 nokta olmalı
    if len(sorted_loads) < 2:
        return None 

    try:
        interp_func = interp1d(sorted_loads, sorted_sfocs, kind='quadratic', fill_value="extrapolate")
        sfoc_value = float(interp_func(load_percentage))
            
        # SFOC 50 g/kWh altında makul olmayan bir alt sınır kabul edilebilir
        if sfoc_value < 50 and load_percentage > 0: # Sıfır yükte SFOC yüksek olabilir, 0 yükü hariç tutalım
            pass # calculate_fuel'de kontrol ediliyor

        return sfoc_value

    except ValueError as e:
        return None 

# Yakıt hesaplama
def calculate_fuel(power_output_kw, load_percent_on_engine, duration_hr, sfoc_data_input):
    if power_output_kw <= 0 or duration_hr <= 0:
        return 0.0 

    sfoc = interpolate_sfoc_non_linear(load_percent_on_engine, sfoc_data_input)

    if sfoc is None or sfoc < 50: # 50 g/kWh SFOC için makul olmayan bir alt sınır
        return 0.0

    return (power_output_kw * duration_hr * sfoc) / 1_000_000

# --- Sayfa İçeriği ---

if page_selection == "Yakıt Analizi":
    # Sidebar arayüz
    st.sidebar.header("Yakıt Analizi Girdi Ayarları")

    # Girdi ayarları (Sidebar'da kalır)
    gen_power_range_input = st.sidebar.slider("Jeneratör Birim Güç Aralığı (kW)", 1800, 3600, (2000, 3400), step=100)
    sea_power_range_input = st.sidebar.slider("Seyir Şaft Güç Aralığı (kW)", 2500, 5500, (3000, 4400), step=100)
    maneuver_power_range_input = st.sidebar.slider("Manevra Şaft Güç Aralığı (kW)", 1500, 3500, (1600, 2700), step=100)
    sea_duration_input = st.sidebar.number_input("Seyir Süresi (saat)", min_value=1.0, value=48.0, step=1.0)
    maneuver_duration_input = st.sidebar.number_input("Manevra Süresi (saat)", min_value=1.0, value=4.0, step=1.0)
    main_engine_mcr_input = st.sidebar.number_input("Ana Makine MCR (kW)", min_value=1000, value=7200, step=100)


    # --- Session State Initialization (Yakıt Analizi İçin) ---
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()
        st.session_state.detailed_df = pd.DataFrame()
        st.session_state.usage_df = pd.DataFrame()

    if "show_fuel_results" not in st.session_state:
        st.session_state.show_fuel_results = False


    # Hesaplama fonksiyonu
    @st.cache_data
    def calculate_all_results(
        current_gen_power_range,
        current_sea_power_range,
        current_maneuver_power_range,
        current_sea_duration,
        current_maneuver_duration,
        current_main_engine_mcr,
        current_sfoc_data
    ):
        results_summary_list = []
        detailed_data_list = []
        generator_usage_data_list = []

        # --- 1. Ana Makine Referans Verilerini Hesapla (Bir Kez) ---
        main_engine_ref_detailed_entries = []
        total_sea_fuel_main_engine_overall = 0
        total_maneuver_fuel_main_engine_overall = 0

        # Seyir Modu - Ana Makine
        for shaft_power_sea in range(current_sea_power_range[0], current_sea_power_range[1] + 1, 100):
            if shaft_power_sea <= 0 or current_main_engine_mcr <= 0: continue
            main_engine_load_sea = (shaft_power_sea / current_main_engine_mcr) * 100
            if shaft_power_sea > 0:
                fuel_main_ref_sea = calculate_fuel(shaft_power_sea, main_engine_load_sea, current_sea_duration, current_sfoc_data)
                if fuel_main_ref_sea > 0:
                    total_sea_fuel_main_engine_overall += fuel_main_ref_sea
                    main_engine_ref_detailed_entries.append({
                        "Combo": "Ana Makine Referans", "Mode": "Seyir", "Shaft Power (kW)": shaft_power_sea,
                        "DE Power (kW)": np.nan, "Fuel (ton)": round(fuel_main_ref_sea, 3), "System Type": "Ana Makine",
                        "Load (%)": round(main_engine_load_sea, 2)
                    })

        # Manevra Modu - Ana Makine
        for shaft_power_maneuver in range(current_maneuver_power_range[0], current_maneuver_power_range[1] + 1, 100):
            if shaft_power_maneuver <= 0 or current_main_engine_mcr <= 0: continue
            main_engine_load_maneuver = (shaft_power_maneuver / current_main_engine_mcr) * 100
            if shaft_power_maneuver > 0:
                fuel_main_ref_maneuver = calculate_fuel(shaft_power_maneuver, main_engine_load_maneuver, current_maneuver_duration, current_sfoc_data)
                if fuel_main_ref_maneuver > 0:
                    total_maneuver_fuel_main_engine_overall += fuel_main_ref_maneuver
                    main_engine_ref_detailed_entries.append({
                        "Combo": "Ana Makine Referans", "Mode": "Manevra", "Shaft Power (kW)": shaft_power_maneuver,
                        "DE Power (kW)": np.nan, "Fuel (ton)": round(fuel_main_ref_maneuver, 3), "System Type": "Ana Makine",
                        "Load (%)": round(main_engine_load_maneuver, 2)
                    })
        detailed_data_list.extend(main_engine_ref_detailed_entries)


        # --- 2. Jeneratör Verilerini Hesapla (Her Jeneratör Kombinasyonu İçin) ---
        for gen_power_unit in range(current_gen_power_range[0], current_gen_power_range[1] + 1, 100):
            if gen_power_unit <= 0: continue

            combo_label = f"3 x {gen_power_unit} kW Jeneratör"
            
            # !!! DÜZELTME: Toplam yakıt değişkenlerini burada, her jeneratör kombosu için initialize et !!!
            current_combo_total_sea_fuel_generators = 0
            current_combo_total_maneuver_fuel_generators = 0
            # !!! DÜZELTME SONU !!!

            current_combo_sea_detailed_entries = []
            current_combo_maneuver_detailed_entries = []
            current_combo_usage_entries = []

            # Sabit verimlilikler kullanılıyor (sizin orijinal kodunuzdaki gibi)
            efficiency_factor = 0.95 / 0.93 


            # Seyir modu - Jeneratörler
            for shaft_power_sea in range(current_sea_power_range[0], current_sea_power_range[1] + 1, 100):
                if shaft_power_sea <= 0: continue

                de_power_sea = shaft_power_sea * efficiency_factor

                ngen_sea, load_sea = determine_generator_usage(de_power_sea, gen_power_unit)
                if ngen_sea is not None: 
                    fuel_gen_sea = calculate_fuel(de_power_sea, load_sea, current_sea_duration, current_sfoc_data)
                    if fuel_gen_sea > 0:
                        current_combo_total_sea_fuel_generators += fuel_gen_sea
                        current_combo_sea_detailed_entries.append({
                            "Combo": combo_label, "Mode": "Seyir", "Shaft Power (kW)": shaft_power_sea,
                            "DE Power (kW)": round(de_power_sea),"Fuel (ton)": round(fuel_gen_sea, 3), "System Type": "Jeneratör",
                            "Load (%)": round(load_sea, 2)
                        })
                        current_combo_usage_entries.append({
                            "Combo": combo_label, "Mode": "Seyir", "DE Power (kW)": round(de_power_sea),
                            "Generators Used": ngen_sea, "Load Per Generator (%)": round(load_sea, 2)
                        })

            # Manevra modu - Jeneratörler
            for shaft_power_maneuver in range(current_maneuver_power_range[0], current_maneuver_power_range[1] + 1, 100):
                if shaft_power_maneuver <= 0: continue

                de_power_maneuver = shaft_power_maneuver * efficiency_factor

                ngen_maneuver, load_maneuver = determine_generator_usage(de_power_maneuver, gen_power_unit)
                if ngen_maneuver is not None:
                    fuel_gen_maneuver = calculate_fuel(de_power_maneuver, load_maneuver, current_maneuver_duration, current_sfoc_data)
                    if fuel_gen_maneuver > 0:
                        current_combo_total_maneuver_fuel_generators += fuel_gen_maneuver
                        current_combo_maneuver_detailed_entries.append({
                            "Combo": combo_label, "Mode": "Manevra", "Shaft Power (kW)": shaft_power_maneuver,
                            "DE Power (kW)": round(de_power_maneuver), "Fuel (ton)": round(fuel_gen_maneuver, 3), "System Type": "Jeneratör",
                            "Load (%)": round(load_maneuver, 2)
                        })
                        current_combo_usage_entries.append({
                            "Combo": combo_label, "Mode": "Manevra", "DE Power (kW)": round(de_power_maneuver),
                            "Generators Used": ngen_maneuver, "Load Per Generator (%)": round(load_maneuver, 2)
                        })

            # Detaylı ve kullanım verilerini ana listelere ekle
            detailed_data_list.extend(current_combo_sea_detailed_entries)
            detailed_data_list.extend(current_combo_maneuver_detailed_entries)
            generator_usage_data_list.extend(current_combo_usage_entries)
            
            # Bu jeneratör kombinasyonu için toplam yakıtları hesapla (aralıktaki tüm noktaların toplamı)
            # Sadece anlamlı (pozitif) toplamlar varsa özete ekle
            # Burada değişkenler artık initialize edildiği için hata vermeyecekler.
            if current_combo_total_sea_fuel_generators > 0 or current_combo_total_maneuver_fuel_generators > 0:
                sea_diff = total_sea_fuel_main_engine_overall - current_combo_total_sea_fuel_generators
                maneuver_diff = total_maneuver_fuel_main_engine_overall - current_combo_total_maneuver_fuel_generators
                results_summary_list.append({
                    "Jeneratör Kombinasyonu": combo_label,
                    "Toplam Seyir Yakıtı (Jeneratörler) (ton)": round(current_combo_total_sea_fuel_generators, 2),
                    "Toplam Manevra Yakıtı (Jeneratörler) (ton)": round(current_combo_total_maneuver_fuel_generators, 2),
                    "Seyir Yakıt Farkı (Jen - Ana M. Ref.) (ton)": round(sea_diff, 2),
                    "Manevra Yakıt Farkı (Jen - Ana M. Ref.) (ton)": round(maneuver_diff, 2)
                })

        return pd.DataFrame(results_summary_list), pd.DataFrame(detailed_data_list), pd.DataFrame(generator_usage_data_list)

    # Hesapla butonu
    if st.sidebar.button("HESAPLA"):
        st.session_state.show_fuel_results = True
        st.session_state.results_df, st.session_state.detailed_df, st.session_state.usage_df = calculate_all_results(
            gen_power_range_input,
            sea_power_range_input,
            maneuver_power_range_input,
            sea_duration_input,
            maneuver_duration_input,
            main_engine_mcr_input,
            sfoc_data_global
        )

    # Başlık
    st.header("Jeneratör ve Ana Makine Yakıt Tüketim Analizi")

    # Sonuçları sadece 'HESAPLA' butonuna basıldıysa ve data boş değilse göster
    if st.session_state.show_fuel_results and not st.session_state.results_df.empty:
        st.subheader("Özet Sonuçlar")
        st.dataframe(st.session_state.results_df, use_container_width=True)

        st.markdown("---")
        st.subheader("Detaylı Grafiksel Analiz")

        available_gen_combos = [
            combo for combo in st.session_state.detailed_df["Combo"].unique()
            if combo != "Ana Makine Referans"
        ]

        if available_gen_combos:
            selected_gen_combo = st.selectbox("Karşılaştırılacak Jeneratör Kombinasyonunu Seçin", available_gen_combos)
            
            plot_mode = st.radio("Analiz Modunu Seçin", ["Seyir", "Manevra"], horizontal=True, key="plot_mode_selector")

            # Karşılaştırma grafiği için veri filtreleme
            plot_data_gen_selected = st.session_state.detailed_df[
                (st.session_state.detailed_df["Combo"] == selected_gen_combo) &
                (st.session_state.detailed_df["System Type"] == "Jeneratör") &
                (st.session_state.detailed_df["Mode"] == plot_mode) &
                (st.session_state.detailed_df["Fuel (ton)"].notna()) & # NaN olmayan yakıt değerleri
                (st.session_state.detailed_df["Fuel (ton)"] > 0) # Pozitif yakıt değerleri
            ]
            plot_data_me_ref = st.session_state.detailed_df[
                (st.session_state.detailed_df["System Type"] == "Ana Makine") &
                (st.session_state.detailed_df["Mode"] == plot_mode) &
                (st.session_state.detailed_df["Fuel (ton)"].notna()) &
                (st.session_state.detailed_df["Fuel (ton)"] > 0)
            ]
            
            combined_fuel_plot_data = pd.concat([plot_data_gen_selected, plot_data_me_ref]).reset_index(drop=True)

            if not combined_fuel_plot_data.empty:
                fig_fuel_comparison = px.bar(
                    combined_fuel_plot_data,
                    x="Shaft Power (kW)",
                    y="Fuel (ton)",
                    color="System Type", # Jeneratör vs Ana Makine
                    barmode="group",
                    title=f"{selected_gen_combo} vs Ana Makine Referans ({plot_mode} Modu)",
                    labels={"Fuel (ton)": "Yakıt (ton)", "Shaft Power (kW)": "Şaft Gücü (kW)", "System Type": "Sistem Tipi"}
                )
                st.plotly_chart(fig_fuel_comparison, use_container_width=True)
            else:
                st.warning(f"{plot_mode} modu için {selected_gen_combo} veya Ana Makine Referansına ait gösterilecek karşılaştırmalı yakıt verisi bulunamadı.")

            # Jeneratör Kullanım Grafiği (Sadece seçilen jeneratör kombinasyonu için)
            gen_usage_plot_data = st.session_state.usage_df[
                (st.session_state.usage_df["Combo"] == selected_gen_combo) &
                (st.session_state.usage_df["Mode"] == plot_mode)
            ]
            if not gen_usage_plot_data.empty:
                fig_usage = px.bar(
                    gen_usage_plot_data,
                    x="DE Power (kW)",
                    y="Generators Used",
                    hover_data=["Load Per Generator (%)"],
                    barmode="group",
                    title=f"{selected_gen_combo} - Jeneratör Kullanımı ({plot_mode} Modu)",
                    labels={"Generators Used": "Kullanılan Jeneratör Sayısı", "DE Power (kW)": "Dizel Elektrik Gücü (kW)"}
                )
                # Yük yüzdesini bar üzerinde göstermek için (Plotly'nin yeni versiyonlarında text_auto=True daha kolay olabilir)
                # fig_usage.update_traces(texttemplate='%{customdata[0]:.2f}%', textposition='outside')
                fig_usage.update_traces(text=gen_usage_plot_data["Load Per Generator (%)"].apply(lambda x: f'{x:.2f}%'), textposition='outside')

                st.plotly_chart(fig_usage, use_container_width=True)
            # else: # Jeneratör kullanım verisi yoksa uyarı verilebilir ama yakıt grafiği ana odak olduğu için opsiyonel.
                # st.warning(f"{selected_gen_combo} için {plot_mode} modunda jeneratör kullanım verisi bulunamadı.")
                
        elif not st.session_state.detailed_df.empty : # Detaylı veri var ama jeneratör kombosu yok (sadece ana makine olabilir)
            st.info("Hesaplama sonucunda jeneratör kombinasyonu bulunamadı, sadece Ana Makine Referans verileri mevcut olabilir.")
            # İsteğe bağlı: Sadece ana makine verilerini gösteren bir grafik eklenebilir.
            plot_data_me_ref_only = st.session_state.detailed_df[
                (st.session_state.detailed_df["System Type"] == "Ana Makine") &
                (st.session_state.detailed_df["Fuel (ton)"].notna()) &
                (st.session_state.detailed_df["Fuel (ton)"] > 0)
            ]
            if not plot_data_me_ref_only.empty:
                plot_mode_me_only = st.radio("Ana Makine Analiz Modunu Seçin", ["Seyir", "Manevra"], horizontal=True, key="plot_mode_me_only_selector")
                plot_data_me_ref_only_filtered = plot_data_me_ref_only[plot_data_me_ref_only["Mode"] == plot_mode_me_only]
                if not plot_data_me_ref_only_filtered.empty:
                    fig_fuel_me_only = px.bar(
                        plot_data_me_ref_only_filtered,
                        x="Shaft Power (kW)", y="Fuel (ton)", color="Mode",
                        title=f"Ana Makine Referans Yakıt Tüketimi ({plot_mode_me_only})",
                        labels={"Fuel (ton)": "Yakıt (ton)", "Shaft Power (kW)": "Şaft Gücü (kW)"}
                    )
                    st.plotly_chart(fig_fuel_me_only, use_container_width=True)



    elif st.session_state.results_df.empty and st.session_state.detailed_df.empty and st.session_state.usage_df.empty and "calculate_all_results" in st.session_state:
        # Bu kontrol, HESAPLA butonuna basıldıktan sonra boş DataFrame'ler döndüğünde çalışır.
        # calculate_all_results'ın session state'e eklenmesi, butonun en az bir kez basıldığını anlamak için bir işaretçi olabilir,
        # ancak daha iyi bir yol, butonun state'ini kontrol etmek veya sonuçların None olup olmadığını kontrol etmektir.
        # Şimdilik basit bir uyarı:
        st.warning("Girilen parametrelerle hesaplanacak uygun bir senaryo bulunamadı veya hesaplama yapılmadı.")

    # --- SFOC - Yük Eğrisi Grafiği ---
    st.markdown("---")
    st.subheader("Özgül Yakıt Tüketimi (SFOC) - Yük Eğrisi")

    # Global SFOC verisini al
    loads_original = list(sfoc_data_global.keys())
    sfocs_original = list(sfoc_data_global.values())

    # Veriyi yük yüzdesine göre sırala
    sorted_indices = np.argsort(loads_original)
    sorted_loads_original = np.array(loads_original)[sorted_indices]
    sorted_sfocs_original = np.array(sfocs_original)[sorted_indices]

    # Orijinal veri noktaları için DataFrame
    df_sfoc_points = pd.DataFrame({
        'Yük (%)': sorted_loads_original,
        'SFOC (g/kWh)': sorted_sfocs_original
    })

    # İnterpolasyon eğrisi için daha sık aralıklı yük değerleri oluştur
    plot_min_load = 0 
    plot_max_load = 110
    interpolated_loads = np.linspace(plot_min_load, plot_max_load, 200)

    # Bu yük değerleri için SFOC değerlerini interpole et
    interpolated_sfocs = [interpolate_sfoc_non_linear(load, sfoc_data_global) for load in interpolated_loads]

    # Anlamsız SFOC değerlerini filtrele (None veya çok düşük/negatif)
    valid_interpolated_data = [(load, sfoc) for load, sfoc in zip(interpolated_loads, interpolated_sfocs) if sfoc is not None and sfoc >= 50] 

    if valid_interpolated_data:
        interpolated_loads_valid, interpolated_sfocs_valid = zip(*valid_interpolated_data)
        df_sfoc_curve = pd.DataFrame({
            'Yük (%)': interpolated_loads_valid,
            'SFOC (g/kWh)': interpolated_sfocs_valid
        })

        fig_sfoc_display = px.line(df_sfoc_curve, x='Yük (%)', y='SFOC (g/kWh)',
                                title='Jeneratör SFOC vs. Yük Yüzdesi',
                                labels={'Yük (%)': 'Jeneratör Yükü (%)', 'SFOC (g/kWh)': 'SFOC (g/kWh)'})

        fig_sfoc_display.add_scatter(x=df_sfoc_points['Yük (%)'], y=df_sfoc_points['SFOC (g/kWh)'],
                                    mode='markers',
                                    name='Orjinal Veri Noktaları',
                                    marker=dict(color='red', size=10, symbol='circle'))

        min_display_sfoc = df_sfoc_points['SFOC (g/kWh)'].min() - 5
        max_display_sfoc = df_sfoc_points['SFOC (g/kWh)'].max() + 5
        min_display_sfoc = max(0, min_display_sfoc) 

        fig_sfoc_display.update_yaxes(range=[min_display_sfoc, max_display_sfoc])
        fig_sfoc_display.update_xaxes(range=[plot_min_load -5, plot_max_load + 5])

        st.plotly_chart(fig_sfoc_display, use_container_width=True)
    else:
        st.warning("SFOC eğrisi çizilemedi. Lütfen SFOC verilerini kontrol edin.")


    # --- Güç Akışı ve Kayıplar Diyagramı ---
    st.markdown("---")
    st.subheader("Dizel Elektrik Güç Akışı ve Kayıpları Diyagramı")

    # Diagram için girdi ayarları (sidebar'da ayrı bir başlık altında)
    st.sidebar.header("Güç Akışı Diyagramı Ayarları")

    # diagram_shaft_power session state değişkenini burada, kullanılmadan önce initialize et
    if "diagram_shaft_power" not in st.session_state:
        st.session_state.diagram_shaft_power = 3000 # Diagram için default başlangıç değeri

    # diagram_shaft_power_input widget'ı artık session state'teki değeri okurken hata vermeyecek
    diagram_shaft_power_input = st.sidebar.number_input("Diyagram için Şaft Gücü (kW)", min_value=100, value=int(st.session_state.diagram_shaft_power), step=50, key="diagram_shaft_power_input_widget")

    # Kayıp yüzdeleri için girdiler (verimlilik olarak alalım) - Bu verimlilikler sadece diyagramı etkiler.
    st.sidebar.subheader("Sistem Verimlilikleri (%) (Diyagram İçin)")
    motor_eff_input_diagram_perc = st.sidebar.slider("Diyagram - Elektrik Motoru Verimliliği (%)", 90.0, 99.9, 97.0, step=0.1, key="diagram_motor_eff_slider")
    converter_eff_input_diagram_perc = st.sidebar.slider("Diyagram - Frekans Konvertörü Verimliliği (%)", 90.0, 99.9, 98.0, step=0.1, key="diagram_converter_eff_slider")
    switchboard_eff_input_diagram_perc = st.sidebar.slider("Diyagram - Pano Verimliliği (%)", 90.0, 99.9, 99.0, step=0.1, key="diagram_switchboard_eff_slider")
    generator_elec_eff_input_diagram_perc = st.sidebar.slider("Diyagram - Jeneratör Elektrik Verimliliği (%)", 90.0, 99.9, 99.0, step=0.1, key="diagram_gen_eff_slider")

    # 0-1 aralığına çevrilmiş verimlilikler (diyagram için)
    motor_eff_input_diagram = motor_eff_input_diagram_perc / 100.0
    converter_eff_input_diagram = converter_eff_input_diagram_perc / 100.0
    switchboard_eff_input_diagram = switchboard_eff_input_diagram_perc / 100.0
    generator_elec_eff_input_diagram = generator_elec_eff_input_diagram_perc / 100.0


    # Diagram hesaplaması
    def calculate_power_flow(shaft_power, motor_eff, converter_eff, switchboard_eff, generator_elec_eff):
        if shaft_power <= 0:
            return None, None 

        if not all([motor_eff > 0, converter_eff > 0, switchboard_eff > 0, generator_elec_eff > 0]):
            return None, None 

        p_shaft = shaft_power
        p_motor_in = p_shaft / motor_eff
        p_converter_in = p_motor_in / converter_eff
        p_switchboard_in = p_converter_in / switchboard_eff
        p_generator_out = p_switchboard_in / generator_elec_eff

        if any([not np.isfinite(p_motor_in), not np.isfinite(p_converter_in), 
                not np.isfinite(p_switchboard_in), not np.isfinite(p_generator_out)]):
            return None, None 

        loss_motor = p_motor_in - p_shaft
        loss_converter = p_converter_in - p_motor_in
        loss_switchboard = p_switchboard_in - p_converter_in
        loss_generators_elec = p_generator_out - p_switchboard_in

        power_values = {
            "shaft": p_shaft,
            "motor_in": p_motor_in,
            "converter_in": p_converter_in,
            "switchboard_in": p_switchboard_in,
            "generator_out": p_generator_out
        }

        loss_values = {
            "motor": loss_motor,
            "converter": loss_converter,
            "switchboard": loss_switchboard,
            "generators_elec": loss_generators_elec
        }

        return power_values, loss_values

    # Hesaplamayı yap
    power_vals, loss_vals = calculate_power_flow(
        diagram_shaft_power_input,           
        motor_eff_input_diagram,             
        converter_eff_input_diagram,
        switchboard_eff_input_diagram,
        generator_elec_eff_input_diagram
    )

    # Graphviz diyagramını oluştur
    if power_vals and loss_vals and all(v is not None and np.isfinite(v) for v in power_vals.values()): 
        dot = graphviz.Digraph('power_flow_diagram', comment='Güç Akışı ve Kayıplar')
        dot.attr(rankdir='LR') 
        dot.attr('node', shape='box', fontsize='12', fontcolor='black', style='filled')
        dot.attr('edge', fontsize='10')

        # Kayıp Yüzdeleri (Gösterim için) 
        motor_loss_percent = (1 - motor_eff_input_diagram) * 100 if motor_eff_input_diagram > 0 else float('nan')
        converter_loss_percent = (1 - converter_eff_input_diagram) * 100 if converter_eff_input_diagram > 0 else float('nan')
        switchboard_loss_percent = (1 - switchboard_eff_input_diagram) * 100 if switchboard_eff_input_diagram > 0 else float('nan')
        generator_elec_loss_percent = (1 - generator_elec_eff_input_diagram) * 100 if generator_elec_eff_input_diagram > 0 else float('nan')

        def format_loss_percent(percent):
            return f'({percent:.1f}%)' if not np.isnan(percent) else '(N/A)'

        color_shaft = "#aaffaa"
        color_motor = "#ffffaa"
        color_converter = "#ffccaa"
        color_switchboard = "#ffaaaa"
        color_generators = "#ffccaa"

        dot.node('generators1', f'Jeneratörler (Toplam Elektrik Çıkışı)\n{power_vals["generator_out"]:.0f} kW', shape="box3d", fillcolor=color_generators, fontsize="14")
        dot.node('generators2', f'Jeneratörler (Toplam Elektrik Çıkışı)\n{power_vals["generator_out"]:.0f} kW', shape="box3d", fillcolor=color_generators, fontsize="14")
        dot.node('generators3', f'Jeneratörler (Toplam Elektrik Çıkışı)\n{power_vals["generator_out"]:.0f} kW', shape="box3d", fillcolor=color_generators, fontsize="14")
        dot.node('switchboard', f'Ana Pano\nKayıp: {loss_vals["switchboard"]:.0f} kW {format_loss_percent(switchboard_loss_percent)}', shape="rectangle", fillcolor=color_switchboard)
        dot.node('converter', f'Frekans Konvertörü\nKayıp: {loss_vals["converter"]:.0f} kW {format_loss_percent(converter_loss_percent)}', shape="square", fillcolor=color_converter, fontsize="8")
        dot.node('motor', f'Elektrik Motoru\nKayıp: {loss_vals["motor"]:.0f} kW {format_loss_percent(motor_loss_percent)}', shape="circle", fillcolor=color_motor, fontsize="6")
        dot.node('shaft', f'Şaft Gücü\n{power_vals["shaft"]:.0f} kW', shape="cylinder", fillcolor=color_shaft)

        dot.edge('generators1', 'switchboard', penwidth="2", arrowhead="normal", color="#00cc44")
        dot.edge('generators2', 'switchboard', label=f'{power_vals["generator_out"]:.0f} kW', penwidth="2", arrowhead="normal", color="#00cc44")
        dot.edge('generators3', 'switchboard', penwidth="2", arrowhead="normal", color="#00cc44")

        dot.edge('switchboard', 'converter', label=f'{power_vals["switchboard_in"]:.0f} kW', penwidth="2", arrowhead="normal", color="#a2d729")
        dot.edge('converter', 'motor', label=f'{power_vals["converter_in"]:.0f} kW', penwidth="2", arrowhead="normal", color="#ffcc00")
        dot.edge('motor', 'shaft', label=f'{power_vals["motor_in"]:.0f} kW', penwidth="2", arrowhead="normal", color="#cc0000") 

        st.graphviz_chart(dot, use_container_width=True)

        total_loss = loss_vals["motor"] + loss_vals["converter"] + loss_vals["switchboard"] + loss_vals["generators_elec"]
        st.info(f"Girilen Şaft Gücü ({power_vals['shaft']:.0f} kW) için Jeneratörlerden üretilmesi gereken Toplam Elektrik Gücü yaklaşık {power_vals['generator_out']:.0f} kW'tır. Toplam Sistem Kaybı: {total_loss:.0f} kW.")

    elif diagram_shaft_power_input > 0:
        st.warning("Güç Akışı diyagramı hesaplanamadı. Lütfen Diyagram için Şaft Gücü'nün pozitif olduğundan ve tüm sistem verimliliklerinin %0'dan büyük olduğundan emin olun.")

    else: 
        st.info("Güç Akışı diyagramını görmek için lütfen sidebar'dan 'Diyagram için Şaft Gücü' değeri girin.")


elif page_selection == "Yeni Jeneratör Kombinasyonları":
    st.header("Yeni Jeneratör Kombinasyonları Analizi")

    # --- Sidebar Inputs for New Page ---
    st.sidebar.header("Yeni Kombinasyon Girdi Ayarları")
    main_gen_mcr_new = st.sidebar.number_input("Ana Jeneratör MCR (kW)", min_value=100, value=2400, step=100, key="main_gen_mcr_new")
    main_gen_qty_new = st.sidebar.number_input("Ana Jeneratör Adedi", min_value=1, value=3, step=1, key="main_gen_qty_new")
    port_gen_mcr_new = st.sidebar.number_input("Liman Jeneratörü MCR (kW)", min_value=50, value=1000, step=50, key="port_gen_mcr_new")
    port_gen_qty_new = st.sidebar.number_input("Liman Jeneratörü Adedi", min_value=0, value=1, step=1, key="port_gen_qty_new")
    sea_power_range_new = st.sidebar.slider("Seyir Şaft Güç Aralığı (kW)", 2500, 5500, (3000, 4400), step=100, key="sea_power_range_new")
    maneuver_power_range_new = st.sidebar.slider("Manevra Şaft Güç Aralığı (kW)", 1500, 3500, (1600, 2700), step=100, key="maneuver_power_range_new")
    sea_duration_new = st.sidebar.number_input("Seyir Süresi (saat)", min_value=1.0, value=48.0, step=1.0, key="sea_duration_new")
    maneuver_duration_new = st.sidebar.number_input("Manevra Süresi (saat)", min_value=1.0, value=4.0, step=1.0, key="maneuver_duration_new")
    main_engine_mcr_new = st.sidebar.number_input("Ana Makine MCR (kW) (Referans İçin)", min_value=1000, value=7200, step=100, key="main_engine_mcr_new")

    st.sidebar.subheader("Sistem Verimlilikleri (%) (Yeni Kombinasyon İçin)")
    motor_eff_new_perc = st.sidebar.slider("Yeni - Elektrik Motoru Verimliliği (%)", 90.0, 99.9, 97.0, step=0.1, key="motor_eff_new_slider")
    converter_eff_new_perc = st.sidebar.slider("Yeni - Frekans Konvertörü Verimliliği (%)", 90.0, 99.9, 98.0, step=0.1, key="converter_eff_new_slider")
    switchboard_eff_new_perc = st.sidebar.slider("Yeni - Pano Verimliliği (%)", 90.0, 99.9, 99.0, step=0.1, key="switchboard_eff_new_slider")
    generator_elec_eff_new_perc = st.sidebar.slider("Yeni - Jeneratör Elektrik Verimliliği (%)", 90.0, 99.9, 99.0, step=0.1, key="generator_elec_eff_new_slider")

    total_elec_eff_new_factor = (motor_eff_new_perc / 100.0) * (converter_eff_new_perc / 100.0) * (switchboard_eff_new_perc / 100.0) * (generator_elec_eff_new_perc / 100.0)

    if total_elec_eff_new_factor <= 0:
        st.warning("Yeni Kombinasyon için toplam sistem verimliliği sıfırdan büyük olmalıdır. Lütfen verimlilikleri kontrol edin.")
        total_elec_eff_new_factor = 1e-6

    if "results_df_new" not in st.session_state:
        st.session_state.results_df_new = pd.DataFrame()
        st.session_state.detailed_df_new = pd.DataFrame()
        st.session_state.usage_df_new = pd.DataFrame()
    if "show_fuel_results_new" not in st.session_state:
        st.session_state.show_fuel_results_new = False

    # --- Helper Functions ---
    def find_min_gens_for_power(required_power, unit_mcr, unit_qty):
        if unit_mcr <= 0 or unit_qty <= 0: return None
        if required_power <= 0: return 0
        min_gens = np.ceil(required_power / unit_mcr)
        return int(min_gens) if min_gens <= unit_qty else None

    def evaluate_combination(required_de_power, running_gens_info, sfoc_data, duration): # sfoc_data ve duration eklendi
        if not running_gens_info: return None
        running_mcrs = [mcr for mcr, gen_type in running_gens_info]
        total_capacity = sum(running_mcrs)
        if total_capacity <= 0 or required_de_power <= 0: return None
        # Kapasitenin yeterli olduğundan emin ol (çok küçük aşımlara izin verilebilir)
        if required_de_power > total_capacity * 1.001: return None

        power_per_gen_list = [(required_de_power * gen_mcr / total_capacity) if total_capacity > 0 else 0 for gen_mcr in running_mcrs]
        load_percent_list = [(power / mcr * 100) if mcr > 0 else 0 for power, mcr in zip(power_per_gen_list, running_mcrs)]

        total_fuel = 0
        loads_info = []
        valid_fuel_count = 0
        for i in range(len(running_gens_info)):
            gen_mcr, gen_type_label = running_gens_info[i]
            load_percent = load_percent_list[i]
            power_output = power_per_gen_list[i]
            
            if load_percent > 110 : # Aşırı yüklenme durumunda bu jeneratörü atla (SFOC eğrisi genellikle %110'a kadar tanımlıdır)
                continue

            # calculate_fuel fonksiyonunun global kapsamda tanımlı olduğu varsayılır
            fuel_part = calculate_fuel(power_output, load_percent, duration, sfoc_data)
            if fuel_part is not None and fuel_part > 0:
                total_fuel += fuel_part
                loads_info.append((gen_mcr, load_percent, gen_type_label))
                valid_fuel_count += 1
        
        return (total_fuel, loads_info) if valid_fuel_count > 0 and total_fuel > 0 else None

    def get_best_combination(required_de_power, main_mcr, main_qty, port_mcr, port_qty, sfoc_data, duration):
        if required_de_power <= 0:
            return 0.0, "0 kW Yük", [], None

        evaluated_options = {}

        def add_option(key, fuel, label, loads, original_info=None):
            # Daha düşük yakıt tüketimine sahip olanı sakla
            if key not in evaluated_options or fuel < evaluated_options[key][0]:
                evaluated_options[key] = (fuel, label, loads, original_info)

        main_only_inefficient_candidate = None # (fuel, label, loads, original_main_load_percent)
        original_main_loads_for_inefficient_case = []


        # --- STRATEGY 1: MAIN GENERATORS ONLY ---
        if main_qty > 0 and main_mcr > 0:
            n_main1 = find_min_gens_for_power(required_de_power, main_mcr, main_qty)
            if n_main1 is not None:
                running_info1 = [(main_mcr, "Ana")] * n_main1
                eval_res1 = evaluate_combination(required_de_power, running_info1, sfoc_data, duration)
                if eval_res1:
                    fuel1, loads1 = eval_res1
                    label1 = f"{n_main1}x {main_mcr}kW Ana"
                    load_pct1 = loads1[0][1] if loads1 else 100.0

                    if 65 <= load_pct1 <= 92: # Verimli aralık
                        add_option("main_eff", fuel1, label1, loads1)
                    elif load_pct1 < 65: # Verimsiz düşük yük
                        main_only_inefficient_candidate = (fuel1, label1, loads1)
                        original_main_loads_for_inefficient_case = [l[1] for l in loads1] # Yükleri sakla
                        add_option("main_ineff_low", fuel1, label1, loads1)
                    elif load_pct1 > 92 and n_main1 + 1 <= main_qty : # Aşırı yük, bir fazla jen dene
                        n_main2 = n_main1 + 1
                        if n_main2 * main_mcr >= required_de_power: # Kapasite kontrolü
                            running_info2 = [(main_mcr, "Ana")] * n_main2
                            eval_res2 = evaluate_combination(required_de_power, running_info2, sfoc_data, duration)
                            if eval_res2:
                                fuel2, loads2 = eval_res2
                                label2 = f"{n_main2}x {main_mcr}kW Ana"
                                load_pct2 = loads2[0][1] if loads2 else 100.0
                                if 65 <= load_pct2 <= 92:
                                    add_option("main_eff_plus_one", fuel2, label2, loads2)
                                elif load_pct2 < 65:
                                    main_only_inefficient_candidate = (fuel2, label2, loads2)
                                    original_main_loads_for_inefficient_case = [l[1] for l in loads2]
                                    add_option("main_ineff_low_plus_one", fuel2, label2, loads2)
                                else: # Hala aşırı yük veya başka bir durum
                                     add_option("main_fallback_plus_one", fuel2, label2, loads2)
                    else: # n_main1 zaten maksimumda ve yük > %92 veya diğer durumlar
                        add_option("main_fallback_at_n_main1", fuel1, label1, loads1) # Fallback olarak ekle

        # --- STRATEGY 2: PORT GENERATOR(S) ONLY ---
        if port_qty > 0 and port_mcr > 0:
            n_port = find_min_gens_for_power(required_de_power, port_mcr, port_qty)
            if n_port is not None and n_port > 0 : # Liman jen. adedi 0 olmamalı
                running_info_port = [(port_mcr, "Liman")] * n_port
                eval_res_port = evaluate_combination(required_de_power, running_info_port, sfoc_data, duration)
                if eval_res_port:
                    fuel_p, loads_p = eval_res_port
                    label_p = f"{n_port}x {port_mcr}kW Liman"
                    add_option("port_only", fuel_p, label_p, loads_p)

        # --- STRATEGY 3: ASSISTED MODE (1 Port Gen + N Main Gens) ---
        if main_only_inefficient_candidate and port_qty >= 1 and port_mcr > 0 and main_qty >= 1:
            base_inefficient_fuel, base_inefficient_label, base_inefficient_loads = main_only_inefficient_candidate
            num_main_in_original_inefficient_case = sum(1 for _, _, gen_type in base_inefficient_loads if gen_type == "Ana")

            if num_main_in_original_inefficient_case > 0:
                # Denenecek ana jeneratör sayıları (genellikle orijinalden bir eksik, veya kapasiteye göre 1)
                n_main_options_assisted = []
                if num_main_in_original_inefficient_case > 1:
                    n_main_options_assisted.append(num_main_in_original_inefficient_case - 1)
                if 1 not in n_main_options_assisted : # Her zaman 1 ana jeneratör ile denemeyi de ekle (eğer kapasite yetiyorsa)
                     n_main_options_assisted.append(1)
                
                n_main_options_assisted = sorted(list(set(n for n in n_main_options_assisted if n > 0 and n <= main_qty)))


                for n_main_assisted_count in n_main_options_assisted:
                    # Liman jeneratörü yükünü %85'ten %60'a doğru dene
                    for target_port_load_pct in range(85, 59, -5): # 85, 80, ..., 60
                        port_power_output = port_mcr * (target_port_load_pct / 100.0)
                        
                        # Liman jeneratörü tek başına tüm gücü karşılıyorsa veya aşıyorsa
                        if port_power_output >= required_de_power -1 : # -1 tolerans
                            if n_main_assisted_count > 0: continue # Ana jen. çalışacak yer kalmadı
                            # Eğer n_main_assisted_count = 0 olsaydı (şu an değil), bu port_only durumu olurdu
                        
                        remaining_power_for_main_gens = required_de_power - port_power_output

                        if remaining_power_for_main_gens <= 1: # Ana jeneratörler için çok az/negatif yük
                            if n_main_assisted_count > 0: continue 
                            # Eğer n_main_assisted_count = 0 olsaydı, bu da port_only durumu olurdu
                            # ve port_power_output == required_de_power olmalıydı.
                            # Bu durum zaten STRATEGY 2'de ele alınır.

                        # Ana jeneratörlerin kapasitesi yeterli mi?
                        if n_main_assisted_count * main_mcr < remaining_power_for_main_gens:
                            continue

                        # Ana jeneratör başına düşen güç ve yük
                        main_power_per_gen = remaining_power_for_main_gens / n_main_assisted_count
                        main_load_percentage = (main_power_per_gen / main_mcr) * 100

                        if main_load_percentage > 100.1 : continue # Ana jen. aşırı yüklenmemeli (SFOC %100'e kadar)
                                                                  # veya %110'a kadar SFOC varsa bu limit artırılabilir.

                        # Yakıt Hesaplaması (Option B - doğrudan)
                        fuel_port_assisted = calculate_fuel(port_power_output, target_port_load_pct, duration, sfoc_data)
                        fuel_main_assisted_total = 0
                        valid_main_fuel_calc = True
                        if main_power_per_gen > 0 : # Sadece pozitif güçte yakıt hesapla
                            for _ in range(n_main_assisted_count):
                                fuel_m_part = calculate_fuel(main_power_per_gen, main_load_percentage, duration, sfoc_data)
                                if fuel_m_part is None or fuel_m_part <=0: # Bir ana jen. için yakıt hesaplanamazsa geçersiz
                                    valid_main_fuel_calc = False
                                    break
                                fuel_main_assisted_total += fuel_m_part
                        elif n_main_assisted_count > 0 and main_power_per_gen <=0 : # Ana jen. var ama güç yok
                             valid_main_fuel_calc = False


                        if not valid_main_fuel_calc : continue
                        if fuel_port_assisted is None or fuel_port_assisted <= 0 : continue # Liman jen. yakıtı da geçerli olmalı

                        total_fuel_assisted = fuel_port_assisted + fuel_main_assisted_total
                        
                        # Koşul Kontrolleri
                        port_load_ok = (60 <= target_port_load_pct <= 85)
                        
                        main_gen_load_conditionally_ok = False
                        if 70 <= main_load_percentage <= 85: # İdeal aralık
                            main_gen_load_conditionally_ok = True
                        elif main_load_percentage >= 65: # Kabul edilebilir alt sınır
                            # Orijinal verimsiz yükten daha iyi mi?
                            if original_main_loads_for_inefficient_case:
                                avg_original_main_load = sum(original_main_loads_for_inefficient_case) / len(original_main_loads_for_inefficient_case)
                                # Daha az sayıda jeneratör veya daha iyi yüklenme durumu
                                if main_load_percentage > avg_original_main_load or n_main_assisted_count < num_main_in_original_inefficient_case :
                                    main_gen_load_conditionally_ok = True
                            else: # Orijinal yük bilgisi yoksa, sadece >= %65 kontrolü
                                main_gen_load_conditionally_ok = True
                        
                        if port_load_ok and main_gen_load_conditionally_ok and total_fuel_assisted < base_inefficient_fuel:
                            loads_info_assisted = [(port_mcr, target_port_load_pct, "Liman")] + \
                                                  [(main_mcr, main_load_percentage, "Ana")] * n_main_assisted_count
                            
                            label_assisted = (f"{n_main_assisted_count}x{main_mcr}kW Ana ({main_load_percentage:.1f}%) + "
                                              f"1x{port_mcr}kW Liman ({target_port_load_pct:.1f}%)")
                            
                            add_option("assisted_optimal", total_fuel_assisted, label_assisted, loads_info_assisted,
                                       original_info=(base_inefficient_fuel, base_inefficient_label))
                            # Bu n_main_assisted_count için iyi bir opsiyon bulundu, liman yükü döngüsünden çık.
                            # Daha iyi bir liman yükü (daha düşük yakıtla) bulunabilir diye devam da edilebilir.
                            # Şimdilik ilk bulunan iyi opsiyonda duralım (range 85'ten başladığı için yüksek port load öncelikli).
                            # Ya da en iyisini bulmak için devam etmeli ve add_option kendi içinde en iyiyi tutmalı.
                            # add_option zaten en iyiyi tutuyor, o yüzden break'e gerek yok.
        
        # --- DECISION LOGIC ---
        final_choice_key = None
        preferred_keys = ["main_eff", "main_eff_plus_one", "assisted_optimal", "port_only"] # Öncelik sırası
        
        for p_key in preferred_keys:
            if p_key in evaluated_options:
                # Eğer assisted_optimal ise, zaten base_inefficient_fuel'den iyi olduğu için seçilebilir.
                # Diğer durumlar için, o anki en iyi ile karşılaştır.
                if final_choice_key is None or evaluated_options[p_key][0] < evaluated_options[final_choice_key][0]:
                    final_choice_key = p_key

        # Eğer yukarıdaki kurallarla bir şey seçilemediyse, tüm opsiyonlar arasından en düşük yakıtlıyı seç
        if final_choice_key is None and evaluated_options:
            final_choice_key = min(evaluated_options, key=lambda k: evaluated_options[k][0])

        if final_choice_key:
            fuel, label, loads, original_info = evaluated_options[final_choice_key]
            return fuel, label, loads, original_info
        else:
            return 0.0, "No viable combination", [], None


    @st.cache_data
    def calculate_all_results_new_gens(
        main_mcr, main_qty, port_mcr, port_qty,
        sea_power_range, maneuver_power_range,
        sea_duration, maneuver_duration,
        main_engine_mcr_ref,
        sfoc_data, total_elec_eff_factor
    ):
        results_summary_list = []
        detailed_data_list = []
        generator_usage_data_list = []
        CONVENTIONAL_SHAFT_EFFICIENCY = 0.95

        main_engine_ref_detailed_entries = []
        total_sea_fuel_main_engine_overall = 0
        total_maneuver_fuel_main_engine_overall = 0

        for shaft_power_sea in range(sea_power_range[0], sea_power_range[1] + 1, 100):
            if shaft_power_sea <= 0 or main_engine_mcr_ref <= 0: continue
            main_engine_load_sea = (shaft_power_sea / main_engine_mcr_ref) * 100
            if main_engine_load_sea > 0:
                fuel_main_ref_sea = calculate_fuel(shaft_power_sea, main_engine_load_sea, sea_duration, sfoc_data)
                if fuel_main_ref_sea is not None and fuel_main_ref_sea > 0:
                    total_sea_fuel_main_engine_overall += fuel_main_ref_sea
                    main_engine_ref_detailed_entries.append({
                        "Combo": "Ana Makine Referans", "SpecificComboUsed": "Ana Makine Referans", "Mode": "Seyir",
                        "Shaft Power (kW)": shaft_power_sea, "Required DE Power (kW)": np.nan,
                        "Fuel (ton)": round(fuel_main_ref_sea, 3), "System Type": "Ana Makine",
                        "Load (%)": round(main_engine_load_sea, 2), "Gen Type": "Ana Makine", "N_running_combo": 1,
                        "OriginalMainOnlyFuel (ton)": np.nan, "OriginalMainOnlyLabel": np.nan, "IsAssisted": False
                    })
        for shaft_power_maneuver in range(maneuver_power_range[0], maneuver_power_range[1] + 1, 100):
            if shaft_power_maneuver <= 0 or main_engine_mcr_ref <= 0: continue
            main_engine_load_maneuver = (shaft_power_maneuver / main_engine_mcr_ref) * 100
            if main_engine_load_maneuver > 0:
                fuel_main_ref_maneuver = calculate_fuel(shaft_power_maneuver, main_engine_load_maneuver, maneuver_duration, sfoc_data)
                if fuel_main_ref_maneuver is not None and fuel_main_ref_maneuver > 0:
                    total_maneuver_fuel_main_engine_overall += fuel_main_ref_maneuver
                    main_engine_ref_detailed_entries.append({
                        "Combo": "Ana Makine Referans", "SpecificComboUsed": "Ana Makine Referans", "Mode": "Manevra",
                        "Shaft Power (kW)": shaft_power_maneuver, "Required DE Power (kW)": np.nan,
                        "Fuel (ton)": round(fuel_main_ref_maneuver, 3), "System Type": "Ana Makine",
                        "Load (%)": round(main_engine_load_maneuver, 2), "Gen Type": "Ana Makine", "N_running_combo": 1,
                        "OriginalMainOnlyFuel (ton)": np.nan, "OriginalMainOnlyLabel": np.nan, "IsAssisted": False
                    })
        detailed_data_list.extend(main_engine_ref_detailed_entries)

        current_combo_total_sea_fuel_generators = 0
        current_combo_total_maneuver_fuel_generators = 0
        config_label = f"{main_qty}x{main_mcr}kW Ana"
        if port_qty > 0 and port_mcr > 0: config_label += f" + {port_qty}x{port_mcr}kW Liman"

        for mode_params in [(sea_power_range, sea_duration, "Seyir"), (maneuver_power_range, maneuver_duration, "Manevra")]:
            power_range_loop, duration_loop, mode_label_loop = mode_params
            for shaft_power_loop in range(power_range_loop[0], power_range_loop[1] + 1, 100):
                if shaft_power_loop <= 0: continue
                power_basis_for_de_calculation = shaft_power_loop * CONVENTIONAL_SHAFT_EFFICIENCY
                required_de_power_loop = power_basis_for_de_calculation / total_elec_eff_factor
                if required_de_power_loop <= 0 or not np.isfinite(required_de_power_loop): continue

                fuel_total, combo_used_label, loads_info, original_main_info = get_best_combination(
                    required_de_power_loop, main_mcr, main_qty, port_mcr, port_qty, sfoc_data, duration_loop
                )
                if fuel_total > 0 and loads_info:
                    if mode_label_loop == "Seyir": current_combo_total_sea_fuel_generators += fuel_total
                    else: current_combo_total_maneuver_fuel_generators += fuel_total
                    original_fuel_val, original_label_val, is_assisted = np.nan, np.nan, False
                    if original_main_info:
                        original_fuel_val, original_label_val, is_assisted = round(original_main_info[0], 3), original_main_info[1], True
                    detailed_data_list.append({
                        "Combo": config_label, "SpecificComboUsed": combo_used_label, "Mode": mode_label_loop,
                        "Shaft Power (kW)": shaft_power_loop, "Required DE Power (kW)": round(required_de_power_loop),
                        "Fuel (ton)": round(fuel_total, 3), "System Type": "Jeneratör", "Load (%)": np.nan,
                        "Gen Type": combo_used_label, "N_running_combo": len(loads_info),
                        "OriginalMainOnlyFuel (ton)": original_fuel_val, "OriginalMainOnlyLabel": original_label_val, "IsAssisted": is_assisted
                    })
                    for gen_mcr_running, load_percent_running, gen_type_running in loads_info:
                        generator_usage_data_list.append({
                            "Combo": config_label, "Mode": mode_label_loop, "Required DE Power (kW)": round(required_de_power_loop),
                            "Gen Type": f"{gen_mcr_running} kW {gen_type_running} Jen", "Load Percent": round(load_percent_running, 2),
                            "N_running_combo": len(loads_info)
                        })
        if current_combo_total_sea_fuel_generators > 0 or current_combo_total_maneuver_fuel_generators > 0:
            sea_diff = current_combo_total_sea_fuel_generators - total_sea_fuel_main_engine_overall
            maneuver_diff = current_combo_total_maneuver_fuel_generators - total_maneuver_fuel_main_engine_overall
            results_summary_list.append({
                "Jeneratör Konfigürasyonu": config_label,
                "Toplam Seyir Yakıtı (Jeneratörler) (ton)": round(current_combo_total_sea_fuel_generators, 2),
                "Toplam Manevra Yakıtı (Jeneratörler) (ton)": round(current_combo_total_maneuver_fuel_generators, 2),
                "Seyir Yakıt Farkı (Jen - Ana M. Ref.) (ton)": round(sea_diff, 2),
                "Manevra Yakıt Farkı (Jen - Ana M. Ref.) (ton)": round(maneuver_diff, 2)
            })
        elif not results_summary_list and (total_sea_fuel_main_engine_overall > 0 or total_maneuver_fuel_main_engine_overall > 0):
            results_summary_list.append({
                "Jeneratör Konfigürasyonu": config_label + " (Hesaplama Yok/Verimsiz)",
                "Toplam Seyir Yakıtı (Jeneratörler) (ton)": 0, "Toplam Manevra Yakıtı (Jeneratörler) (ton)": 0,
                "Seyir Yakıt Farkı (Jen - Ana M. Ref.) (ton)": round(-total_sea_fuel_main_engine_overall, 2),
                "Manevra Yakıt Farkı (Jen - Ana M. Ref.) (ton)": round(-total_maneuver_fuel_main_engine_overall, 2)
            })
        return pd.DataFrame(results_summary_list), pd.DataFrame(detailed_data_list), pd.DataFrame(generator_usage_data_list)

    # --- Calculation Button ---
    if st.sidebar.button("Yeni Kombinasyon HESAPLA", key="calculate_button_new"):
        if 'sfoc_data_global' not in globals() or not isinstance(sfoc_data_global, dict) or not sfoc_data_global:
             st.error("SFOC verisi yüklenmemiş veya hatalı. Lütfen ana programda SFOC verilerini kontrol edin.")
        elif total_elec_eff_new_factor < 1e-5:
             st.error("Toplam elektriksel verimlilik çok düşük. Lütfen verimlilik ayarlarını kontrol edin.")
        else:
            st.session_state.results_df_new, st.session_state.detailed_df_new, st.session_state.usage_df_new = calculate_all_results_new_gens(
                main_gen_mcr_new, main_gen_qty_new, port_gen_mcr_new, port_gen_qty_new,
                sea_power_range_new, maneuver_power_range_new,
                sea_duration_new, maneuver_duration_new,
                main_engine_mcr_new,
                sfoc_data_global, total_elec_eff_new_factor
            )
            st.session_state.show_fuel_results_new = True
            if st.session_state.results_df_new.empty and st.session_state.detailed_df_new.empty :
                 st.warning("Hesaplama yapıldı ancak gösterilecek sonuç bulunamadı. Girdi değerlerinizi, SFOC verilerini ve jeneratör konfigürasyonunu kontrol edin.")

    # --- Display Results ---
    if st.session_state.show_fuel_results_new and not st.session_state.results_df_new.empty:
        st.subheader("Özet Sonuçlar (Yeni Kombinasyon)")
        st.dataframe(st.session_state.results_df_new.style.format({
            "Toplam Seyir Yakıtı (Jeneratörler) (ton)": "{:.2f}", "Toplam Manevra Yakıtı (Jeneratörler) (ton)": "{:.2f}",
            "Seyir Yakıt Farkı (Jen - Ana M. Ref.) (ton)": "{:.2f}", "Manevra Yakıt Farkı (Jen - Ana M. Ref.) (ton)": "{:.2f}"
        }), use_container_width=True)
        st.markdown("---")
        st.subheader("Detaylı Grafiksel Analiz (Yeni Kombinasyon)")
        plot_data_source = st.session_state.detailed_df_new[
            (st.session_state.detailed_df_new["Fuel (ton)"].notna()) & (st.session_state.detailed_df_new["Fuel (ton)"] > 0)
        ].copy()
        if not plot_data_source.empty:
            plot_mode_new = st.radio("Analiz Modunu Seçin (Yeni Kombinasyon)", ["Seyir", "Manevra"], horizontal=True, key="plot_mode_new_selector")
            mode_filtered_data = plot_data_source[plot_data_source["Mode"] == plot_mode_new].sort_values(by="Shaft Power (kW)").copy()
            plot_df_transformed_list = []
            if not mode_filtered_data.empty:
                for _, row in mode_filtered_data.iterrows():
                    if row["System Type"] == "Ana Makine":
                        plot_df_transformed_list.append({"Shaft Power (kW)": row["Shaft Power (kW)"], "Fuel (ton)": row["Fuel (ton)"], "DisplayCombo": "Ana Makine Referans", "Mode": row["Mode"]})
                    elif row["System Type"] == "Jeneratör":
                        if row["IsAssisted"] and pd.notna(row["OriginalMainOnlyFuel (ton)"]):
                            plot_df_transformed_list.append({"Shaft Power (kW)": row["Shaft Power (kW)"], "Fuel (ton)": row["Fuel (ton)"], "DisplayCombo": f"{row['SpecificComboUsed']} (Destekli)", "Mode": row["Mode"]})
                            plot_df_transformed_list.append({"Shaft Power (kW)": row["Shaft Power (kW)"], "Fuel (ton)": row["OriginalMainOnlyFuel (ton)"], "DisplayCombo": f"{row['OriginalMainOnlyLabel']} (Karşılaştırma)", "Mode": row["Mode"]})
                        else: 
                            plot_df_transformed_list.append({"Shaft Power (kW)": row["Shaft Power (kW)"], "Fuel (ton)": row["Fuel (ton)"], "DisplayCombo": row["SpecificComboUsed"], "Mode": row["Mode"]})
            transformed_plot_df = pd.DataFrame(plot_df_transformed_list)
            # --- GEÇİCİ HATA AYIKLAMA BÖLÜMÜ BAŞLANGICI ---
            if not transformed_plot_df.empty:
                fig_fuel_comparison_new = px.bar(
                    transformed_plot_df, x="Shaft Power (kW)", y="Fuel (ton)", color="DisplayCombo", barmode="group",
                    title=f"Yakıt Tüketimi Karşılaştırması ({plot_mode_new} Modu - Yeni Kombinasyon)",
                    labels={"Fuel (ton)": "Yakıt (ton)", "Shaft Power (kW)": "Şaft Gücü (kW)", "DisplayCombo": "Sistem / Kombinasyon"}
                )
                fig_fuel_comparison_new.update_layout(bargap=0, bargroupgap=0)
                st.plotly_chart(fig_fuel_comparison_new, use_container_width=True)
            else: st.warning(f"{plot_mode_new} modu için gösterilecek karşılaştırmalı yakıt verisi bulunamadı (transformasyon sonrası).")
        else: st.warning("Yeni jeneratör kombinasyonu veya Ana Makine Referansına ait gösterilecek karşılaştırmalı yakıt verisi bulunamadı (kaynak veri boş).")

        usage_plot_data_raw_new = st.session_state.usage_df_new[st.session_state.usage_df_new["Mode"] == plot_mode_new]

        if not usage_plot_data_raw_new.empty:
            if plot_mode_new == "Seyir":
                # Seyir modunda her güç seviyesi için özetlenmiş veri oluştur
                usage_plot_data_transformed_list = []
                # Required DE Power (kW) değerlerine göre grupla
                for power_level, group in usage_plot_data_raw_new.groupby("Required DE Power (kW)"):
                    if not group.empty:
                        # Bu güç seviyesi için çalışan jeneratörleri ve yüklerini özetle
                        # Jeneratör tiplerine göre grupla ve say
                        gen_summary = {}
                        representative_load_percent = 0.0 # Temsili yük yüzdesi
                        
                        for _, row in group.iterrows():
                            # 'X kW Gen Type' formatından sadece 'X kW Gen Type' kısmını al (e.g., '2400 kW Ana Jen')
                            gen_type_clean = row['Gen Type'].split(' @')[0] 
                            if gen_type_clean not in gen_summary:
                                gen_summary[gen_type_clean] = {'count': 0, 'load_percent': row['Load Percent']}
                            gen_summary[gen_type_clean]['count'] += 1
                            
                        # Özet açıklamayı oluştur (örn: 2x 2400kW Ana @ 75.0%) ve temsili yükü belirle
                        simplified_description_parts = []
                        # Temsili yük olarak (eğer varsa) Ana jeneratörün yükünü, yoksa Liman jeneratörünün yükünü al
                        representative_load_found = False
                        for gen_type, info in gen_summary.items():
                            simplified_description_parts.append(f"{info['count']}x {gen_type.replace(' Jen', '')} @ {info['load_percent']:.1f}%")
                            if not representative_load_found and "Ana" in gen_type:
                                representative_load_percent = info['load_percent']
                                representative_load_found = True
                        # Eğer Ana jen yoksa veya bulunamadıysa Liman jen yükünü al (eğer varsa)
                        if not representative_load_found and "Liman" in gen_summary:
                             representative_load_percent = gen_summary["Liman"]['load_percent']


                        usage_plot_data_transformed_list.append({
                            "Required DE Power (kW)": power_level,
                            "Mode": plot_mode_new,
                            "Running Config": ", ".join(simplified_description_parts),
                            "Representative Load (%)": representative_load_percent, # Tek bir jeneratörün yükü
                            "Number of Generators": group['N_running_combo'].iloc[0] # Toplam çalışan jeneratör sayısı
                        })

                usage_plot_data_to_plot = pd.DataFrame(usage_plot_data_transformed_list).sort_values(by="Required DE Power (kW)")

                if not usage_plot_data_to_plot.empty:
                    fig_usage_new = px.bar(
                        usage_plot_data_to_plot,
                        x="Required DE Power (kW)",
                        y="Representative Load (%)", # Y ekseninde temsili yük
                        hover_data=["Running Config", "Number of Generators"],
                        title=f"Jeneratör Kullanımı ve Yük Dağılımı ({plot_mode_new} Modu - Yeni Kombinasyon)",
                        labels={
                            "Representative Load (%)": "Temsili Jeneratör Yükü (%)",
                            "Required DE Power (kW)": "Gerekli Dizel Elektrik Gücü (kW)",
                            "Running Config": "Çalışan Kombinasyon"
                        },
                    )
                    # Barların üzerine temsili yük yüzdesini ve jeneratör sayısını ekle
                    fig_usage_new.update_traces(text=[f"{row['Representative Load (%)']:.1f}% ({row['Number of Generators']} Jen)" for index, row in usage_plot_data_to_plot.iterrows()],
                                                textposition='outside')
                    fig_usage_new.update_yaxes(range=[0, 110])
                    fig_usage_new.update_layout(bargap=0.1) # Seyir modunda barlar arası hafif boşluk kalsın
                    st.plotly_chart(fig_usage_new, use_container_width=True)
                else:
                     st.warning(f"{plot_mode_new} modu için jeneratör kullanım verisi bulunamadı.")

            else: # Manevra modu - Orijinal mantık ve gruplama devam eder
                usage_plot_data_to_plot = usage_plot_data_raw_new.sort_values(by="Required DE Power (kW)")
                if not usage_plot_data_to_plot.empty:
                     fig_usage_new = px.bar(
                        usage_plot_data_to_plot, x="Required DE Power (kW)", y="Load Percent", color="Gen Type", text_auto=".1f", barmode="group",
                        title=f"Jeneratör Kullanımı ve Yük Dağılımı ({plot_mode_new} Modu - Yeni Kombinasyon)",
                        labels={"Load Percent": "Yük Yüzdesi (%)", "Required DE Power (kW)": "Gerekli Dizel Elektrik Gücü (kW)", "Gen Type": "Jeneratör Tipi"},
                    )
                     fig_usage_new.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
                     fig_usage_new.update_yaxes(range=[0, 110])
                     fig_usage_new.update_layout(bargroupgap=0.05) # Manevra için gruplar arası boşluk
                     st.plotly_chart(fig_usage_new, use_container_width=True)
                else: st.warning(f"{plot_mode_new} modu için jeneratör kullanım verisi bulunamadı.")

        else:
             st.warning(f"{plot_mode_new} modu için jeneratör kullanım verisi bulunamadı.")
        
    elif st.session_state.show_fuel_results_new and st.session_state.results_df_new.empty :
         st.warning("Yeni kombinasyon için hesaplama yapıldı ancak özetlenecek sonuç bulunamadı...")
         if st.session_state.detailed_df_new.empty:
              st.error("Detaylı sonuçlar da boş. Girdi değerlerinizi, SFOC verilerini ve jeneratör konfigürasyonunu tekrar kontrol edin.")

# else:
#    st.info("Lütfen sidebar'dan bir analiz sayfası seçin.")


