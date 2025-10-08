"""
Módulo de operações de processamento e análise de dados de descargas atmosféricas
Autor: Sistema de processamento automático
Data: 2025-10-08
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from datetime import datetime, time
import os
import logging

# Configurar estilo dos gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Classe para análises e operações nos dados de descargas atmosféricas"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa o analisador com os dados
        
        Args:
            df: DataFrame com os dados de descargas atmosféricas
        """
        self.df = df.copy()
        
    def remove_duplicates(self, output_dir: str) -> pd.DataFrame:
        """
        Remove registros duplicados dos dados
        
        Args:
            output_dir: Diretório para salvar relatório
            
        Returns:
            DataFrame sem duplicatas
        """
        logger.info("Iniciando remoção de duplicatas...")
        
        initial_count = len(self.df)
        
        # Identificar duplicatas baseado em datetime, latitude, longitude, classificacao e intensidade
        duplicate_columns = ['datetime', 'latitude', 'longitude', 'classificacao', 'intensidade']
        
        # Marcar duplicatas
        self.df['is_duplicate'] = self.df.duplicated(subset=duplicate_columns, keep='first')
        duplicates_count = self.df['is_duplicate'].sum()
        
        # Remover duplicatas
        df_clean = self.df[~self.df['is_duplicate']].drop('is_duplicate', axis=1)
        final_count = len(df_clean)
        
        # Gerar relatório
        report = []
        report.append("=== RELATÓRIO DE REMOÇÃO DE DUPLICATAS ===")
        report.append(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append(f"Registros iniciais: {initial_count:,}")
        report.append(f"Duplicatas encontradas: {duplicates_count:,}")
        report.append(f"Registros após remoção: {final_count:,}")
        report.append(f"Taxa de duplicação: {(duplicates_count/initial_count)*100:.2f}%")
        report.append(f"Redução de dados: {duplicates_count:,} registros ({(duplicates_count/initial_count)*100:.2f}%)")
        
        # Salvar relatório
        report_path = os.path.join(output_dir, 'relatorio_remocao_duplicatas.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Duplicatas removidas: {duplicates_count:,} de {initial_count:,} registros")
        logger.info(f"Relatório salvo em: {report_path}")
        
        return df_clean
    
    def filter_by_time_period(self, df: pd.DataFrame, start_time: str = "12:00:00", 
                            end_time: str = "23:59:59", output_dir: str = "") -> pd.DataFrame:
        """
        Filtra descargas por período do dia
        
        Args:
            df: DataFrame com os dados
            start_time: Hora de início (formato HH:MM:SS)
            end_time: Hora de fim (formato HH:MM:SS)
            output_dir: Diretório para salvar relatório
            
        Returns:
            DataFrame filtrado
        """
        logger.info(f"Filtrando dados entre {start_time} e {end_time}...")
        
        initial_count = len(df)
        
        # Converter strings de tempo para objetos time
        start_time_obj = datetime.strptime(start_time, "%H:%M:%S").time()
        end_time_obj = datetime.strptime(end_time, "%H:%M:%S").time()
        
        # Filtrar por horário
        mask = (df['datetime'].dt.time >= start_time_obj) & (df['datetime'].dt.time <= end_time_obj)
        df_filtered = df[mask].copy()
        final_count = len(df_filtered)
        
        removed_count = initial_count - final_count
        
        # Gerar relatório
        report = []
        report.append("=== RELATÓRIO DE FILTRO POR HORÁRIO ===")
        report.append(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append(f"Período selecionado: {start_time} às {end_time}")
        report.append(f"Registros iniciais: {initial_count:,}")
        report.append(f"Registros após filtro: {final_count:,}")
        report.append(f"Registros removidos: {removed_count:,}")
        report.append(f"Taxa de retenção: {(final_count/initial_count)*100:.2f}%")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, 'relatorio_filtro_horario.txt')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(report))
            logger.info(f"Relatório salvo em: {report_path}")
        
        logger.info(f"Filtro aplicado: {final_count:,} de {initial_count:,} registros mantidos")
        
        return df_filtered
    
    def create_daily_summary(self, df: pd.DataFrame, output_dir: str):
        """
        Cria arquivo com dados agrupados por dia do ano
        
        Args:
            df: DataFrame com os dados
            output_dir: Diretório para salvar o arquivo
        """
        logger.info("Criando resumo por dia do ano...")
        
        # Agrupar por dia do ano
        daily_summary = df.groupby('dia_do_ano').agg({
            'datetime': 'count',
            'classificacao': lambda x: x.value_counts().to_dict(),
            'sinal_intensidade': lambda x: x.value_counts().to_dict(),
            'intensidade': ['mean', 'std', 'min', 'max'],
            'latitude': ['mean', 'min', 'max'],
            'longitude': ['mean', 'min', 'max']
        }).round(4)
        
        # Renomear colunas
        daily_summary.columns = [
            'total_registros', 'classificacoes', 'sinais_intensidade',
            'intensidade_media', 'intensidade_std', 'intensidade_min', 'intensidade_max',
            'latitude_media', 'latitude_min', 'latitude_max',
            'longitude_media', 'longitude_min', 'longitude_max'
        ]
        
        # Salvar arquivo
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'dados_por_dia_do_ano.csv')
        daily_summary.to_csv(output_path, encoding='utf-8')
        
        # Criar relatório detalhado
        report_lines = []
        report_lines.append("=== RESUMO DE DADOS POR DIA DO ANO ===")
        report_lines.append(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        report_lines.append(f"Total de dias únicos: {len(daily_summary)}")
        report_lines.append(f"Dia com mais registros: {daily_summary['total_registros'].idxmax()} ({daily_summary['total_registros'].max():,} registros)")
        report_lines.append(f"Dia com menos registros: {daily_summary['total_registros'].idxmin()} ({daily_summary['total_registros'].min():,} registros)")
        report_lines.append(f"Média de registros por dia: {daily_summary['total_registros'].mean():.1f}")
        report_lines.append("")
        report_lines.append(f"Arquivo salvo em: {output_path}")
        
        report_path = os.path.join(output_dir, 'relatorio_dados_diarios.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Resumo diário criado: {len(daily_summary)} dias únicos")
        logger.info(f"Arquivo salvo em: {output_path}")
        
    def create_coverage_map(self, df: pd.DataFrame, output_dir: str):
        """
        Cria mapa com bounding box da área de cobertura
        
        Args:
            df: DataFrame com os dados
            output_dir: Diretório para salvar o mapa
        """
        logger.info("Criando mapa de cobertura espacial...")
        
        # Calcular bounding box
        lat_min, lat_max = df['latitude'].min(), df['latitude'].max()
        lon_min, lon_max = df['longitude'].min(), df['longitude'].max()
        
        # Centro do mapa
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        
        # Criar mapa
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=6,
            tiles='OpenStreetMap'
        )
        
        # Adicionar bounding box
        folium.Rectangle(
            bounds=[[lat_min, lon_min], [lat_max, lon_max]],
            color='red',
            weight=3,
            fill=True,
            fillColor='red',
            fillOpacity=0.1,
            popup=f'Área de Cobertura<br>Lat: {lat_min:.3f} a {lat_max:.3f}<br>Lon: {lon_min:.3f} a {lon_max:.3f}'
        ).add_to(m)
        
        # Adicionar marcadores nos cantos
        folium.Marker(
            [lat_min, lon_min],
            popup=f'SW: ({lat_min:.3f}, {lon_min:.3f})',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        folium.Marker(
            [lat_max, lon_max],
            popup=f'NE: ({lat_max:.3f}, {lon_max:.3f})',
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        
        # Salvar mapa
        map_path = os.path.join(output_dir, 'mapa_cobertura_espacial.html')
        m.save(map_path)
        
        # Criar relatório
        report = []
        report.append("=== MAPA DE COBERTURA ESPACIAL ===")
        report.append(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("BOUNDING BOX:")
        report.append(f"  Latitude mínima: {lat_min:.6f}°")
        report.append(f"  Latitude máxima: {lat_max:.6f}°")
        report.append(f"  Longitude mínima: {lon_min:.6f}°")
        report.append(f"  Longitude máxima: {lon_max:.6f}°")
        report.append("")
        report.append(f"  Extensão latitudinal: {lat_max - lat_min:.6f}° ({(lat_max - lat_min) * 111:.1f} km)")
        report.append(f"  Extensão longitudinal: {lon_max - lon_min:.6f}° ({(lon_max - lon_min) * 111 * np.cos(np.radians(center_lat)):.1f} km)")
        report.append("")
        report.append(f"Centro aproximado: ({center_lat:.6f}°, {center_lon:.6f}°)")
        report.append("")
        report.append(f"Mapa salvo em: {map_path}")
        
        report_path = os.path.join(output_dir, 'relatorio_cobertura_espacial.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Mapa de cobertura criado: {map_path}")
        
    def create_yearly_plots(self, df: pd.DataFrame, output_dir: str):
        """
        Cria gráficos de barras com total de raios por ano
        
        Args:
            df: DataFrame com os dados
            output_dir: Diretório para salvar os gráficos
        """
        logger.info("Criando gráficos por ano...")
        
        # Agrupar dados por ano, classificação e sinal
        yearly_data = df.groupby(['ano', 'classificacao', 'sinal_intensidade']).size().reset_index(name='count')
        
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise de Descargas Atmosféricas por Ano', fontsize=16, fontweight='bold')
        
        # 1. Total por ano
        total_by_year = df.groupby('ano').size()
        axes[0,0].bar(total_by_year.index, total_by_year.values, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Total de Descargas por Ano')
        axes[0,0].set_xlabel('Ano')
        axes[0,0].set_ylabel('Número de Descargas')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Por classificação e ano
        pivot_class = df.pivot_table(values='datetime', index='ano', columns='classificacao', 
                                   aggfunc='count', fill_value=0)
        pivot_class.plot(kind='bar', ax=axes[0,1], width=0.8)
        axes[0,1].set_title('Descargas por Ano e Classificação')
        axes[0,1].set_xlabel('Ano')
        axes[0,1].set_ylabel('Número de Descargas')
        axes[0,1].legend(title='Classificação')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Por sinal de intensidade e ano
        pivot_signal = df.pivot_table(values='datetime', index='ano', columns='sinal_intensidade', 
                                    aggfunc='count', fill_value=0)
        pivot_signal.plot(kind='bar', ax=axes[1,0], width=0.8, color=['red', 'blue'])
        axes[1,0].set_title('Descargas por Ano e Sinal de Intensidade')
        axes[1,0].set_xlabel('Ano')
        axes[1,0].set_ylabel('Número de Descargas')
        axes[1,0].legend(title='Sinal')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Heatmap combinado
        pivot_combined = df.pivot_table(values='datetime', 
                                      index='ano', 
                                      columns=['classificacao', 'sinal_intensidade'], 
                                      aggfunc='count', fill_value=0)
        sns.heatmap(pivot_combined.T, ax=axes[1,1], cmap='YlOrRd', annot=True, fmt='d')
        axes[1,1].set_title('Heatmap: Classificação × Sinal × Ano')
        axes[1,1].set_xlabel('Ano')
        
        plt.tight_layout()
        
        # Salvar gráfico
        plot_path = os.path.join(output_dir, 'graficos_analise_anual.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráficos anuais salvos em: {plot_path}")
        
    def create_monthly_plots(self, df: pd.DataFrame, output_dir: str):
        """
        Cria gráficos de barras com total de raios por mês
        
        Args:
            df: DataFrame com os dados
            output_dir: Diretório para salvar os gráficos
        """
        logger.info("Criando gráficos por mês...")
        
        # Criar figura com subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Análise de Descargas Atmosféricas por Mês', fontsize=16, fontweight='bold')
        
        # 1. Total por mês
        total_by_month = df.groupby('mes').size()
        month_names = ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 
                      'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez']
        axes[0,0].bar(range(1, 13), [total_by_month.get(i, 0) for i in range(1, 13)], 
                     color='lightcoral', alpha=0.7)
        axes[0,0].set_title('Total de Descargas por Mês')
        axes[0,0].set_xlabel('Mês')
        axes[0,0].set_ylabel('Número de Descargas')
        axes[0,0].set_xticks(range(1, 13))
        axes[0,0].set_xticklabels(month_names)
        
        # 2. Por classificação e mês
        pivot_class = df.pivot_table(values='datetime', index='mes', columns='classificacao', 
                                   aggfunc='count', fill_value=0)
        pivot_class.plot(kind='bar', ax=axes[0,1], width=0.8)
        axes[0,1].set_title('Descargas por Mês e Classificação')
        axes[0,1].set_xlabel('Mês')
        axes[0,1].set_ylabel('Número de Descargas')
        axes[0,1].legend(title='Classificação')
        # Usar os labels dos meses que realmente têm dados
        month_labels = [month_names[i-1] for i in pivot_class.index]
        axes[0,1].set_xticklabels(month_labels, rotation=45)
        
        # 3. Por sinal de intensidade e mês
        pivot_signal = df.pivot_table(values='datetime', index='mes', columns='sinal_intensidade', 
                                    aggfunc='count', fill_value=0)
        pivot_signal.plot(kind='bar', ax=axes[1,0], width=0.8, color=['red', 'blue'])
        axes[1,0].set_title('Descargas por Mês e Sinal de Intensidade')
        axes[1,0].set_xlabel('Mês')
        axes[1,0].set_ylabel('Número de Descargas')
        axes[1,0].legend(title='Sinal')
        # Usar os labels dos meses que realmente têm dados
        month_labels = [month_names[i-1] for i in pivot_signal.index]
        axes[1,0].set_xticklabels(month_labels, rotation=45)
        
        # 4. Distribuição sazonal
        season_mapping = {
            12: 'Verão', 1: 'Verão', 2: 'Verão',
            3: 'Outono', 4: 'Outono', 5: 'Outono',
            6: 'Inverno', 7: 'Inverno', 8: 'Inverno',
            9: 'Primavera', 10: 'Primavera', 11: 'Primavera'
        }
        df['estacao'] = df['mes'].map(season_mapping)
        seasonal_data = df.groupby(['estacao', 'classificacao']).size().unstack(fill_value=0)
        seasonal_data.plot(kind='bar', ax=axes[1,1], width=0.8)
        axes[1,1].set_title('Descargas por Estação e Classificação')
        axes[1,1].set_xlabel('Estação')
        axes[1,1].set_ylabel('Número de Descargas')
        axes[1,1].legend(title='Classificação')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Salvar gráfico
        plot_path = os.path.join(output_dir, 'graficos_analise_mensal.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráficos mensais salvos em: {plot_path}")
        
    def create_filtered_files(self, df: pd.DataFrame, output_dir: str):
        """
        Cria arquivos filtrados para classificação G com intensidades positiva e negativa
        
        Args:
            df: DataFrame com os dados
            output_dir: Diretório para salvar os arquivos
        """
        logger.info("Criando arquivos filtrados para classificação G...")
        
        # Filtrar classificação G
        df_g = df[df['classificacao'] == 'G'].copy()
        
        # Filtrar intensidade positiva
        df_g_pos = df_g[df_g['sinal_intensidade'] == 'positivo'].copy()
        
        # Filtrar intensidade negativa  
        df_g_neg = df_g[df_g['sinal_intensidade'] == 'negativo'].copy()
        
        # Salvar arquivos
        pos_path = os.path.join(output_dir, 'descargas_G_intensidade_positiva.csv')
        neg_path = os.path.join(output_dir, 'descargas_G_intensidade_negativa.csv')
        
        df_g_pos.to_csv(pos_path, index=False, encoding='utf-8')
        df_g_neg.to_csv(neg_path, index=False, encoding='utf-8')
        
        # Criar relatório
        report = []
        report.append("=== ARQUIVOS FILTRADOS - CLASSIFICAÇÃO G ===")
        report.append(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append(f"Total de registros classificação G: {len(df_g):,}")
        report.append("")
        report.append("INTENSIDADE POSITIVA:")
        report.append(f"  Registros: {len(df_g_pos):,}")
        report.append(f"  Arquivo: {pos_path}")
        report.append(f"  Período: {df_g_pos['datetime'].min()} a {df_g_pos['datetime'].max()}")
        report.append("")
        report.append("INTENSIDADE NEGATIVA:")
        report.append(f"  Registros: {len(df_g_neg):,}")
        report.append(f"  Arquivo: {neg_path}")
        report.append(f"  Período: {df_g_neg['datetime'].min()} a {df_g_neg['datetime'].max()}")
        
        report_path = os.path.join(output_dir, 'relatorio_arquivos_filtrados.txt')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Arquivos filtrados criados:")
        logger.info(f"  Positivos: {len(df_g_pos):,} registros → {pos_path}")
        logger.info(f"  Negativos: {len(df_g_neg):,} registros → {neg_path}")