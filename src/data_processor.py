"""
Módulo para processamento de dados de descargas atmosféricas RINDAT
Autor: Sistema de processamento automático
Data: 2025-10-08
"""

import pandas as pd
import numpy as np
import os
import re
from datetime import datetime
from typing import List, Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RindatDataProcessor:
    """Classe principal para processamento dos dados RINDAT"""
    
    def __init__(self, data_path: str):
        """
        Inicializa o processador de dados RINDAT
        
        Args:
            data_path: Caminho para o diretório contendo os arquivos RINDAT
        """
        self.data_path = data_path
        self.columns_old_format = [
            'data', 'horario', 'latitude', 'longitude', 'classificacao', 
            'intensidade', 'unidade', 'col8', 'col9'
        ]
        self.columns_new_format = [
            'datetime', 'latitude', 'longitude', 'intensidade_completa', 
            'classificacao', 'col8', 'col9'
        ]
        
    def identify_file_format(self, filepath: str) -> str:
        """
        Identifica o formato do arquivo baseado na primeira linha
        
        Args:
            filepath: Caminho para o arquivo
            
        Returns:
            'old' para formato antigo (DD/MM/YY), 'new' para formato novo (YYYY-MM-DD)
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                
            # Verifica se é formato novo (YYYY-MM-DD)
            if re.match(r'^\d{4}-\d{2}-\d{2}', first_line):
                return 'new'
            # Verifica se é formato antigo (DD/MM/YY)
            elif re.match(r'^\d{2}/\d{2}/\d{2}', first_line):
                return 'old'
            else:
                raise ValueError(f"Formato não reconhecido no arquivo: {filepath}")
                
        except Exception as e:
            logger.error(f"Erro ao identificar formato do arquivo {filepath}: {e}")
            raise
            
    def parse_old_format(self, filepath: str) -> pd.DataFrame:
        """
        Processa arquivo no formato antigo (DD/MM/YY HH:MM:SS.mmm)
        
        Args:
            filepath: Caminho para o arquivo
            
        Returns:
            DataFrame com os dados processados
        """
        logger.info(f"Processando arquivo formato antigo: {filepath}")
        
        # Ler arquivo como texto simples e processar linha por linha
        data_rows = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Dividir a linha em partes
                    parts = line.split()
                    
                    if len(parts) >= 9:  # Garantir que temos pelo menos 9 colunas
                        # Extrair componentes
                        data = parts[0]  # DD/MM/YY
                        horario = parts[1]  # HH:MM:SS.mmm
                        latitude = float(parts[2])
                        longitude = float(parts[3])
                        classificacao = parts[4]
                        
                        # Processar intensidade (ex: "-15.2" ou "+0.0")
                        intensidade_str = parts[5]
                        intensidade = float(intensidade_str)
                        
                        unidade = parts[6]  # "kA"
                        col8 = float(parts[7]) if parts[7] != '' else 0.0
                        col9 = int(parts[8]) if parts[8] != '' else 0
                        
                        # Converter data/hora para datetime
                        # Ajustar ano: assumir 20xx se YY > 50, senão 19xx
                        year_part = data.split('/')[2]
                        if int(year_part) <= 30:  # Anos 00-30 = 2000-2030
                            full_year = f"20{year_part}"
                        else:  # Anos 31-99 = 1931-1999
                            full_year = f"19{year_part}"
                        
                        full_date = data.replace(f"/{year_part}", f"/{full_year}")
                        
                        # Tratar milissegundos - formato é MM/DD/YYYY
                        if '.' in horario:
                            base_time, milliseconds = horario.split('.')
                            # Milissegundos (3 dígitos) -> microssegundos (6 dígitos)
                            microseconds = milliseconds.ljust(6, '0')[:6]
                            horario_fixed = f"{base_time}.{microseconds}"
                            datetime_obj = datetime.strptime(f"{full_date} {horario_fixed}", "%m/%d/%Y %H:%M:%S.%f")
                        else:
                            datetime_obj = datetime.strptime(f"{full_date} {horario}", "%m/%d/%Y %H:%M:%S")
                        
                        data_rows.append({
                            'datetime': datetime_obj,
                            'latitude': latitude,
                            'longitude': longitude,
                            'classificacao': classificacao,
                            'intensidade': intensidade,
                            'unidade': unidade,
                            'col8': col8,
                            'col9': col9,
                            'sinal_intensidade': 'positivo' if intensidade >= 0 else 'negativo'
                        })
                        
                except Exception as e:
                    logger.warning(f"Erro na linha {line_num} do arquivo {filepath}: {e}")
                    continue
                    
        return pd.DataFrame(data_rows)
    
    def parse_new_format(self, filepath: str) -> pd.DataFrame:
        """
        Processa arquivo no formato novo (YYYY-MM-DD HH:MM:SS.nnnnnnnnn)
        
        Args:
            filepath: Caminho para o arquivo
            
        Returns:
            DataFrame com os dados processados
        """
        logger.info(f"Processando arquivo formato novo: {filepath}")
        
        data_rows = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # Dividir a linha em partes
                    parts = line.split()
                    
                    if len(parts) >= 9:  # Garantir que temos pelo menos 9 colunas
                        # Extrair componentes
                        date_part = parts[0]  # YYYY-MM-DD
                        time_part = parts[1]  # HH:MM:SS.nnnnnnnnn
                        latitude = float(parts[2])
                        longitude = float(parts[3])
                        
                        # Intensidade vem com unidade junto (ex: "-36.2 kA")
                        intensidade_str = parts[4]
                        unidade = parts[5]
                        intensidade = float(intensidade_str)
                        
                        classificacao = parts[6]
                        col8 = int(parts[7]) if parts[7] != '' else 0
                        col9 = int(parts[8]) if parts[8] != '' else 0
                        
                        # Converter data/hora para datetime
                        datetime_str = f"{date_part} {time_part}"
                        try:
                            datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
                        except ValueError:
                            # Lidar com casos onde há muitos dígitos nos microssegundos
                            # Truncar para 6 dígitos
                            if '.' in time_part:
                                base_time, microseconds = time_part.split('.')
                                if len(microseconds) > 6:
                                    microseconds = microseconds[:6]
                                time_part_fixed = f"{base_time}.{microseconds}"
                                datetime_str = f"{date_part} {time_part_fixed}"
                            datetime_obj = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S.%f")
                        
                        data_rows.append({
                            'datetime': datetime_obj,
                            'latitude': latitude,
                            'longitude': longitude,
                            'classificacao': classificacao,
                            'intensidade': intensidade,
                            'unidade': unidade,
                            'col8': col8,
                            'col9': col9,
                            'sinal_intensidade': 'positivo' if intensidade >= 0 else 'negativo'
                        })
                        
                except Exception as e:
                    logger.warning(f"Erro na linha {line_num} do arquivo {filepath}: {e}")
                    continue
                    
        return pd.DataFrame(data_rows)
    
    def load_single_file(self, filepath: str) -> pd.DataFrame:
        """
        Carrega um único arquivo RINDAT
        
        Args:
            filepath: Caminho para o arquivo
            
        Returns:
            DataFrame com os dados do arquivo
        """
        file_format = self.identify_file_format(filepath)
        
        if file_format == 'old':
            return self.parse_old_format(filepath)
        elif file_format == 'new':
            return self.parse_new_format(filepath)
        else:
            raise ValueError(f"Formato não suportado para o arquivo: {filepath}")
    
    def load_all_files(self) -> pd.DataFrame:
        """
        Carrega todos os arquivos RINDAT do diretório
        
        Returns:
            DataFrame combinado com todos os dados
        """
        rindat_path = os.path.join(self.data_path, 'RINDAT')
        
        if not os.path.exists(rindat_path):
            raise FileNotFoundError(f"Diretório RINDAT não encontrado: {rindat_path}")
        
        all_dataframes = []
        files_processed = 0
        total_records = 0
        
        # Listar todos os arquivos .dat
        dat_files = [f for f in os.listdir(rindat_path) if f.endswith('.dat')]
        dat_files.sort()  # Ordenar para processamento consistente
        
        logger.info(f"Encontrados {len(dat_files)} arquivos para processar")
        
        for filename in dat_files:
            filepath = os.path.join(rindat_path, filename)
            
            try:
                df = self.load_single_file(filepath)
                if not df.empty:
                    all_dataframes.append(df)
                    files_processed += 1
                    total_records += len(df)
                    logger.info(f"Arquivo {filename}: {len(df):,} registros carregados")
                else:
                    logger.warning(f"Arquivo {filename} está vazio")
                    
            except Exception as e:
                logger.error(f"Erro ao processar arquivo {filename}: {e}")
                continue
        
        if not all_dataframes:
            raise ValueError("Nenhum arquivo foi carregado com sucesso")
        
        # Combinar todos os DataFrames
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        
        # Adicionar colunas derivadas
        combined_df['ano'] = combined_df['datetime'].dt.year
        combined_df['mes'] = combined_df['datetime'].dt.month
        combined_df['dia'] = combined_df['datetime'].dt.day
        combined_df['hora'] = combined_df['datetime'].dt.hour
        combined_df['minuto'] = combined_df['datetime'].dt.minute
        combined_df['dia_do_ano'] = combined_df['datetime'].dt.strftime('%d/%m')
        
        logger.info(f"Total: {files_processed} arquivos processados, {total_records:,} registros carregados")
        
        return combined_df
    
    def generate_loading_report(self, df: pd.DataFrame, output_dir: str):
        """
        Gera relatório de carregamento de dados
        
        Args:
            df: DataFrame com os dados carregados
            output_dir: Diretório para salvar o relatório
        """
        report = []
        report.append("=== RELATÓRIO DE CARREGAMENTO DE DADOS ===")
        report.append(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Estatísticas gerais
        report.append("ESTATÍSTICAS GERAIS:")
        report.append(f"Total de registros carregados: {len(df):,}")
        
        if len(df) > 0:
            report.append(f"Período dos dados: {df['datetime'].min()} a {df['datetime'].max()}")
            report.append(f"Anos únicos: {sorted(df['ano'].unique())}")
        else:
            report.append("Período dos dados: N/A (dados vazios)")
            report.append("Anos únicos: N/A (dados vazios)")
        report.append("")
        
        # Estatísticas por ano
        report.append("REGISTROS POR ANO:")
        if len(df) > 0:
            year_counts = df.groupby('ano').size().sort_index()
            for year, count in year_counts.items():
                report.append(f"  {year}: {count:,} registros")
        else:
            report.append("  N/A (dados vazios)")
        report.append("")
        
        # Estatísticas por classificação
        report.append("REGISTROS POR CLASSIFICAÇÃO:")
        if len(df) > 0:
            class_counts = df['classificacao'].value_counts()
            for classification, count in class_counts.items():
                report.append(f"  {classification}: {count:,} registros")
        else:
            report.append("  N/A (dados vazios)")
        report.append("")
        
        # Estatísticas por sinal
        report.append("REGISTROS POR SINAL DE INTENSIDADE:")
        if len(df) > 0:
            signal_counts = df['sinal_intensidade'].value_counts()
            for signal, count in signal_counts.items():
                report.append(f"  {signal}: {count:,} registros")
        else:
            report.append("  N/A (dados vazios)")
        report.append("")
        
        # Área de cobertura
        report.append("ÁREA DE COBERTURA:")
        if len(df) > 0:
            report.append(f"  Latitude mín: {df['latitude'].min():.4f}°")
            report.append(f"  Latitude máx: {df['latitude'].max():.4f}°")
            report.append(f"  Longitude mín: {df['longitude'].min():.4f}°")
            report.append(f"  Longitude máx: {df['longitude'].max():.4f}°")
        else:
            report.append("  N/A (dados vazios)")
        
        # Salvar relatório
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'loading_report.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Relatório de carregamento salvo em: {output_path}")
    
    def remove_duplicates(self, df: pd.DataFrame, output_dir: str) -> Tuple[pd.DataFrame, str]:
        """
        Remove duplicados baseado em datetime, latitude e longitude
        
        Args:
            df: DataFrame com os dados
            output_dir: Diretório para salvar o relatório
            
        Returns:
            Tupla com (DataFrame limpo, diretório do relatório)
        """
        logger.info("Removendo duplicados...")
        
        initial_count = len(df)
        
        # Remover duplicados baseado em datetime, latitude e longitude
        df_clean = df.drop_duplicates(subset=['datetime', 'latitude', 'longitude'], keep='first')
        
        final_count = len(df_clean)
        duplicates_removed = initial_count - final_count
        
        # Gerar relatório
        reports_dir = os.path.join(output_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        report = []
        report.append("=== RELATÓRIO DE REMOÇÃO DE DUPLICADOS ===")
        report.append(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append(f"Registros iniciais: {initial_count:,}")
        report.append(f"Registros finais: {final_count:,}")
        report.append(f"Duplicados removidos: {duplicates_removed:,}")
        report.append(f"Percentual removido: {(duplicates_removed/initial_count)*100:.2f}%")
        
        output_path = os.path.join(reports_dir, 'duplicates_report.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Duplicados removidos: {duplicates_removed:,} ({(duplicates_removed/initial_count)*100:.2f}%)")
        
        return df_clean, reports_dir
    
    def filter_by_time(self, df: pd.DataFrame, output_dir: str) -> Tuple[pd.DataFrame, str]:
        """
        Filtra dados por período de tempo (12:00 às 23:59)
        
        Args:
            df: DataFrame com os dados
            output_dir: Diretório para salvar o relatório
            
        Returns:
            Tupla com (DataFrame filtrado, diretório do relatório)
        """
        logger.info("Filtrando dados por período de tempo (12:00-23:59)...")
        
        initial_count = len(df)
        
        # Filtrar por horário
        mask = (df['datetime'].dt.hour >= 12)
        df_filtered = df[mask].copy()
        
        final_count = len(df_filtered)
        filtered_out = initial_count - final_count
        
        # Gerar relatório
        reports_dir = os.path.join(output_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        report = []
        report.append("=== RELATÓRIO DE FILTRO POR HORÁRIO ===")
        report.append(f"Data/Hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append(f"Período: 12:00:00 - 23:59:59")
        report.append(f"Registros iniciais: {initial_count:,}")
        report.append(f"Registros após filtro: {final_count:,}")
        report.append(f"Registros removidos: {filtered_out:,}")
        report.append(f"Percentual mantido: {(final_count/initial_count)*100:.2f}%")
        
        output_path = os.path.join(reports_dir, 'time_filter_report.txt')
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Filtro por horário: {final_count:,} registros mantidos ({(final_count/initial_count)*100:.2f}%)")
        
        return df_filtered, reports_dir