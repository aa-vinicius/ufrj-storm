#!/usr/bin/env python3
"""
Script principal para processamento de dados de descargas atmosféricas RINDAT
Executa todo o pipeline de processamento conforme especificado

Autor: Sistema de processamento automático
Data: 2025-10-08
"""

import os
import sys
import logging
from datetime import datetime

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import RindatDataProcessor
from src.data_analyzer import DataAnalyzer

def setup_logging():
    """Configura o sistema de logging"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler('processing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def create_output_directories():
    """Cria os diretórios de saída necessários"""
    directories = [
        'output',
        'output/reports',
        'output/plots',
        'output/maps',
        'output/filtered_data'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Diretório criado/verificado: {directory}")

def main():
    """Função principal que executa todo o pipeline"""
    
    # Configurar logging
    setup_logging()
    logging.info("=== INICIANDO PROCESSAMENTO DE DADOS RINDAT ===")
    logging.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Criar diretórios de saída
        create_output_directories()
        
        # Passo 1: Carregar dados
        logging.info("\n--- PASSO 1: CARREGAMENTO DOS DADOS ---")
        data_path = 'data'
        processor = RindatDataProcessor(data_path)
        
        logging.info("Carregando todos os arquivos RINDAT...")
        df_raw = processor.load_all_files()
        
        # Gerar relatório de carregamento
        processor.generate_loading_report(df_raw, 'output/reports')
        logging.info(f"Dados carregados: {len(df_raw):,} registros")
        
        # Passo 2: Inicializar analisador
        logging.info("\n--- PASSO 2: INICIALIZANDO ANALISADOR ---")
        analyzer = DataAnalyzer(df_raw)
        
        # Passo 3: Remover duplicatas
        logging.info("\n--- PASSO 3: REMOÇÃO DE DUPLICATAS ---")
        df_no_duplicates = analyzer.remove_duplicates('output/reports')
        
        # Passo 4: Filtrar por horário (12:00 às 23:59)
        logging.info("\n--- PASSO 4: FILTRO POR HORÁRIO (12:00-23:59) ---")
        df_filtered = analyzer.filter_by_time_period(
            df_no_duplicates, 
            "12:00:00", 
            "23:59:59", 
            'output/reports'
        )
        
        # Passo 5: Criar arquivo por dia do ano
        logging.info("\n--- PASSO 5: DADOS POR DIA DO ANO ---")
        analyzer_filtered = DataAnalyzer(df_filtered)
        analyzer_filtered.create_daily_summary(df_filtered, 'output/reports')
        
        # Passo 6: Criar mapa de cobertura
        logging.info("\n--- PASSO 6: MAPA DE COBERTURA ESPACIAL ---")
        analyzer_filtered.create_coverage_map(df_filtered, 'output/maps')
        
        # Passo 7: Gráficos por ano
        logging.info("\n--- PASSO 7: GRÁFICOS POR ANO ---")
        analyzer_filtered.create_yearly_plots(df_filtered, 'output/plots')
        
        # Passo 8: Gráficos por mês
        logging.info("\n--- PASSO 8: GRÁFICOS POR MÊS ---")
        analyzer_filtered.create_monthly_plots(df_filtered, 'output/plots')
        
        # Passo 9: Arquivos filtrados (classificação G)
        logging.info("\n--- PASSO 9: ARQUIVOS FILTRADOS (CLASSIFICAÇÃO G) ---")
        analyzer_filtered.create_filtered_files(df_filtered, 'output/filtered_data')
        
        # Relatório final
        logging.info("\n--- PROCESSAMENTO CONCLUÍDO ---")
        
        final_report = [
            "=== RELATÓRIO FINAL DE PROCESSAMENTO ===",
            f"Data/Hora de conclusão: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "PIPELINE EXECUTADO:",
            "✓ 1. Carregamento dos dados RINDAT",
            "✓ 2. Remoção de duplicatas",
            "✓ 3. Filtro por horário (12:00-23:59)",
            "✓ 4. Criação de arquivo por dia do ano",
            "✓ 5. Mapa de cobertura espacial",
            "✓ 6. Gráficos de análise por ano",
            "✓ 7. Gráficos de análise por mês", 
            "✓ 8. Arquivos filtrados por classificação G",
            "",
            "ESTATÍSTICAS FINAIS:",
            f"Registros iniciais: {len(df_raw):,}",
            f"Após remoção de duplicatas: {len(df_no_duplicates):,}",
            f"Após filtro de horário: {len(df_filtered):,}",
            "",
            "ARQUIVOS GERADOS:",
            "📁 output/reports/ - Relatórios detalhados",
            "📁 output/plots/ - Gráficos e visualizações",
            "📁 output/maps/ - Mapas de cobertura",
            "📁 output/filtered_data/ - Dados filtrados",
            "",
            "Processamento concluído com sucesso! ✅"
        ]
        
        final_report_path = 'output/relatorio_final_processamento.txt'
        with open(final_report_path, 'w', encoding='utf-8') as f:
            f.write('\\n'.join(final_report))
        
        logging.info("\\n".join(final_report))
        logging.info(f"\\nRelatório final salvo em: {final_report_path}")
        
    except Exception as e:
        logging.error(f"Erro durante o processamento: {e}")
        logging.error("Stacktrace:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()