"""
Testes de validação completa do pipeline RINDAT
Valida todas as criações de arquivos e operações do pipeline
"""

import pytest
import pandas as pd
import os
import tempfile
import shutil
import sys
import unittest

# Adicionar src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import RindatDataProcessor
from data_analyzer import DataAnalyzer

class TestFullPipeline(unittest.TestCase):
    
    def setUp(self):
        """Setup para cada teste (unittest)"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Criar dados de teste com formato atual esperado
        self.sample_data = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2023-01-15 15:30:45.123',
                '2023-02-20 18:15:30.456',
                '2023-03-10 21:45:15.789',
                '2023-04-05 14:20:10.012'
            ]),
            'latitude': [-22.9068, -23.5505, -22.9068, -23.5505],
            'longitude': [-43.1729, -46.6333, -43.1729, -46.6333],
            'classificacao': ['G', 'N', 'G', 'G'],
            'intensidade': [15.5, -8.2, 22.1, -12.7],
            'sinal_intensidade': ['positivo', 'negativo', 'positivo', 'negativo'],
            'ano': [2023, 2023, 2023, 2023],
            'mes': [1, 2, 3, 4],
            'dia_do_ano': [15, 51, 69, 95]
        })
    
    def tearDown(self):
        """Cleanup após cada teste (unittest)"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_directory_creation(self):
        """Testa se todos os diretórios necessários são criados"""
        output_dirs = ['reports', 'plots', 'maps', 'filtered_data']
        
        for dir_name in output_dirs:
            dir_path = os.path.join(self.temp_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
            
            assert os.path.exists(dir_path), f"Diretório {dir_name} não foi criado"
            assert os.path.isdir(dir_path), f"{dir_name} não é um diretório válido"
    
    def test_processor_initialization(self):
        """Testa inicialização do processador"""
        processor = RindatDataProcessor('data')
        assert processor is not None, "RindatDataProcessor não foi inicializado"
        
    def test_analyzer_initialization(self):
        """Testa inicialização do analisador"""
        analyzer = DataAnalyzer(self.sample_data)
        assert analyzer is not None, "DataAnalyzer não foi inicializado"
    
    def test_loading_report_creation(self):
        """Testa criação do relatório de carregamento"""
        processor = RindatDataProcessor('data')
        reports_dir = os.path.join(self.temp_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Testar criação do relatório
        processor.generate_loading_report(self.sample_data, reports_dir)
        
        report_path = os.path.join(reports_dir, 'loading_report.txt')
        assert os.path.exists(report_path), "Relatório de carregamento não foi criado"
        
        # Verificar conteúdo
        with open(report_path, 'r', encoding='utf-8') as f:
            content = f.read()
            assert "RELATÓRIO DE CARREGAMENTO" in content
            assert "Total de registros carregados: 4" in content
    
    def test_duplicate_removal(self):
        """Testa remoção de duplicados"""
        processor = RindatDataProcessor('data')
        
        # Adicionar linha duplicada aos dados de teste
        duplicated_data = pd.concat([self.sample_data, self.sample_data.iloc[[0]]], ignore_index=True)
        
        original_count = len(duplicated_data)
        cleaned_data, reports_dir = processor.remove_duplicates(duplicated_data, self.temp_dir)
        
        # Verificar se duplicados foram removidos
        self.assertEqual(len(cleaned_data), original_count - 1)
        self.assertTrue(os.path.exists(reports_dir))
    
    def test_time_filter_functionality(self):
        """Testa funcionalidade de filtro por horário"""
        analyzer = DataAnalyzer(self.sample_data)
        
        reports_dir = os.path.join(self.temp_dir, 'reports')
        df_filtered = analyzer.filter_by_time_period(self.sample_data, "12:00:00", "23:59:59", reports_dir)
        
        # Todos os dados de teste estão no período 12:00-23:59, então devem ser mantidos
        assert len(df_filtered) == len(self.sample_data), "Filtro por horário removeu dados válidos"
        
        report_path = os.path.join(reports_dir, 'relatorio_filtro_horario.txt')
        assert os.path.exists(report_path), "Relatório de filtro por horário não foi criado"
    
    def test_daily_data_file_creation(self):
        """Testa criação do arquivo de dados diários"""
        analyzer = DataAnalyzer(self.sample_data)
        
        reports_dir = os.path.join(self.temp_dir, 'reports')
        analyzer.create_daily_summary(self.sample_data, reports_dir)
        
        daily_output_path = os.path.join(reports_dir, 'dados_por_dia_do_ano.csv')
        assert os.path.exists(daily_output_path), "Arquivo de dados diários não foi criado"
        
        # Verificar se pode ser lido novamente
        df_read = pd.read_csv(daily_output_path)
        assert not df_read.empty, "Arquivo de dados diários está vazio"
    
    def test_spatial_coverage_map_creation(self):
        """Testa criação do mapa de cobertura espacial"""
        analyzer = DataAnalyzer(self.sample_data)
        
        maps_dir = os.path.join(self.temp_dir, 'maps')
        os.makedirs(maps_dir, exist_ok=True)
        
        # Criar mapa
        analyzer.create_coverage_map(self.sample_data, maps_dir)
        
        map_path = os.path.join(maps_dir, 'mapa_cobertura_espacial.html')
        assert os.path.exists(map_path), "Arquivo do mapa não foi criado"
        
        # Verificar tamanho do arquivo (deve ter conteúdo)
        assert os.path.getsize(map_path) > 100, "Arquivo do mapa parece estar vazio"
    
    def test_yearly_plots_creation(self):
        """Testa criação dos gráficos anuais"""
        analyzer = DataAnalyzer(self.sample_data)
        
        plots_dir = os.path.join(self.temp_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Criar gráficos anuais
        analyzer.create_yearly_plots(self.sample_data, plots_dir)
        
        plot_path = os.path.join(plots_dir, 'graficos_analise_anual.png')
        assert os.path.exists(plot_path), "Gráfico anual não foi criado"
        assert os.path.getsize(plot_path) > 1000, "Gráfico anual parece estar corrompido"
    
    def test_monthly_plots_creation(self):
        """Testa criação dos gráficos mensais"""
        analyzer = DataAnalyzer(self.sample_data)
        
        plots_dir = os.path.join(self.temp_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # Criar gráficos mensais
        analyzer.create_monthly_plots(self.sample_data, plots_dir)
        
        plot_path = os.path.join(plots_dir, 'graficos_analise_mensal.png')
        assert os.path.exists(plot_path), "Gráfico mensal não foi criado"
        assert os.path.getsize(plot_path) > 1000, "Gráfico mensal parece estar corrompido"
    
    def test_classification_g_files_creation(self):
        """Testa criação dos arquivos de classificação G"""
        analyzer = DataAnalyzer(self.sample_data)
        
        filtered_data_dir = os.path.join(self.temp_dir, 'filtered_data')
        os.makedirs(filtered_data_dir, exist_ok=True)
        
        # Criar arquivos filtrados
        analyzer.create_filtered_files(self.sample_data, filtered_data_dir)
        
        # Verificar arquivos G positivos e negativos
        g_pos_path = os.path.join(filtered_data_dir, 'descargas_G_intensidade_positiva.csv')
        g_neg_path = os.path.join(filtered_data_dir, 'descargas_G_intensidade_negativa.csv')
        
        assert os.path.exists(g_pos_path), "Arquivo G positivo não foi criado"
        assert os.path.exists(g_neg_path), "Arquivo G negativo não foi criado"
        
        # Verificar conteúdo
        df_pos = pd.read_csv(g_pos_path)
        df_neg = pd.read_csv(g_neg_path)
        
        # Pelo menos um dos arquivos deve ter dados (baseado nos dados de teste)
        total_g_records = len(df_pos) + len(df_neg)
        assert total_g_records > 0, "Nenhum registro de classificação G foi encontrado"
    
    def test_file_permissions_and_encoding(self):
        """Testa permissões e encoding dos arquivos criados"""
        processor = RindatDataProcessor('data')
        reports_dir = os.path.join(self.temp_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Criar um relatório
        processor.generate_loading_report(self.sample_data, reports_dir)
        
        report_path = os.path.join(reports_dir, 'loading_report.txt')
        
        # Verificar se o arquivo é legível
        assert os.access(report_path, os.R_OK), "Arquivo não tem permissão de leitura"
        
        # Verificar encoding UTF-8
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert content, "Não foi possível ler o arquivo com encoding UTF-8"
        except UnicodeDecodeError:
            pytest.fail("Arquivo não está em encoding UTF-8")
    
    def test_empty_data_handling(self):
        """Testa tratamento de dados vazios"""
        processor = RindatDataProcessor('data')
        empty_df = pd.DataFrame()
        
        reports_dir = os.path.join(self.temp_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Tentar criar relatório com dados vazios (não deve gerar erro)
        try:
            processor.generate_loading_report(empty_df, reports_dir)
            report_path = os.path.join(reports_dir, 'loading_report.txt')
            assert os.path.exists(report_path), "Relatório não foi criado para dados vazios"
        except Exception as e:
            pytest.fail(f"Erro ao processar dados vazios: {e}")

class TestPipelineIntegration:
    """Testes de integração do pipeline completo"""
    
    def setup_method(self):
        """Setup para testes de integração"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Dados de teste mais completos
        self.sample_data = pd.DataFrame({
            'datetime': pd.to_datetime(['2000-06-15 13:30:45.123', 
                                      '2000-07-20 14:15:30.456',
                                      '2000-08-10 15:45:15.789',
                                      '2001-01-05 16:20:05.012',
                                      '2001-02-10 17:30:15.345']),
            'ano': [2000, 2000, 2000, 2001, 2001],
            'mes': [6, 7, 8, 1, 2],
            'dia': [15, 20, 10, 5, 10],
            'hora': [13, 14, 15, 16, 17],
            'minuto': [30, 15, 45, 20, 30],
            'dia_do_ano': ['15/06', '20/07', '10/08', '05/01', '10/02'],
            'latitude': [-22.5, -23.0, -22.8, -23.2, -22.7],
            'longitude': [-43.2, -43.5, -43.1, -43.7, -43.3],
            'classificacao': ['G', 'C', 'G', 'G', 'C'],
            'intensidade': [15.5, -12.3, 20.1, -18.7, 25.2],
            'unidade': ['kA', 'kA', 'kA', 'kA', 'kA'],
            'col8': [1.0, 1.1, 1.2, 1.3, 1.4],
            'col9': [4, 5, 6, 7, 8],
            'sinal_intensidade': ['positivo', 'negativo', 'positivo', 'negativo', 'positivo']
        })
    
    def teardown_method(self):
        """Cleanup após testes de integração"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline_simulation(self):
        """Testa simulação completa do pipeline"""
        # Criar estrutura de diretórios
        output_dirs = ['reports', 'plots', 'maps', 'filtered_data']
        for dir_name in output_dirs:
            dir_path = os.path.join(self.temp_dir, dir_name)
            os.makedirs(dir_path, exist_ok=True)
        
        try:
            analyzer = DataAnalyzer(self.sample_data)
            
            # 1. Remoção de duplicatas
            df_no_duplicates = analyzer.remove_duplicates(os.path.join(self.temp_dir, 'reports'))
            
            # 2. Filtro por horário
            df_filtered = analyzer.filter_by_time_period(
                df_no_duplicates, "12:00:00", "23:59:59", 
                os.path.join(self.temp_dir, 'reports')
            )
            
            # 3. Criar resumo por dia do ano
            analyzer.create_daily_summary(df_filtered, os.path.join(self.temp_dir, 'reports'))
            
            # 4. Criar mapa de cobertura espacial
            analyzer.create_coverage_map(df_filtered, os.path.join(self.temp_dir, 'maps'))
            
            # 5. Criar gráficos anuais
            analyzer.create_yearly_plots(df_filtered, os.path.join(self.temp_dir, 'plots'))
            
            # 6. Criar gráficos mensais
            analyzer.create_monthly_plots(df_filtered, os.path.join(self.temp_dir, 'plots'))
            
            # 7. Criar arquivos filtrados G
            analyzer.create_filtered_files(df_filtered, os.path.join(self.temp_dir, 'filtered_data'))
            
            # Verificar se todos os arquivos principais foram criados
            expected_files = [
                'reports/relatorio_remocao_duplicatas.txt',
                'reports/relatorio_filtro_horario.txt',
                'reports/dados_por_dia_do_ano.csv',
                'maps/mapa_cobertura_espacial.html',
                'plots/graficos_analise_anual.png',
                'plots/graficos_analise_mensal.png',
                'filtered_data/descargas_G_intensidade_positiva.csv',
                'filtered_data/descargas_G_intensidade_negativa.csv'
            ]
            
            missing_files = []
            for file_path in expected_files:
                full_path = os.path.join(self.temp_dir, file_path)
                if not os.path.exists(full_path):
                    missing_files.append(file_path)
            
            assert not missing_files, f"Arquivos não foram criados no pipeline: {missing_files}"
            
            print("✅ Simulação completa do pipeline executada com sucesso!")
            
        except Exception as e:
            pytest.fail(f"Erro na simulação do pipeline completo: {e}")

if __name__ == "__main__":
    # Executar testes diretamente
    pytest.main([__file__, "-v"])