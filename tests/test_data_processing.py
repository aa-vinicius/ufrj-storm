"""
Testes unitários para o módulo de processamento de dados RINDAT
"""

import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime
import sys

# Adicionar o diretório src ao path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_processor import RindatDataProcessor
from data_analyzer import DataAnalyzer

class TestRindatDataProcessor(unittest.TestCase):
    """Testes para a classe RindatDataProcessor"""
    
    def setUp(self):
        """Configuração inicial para os testes"""
        self.temp_dir = tempfile.mkdtemp()
        self.rindat_dir = os.path.join(self.temp_dir, 'RINDAT')
        os.makedirs(self.rindat_dir)
        
        # Criar dados de teste para formato antigo
        old_format_data = """04/18/00 00:00:04.338 -23.6257 -52.7229 G -15.2 kA 1.3 6
04/18/00 00:00:04.338 -23.4782 -52.4854 G +0.0 kA 1.1 4
04/18/00 00:00:04.457 -23.6267 -52.7494 G -28.6 kA 3.0 8"""
        
        with open(os.path.join(self.rindat_dir, 'test_old.dat'), 'w') as f:
            f.write(old_format_data)
        
        # Criar dados de teste para formato novo
        new_format_data = """2021-01-01 00:00:05.859154688 -26.6665 -44.2377 -36.2 kA G 1 4
2021-01-01 00:00:05.908600576 -26.6903 -44.2011 -24.0 kA G 1 4
2021-01-01 00:00:06.011168768 -26.6774 -44.2105 +29.7 kA C 1 4"""
        
        with open(os.path.join(self.rindat_dir, 'test_new.dat'), 'w') as f:
            f.write(new_format_data)
        
        self.processor = RindatDataProcessor(self.temp_dir)
    
    def tearDown(self):
        """Limpeza após os testes"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_identify_file_format(self):
        """Testa identificação do formato dos arquivos"""
        old_file = os.path.join(self.rindat_dir, 'test_old.dat')
        new_file = os.path.join(self.rindat_dir, 'test_new.dat')
        
        self.assertEqual(self.processor.identify_file_format(old_file), 'old')
        self.assertEqual(self.processor.identify_file_format(new_file), 'new')
    
    def test_parse_old_format(self):
        """Testa processamento do formato antigo"""
        old_file = os.path.join(self.rindat_dir, 'test_old.dat')
        df = self.processor.parse_old_format(old_file)
        
        self.assertEqual(len(df), 3)
        self.assertIn('datetime', df.columns)
        self.assertIn('latitude', df.columns)
        self.assertIn('longitude', df.columns)
        self.assertIn('classificacao', df.columns)
        self.assertIn('intensidade', df.columns)
        self.assertIn('sinal_intensidade', df.columns)
        
        # Verificar tipos de dados
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['datetime']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['latitude']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['longitude']))
        
        # Verificar valores específicos
        self.assertEqual(df.iloc[0]['classificacao'], 'G')
        self.assertEqual(df.iloc[0]['intensidade'], -15.2)
        self.assertEqual(df.iloc[0]['sinal_intensidade'], 'negativo')
        self.assertEqual(df.iloc[1]['sinal_intensidade'], 'positivo')  # +0.0
    
    def test_parse_new_format(self):
        """Testa processamento do formato novo"""
        new_file = os.path.join(self.rindat_dir, 'test_new.dat')
        df = self.processor.parse_new_format(new_file)
        
        self.assertEqual(len(df), 3)
        self.assertIn('datetime', df.columns)
        self.assertIn('latitude', df.columns)
        self.assertIn('longitude', df.columns)
        self.assertIn('classificacao', df.columns)
        self.assertIn('intensidade', df.columns)
        self.assertIn('sinal_intensidade', df.columns)
        
        # Verificar valores específicos
        self.assertEqual(df.iloc[0]['classificacao'], 'G')
        self.assertEqual(df.iloc[0]['intensidade'], -36.2)
        self.assertEqual(df.iloc[0]['sinal_intensidade'], 'negativo')
        self.assertEqual(df.iloc[2]['classificacao'], 'C')
        self.assertEqual(df.iloc[2]['sinal_intensidade'], 'positivo')  # +29.7
    
    def test_load_all_files(self):
        """Testa carregamento de todos os arquivos"""
        df = self.processor.load_all_files()
        
        # Deve ter 6 registros no total (3 + 3)
        self.assertEqual(len(df), 6)
        
        # Verificar se tem dados de ambos os formatos
        years = df['ano'].unique()
        self.assertIn(2000, years)
        self.assertIn(2021, years)
        
        # Verificar colunas derivadas
        self.assertIn('ano', df.columns)
        self.assertIn('mes', df.columns)
        self.assertIn('dia', df.columns)
        self.assertIn('hora', df.columns)
        self.assertIn('dia_do_ano', df.columns)

class TestDataAnalyzer(unittest.TestCase):
    """Testes para a classe DataAnalyzer"""
    
    def setUp(self):
        """Configuração inicial para os testes"""
        # Criar dados de teste
        dates = pd.date_range('2020-01-01 10:00:00', periods=100, freq='h')
        
        self.test_data = pd.DataFrame({
            'datetime': dates,
            'latitude': np.random.uniform(-30, -20, 100),
            'longitude': np.random.uniform(-55, -40, 100),
            'classificacao': np.random.choice(['G', 'C'], 100),
            'intensidade': np.random.uniform(-50, 50, 100),
            'unidade': ['kA'] * 100,
            'col8': np.random.uniform(0, 5, 100),
            'col9': np.random.randint(1, 10, 100),
            'sinal_intensidade': np.random.choice(['positivo', 'negativo'], 100),
            'ano': [2020] * 100,
            'mes': [d.month for d in dates],
            'dia': [d.day for d in dates],
            'hora': [d.hour for d in dates],
            'minuto': [d.minute for d in dates],
            'dia_do_ano': [d.strftime('%d/%m') for d in dates]
        })
        
        # Adicionar algumas duplicatas intencionalmente
        self.test_data = pd.concat([self.test_data, self.test_data.iloc[:5]], ignore_index=True)
        
        self.temp_dir = tempfile.mkdtemp()
        self.analyzer = DataAnalyzer(self.test_data)
    
    def tearDown(self):
        """Limpeza após os testes"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_remove_duplicates(self):
        """Testa remoção de duplicatas"""
        initial_count = len(self.analyzer.df)
        df_clean = self.analyzer.remove_duplicates(self.temp_dir)
        
        # Deve ter removido 5 duplicatas
        self.assertEqual(len(df_clean), initial_count - 5)
        
        # Verificar se o relatório foi criado
        report_path = os.path.join(self.temp_dir, 'relatorio_remocao_duplicatas.txt')
        self.assertTrue(os.path.exists(report_path))
    
    def test_filter_by_time_period(self):
        """Testa filtro por período de horário"""
        df_clean = self.analyzer.remove_duplicates(self.temp_dir)
        
        # Filtrar entre 12:00 e 15:00
        df_filtered = self.analyzer.filter_by_time_period(
            df_clean, "12:00:00", "15:00:00", self.temp_dir
        )
        
        # Verificar se todos os registros estão no período correto
        for _, row in df_filtered.iterrows():
            hour = row['datetime'].hour
            self.assertTrue(12 <= hour <= 15)
        
        # Verificar se o relatório foi criado
        report_path = os.path.join(self.temp_dir, 'relatorio_filtro_horario.txt')
        self.assertTrue(os.path.exists(report_path))
    
    def test_create_daily_summary(self):
        """Testa criação do resumo diário"""
        df_clean = self.analyzer.remove_duplicates(self.temp_dir)
        
        self.analyzer.create_daily_summary(df_clean, self.temp_dir)
        
        # Verificar se os arquivos foram criados
        summary_path = os.path.join(self.temp_dir, 'dados_por_dia_do_ano.csv')
        report_path = os.path.join(self.temp_dir, 'relatorio_dados_diarios.txt')
        
        self.assertTrue(os.path.exists(summary_path))
        self.assertTrue(os.path.exists(report_path))
        
        # Verificar conteúdo do arquivo CSV
        df_summary = pd.read_csv(summary_path)
        self.assertGreater(len(df_summary), 0)
        self.assertIn('total_registros', df_summary.columns)
    
    def test_create_filtered_files(self):
        """Testa criação de arquivos filtrados"""
        df_clean = self.analyzer.remove_duplicates(self.temp_dir)
        
        self.analyzer.create_filtered_files(df_clean, self.temp_dir)
        
        # Verificar se os arquivos foram criados
        pos_path = os.path.join(self.temp_dir, 'descargas_G_intensidade_positiva.csv')
        neg_path = os.path.join(self.temp_dir, 'descargas_G_intensidade_negativa.csv')
        report_path = os.path.join(self.temp_dir, 'relatorio_arquivos_filtrados.txt')
        
        self.assertTrue(os.path.exists(pos_path))
        self.assertTrue(os.path.exists(neg_path))
        self.assertTrue(os.path.exists(report_path))

if __name__ == '__main__':
    # Executar todos os testes
    unittest.main(verbosity=2)