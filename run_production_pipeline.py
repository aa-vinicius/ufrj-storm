#!/usr/bin/env python3
"""
Script de produção para pipeline completo do UFRJ Storm
Executa todo o pipeline de ML e gera visualizações dos resultados.

Uso:
    python run_production_pipeline.py

Saídas:
    - Modelos treinados em models/
    - Plots em results/
    - Relatório de performance
"""

import sys
import os
from pathlib import Path
import warnings
import json
from datetime import datetime
import argparse

# Adicionar diretórios ao path
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / 'src'))
sys.path.append(str(BASE_DIR / 'config'))

# Imports de bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Imports locais
try:
    from models import LightningPredictor, UncertaintyPredictor
    from settings import Config, DataConfig, ModelConfig
except ImportError as e:
    print(f"Erro na importação: {e}")
    print("Verifique se os módulos estão no caminho correto")
    sys.exit(1)

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ProductionPipeline:
    """Pipeline completo de produção para previsão de raios"""
    
    def __init__(self, config_file=None, output_dir="results"):
        """
        Inicializar pipeline de produção
        
        Args:
            config_file: Caminho para arquivo de configuração (opcional)
            output_dir: Diretório para salvar resultados
        """
        self.base_dir = BASE_DIR
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.output_dir = self.base_dir / output_dir
        
        # Criar diretórios necessários
        self.models_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Carregar configurações
        self.config = DataConfig()
        
        # Variáveis do pipeline
        self.df = None
        self.df_processed = None
        self.preprocessor = None
        self.lightning_predictor = None
        self.uncertainty_predictor = None
        self.results = {}
        
        print(f"🚀 PIPELINE DE PRODUÇÃO UFRJ STORM")
        print(f"📁 Diretório base: {self.base_dir}")
        print(f"📁 Saída: {self.output_dir}")
        print("=" * 60)
    
    def load_data(self):
        """Carregar e validar dados"""
        print("\n📊 ETAPA 1: CARREGAMENTO DOS DADOS")
        
        file_path = self.data_dir / self.config.input_file
        
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset não encontrado: {file_path}")
        
        # Carregar dados
        self.df = pd.read_csv(file_path)
        self.df[self.config.date_column] = pd.to_datetime(self.df[self.config.date_column])
        
        print(f"✅ Dados carregados: {self.df.shape[0]} registros, {self.df.shape[1]} colunas")
        print(f"📅 Período: {self.df[self.config.date_column].min().date()} a {self.df[self.config.date_column].max().date()}")
        
        # Validações básicas
        assert self.df.shape[0] > 5000, "Dataset muito pequeno"
        assert self.config.target_column in self.df.columns, "Coluna target não encontrada"
        assert self.config.date_column in self.df.columns, "Coluna de data não encontrada"
        
        print("✅ Validações básicas aprovadas")
    
    def preprocess_data(self):
        """Pré-processar dados"""
        print("\n🔧 ETAPA 2: PRÉ-PROCESSAMENTO")
        
        # Criar pré-processador
        self.preprocessor = DataPreprocessor(self.config)
        
        # Aplicar pré-processamento
        self.df_processed = self.preprocessor.prepare_features(self.df)
        feature_cols = self.preprocessor.get_feature_columns(self.df_processed)
        
        print(f"✅ Features processadas: {len(feature_cols)} colunas")
        print(f"🆕 Features criadas: {len([c for c in feature_cols if c not in self.df.columns])}")
        
        # Dividir dados temporalmente
        train_start = pd.to_datetime(self.config.train_start)
        train_end = pd.to_datetime(self.config.train_end)
        test_start = pd.to_datetime(self.config.test_start)
        test_end = pd.to_datetime(self.config.test_end)
        
        # Filtros temporais
        train_mask = (self.df_processed[self.config.date_column] >= train_start) & \
                     (self.df_processed[self.config.date_column] <= train_end)
        test_mask = (self.df_processed[self.config.date_column] >= test_start) & \
                    (self.df_processed[self.config.date_column] <= test_end)
        
        self.train_data = self.df_processed[train_mask].copy()
        self.test_data = self.df_processed[test_mask].copy()
        
        # Preparar features e targets
        self.X_train = self.train_data[feature_cols]
        self.y_train = self.train_data[self.config.target_column]
        self.X_test = self.test_data[feature_cols]
        self.y_test = self.test_data[self.config.target_column]
        
        # Escalar features
        from sklearn.preprocessing import RobustScaler
        scaler = RobustScaler()
        self.X_train_scaled = pd.DataFrame(
            scaler.fit_transform(self.X_train),
            columns=feature_cols,
            index=self.X_train.index
        )
        self.X_test_scaled = pd.DataFrame(
            scaler.transform(self.X_test),
            columns=feature_cols,
            index=self.X_test.index
        )
        
        print(f"📊 Treino: {self.X_train.shape[0]} registros")
        print(f"📊 Teste: {self.X_test.shape[0]} registros")
        print("✅ Pré-processamento concluído")
    
    def train_models(self):
        """Treinar modelos de ML"""
        print("\n🤖 ETAPA 3: TREINAMENTO DOS MODELOS")
        
        # Modelo 1: Previsão de quantidade de raios
        print("\n📈 Treinando Modelo 1 (Quantidade de Raios):")
        self.lightning_predictor = LightningPredictor(self.config, random_state=42)
        self.lightning_predictor.train_multiple_models(self.X_train_scaled, self.y_train, cv_folds=5)
        
        # Salvar melhor modelo
        model_path = self.models_dir / f"lightning_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        self.lightning_predictor.save_model(filepath=str(model_path))
        
        # Modelo 2: Intervalo de confiança
        print("\n🎯 Treinando Modelo 2 (Intervalo de Confiança):")
        self.uncertainty_predictor = UncertaintyPredictor(self.lightning_predictor, confidence_level=0.95)
        self.uncertainty_predictor.train_uncertainty_model(self.X_train_scaled, self.y_train)
        
        print("✅ Todos os modelos treinados")
    
    def evaluate_models(self):
        """Avaliar performance dos modelos"""
        print("\n📊 ETAPA 4: AVALIAÇÃO DOS MODELOS")
        
        # Avaliação do modelo principal
        test_results = self.lightning_predictor.evaluate_all_models(self.X_test_scaled, self.y_test)
        best_model_name = self.lightning_predictor.best_model_name
        
        # Resultados do melhor modelo
        best_predictions = test_results[best_model_name]['predictions']
        best_metrics = test_results[best_model_name]['metrics']
        
        # Avaliação do modelo de incerteza
        uncertainty_results = self.uncertainty_predictor.evaluate_uncertainty(self.X_test_scaled, self.y_test)
        
        # Armazenar resultados
        self.results = {
            'best_model': best_model_name,
            'predictions': {
                'train': self.lightning_predictor.best_model.predict(self.X_train_scaled),
                'test': best_predictions
            },
            'metrics': best_metrics,
            'uncertainty': uncertainty_results,
            'dates': {
                'train': self.train_data[self.config.date_column],
                'test': self.test_data[self.config.date_column]
            },
            'actuals': {
                'train': self.y_train,
                'test': self.y_test
            }
        }
        
        print(f"\n🏆 MELHOR MODELO: {best_model_name}")
        print(f"   RMSE: {best_metrics['rmse']:.2f}")
        print(f"   MAE:  {best_metrics['mae']:.2f}")
        print(f"   R²:   {best_metrics['r2']:.4f}")
        print(f"   Cobertura IC: {uncertainty_results['coverage']:.1%}")
        print("✅ Avaliação concluída")
    
    def generate_plots(self):
        """Gerar plots da série temporal com predições e intervalos de confiança"""
        print("\n📈 ETAPA 5: GERAÇÃO DE VISUALIZAÇÕES")
        
        # Configurar style
        plt.rcParams.update({
            'figure.figsize': (20, 12),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11
        })
        
        # Plot 1: Série temporal completa (treino + teste)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
        
        # Dados para plot
        train_dates = self.results['dates']['train']
        test_dates = self.results['dates']['test']
        train_actual = self.results['actuals']['train']
        test_actual = self.results['actuals']['test']
        train_pred = self.results['predictions']['train']
        test_pred = self.results['predictions']['test']
        
        # Plot superior: Série temporal completa
        ax1.scatter(train_dates, train_actual, c='blue', alpha=0.6, s=20, label='Observado (Treino)', marker='o')
        ax1.scatter(test_dates, test_actual, c='red', alpha=0.6, s=20, label='Observado (Teste)', marker='o')
        ax1.scatter(train_dates, train_pred, c='lightblue', alpha=0.4, s=15, label='Predito (Treino)', marker='^')
        ax1.scatter(test_dates, test_pred, c='orange', alpha=0.6, s=15, label='Predito (Teste)', marker='^')
        
        ax1.set_title('Série Temporal Completa - Valores Observados vs Preditos', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Data')
        ax1.set_ylabel('Contagem de Raios')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Linha divisória entre treino e teste
        division_date = pd.to_datetime(self.config.test_start)
        ax1.axvline(x=division_date, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Divisão Treino/Teste')
        
        # Plot inferior: Série temporal de teste com intervalo de confiança
        uncertainty_results = self.results['uncertainty']
        
        # Ordenar por data para plot contínuo
        test_data_sorted = self.test_data.sort_values(self.config.date_column)
        sorted_indices = test_data_sorted.index
        
        # Obter dados ordenados
        test_dates_sorted = test_data_sorted[self.config.date_column]
        test_actual_sorted = self.y_test.loc[sorted_indices]
        test_pred_sorted = uncertainty_results['predictions'][self.y_test.index.get_indexer(sorted_indices)]
        lower_bound_sorted = uncertainty_results['lower_bound'][self.y_test.index.get_indexer(sorted_indices)]
        upper_bound_sorted = uncertainty_results['upper_bound'][self.y_test.index.get_indexer(sorted_indices)]
        
        # Plot da região de confiança
        ax2.fill_between(test_dates_sorted, lower_bound_sorted, upper_bound_sorted, 
                        alpha=0.3, color='gray', label='Intervalo de Confiança 95%')
        
        # Plot dos valores
        ax2.scatter(test_dates_sorted, test_actual_sorted, c='red', alpha=0.8, s=30, 
                   label='Observado (Teste)', marker='o', edgecolors='darkred', linewidth=0.5)
        ax2.scatter(test_dates_sorted, test_pred_sorted, c='orange', alpha=0.7, s=25, 
                   label='Predito (Teste)', marker='^', edgecolors='darkorange', linewidth=0.5)
        
        ax2.set_title('Período de Teste - Predições com Intervalo de Confiança 95%', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Data')
        ax2.set_ylabel('Contagem de Raios')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar plot
        plot_path = self.output_dir / f"temporal_series_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot salvo: {plot_path}")
        
        # Plot 2: Análise detalhada do período de teste
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # Subplot 1: Predito vs Real
        ax1.scatter(test_actual, test_pred, alpha=0.6, s=30, c='blue', edgecolors='navy', linewidth=0.5)
        ax1.plot([test_actual.min(), test_actual.max()], [test_actual.min(), test_actual.max()], 
                'r--', lw=2, label='Linha Perfeita')
        ax1.set_xlabel('Valores Observados')
        ax1.set_ylabel('Valores Preditos')
        ax1.set_title('Predito vs Observado - Período de Teste')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Adicionar R²
        r2 = self.results['metrics']['r2']
        ax1.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax1.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Subplot 2: Resíduos
        residuals = test_actual - test_pred
        ax2.scatter(test_pred, residuals, alpha=0.6, s=30, c='green', edgecolors='darkgreen', linewidth=0.5)
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax2.set_xlabel('Valores Preditos')
        ax2.set_ylabel('Resíduos')
        ax2.set_title('Análise de Resíduos')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Distribuição dos resíduos
        ax3.hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
        ax3.axvline(residuals.mean(), color='red', linestyle='--', 
                   label=f'Média: {residuals.mean():.1f}')
        ax3.set_xlabel('Resíduos')
        ax3.set_ylabel('Frequência')
        ax3.set_title('Distribuição dos Resíduos')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Cobertura do IC por faixas
        bins = [0, 100, 500, 1000, 5000, float('inf')]
        labels = ['0-100', '100-500', '500-1K', '1K-5K', '5K+']
        coverages = []
        counts = []
        
        for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (test_actual >= low) & (test_actual < high)
            if mask.sum() > 0:
                coverage = uncertainty_results['within_interval'][mask].mean()
                coverages.append(coverage)
                counts.append(mask.sum())
            else:
                coverages.append(0)
                counts.append(0)
        
        bars = ax4.bar(labels, coverages, alpha=0.7, color='lightcoral', edgecolor='darkred')
        ax4.axhline(y=0.95, color='green', linestyle='--', linewidth=2, label='95% Esperado')
        ax4.set_ylabel('Taxa de Cobertura')
        ax4.set_title('Cobertura do IC por Faixa de Valores')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Adicionar contagem nas barras
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'n={count}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        # Salvar plot detalhado
        detail_plot_path = self.output_dir / f"detailed_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(detail_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot detalhado salvo: {detail_plot_path}")
        
        plt.show()
    
    def generate_monthly_plots(self):
        """Gerar plots de acumulados mensais"""
        print("\n📊 GERANDO PLOTS DE ACUMULADOS MENSAIS")
        
        # Criar DataFrame com resultados
        train_results = pd.DataFrame({
            'date': self.results['dates']['train'],
            'observed': self.results['actuals']['train'],
            'predicted': self.results['predictions']['train'],
            'dataset': 'train'
        })
        
        test_results = pd.DataFrame({
            'date': self.results['dates']['test'],
            'observed': self.results['actuals']['test'],
            'predicted': self.results['predictions']['test'],
            'dataset': 'test'
        })
        
        # Combinar resultados
        all_results = pd.concat([train_results, test_results], ignore_index=True)
        
        # Calcular acumulados mensais
        all_results['year_month'] = all_results['date'].dt.to_period('M')
        monthly_results = all_results.groupby(['year_month', 'dataset']).agg({
            'observed': 'sum',
            'predicted': 'sum'
        }).reset_index()
        
        # Separar treino e teste
        monthly_train = monthly_results[monthly_results['dataset'] == 'train'].copy()
        monthly_test = monthly_results[monthly_results['dataset'] == 'test'].copy()
        monthly_train['plot_date'] = monthly_train['year_month'].dt.to_timestamp()
        monthly_test['plot_date'] = monthly_test['year_month'].dt.to_timestamp()
        
        # Calcular incerteza mensal simplificada
        uncertainty_results = self.results['uncertainty']
        
        # Criar DataFrame com incerteza
        train_uncertainty_df = pd.DataFrame({
            'date': self.results['dates']['train'],
            'uncertainty': np.full(len(self.results['dates']['train']), 
                                 uncertainty_results['avg_interval_width'] / (2 * 1.96)),
            'dataset': 'train'
        })
        
        test_uncertainty_df = pd.DataFrame({
            'date': self.results['dates']['test'],
            'uncertainty': (uncertainty_results['upper_bound'] - uncertainty_results['lower_bound']) / (2 * 1.96),
            'dataset': 'test'
        })
        
        all_uncertainty = pd.concat([train_uncertainty_df, test_uncertainty_df], ignore_index=True)
        all_uncertainty['year_month'] = all_uncertainty['date'].dt.to_period('M')
        
        # Somar incertezas mensais
        monthly_uncertainty = all_uncertainty.groupby(['year_month', 'dataset']).agg({
            'uncertainty': lambda x: np.sqrt(np.sum(x**2))
        }).reset_index()
        
        # Mesclar com resultados mensais
        monthly_train = monthly_train.merge(
            monthly_uncertainty[monthly_uncertainty['dataset'] == 'train'][['year_month', 'uncertainty']], 
            on='year_month', how='left'
        )
        monthly_test = monthly_test.merge(
            monthly_uncertainty[monthly_uncertainty['dataset'] == 'test'][['year_month', 'uncertainty']], 
            on='year_month', how='left'
        )
        
        # Calcular intervalos de confiança
        z_score = 1.96
        monthly_train['lower_bound'] = np.maximum(monthly_train['predicted'] - z_score * monthly_train['uncertainty'], 0)
        monthly_train['upper_bound'] = monthly_train['predicted'] + z_score * monthly_train['uncertainty']
        monthly_test['lower_bound'] = np.maximum(monthly_test['predicted'] - z_score * monthly_test['uncertainty'], 0)
        monthly_test['upper_bound'] = monthly_test['predicted'] + z_score * monthly_test['uncertainty']
        
        # Criar visualização mensal
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
        
        # Plot superior: Série temporal completa mensal
        ax1.scatter(monthly_train['plot_date'], monthly_train['observed'], 
                   c='blue', alpha=0.7, s=60, label='Observado (Treino)', marker='o', edgecolors='darkblue', linewidth=1)
        ax1.scatter(monthly_train['plot_date'], monthly_train['predicted'], 
                   c='lightblue', alpha=0.8, s=45, label='Predito (Treino)', marker='^', edgecolors='blue', linewidth=1)
        ax1.scatter(monthly_test['plot_date'], monthly_test['observed'], 
                   c='red', alpha=0.8, s=60, label='Observado (Teste)', marker='o', edgecolors='darkred', linewidth=1)
        ax1.scatter(monthly_test['plot_date'], monthly_test['predicted'], 
                   c='orange', alpha=0.8, s=45, label='Predito (Teste)', marker='^', edgecolors='darkorange', linewidth=1)
        
        division_date = pd.to_datetime(self.config.test_start)
        ax1.axvline(x=division_date, color='green', linestyle='--', linewidth=3, alpha=0.8, label='Divisão Treino/Teste')
        
        ax1.set_title('UFRJ Storm - Acumulados Mensais de Raios (2000-2019)', fontsize=18, fontweight='bold')
        ax1.set_xlabel('Data', fontsize=14)
        ax1.set_ylabel('Raios Acumulados por Mês', fontsize=14)
        ax1.legend(loc='upper right', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Plot inferior: Período de teste com intervalo de confiança
        ax2.fill_between(monthly_test['plot_date'], monthly_test['lower_bound'], monthly_test['upper_bound'], 
                        alpha=0.3, color='gray', label='Intervalo de Confiança 95%')
        ax2.scatter(monthly_test['plot_date'], monthly_test['observed'], 
                   c='red', alpha=0.9, s=70, label='Observado (Teste)', marker='o', edgecolors='darkred', linewidth=1.2)
        ax2.scatter(monthly_test['plot_date'], monthly_test['predicted'], 
                   c='orange', alpha=0.9, s=60, label='Predito (Teste)', marker='^', edgecolors='darkorange', linewidth=1.2)
        ax2.plot(monthly_test['plot_date'], monthly_test['observed'], 'r-', alpha=0.6, linewidth=2, label='_nolegend_')
        ax2.plot(monthly_test['plot_date'], monthly_test['predicted'], 'orange', alpha=0.6, linewidth=2, linestyle='--', label='_nolegend_')
        
        # Calcular cobertura mensal
        within_interval = ((monthly_test['observed'] >= monthly_test['lower_bound']) & 
                          (monthly_test['observed'] <= monthly_test['upper_bound']))
        monthly_coverage = within_interval.mean()
        
        ax2.set_title(f'Acumulados Mensais - Período de Teste com IC 95% (Cobertura: {monthly_coverage:.1%})', 
                      fontsize=18, fontweight='bold')
        ax2.set_xlabel('Data', fontsize=14)
        ax2.set_ylabel('Raios Acumulados por Mês', fontsize=14)
        ax2.legend(loc='upper right', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Salvar plot mensal
        monthly_plot_path = self.output_dir / f"monthly_accumulations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(monthly_plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Plot mensal salvo: {monthly_plot_path}")
        
        plt.show()
        
        # Estatísticas mensais
        print(f"\n📊 ESTATÍSTICAS DOS ACUMULADOS MENSAIS:")
        print(f"   🔵 Treino - Observado: {monthly_train['observed'].mean():.0f} ± {monthly_train['observed'].std():.0f} raios/mês")
        print(f"   🔵 Treino - Predito: {monthly_train['predicted'].mean():.0f} ± {monthly_train['predicted'].std():.0f} raios/mês")
        print(f"   🔴 Teste - Observado: {monthly_test['observed'].mean():.0f} ± {monthly_test['observed'].std():.0f} raios/mês")
        print(f"   🔴 Teste - Predito: {monthly_test['predicted'].mean():.0f} ± {monthly_test['predicted'].std():.0f} raios/mês")
        print(f"   🎯 Cobertura IC Mensal: {monthly_coverage:.1%}")
        
        return monthly_plot_path
    
    def generate_30day_plot(self):
        """Gerar plot detalhado de 30 dias contínuos"""
        print("\n🔍 GERANDO PLOT DE 30 DIAS CONTÍNUOS")
        
        # Dados do período de teste ordenados por data
        test_data_sorted = self.test_data.sort_values(self.config.date_column)
        
        # Procurar janela de 30 dias com mais pontos
        best_start = None
        max_points = 0
        window_size = 30
        
        for i in range(len(test_data_sorted) - 1):
            start_date = test_data_sorted.iloc[i][self.config.date_column]
            end_date = start_date + pd.Timedelta(days=window_size)
            
            # Contar pontos nesta janela
            window_mask = ((test_data_sorted[self.config.date_column] >= start_date) & 
                           (test_data_sorted[self.config.date_column] <= end_date))
            points_in_window = window_mask.sum()
            
            if points_in_window > max_points:
                max_points = points_in_window
                best_start = start_date
        
        # Se não encontrou uma boa janela, usar o meio do período
        if best_start is None or max_points < 5:
            mid_point = len(test_data_sorted) // 2
            best_start = test_data_sorted.iloc[mid_point][self.config.date_column]
        
        end_date = best_start + pd.Timedelta(days=window_size)
        
        # Filtrar dados para a janela de 30 dias
        window_mask = ((test_data_sorted[self.config.date_column] >= best_start) & 
                       (test_data_sorted[self.config.date_column] <= end_date))
        window_data = test_data_sorted[window_mask].copy()
        
        if len(window_data) > 0:
            # Obter dados da janela
            window_indices = window_data.index
            window_dates = window_data[self.config.date_column]
            window_observed = self.y_test.loc[window_indices]
            
            # Obter predições para esta janela
            test_pred = self.results['predictions']['test']
            test_lower = self.results['uncertainty']['lower_bound']
            test_upper = self.results['uncertainty']['upper_bound']
            
            window_pred = test_pred[self.y_test.index.get_indexer(window_indices)]
            window_lower = test_lower[self.y_test.index.get_indexer(window_indices)]
            window_upper = test_upper[self.y_test.index.get_indexer(window_indices)]
            
            # Criar plot detalhado
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            
            # Região do intervalo de confiança
            ax.fill_between(window_dates, window_lower, window_upper, 
                           alpha=0.3, color='lightblue', label='Intervalo de Confiança 95%')
            
            # Pontos observados e preditos
            ax.scatter(window_dates, window_observed, c='red', alpha=0.9, s=80, 
                      label='Observado', marker='o', edgecolors='darkred', linewidth=1.5, zorder=5)
            ax.scatter(window_dates, window_pred, c='orange', alpha=0.9, s=70, 
                      label='Predito', marker='^', edgecolors='darkorange', linewidth=1.5, zorder=5)
            
            # Linhas conectando os pontos
            ax.plot(window_dates, window_observed, 'r-', alpha=0.7, linewidth=3, 
                   label='Tendência Observada', zorder=4)
            ax.plot(window_dates, window_pred, color='orange', alpha=0.7, linewidth=3, 
                   linestyle='--', label='Tendência Predita', zorder=4)
            
            # Calcular métricas para esta janela
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            window_rmse = np.sqrt(mean_squared_error(window_observed, window_pred))
            window_mae = mean_absolute_error(window_observed, window_pred)
            window_r2 = r2_score(window_observed, window_pred) if len(window_observed) > 1 else 0
            
            # Cobertura do IC nesta janela
            window_within = ((window_observed >= window_lower) & (window_observed <= window_upper))
            window_coverage = window_within.mean()
            
            ax.set_title(f'Zoom: {window_size} Dias Contínuos ({best_start.strftime("%Y-%m-%d")} a {end_date.strftime("%Y-%m-%d")})\n'
                        f'RMSE: {window_rmse:.1f} | R²: {window_r2:.3f} | Cobertura IC: {window_coverage:.1%}', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Data', fontsize=12)
            ax.set_ylabel('Contagem de Raios', fontsize=12)
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Rotacionar labels das datas
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            
            # Salvar plot detalhado
            detail30_plot_path = self.output_dir / f"detailed_30days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(detail30_plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot de 30 dias salvo: {detail30_plot_path}")
            
            plt.show()
            
            print(f"📊 Janela analisada: {len(window_data)} pontos em {window_size} dias")
            print(f"📈 Período: {best_start.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
            print(f"🎯 Métricas da janela - RMSE: {window_rmse:.1f}, R²: {window_r2:.3f}")
            
            return detail30_plot_path
        else:
            print("⚠️ Não foi possível criar plot de 30 dias - dados insuficientes")
            return None
    
    def generate_specific_period_plot(self):
        """Gerar plot para período específico 2018-11-01 a 2019-03-31"""
        print("\n📅 GERANDO PLOT DO PERÍODO ESPECÍFICO (2018-11-01 a 2019-03-31)")
        
        # Definir período específico
        specific_start = pd.to_datetime("2018-11-01")
        specific_end = pd.to_datetime("2019-03-31")
        
        # Dados do período de teste ordenados por data
        test_data_sorted = self.test_data.sort_values(self.config.date_column)
        
        # Filtrar dados para o período específico
        specific_mask = ((test_data_sorted[self.config.date_column] >= specific_start) & 
                        (test_data_sorted[self.config.date_column] <= specific_end))
        specific_data = test_data_sorted[specific_mask].copy()
        
        if len(specific_data) > 0:
            # Obter dados do período específico
            specific_indices = specific_data.index
            specific_dates = specific_data[self.config.date_column]
            specific_observed = self.y_test.loc[specific_indices]
            
            # Obter predições para este período específico
            test_pred = self.results['predictions']['test']
            test_lower = self.results['uncertainty']['lower_bound']
            test_upper = self.results['uncertainty']['upper_bound']
            
            specific_pred = test_pred[self.y_test.index.get_indexer(specific_indices)]
            specific_lower = test_lower[self.y_test.index.get_indexer(specific_indices)]
            specific_upper = test_upper[self.y_test.index.get_indexer(specific_indices)]
            
            # Criar plot específico
            fig, ax = plt.subplots(1, 1, figsize=(16, 8))
            
            # Região do intervalo de confiança
            ax.fill_between(specific_dates, specific_lower, specific_upper, 
                           alpha=0.3, color='lightcoral', label='Intervalo de Confiança 95%')
            
            # Pontos observados e preditos
            ax.scatter(specific_dates, specific_observed, c='darkred', alpha=0.9, s=80, 
                      label='Observado', marker='o', edgecolors='maroon', linewidth=1.5, zorder=5)
            ax.scatter(specific_dates, specific_pred, c='darkorange', alpha=0.9, s=70, 
                      label='Predito', marker='^', edgecolors='orangered', linewidth=1.5, zorder=5)
            
            # Linhas conectando os pontos
            ax.plot(specific_dates, specific_observed, 'darkred', alpha=0.7, linewidth=3, 
                   label='Tendência Observada', zorder=4)
            ax.plot(specific_dates, specific_pred, color='darkorange', alpha=0.7, linewidth=3, 
                   linestyle='--', label='Tendência Predita', zorder=4)
            
            # Calcular métricas para este período
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            specific_rmse = np.sqrt(mean_squared_error(specific_observed, specific_pred))
            specific_mae = mean_absolute_error(specific_observed, specific_pred)
            specific_r2 = r2_score(specific_observed, specific_pred) if len(specific_observed) > 1 else 0
            
            # Cobertura do IC neste período
            specific_within = ((specific_observed >= specific_lower) & (specific_observed <= specific_upper))
            specific_coverage = specific_within.mean()
            
            # Adicionar marcadores para início de cada mês
            months_in_period = pd.date_range(start=specific_start, end=specific_end, freq='MS')
            for month_start in months_in_period:
                ax.axvline(x=month_start, color='gray', linestyle=':', alpha=0.5, linewidth=1)
            
            ax.set_title(f'Período Específico: Nov/2018 - Mar/2019 ({len(specific_data)} pontos)\n'
                        f'RMSE: {specific_rmse:.1f} | MAE: {specific_mae:.1f} | R²: {specific_r2:.3f} | Cobertura IC: {specific_coverage:.1%}', 
                        fontsize=16, fontweight='bold')
            ax.set_xlabel('Data', fontsize=12)
            ax.set_ylabel('Contagem de Raios', fontsize=12)
            ax.legend(loc='upper right', fontsize=11)
            ax.grid(True, alpha=0.3)
            
            # Rotacionar labels das datas
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            # Configurar formato das datas no eixo x
            import matplotlib.dates as mdates
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            
            plt.tight_layout()
            
            # Salvar plot específico
            specific_plot_path = self.output_dir / f"specific_period_2018-2019_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(specific_plot_path, dpi=300, bbox_inches='tight')
            print(f"✅ Plot do período específico salvo: {specific_plot_path}")
            
            plt.show()
            
            print(f"📊 Período analisado: {len(specific_data)} pontos de {specific_start.strftime('%d/%m/%Y')} a {specific_end.strftime('%d/%m/%Y')}")
            print(f"📈 Duração: {(specific_end - specific_start).days} dias")
            print(f"🎯 Métricas do período - RMSE: {specific_rmse:.1f}, MAE: {specific_mae:.1f}, R²: {specific_r2:.3f}")
            
            return specific_plot_path
        else:
            print("⚠️ Não foi possível criar plot do período específico - dados insuficientes")
            return None
    
    def generate_report(self):
        """Gerar relatório final"""
        print("\n📋 ETAPA 6: GERAÇÃO DE RELATÓRIO")
        
        # Criar relatório
        report = {
            'execution_timestamp': datetime.now().isoformat(),
            'data_info': {
                'total_records': len(self.df),
                'train_records': len(self.train_data),
                'test_records': len(self.test_data),
                'features_count': len(self.X_train.columns),
                'train_period': f"{self.config.train_start} to {self.config.train_end}",
                'test_period': f"{self.config.test_start} to {self.config.test_end}"
            },
            'model_performance': {
                'best_model': self.results['best_model'],
                'rmse': float(self.results['metrics']['rmse']),
                'mae': float(self.results['metrics']['mae']),
                'r2': float(self.results['metrics']['r2'])
            },
            'uncertainty_model': {
                'coverage': float(self.results['uncertainty']['coverage']),
                'avg_interval_width': float(self.results['uncertainty']['avg_interval_width']),
                'expected_coverage': 0.95
            },
            'feature_importance': {},
            'data_quality': {
                'missing_values': self.df.isnull().sum().sum(),
                'outliers_detected': len(self.df[self.df[self.config.target_column] > 
                                                self.df[self.config.target_column].quantile(0.75) + 
                                                1.5 * (self.df[self.config.target_column].quantile(0.75) - 
                                                      self.df[self.config.target_column].quantile(0.25))]),
                'zero_lightning_days': (self.df[self.config.target_column] == 0).sum()
            }
        }
        
        # Adicionar importância das features se disponível
        try:
            feature_importance = self.lightning_predictor.get_feature_importance()
            if feature_importance is not None:
                feature_names = self.X_train.columns
                importance_dict = {name: float(imp) for name, imp in zip(feature_names, feature_importance)}
                # Top 10 features
                sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
                report['feature_importance'] = dict(sorted_features[:10])
        except:
            pass
        
        # Salvar relatório
        report_path = self.output_dir / f"production_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Relatório salvo: {report_path}")
        
        # Imprimir resumo
        print(f"\n🎯 RESUMO EXECUTIVO:")
        print(f"   Modelo selecionado: {report['model_performance']['best_model']}")
        print(f"   RMSE: {report['model_performance']['rmse']:.2f}")
        print(f"   R²: {report['model_performance']['r2']:.4f}")
        print(f"   Cobertura IC: {report['uncertainty_model']['coverage']:.1%}")
        print(f"   Registros processados: {report['data_info']['total_records']}")
    
    def run_complete_pipeline(self):
        """Executar pipeline completo"""
        start_time = datetime.now()
        
        try:
            # Executar todas as etapas
            self.load_data()
            self.preprocess_data()
            self.train_models()
            self.evaluate_models()
            self.generate_plots()
            self.generate_monthly_plots()
            self.generate_30day_plot()
            self.generate_specific_period_plot()
            self.generate_report()
            
            # Tempo total
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
            print(f"⏱️  Tempo total: {duration}")
            print(f"📁 Resultados salvos em: {self.output_dir}")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n❌ ERRO NO PIPELINE: {e}")
            raise

# Classe para pré-processamento (compatibilidade)
class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.column_mapping = {}
    
    def clean_feature_names(self, df):
        df_clean = df.copy()
        column_mapping = {}
        for col in df_clean.columns:
            clean_col = col.replace('[', '_').replace(']', '_').replace('(', '_').replace(')', '_')
            clean_col = clean_col.replace(' ', '_').replace('-', '_')
            if clean_col != col:
                column_mapping[col] = clean_col
        
        if column_mapping:
            df_clean = df_clean.rename(columns=column_mapping)
        
        return df_clean, column_mapping
    
    def prepare_features(self, df):
        df_processed = df.copy()
        df_processed, self.column_mapping = self.clean_feature_names(df_processed)
        
        date_col = self.column_mapping.get(self.config.date_column, self.config.date_column)
        df_processed['year'] = df_processed[date_col].dt.year
        df_processed['month'] = df_processed[date_col].dt.month
        df_processed['day_of_year'] = df_processed[date_col].dt.dayofyear
        df_processed['day_of_month'] = df_processed[date_col].dt.day
        df_processed['weekday'] = df_processed[date_col].dt.weekday
        
        df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
        df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
        df_processed['day_of_year_sin'] = np.sin(2 * np.pi * df_processed['day_of_year'] / 365)
        df_processed['day_of_year_cos'] = np.cos(2 * np.pi * df_processed['day_of_year'] / 365)
        
        binary_col = self.column_mapping.get(self.config.binary_target_column, self.config.binary_target_column)
        df_processed['has_lightning'] = df_processed[binary_col].map({'sim': 1, 'nao': 0})
        
        target_col = self.column_mapping.get(self.config.target_column, self.config.target_column)
        df_processed['log_target'] = np.log1p(df_processed[target_col])
        
        return df_processed
    
    def get_feature_columns(self, df_processed):
        date_col = self.column_mapping.get(self.config.date_column, self.config.date_column)
        target_col = self.column_mapping.get(self.config.target_column, self.config.target_column)
        binary_col = self.column_mapping.get(self.config.binary_target_column, self.config.binary_target_column)
        
        exclude_cols = [date_col, target_col, binary_col, 'log_target']
        feature_cols = [col for col in df_processed.columns if col not in exclude_cols]
        return feature_cols

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description='Pipeline de Produção UFRJ Storm')
    parser.add_argument('--output-dir', default='results', help='Diretório de saída')
    parser.add_argument('--config', help='Arquivo de configuração (opcional)')
    
    args = parser.parse_args()
    
    # Executar pipeline
    pipeline = ProductionPipeline(
        config_file=args.config,
        output_dir=args.output_dir
    )
    
    pipeline.run_complete_pipeline()

if __name__ == "__main__":
    main()