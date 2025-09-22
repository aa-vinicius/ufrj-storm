#!/usr/bin/env python3
"""
Script de demonstração rápida do pipeline UFRJ Storm
Versão otimizada para execução rápida com visualizações completas.
"""

import sys
import os
from pathlib import Path
import warnings
import json
from datetime import datetime

# Setup
BASE_DIR = Path(__file__).parent
sys.path.append(str(BASE_DIR / 'src'))
sys.path.append(str(BASE_DIR / 'config'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')

print("🚀 PIPELINE RÁPIDO UFRJ STORM - DEMONSTRAÇÃO")
print("=" * 60)

# Configurações básicas
class SimpleConfig:
    def __init__(self):
        self.input_file = "tma_sp.csv"
        self.date_column = "data"
        self.target_column = "contagem_raios"
        self.binary_target_column = "sim_nao"
        self.train_start = "2000-01-01"
        self.train_end = "2014-12-31"
        self.test_start = "2015-01-01"
        self.test_end = "2019-12-31"

config = SimpleConfig()

# 1. CARREGAMENTO DOS DADOS
print("\n📊 ETAPA 1: CARREGAMENTO DOS DADOS")
data_path = BASE_DIR / "data" / config.input_file
df = pd.read_csv(data_path)
df[config.date_column] = pd.to_datetime(df[config.date_column])

print(f"✅ Dados carregados: {df.shape[0]} registros, {df.shape[1]} colunas")
print(f"📅 Período: {df[config.date_column].min().date()} a {df[config.date_column].max().date()}")

# 2. PRÉ-PROCESSAMENTO SIMPLIFICADO
print("\n🔧 ETAPA 2: PRÉ-PROCESSAMENTO")

# Limpar nomes das colunas
df_clean = df.copy()
column_mapping = {}
for col in df_clean.columns:
    clean_col = col.replace('[', '_').replace(']', '_').replace('(', '_').replace(')', '_')
    clean_col = clean_col.replace(' ', '_')
    if clean_col != col:
        column_mapping[col] = clean_col

if column_mapping:
    df_clean = df_clean.rename(columns=column_mapping)

# Features temporais básicas
df_clean['year'] = df_clean[config.date_column].dt.year
df_clean['month'] = df_clean[config.date_column].dt.month
df_clean['day_of_year'] = df_clean[config.date_column].dt.dayofyear

# Variável binária numérica
df_clean['has_lightning'] = df_clean[config.binary_target_column].map({'sim': 1, 'nao': 0})

# Selecionar features (excluir colunas não numéricas e target)
exclude_cols = [config.date_column, config.target_column, config.binary_target_column]
feature_cols = [col for col in df_clean.columns if col not in exclude_cols and df_clean[col].dtype in ['float64', 'int64']]

print(f"✅ Features selecionadas: {len(feature_cols)} colunas")

# 3. DIVISÃO TEMPORAL DOS DADOS
print("\n📊 ETAPA 3: DIVISÃO DOS DADOS")

train_start = pd.to_datetime(config.train_start)
train_end = pd.to_datetime(config.train_end)
test_start = pd.to_datetime(config.test_start)
test_end = pd.to_datetime(config.test_end)

train_mask = (df_clean[config.date_column] >= train_start) & (df_clean[config.date_column] <= train_end)
test_mask = (df_clean[config.date_column] >= test_start) & (df_clean[config.date_column] <= test_end)

train_data = df_clean[train_mask].copy()
test_data = df_clean[test_mask].copy()

# Preparar dados para modelo
X_train = train_data[feature_cols]
y_train = train_data[config.target_column]
X_test = test_data[feature_cols]
y_test = test_data[config.target_column]

# Escalar dados
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"📊 Dados de treino: {X_train.shape[0]} registros")
print(f"📊 Dados de teste: {X_test.shape[0]} registros")

# Importar as classes da biblioteca
from models import LightningPredictor, UncertaintyPredictor

# 4. TREINAMENTO DO MODELO
print("\n🤖 ETAPA 4: TREINAMENTO DO MODELO")

# Usar a biblioteca models.py - versão rápida só com Random Forest
lightning_predictor = LightningPredictor(config, random_state=42)

# Treinar apenas Random Forest para velocidade
models = {
    'random_forest': RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
}

# Treinar o modelo
rf_model = models['random_forest']
rf_model.fit(X_train_scaled, y_train)

# Simular a estrutura da classe para compatibilidade
lightning_predictor.models = models
lightning_predictor.best_model = rf_model
lightning_predictor.best_model_name = 'random_forest'

print("✅ Modelo Random Forest treinado")

# 5. PREDIÇÕES
print("\n📈 ETAPA 5: GERANDO PREDIÇÕES")

# Usar o método da classe para avaliação
train_pred_raw, train_metrics = lightning_predictor.evaluate_model(rf_model, X_train_scaled, y_train, "TREINO")
test_pred_raw, test_metrics = lightning_predictor.evaluate_model(rf_model, X_test_scaled, y_test, "TESTE")

# Extrair predições
train_pred = train_pred_raw
test_pred = test_pred_raw

# Extrair métricas
test_rmse = test_metrics['rmse']
test_mae = test_metrics['mae']
test_r2 = test_metrics['r2']

# 6. INTERVALO DE CONFIANÇA USANDO A BIBLIOTECA
print("\n🎯 ETAPA 6: CALCULANDO INTERVALO DE CONFIANÇA")

# Usar a classe UncertaintyPredictor
uncertainty_predictor = UncertaintyPredictor(lightning_predictor, confidence_level=0.95)
uncertainty_predictor.train_uncertainty_model(X_train_scaled, y_train)

# Avaliar incerteza
uncertainty_results = uncertainty_predictor.evaluate_uncertainty(X_test_scaled, y_test)

# Extrair resultados
test_pred = uncertainty_results['predictions']  # Sobrescrever com predições da biblioteca
lower_bound = uncertainty_results['lower_bound']
upper_bound = uncertainty_results['upper_bound']
coverage = uncertainty_results['coverage']

# 7. GERAR VISUALIZAÇÕES
print("\n📈 ETAPA 7: GERANDO VISUALIZAÇÕES")

# Criar diretório de resultados
results_dir = BASE_DIR / "results"
results_dir.mkdir(exist_ok=True)

# Configurar plots
plt.rcParams.update({
    'figure.figsize': (20, 12),
    'font.size': 12,
    'axes.titlesize': 16,
    'axes.labelsize': 14
})

# PLOT 1: SÉRIE TEMPORAL COMPLETA
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))

# Dados para plot
train_dates = train_data[config.date_column]
test_dates = test_data[config.date_column]

# Plot superior: Série temporal completa
ax1.scatter(train_dates, y_train, c='blue', alpha=0.6, s=30, label='Observado (Treino)', marker='o')
ax1.scatter(test_dates, y_test, c='red', alpha=0.7, s=30, label='Observado (Teste)', marker='o')
ax1.scatter(train_dates, train_pred, c='lightblue', alpha=0.4, s=20, label='Predito (Treino)', marker='^')
ax1.scatter(test_dates, test_pred, c='orange', alpha=0.7, s=20, label='Predito (Teste)', marker='^')

# Linha divisória
division_date = pd.to_datetime(config.test_start)
ax1.axvline(x=division_date, color='green', linestyle='--', linewidth=3, alpha=0.8, label='Divisão Treino/Teste')

ax1.set_title('UFRJ Storm - Série Temporal Completa de Raios (2000-2019)', fontsize=18, fontweight='bold')
ax1.set_xlabel('Data', fontsize=14)
ax1.set_ylabel('Contagem de Raios', fontsize=14)
ax1.legend(loc='upper right', fontsize=12)
ax1.grid(True, alpha=0.3)

# Plot inferior: Período de teste com intervalo de confiança
# Ordenar por data para plot contínuo
test_data_sorted = test_data.sort_values(config.date_column)
sorted_indices = test_data_sorted.index

test_dates_sorted = test_data_sorted[config.date_column]
y_test_sorted = y_test.loc[sorted_indices]
test_pred_sorted = test_pred[y_test.index.get_indexer(sorted_indices)]
lower_sorted = lower_bound[y_test.index.get_indexer(sorted_indices)]
upper_sorted = upper_bound[y_test.index.get_indexer(sorted_indices)]

# Região de confiança
ax2.fill_between(test_dates_sorted, lower_sorted, upper_sorted, 
                alpha=0.3, color='gray', label='Intervalo de Confiança 95%')

# Valores observados e preditos
ax2.scatter(test_dates_sorted, y_test_sorted, c='red', alpha=0.8, s=40, 
           label='Observado (Teste)', marker='o', edgecolors='darkred', linewidth=0.8)
ax2.scatter(test_dates_sorted, test_pred_sorted, c='orange', alpha=0.8, s=35, 
           label='Predito (Teste)', marker='^', edgecolors='darkorange', linewidth=0.8)

ax2.set_title(f'Período de Teste com IC 95% - Cobertura: {coverage:.1%}', fontsize=18, fontweight='bold')
ax2.set_xlabel('Data', fontsize=14)
ax2.set_ylabel('Contagem de Raios', fontsize=14)
ax2.legend(loc='upper right', fontsize=12)
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# Salvar plot
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
plot1_path = results_dir / f"temporal_series_complete_{timestamp}.png"
plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
print(f"✅ Plot 1 salvo: {plot1_path}")

plt.show()

# PLOT 2: ANÁLISE DETALHADA
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

# Subplot 1: Predito vs Real
ax1.scatter(y_test, test_pred, alpha=0.6, s=40, c='blue', edgecolors='navy', linewidth=0.8)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3, label='Linha Perfeita')
ax1.set_xlabel('Valores Observados', fontsize=12)
ax1.set_ylabel('Valores Preditos', fontsize=12)
ax1.set_title('Predito vs Observado - Período de Teste', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, f'R² = {test_r2:.3f}', transform=ax1.transAxes, fontsize=14,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

# Subplot 2: Resíduos
residuals = y_test - test_pred
ax2.scatter(test_pred, residuals, alpha=0.6, s=40, c='green', edgecolors='darkgreen', linewidth=0.8)
ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
ax2.set_xlabel('Valores Preditos', fontsize=12)
ax2.set_ylabel('Resíduos', fontsize=12)
ax2.set_title('Análise de Resíduos', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Subplot 3: Distribuição dos resíduos
ax3.hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='skyblue')
ax3.axvline(residuals.mean(), color='red', linestyle='--', linewidth=2,
           label=f'Média: {residuals.mean():.1f}')
ax3.set_xlabel('Resíduos', fontsize=12)
ax3.set_ylabel('Frequência', fontsize=12)
ax3.set_title('Distribuição dos Resíduos', fontsize=14, fontweight='bold')
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

# Subplot 4: Série temporal focada (últimos 200 pontos)
recent_indices = test_data_sorted.tail(200).index
recent_dates = test_dates_sorted.tail(200)
recent_observed = y_test_sorted.tail(200)
recent_predicted = test_pred_sorted[-200:]
recent_lower = lower_sorted[-200:]
recent_upper = upper_sorted[-200:]

ax4.fill_between(recent_dates, recent_lower, recent_upper, alpha=0.3, color='gray', label='IC 95%')
ax4.plot(recent_dates, recent_observed, 'ro-', alpha=0.8, markersize=6, linewidth=2, label='Observado')
ax4.plot(recent_dates, recent_predicted, '^-', color='orange', alpha=0.8, markersize=5, linewidth=2, label='Predito')
ax4.set_xlabel('Data', fontsize=12)
ax4.set_ylabel('Contagem de Raios', fontsize=12)
ax4.set_title('Zoom: Últimos 200 Pontos do Período de Teste', fontsize=14, fontweight='bold')
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()

# Salvar plot detalhado
plot2_path = results_dir / f"detailed_analysis_{timestamp}.png"
plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
print(f"✅ Plot 2 salvo: {plot2_path}")

plt.show()

# PLOT 3: ACUMULADOS MENSAIS
print("\n📊 Gerando plot de acumulados mensais...")

# Criar DataFrame com resultados
train_results = pd.DataFrame({
    'date': train_data[config.date_column],
    'observed': y_train.values,
    'predicted': train_pred,
    'dataset': 'train'
})

test_results = pd.DataFrame({
    'date': test_data[config.date_column],
    'observed': y_test.values,
    'predicted': test_pred,
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

# Calcular largura do intervalo de confiança para análise mensal
test_uncertainty = upper_bound - lower_bound

# Calcular incerteza mensal
train_uncertainty_df = pd.DataFrame({
    'date': train_data[config.date_column],
    'uncertainty': np.repeat(test_uncertainty.mean(), len(train_data)),
    'dataset': 'train'
})

test_uncertainty_df = pd.DataFrame({
    'date': test_data[config.date_column],
    'uncertainty': test_uncertainty,
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

division_date = pd.to_datetime(config.test_start)
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
plot3_path = results_dir / f"monthly_accumulations_{timestamp}.png"
plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
print(f"✅ Plot 3 (Mensal) salvo: {plot3_path}")

plt.show()

# PLOT 4: ZOOM EM 30 DIAS CONTÍNUOS DO PERÍODO DE TESTE
print("\n🔍 Gerando plot detalhado de 30 dias...")

# Encontrar período contínuo de 30 dias com mais dados
test_data_sorted = test_data.sort_values(config.date_column)

# Procurar janela de 30 dias com mais pontos
best_start = None
max_points = 0
window_size = 30

for i in range(len(test_data_sorted) - 1):
    start_date = test_data_sorted.iloc[i][config.date_column]
    end_date = start_date + pd.Timedelta(days=window_size)
    
    # Contar pontos nesta janela
    window_mask = ((test_data_sorted[config.date_column] >= start_date) & 
                   (test_data_sorted[config.date_column] <= end_date))
    points_in_window = window_mask.sum()
    
    if points_in_window > max_points:
        max_points = points_in_window
        best_start = start_date

# Se não encontrou uma boa janela, usar o meio do período
if best_start is None or max_points < 5:
    mid_point = len(test_data_sorted) // 2
    best_start = test_data_sorted.iloc[mid_point][config.date_column]

end_date = best_start + pd.Timedelta(days=window_size)

# Filtrar dados para a janela de 30 dias
window_mask = ((test_data_sorted[config.date_column] >= best_start) & 
               (test_data_sorted[config.date_column] <= end_date))
window_data = test_data_sorted[window_mask].copy()

if len(window_data) > 0:
    # Obter predições para esta janela
    window_indices = window_data.index
    window_dates = window_data[config.date_column]
    window_observed = y_test.loc[window_indices]
    window_pred = test_pred[y_test.index.get_indexer(window_indices)]
    window_lower = lower_bound[y_test.index.get_indexer(window_indices)]
    window_upper = upper_bound[y_test.index.get_indexer(window_indices)]
    
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
    plot4_path = results_dir / f"detailed_30days_{timestamp}.png"
    plt.savefig(plot4_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot 4 (30 dias) salvo: {plot4_path}")
    
    plt.show()
    
    print(f"📊 Janela analisada: {len(window_data)} pontos em {window_size} dias")
    print(f"📈 Período: {best_start.strftime('%d/%m/%Y')} a {end_date.strftime('%d/%m/%Y')}")
    print(f"🎯 Métricas da janela - RMSE: {window_rmse:.1f}, R²: {window_r2:.3f}")
    
else:
    print("⚠️ Não foi possível criar plot de 30 dias - dados insuficientes")
    plot4_path = None

# PLOT 5: PERÍODO ESPECÍFICO 2018-11-01 a 2019-03-31
print("\n📅 Gerando plot do período específico (2018-11-01 a 2019-03-31)...")

# Definir período específico
specific_start = pd.to_datetime("2018-11-01")
specific_end = pd.to_datetime("2019-03-31")

# Filtrar dados para o período específico
specific_mask = ((test_data_sorted[config.date_column] >= specific_start) & 
                (test_data_sorted[config.date_column] <= specific_end))
specific_data = test_data_sorted[specific_mask].copy()

if len(specific_data) > 0:
    # Obter predições para este período específico
    specific_indices = specific_data.index
    specific_dates = specific_data[config.date_column]
    specific_observed = y_test.loc[specific_indices]
    specific_pred = test_pred[y_test.index.get_indexer(specific_indices)]
    specific_lower = lower_bound[y_test.index.get_indexer(specific_indices)]
    specific_upper = upper_bound[y_test.index.get_indexer(specific_indices)]
    
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
    plot5_path = results_dir / f"specific_period_2018-2019_{timestamp}.png"
    plt.savefig(plot5_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot 5 (Período Específico) salvo: {plot5_path}")
    
    plt.show()
    
    print(f"📊 Período analisado: {len(specific_data)} pontos de {specific_start.strftime('%d/%m/%Y')} a {specific_end.strftime('%d/%m/%Y')}")
    print(f"📈 Duração: {(specific_end - specific_start).days} dias")
    print(f"🎯 Métricas do período - RMSE: {specific_rmse:.1f}, MAE: {specific_mae:.1f}, R²: {specific_r2:.3f}")
    
else:
    print("⚠️ Não foi possível criar plot do período específico - dados insuficientes")
    plot5_path = None

# 8. RELATÓRIO FINAL
print("\n📋 ETAPA 8: RELATÓRIO FINAL")

# Feature importance usando a biblioteca
feature_importance = lightning_predictor.get_feature_importance()
top_features = sorted(zip(feature_cols, feature_importance), key=lambda x: x[1], reverse=True)[:10]

# Criar relatório
report = {
    'execution_timestamp': datetime.now().isoformat(),
    'model_info': {
        'algorithm': 'Random Forest',
        'n_estimators': 50,
        'features_used': len(feature_cols)
    },
    'data_summary': {
        'total_records': len(df),
        'train_records': len(train_data),
        'test_records': len(test_data),
        'train_period': f"{config.train_start} to {config.train_end}",
        'test_period': f"{config.test_start} to {config.test_end}"
    },
    'performance_metrics': {
        'rmse': float(test_rmse),
        'mae': float(test_mae),
        'r2': float(test_r2),
        'uncertainty_coverage': float(coverage)
    },
    'top_features': [{'feature': name, 'importance': float(imp)} for name, imp in top_features],
    'files_generated': [
        str(plot1_path.name),
        str(plot2_path.name),
        str(plot3_path.name)
    ] + ([str(plot4_path.name)] if plot4_path else []) + ([str(plot5_path.name)] if plot5_path else [])
}

# Salvar relatório
report_path = results_dir / f"quick_report_{timestamp}.json"
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"✅ Relatório salvo: {report_path}")

# Salvar modelo usando a biblioteca
model_path = results_dir / f"lightning_model_{timestamp}.joblib"
lightning_predictor.models['scaler'] = scaler  # Adicionar scaler para salvar junto
joblib.dump({
    'model': lightning_predictor.best_model,
    'uncertainty_model': uncertainty_predictor.uncertainty_model,
    'scaler': scaler,
    'feature_columns': feature_cols,
    'config': config
}, model_path)

print(f"✅ Modelo salvo: {model_path}")

# RESUMO FINAL
print(f"\n🎉 PIPELINE CONCLUÍDO COM SUCESSO!")
print("=" * 60)
print(f"🏆 RESULTADOS FINAIS:")
print(f"   📊 RMSE: {test_rmse:.2f}")
print(f"   📊 MAE: {test_mae:.2f}")
print(f"   📊 R²: {test_r2:.4f}")
print(f"   🎯 Cobertura IC: {coverage:.1%}")
print(f"   📁 Arquivos gerados em: {results_dir}")
total_plots = 3 + (1 if plot4_path else 0) + (1 if plot5_path else 0)
print(f"   📈 {total_plots} plots da série temporal criados")
print(f"   📋 Relatório JSON completo disponível")
print(f"   💾 Modelo treinado salvo")

print(f"\n📊 TOP 5 FEATURES MAIS IMPORTANTES:")
for i, (feature, importance) in enumerate(top_features[:5], 1):
    print(f"   {i}. {feature}: {importance:.4f}")

print("=" * 60)
print("✨ VISUALIZAÇÕES CRIADAS:")
print("   📈 Plot 1: Série temporal completa (2000-2019)")
print("   📈 Plot 2: Análise detalhada do período de teste")
print("   📈 Plot 3: Acumulados mensais com IC 95%")
if plot4_path:
    print("   📈 Plot 4: Zoom em 30 dias contínuos")
if plot5_path:
    print("   📈 Plot 5: Período específico Nov/2018 - Mar/2019")
print("   🎯 Todos incluem círculos coloridos e IC 95%")
print("=" * 60)