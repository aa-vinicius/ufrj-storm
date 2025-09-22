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

# 4. TREINAMENTO DO MODELO
print("\n🤖 ETAPA 4: TREINAMENTO DO MODELO")

# Usar apenas Random Forest (mais rápido)
model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

print("✅ Modelo Random Forest treinado")

# 5. PREDIÇÕES
print("\n📈 ETAPA 5: GERANDO PREDIÇÕES")

train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

# Métricas
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

print(f"📊 RMSE: {test_rmse:.2f}")
print(f"📊 MAE: {test_mae:.2f}")
print(f"📊 R²: {test_r2:.4f}")

# 6. INTERVALO DE CONFIANÇA SIMPLIFICADO
print("\n🎯 ETAPA 6: CALCULANDO INTERVALO DE CONFIANÇA")

# Calcular resíduos de treino
train_residuals = np.abs(y_train - train_pred)

# Modelo simples para incerteza
uncertainty_model = RandomForestRegressor(n_estimators=30, random_state=42)
uncertainty_model.fit(X_train_scaled, train_residuals)

# Predições de incerteza
test_uncertainty = uncertainty_model.predict(X_test_scaled)

# Intervalo de confiança 95% (aproximado)
z_score = 1.96
lower_bound = test_pred - z_score * test_uncertainty
upper_bound = test_pred + z_score * test_uncertainty

# Garantir que não sejam negativos
lower_bound = np.maximum(lower_bound, 0)

# Calcular cobertura
within_interval = (y_test >= lower_bound) & (y_test <= upper_bound)
coverage = within_interval.mean()

print(f"🎯 Cobertura do IC 95%: {coverage:.1%}")

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

# 8. RELATÓRIO FINAL
print("\n📋 ETAPA 8: RELATÓRIO FINAL")

# Feature importance
feature_importance = model.feature_importances_
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
        str(plot2_path.name)
    ]
}

# Salvar relatório
report_path = results_dir / f"quick_report_{timestamp}.json"
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"✅ Relatório salvo: {report_path}")

# Salvar modelo
model_path = results_dir / f"lightning_model_{timestamp}.joblib"
joblib.dump({
    'model': model,
    'uncertainty_model': uncertainty_model,
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
print(f"   📈 2 plots da série temporal criados")
print(f"   📋 Relatório JSON completo disponível")
print(f"   💾 Modelo treinado salvo")

print(f"\n📊 TOP 5 FEATURES MAIS IMPORTANTES:")
for i, (feature, importance) in enumerate(top_features[:5], 1):
    print(f"   {i}. {feature}: {importance:.4f}")

print("=" * 60)
print("✨ VISUALIZAÇÕES CRIADAS:")
print("   📈 Plot 1: Série temporal completa (2000-2019)")
print("   📈 Plot 2: Análise detalhada do período de teste")
print("   🎯 Ambos incluem círculos coloridos e IC 95%")
print("=" * 60)