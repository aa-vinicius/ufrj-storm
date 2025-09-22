# Guia de Produção - UFRJ Storm

Este documento descreve como executar o pipeline completo de Machine Learning em ambiente de produção, fora do ambiente de notebook.

## 🎯 Objetivo

Executar todo o pipeline de ML de forma automatizada, gerando:
- Modelos treinados e otimizados
- Predições com intervalos de confiança
- Visualizações da série temporal completa
- Relatórios de performance detalhados

## 📋 Pré-requisitos Produtivos

### Ambiente Técnico
- Python 3.8+
- Memória RAM: mínimo 8GB (recomendado 16GB)
- Espaço em disco: 2GB livres
- CPU: multi-core recomendado para treinamento

### Dependências
```bash
pip install -r requirements.txt
```

### Dados de Entrada
- Arquivo CSV com estrutura idêntica ao `tma_sp.csv`
- Mínimo 1000 registros para treinamento
- Features meteorológicas completas

## 🔄 Pipeline Produtivo

### Fase 1: Validação e Carregamento

```python
# 1.1 Executar testes de qualidade
from notebooks.data_quality_tests import run_data_quality_tests
test_results = run_data_quality_tests()

# Critério de parada: 100% dos testes aprovados
assert test_results['success_rate'] == 1.0, "Falha nos testes de qualidade"
```

**Critérios de Aprovação:**
- ✅ 100% dos testes automatizados aprovados
- ✅ Missing values < 5% por coluna
- ✅ Consistência temporal > 90%
- ✅ Variáveis target sem valores inválidos

### Fase 2: Pré-processamento

```python
# 2.1 Aplicar limpeza dos dados
preprocessor = DataPreprocessor(config)
df_processed = preprocessor.prepare_features(df_raw)

# 2.2 Gerar features temporais
feature_cols = preprocessor.get_feature_columns(df_processed)

# 2.3 Dividir dados temporalmente
train_data, test_data = split_data_temporal(df_processed, config)

# 2.4 Escalonar features
X_train_scaled, X_test_scaled, scaler = preprocessor.scale_features(
    train_data[feature_cols], 
    test_data[feature_cols]
)
```

**Validações da Fase 2:**
- Número de features: 40 (esperado)
- Shape treino: (>=1000, 40)
- Shape teste: (>=200, 40)
- Escalamento aplicado com sucesso

### Fase 3: Treinamento dos Modelos

```python
# 3.1 Treinar modelo principal
lightning_predictor = LightningPredictor(config)
lightning_predictor.train_multiple_models(X_train_scaled, y_train)

# 3.2 Validar performance mínima
best_model_rmse = lightning_predictor.model_scores[lightning_predictor.best_model_name]['rmse_cv_mean']
assert best_model_rmse < 3000, f"RMSE muito alto: {best_model_rmse}"

# 3.3 Treinar modelo de incerteza
uncertainty_predictor = UncertaintyPredictor(lightning_predictor)
uncertainty_predictor.train_uncertainty_model(X_train_scaled, y_train)
```

**Critérios de Performance:**
- RMSE do melhor modelo < 3000
- R² > 0.25
- Modelo de incerteza treinado sem erros

### Fase 4: Validação Final

```python
# 4.1 Avaliar no conjunto de teste
test_results = lightning_predictor.evaluate_all_models(X_test_scaled, y_test)
uncertainty_results = uncertainty_predictor.evaluate_uncertainty(X_test_scaled, y_test)

# 4.2 Validar métricas finais
best_r2 = test_results[lightning_predictor.best_model_name]['metrics']['r2']
coverage = uncertainty_results['coverage']

assert best_r2 > 0.20, f"R² insuficiente: {best_r2}"
assert coverage > 0.50, f"Cobertura baixa: {coverage}"
```

**Métricas Mínimas para Produção:**
- R² > 0.20
- RMSE < 3500
- MAE < 2000
- Cobertura do IC > 50%

### Fase 5: Salvamento e Deploy

```python
# 5.1 Salvar modelos treinados
lightning_predictor.save_model()
joblib.dump(uncertainty_predictor, '../models/uncertainty_model.joblib')
joblib.dump(scaler, '../models/scaler.joblib')

# 5.2 Salvar metadados
metadata = {
    'training_date': datetime.now().isoformat(),
    'model_performance': test_results[lightning_predictor.best_model_name]['metrics'],
    'uncertainty_coverage': uncertainty_results['coverage'],
    'feature_columns': feature_cols,
    'data_stats': {
        'train_samples': len(X_train_scaled),
        'test_samples': len(X_test_scaled),
        'features': len(feature_cols)
    }
}
with open('../models/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

## 🚀 Execução em Produção

### Script Principal

```python
def run_production_pipeline(input_file):
    """Executar pipeline completo de produção"""
    
    print("🚀 INICIANDO PIPELINE PRODUTIVO UFRJ STORM")
    print("=" * 60)
    
    try:
        # Fase 1: Validação
        print("📊 Fase 1: Validação dos dados...")
        df = load_and_validate_data(input_file)
        
        # Fase 2: Pré-processamento
        print("🔧 Fase 2: Pré-processamento...")
        X_train, X_test, y_train, y_test = preprocess_data(df)
        
        # Fase 3: Treinamento
        print("🤖 Fase 3: Treinamento dos modelos...")
        model1, model2 = train_models(X_train, y_train)
        
        # Fase 4: Validação
        print("📈 Fase 4: Validação final...")
        metrics = validate_models(model1, model2, X_test, y_test)
        
        # Fase 5: Deploy
        print("💾 Fase 5: Salvamento dos modelos...")
        save_models(model1, model2, metrics)
        
        print("✅ PIPELINE EXECUTADO COM SUCESSO!")
        return True
        
    except Exception as e:
        print(f"❌ ERRO NO PIPELINE: {str(e)}")
        return False

# Executar
success = run_production_pipeline('data/tma_sp.csv')
```

### Uso dos Modelos Treinados

```python
def predict_lightning(input_features):
    """Fazer predições com modelos treinados"""
    
    # Carregar modelos
    lightning_predictor = LightningPredictor.load_model('../models/model_random_forest.joblib')
    uncertainty_predictor = joblib.load('../models/uncertainty_model.joblib')
    scaler = joblib.load('../models/scaler.joblib')
    
    # Pré-processar entrada
    features_scaled = scaler.transform(input_features)
    
    # Predições
    base_prediction = lightning_predictor.best_model.predict(features_scaled)
    uncertainty_result = uncertainty_predictor.predict_with_uncertainty(features_scaled)
    
    return {
        'prediction': base_prediction[0],
        'confidence_interval': {
            'lower': uncertainty_result['lower_bound'][0],
            'upper': uncertainty_result['upper_bound'][0]
        },
        'uncertainty': uncertainty_result['uncertainty'][0]
    }
```

## 📊 Monitoramento Produtivo

### Métricas a Acompanhar

1. **Performance do Modelo**
   - RMSE mensal
   - R² mensal  
   - Drift das features
   - Cobertura do IC

2. **Qualidade dos Dados**
   - Taxa de valores ausentes
   - Outliers detectados
   - Gaps temporais
   - Consistência das variáveis

3. **Sistema**
   - Tempo de execução
   - Uso de memória
   - Taxa de erro
   - Disponibilidade

### Alertas Críticos

```python
# Configurar alertas
ALERT_THRESHOLDS = {
    'rmse_max': 4000,
    'r2_min': 0.15,
    'coverage_min': 0.40,
    'missing_rate_max': 0.10,
    'execution_time_max': 3600  # 1 hora
}

def check_alerts(metrics):
    """Verificar se métricas estão dentro dos limites"""
    alerts = []
    
    if metrics['rmse'] > ALERT_THRESHOLDS['rmse_max']:
        alerts.append(f"RMSE alto: {metrics['rmse']}")
    
    if metrics['r2'] < ALERT_THRESHOLDS['r2_min']:
        alerts.append(f"R² baixo: {metrics['r2']}")
    
    # ... outras verificações
    
    return alerts
```

## 🔄 Manutenção e Retreinamento

### Critérios para Retreinamento

- **Drift de Performance**: R² cai abaixo de 0.20 por 2 semanas consecutivas
- **Drift de Dados**: >30% das features com distribuição significativamente diferente
- **Novos Dados**: Acúmulo de >1000 novos registros validados
- **Periodicidade**: Retreinamento trimestral obrigatório

### Processo de Retreinamento

```python
def retrain_models():
    """Processo de retreinamento dos modelos"""
    
    # 1. Carregar novos dados
    new_data = load_new_data()
    
    # 2. Validar qualidade
    if not validate_new_data(new_data):
        raise ValueError("Dados novos não passaram na validação")
    
    # 3. Retreinar modelos
    new_models = train_models(new_data)
    
    # 4. Validar performance
    if new_models['performance'] > current_models['performance']:
        # Deploy novos modelos
        deploy_models(new_models)
        # Backup modelos antigos
        backup_models(current_models)
    else:
        print("Novos modelos não superam os atuais")
```

## 🚨 Troubleshooting Produtivo

### Problemas Comuns

#### 1. Falha no Carregamento dos Dados
```python
# Diagnóstico
- Verificar existência do arquivo
- Validar formato CSV
- Checar encoding
- Verificar permissões
```

#### 2. Erro no Pré-processamento
```python
# Soluções
- Verificar nomes das colunas
- Validar tipos de dados
- Checar valores extremos
- Revisar features engineered
```

#### 3. Performance Degradada
```python
# Investigação
- Comparar distribuições das features
- Analisar drift temporal
- Verificar outliers novos
- Avaliar missing patterns
```

#### 4. Erro na Predição
```python
# Debug
- Validar formato da entrada
- Verificar escalamento
- Checar dimensões dos arrays
- Confirmar modelos carregados
```

## 📈 Otimizações Futuras

### Curto Prazo (1-3 meses)
- [ ] Otimização de hiperparâmetros automatizada
- [ ] Pipeline de validação cruzada temporal
- [ ] Monitoramento automático de drift
- [ ] API REST para predições

### Médio Prazo (3-6 meses)
- [ ] Ensemble de modelos
- [ ] Feature selection automática
- [ ] Calibração avançada da incerteza
- [ ] Dashboard de monitoramento

### Longo Prazo (6+ meses)
- [ ] Modelos deep learning
- [ ] Predição multi-step
- [ ] Incorporação de dados satelitais
- [ ] Sistema de alertas geográficos

---

**Mantenha este documento atualizado** conforme evoluções no sistema produtivo.