# Guia de Produção - UFRJ Storm

Este documento descreve como executar os pipelines de Machine Learning automatizados em ambiente de produção.

## 🎯 Objetivo

Executar pipelines ML completamente automatizados usando as classes `LightningPredictor` e `UncertaintyPredictor` da biblioteca `src/models.py`, gerando:
- Modelos treinados e otimizados
- Predições com intervalos de confiança 95%
- 5 visualizações automáticas da série temporal
- Relatórios JSON/texto detalhados

## ⚡ Pipelines Disponíveis

### Pipeline Rápido (`run_quick_pipeline.py`)
- **Uso**: Demonstração e validação rápida
- **Algoritmo**: Random Forest otimizado
- **Tempo**: 2-3 minutos
- **Saída**: 5 plots + relatório JSON

### Pipeline de Produção (`run_production_pipeline.py`) 
- **Uso**: Ambiente produtivo completo
- **Algoritmos**: Todos os 7 algoritmos disponíveis
- **Tempo**: 10-15 minutos  
- **Saída**: Seleção automática do melhor modelo + todos os plots

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

## � Execução dos Pipelines

### Execução Simples

```bash
# Pipeline rápido (recomendado para validação)
python run_quick_pipeline.py

# Pipeline completo (produção)
python run_production_pipeline.py
```

### Saídas Geradas

#### Visualizações (5 plots automáticos):
1. **Série temporal completa** (2000-2019)
2. **Análise detalhada** do período de teste
3. **Acumulados mensais** com IC 95%
4. **Período contínuo** de 30 dias
5. **Período específico** (Nov/2018 - Mar/2019)

#### Relatórios:
- **JSON**: Métricas, features importantes, arquivos gerados
- **Modelo treinado**: Arquivo `.joblib` para uso posterior

## 🔄 Arquitetura do Pipeline

### Fase 1: Validação e Carregamento (Automática)

```python
# 1.1 A biblioteca src/models.py executa automaticamente:
# - Carregamento do CSV
# - Validação da estrutura dos dados
# - Testes de qualidade TDD (8 testes)
# - Pré-processamento e limpeza
```

**Critérios de Aprovação:**
- ✅ 100% dos testes automatizados aprovados
- ✅ Missing values < 5% por coluna
- ✅ Consistência temporal > 90%
- ✅ Variáveis target sem valores inválidos

### Fase 2: Pré-processamento (Automático)

```python
# 2.1 A classe LightningPredictor executa automaticamente:
from src.models import LightningPredictor, UncertaintyPredictor

# Carregamento e pré-processamento integrados
lightning_predictor = LightningPredictor(config)
# Interno: limpeza, features temporais, escalamento, divisão temporal
```

**Validações da Fase 2:**
- Número de features: 40 (esperado)
- Shape treino: (>=1000, 40)
- Shape teste: (>=200, 40)
- Escalamento aplicado com sucesso

### Fase 3: Treinamento dos Modelos (Automático)

```python
# 3.1 Pipeline rápido
lightning_predictor.train_model()  # Random Forest otimizado

# 3.2 Pipeline produção  
lightning_predictor.train_multiple_models()  # 7 algoritmos + seleção automática

# 3.3 Modelo de incerteza (ambos pipelines)
uncertainty_predictor = UncertaintyPredictor(lightning_predictor)
uncertainty_predictor.train_uncertainty_model()
```

**Critérios de Performance:**
- RMSE do melhor modelo < 3000
- R² > 0.25
- Modelo de incerteza treinado sem erros

### Fase 4: Avaliação e Visualização (Automática)

```python
# 4.1 Avaliação automática
# - Métricas de todos os modelos
# - Seleção do melhor modelo
# - Intervalos de confiança 95%

# 4.2 Geração automática de 5 plots:
# - Série temporal completa
# - Análise detalhada do teste  
# - Acumulados mensais
# - Período de 30 dias contínuos
# - Período específico Nov/2018-Mar/2019

# 4.3 Relatórios automáticos
# - JSON com métricas e metadados
# - Salvamento do modelo treinado
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

### Execução Direta dos Pipelines

```bash
# Pipeline rápido: demonstração com Random Forest
python run_quick_pipeline.py

# Saída esperada:
# 🚀 PIPELINE RÁPIDO UFRJ STORM - DEMONSTRAÇÃO
# ✅ Dados carregados: 5857 registros, 33 colunas
# ✅ Modelo Random Forest treinado
# ✅ 5 plots gerados em /results/
# 🎉 PIPELINE CONCLUÍDO COM SUCESSO!
```

```bash
# Pipeline de produção: todos os algoritmos
python run_production_pipeline.py

# Saída esperada:
# 🚀 PIPELINE PRODUTIVO UFRJ STORM
# ✅ Dados processados e divididos
# 🤖 Treinando 7 algoritmos ML...
# 🏆 Melhor modelo: random_forest (RMSE: 2888.85)
# 📈 5 visualizações geradas
# 💾 Modelo e relatórios salvos
# 🎉 PIPELINE CONCLUÍDO COM SUCESSO!
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