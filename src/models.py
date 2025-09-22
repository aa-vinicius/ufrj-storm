"""
Módulo para implementação dos modelos de Machine Learning do projeto UFRJ Storm
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import os
from pathlib import Path

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

class LightningPredictor:
    """Classe principal para previsão de raios usando múltiplos algoritmos em cascata"""
    
    def __init__(self, config, random_state=42):
        self.config = config
        self.random_state = random_state
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        
    def _get_models(self):
        """Definir os modelos disponíveis"""
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'lasso': Lasso(alpha=1.0, random_state=self.random_state),
            'random_forest': RandomForestRegressor(
                n_estimators=100, 
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            )
        }
        
        # Adicionar XGBoost se disponível
        if XGBOOST_AVAILABLE:
            models['xgboost'] = xgb.XGBRegressor(
                n_estimators=100,
                random_state=self.random_state,
                verbosity=0
            )
        
        # Adicionar LightGBM se disponível
        if LIGHTGBM_AVAILABLE:
            models['lightgbm'] = lgb.LGBMRegressor(
                n_estimators=100,
                random_state=self.random_state,
                verbosity=-1
            )
        
        return models
    
    def train_multiple_models(self, X_train, y_train, cv_folds=5):
        """Treinar múltiplos modelos e avaliar performance"""
        models = self._get_models()
        
        print(f"🤖 TREINANDO {len(models)} MODELOS")
        print("=" * 50)
        
        for name, model in models.items():
            print(f"📊 Treinando {name}...")
            
            try:
                # Treinar modelo
                model.fit(X_train, y_train)
                
                # Validação cruzada
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=cv_folds, 
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
                
                rmse_cv = np.sqrt(-cv_scores.mean())
                rmse_std = np.sqrt(cv_scores.std())
                
                # Armazenar modelo e scores
                self.models[name] = model
                self.model_scores[name] = {
                    'rmse_cv_mean': rmse_cv,
                    'rmse_cv_std': rmse_std,
                    'cv_scores': cv_scores
                }
                
                print(f"   ✅ {name}: RMSE CV = {rmse_cv:.2f} ± {rmse_std:.2f}")
                
            except Exception as e:
                print(f"   ❌ {name}: Erro - {str(e)}")
                continue
        
        # Encontrar melhor modelo
        if self.model_scores:
            best_name = min(self.model_scores.keys(), 
                          key=lambda x: self.model_scores[x]['rmse_cv_mean'])
            self.best_model_name = best_name
            self.best_model = self.models[best_name]
            
            print(f"\n🏆 MELHOR MODELO: {best_name}")
            print(f"   RMSE CV: {self.model_scores[best_name]['rmse_cv_mean']:.2f}")
    
    def evaluate_model(self, model, X_test, y_test, model_name="Modelo"):
        """Avaliar um modelo específico"""
        y_pred = model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        print(f"📊 AVALIAÇÃO {model_name.upper()}:")
        print(f"   RMSE: {metrics['rmse']:.2f}")
        print(f"   MAE:  {metrics['mae']:.2f}")
        print(f"   R²:   {metrics['r2']:.4f}")
        
        return y_pred, metrics
    
    def evaluate_all_models(self, X_test, y_test):
        """Avaliar todos os modelos treinados"""
        results = {}
        
        print("📊 AVALIAÇÃO DE TODOS OS MODELOS NO CONJUNTO DE TESTE")
        print("=" * 60)
        
        for name, model in self.models.items():
            y_pred, metrics = self.evaluate_model(model, X_test, y_test, name)
            results[name] = {
                'predictions': y_pred,
                'metrics': metrics
            }
        
        return results
    
    def get_feature_importance(self, model_name=None, top_k=20):
        """Obter importância das features"""
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models.get(model_name)
        
        if model is None:
            print(f"Modelo '{model_name}' não encontrado")
            return None
        
        # Verificar se o modelo tem feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            print(f"Modelo '{model_name}' não possui informação de importância")
            return None
        
        return importances
    
    def save_model(self, model_name=None, filepath=None):
        """Salvar modelo treinado"""
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models.get(model_name)
        
        if model is None:
            print(f"Modelo '{model_name}' não encontrado")
            return False
        
        if filepath is None:
            filepath = f"../models/model_{model_name}.joblib"
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Salvar modelo
        joblib.dump({
            'model': model,
            'scores': self.model_scores.get(model_name, {}),
            'config': self.config,
            'model_name': model_name
        }, filepath)
        
        print(f"✅ Modelo '{model_name}' salvo em: {filepath}")
        return True
    
    @classmethod
    def load_model(cls, filepath):
        """Carregar modelo salvo"""
        data = joblib.load(filepath)
        
        predictor = cls(data['config'])
        predictor.models[data['model_name']] = data['model']
        predictor.model_scores[data['model_name']] = data['scores']
        predictor.best_model_name = data['model_name']
        predictor.best_model = data['model']
        
        return predictor

class UncertaintyPredictor:
    """Classe para prever intervalos de confiança da incerteza"""
    
    def __init__(self, base_predictor, confidence_level=0.95):
        self.base_predictor = base_predictor
        self.confidence_level = confidence_level
        self.uncertainty_model = None
        self.residuals_stats = None
        
    def calculate_residuals(self, X_train, y_train):
        """Calcular resíduos do modelo base para treinar modelo de incerteza"""
        # Fazer predições com o modelo base
        base_predictions = self.base_predictor.best_model.predict(X_train)
        
        # Calcular resíduos absolutos
        residuals = np.abs(y_train - base_predictions)
        
        # Estatísticas dos resíduos
        self.residuals_stats = {
            'mean': residuals.mean(),
            'std': residuals.std(),
            'quantiles': np.percentile(residuals, [25, 50, 75, 90, 95, 99])
        }
        
        return residuals, base_predictions
    
    def train_uncertainty_model(self, X_train, y_train):
        """Treinar modelo para prever incerteza"""
        print("🎯 TREINANDO MODELO DE INCERTEZA")
        print("=" * 40)
        
        # Calcular resíduos
        residuals, base_predictions = self.calculate_residuals(X_train, y_train)
        
        # Treinar modelo para prever resíduos
        uncertainty_models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        best_score = float('inf')
        best_model = None
        best_name = None
        
        for name, model in uncertainty_models.items():
            # Treinar modelo de incerteza
            model.fit(X_train, residuals)
            
            # Avaliar com validação cruzada
            cv_scores = cross_val_score(
                model, X_train, residuals,
                cv=5, scoring='neg_mean_squared_error'
            )
            rmse_cv = np.sqrt(-cv_scores.mean())
            
            print(f"   {name}: RMSE CV = {rmse_cv:.2f}")
            
            if rmse_cv < best_score:
                best_score = rmse_cv
                best_model = model
                best_name = name
        
        self.uncertainty_model = best_model
        print(f"\n🏆 Melhor modelo de incerteza: {best_name}")
        
    def predict_with_uncertainty(self, X_test):
        """Fazer predições com intervalos de confiança"""
        if self.uncertainty_model is None:
            raise ValueError("Modelo de incerteza não foi treinado")
        
        # Predição do modelo base
        base_predictions = self.base_predictor.best_model.predict(X_test)
        
        # Predição da incerteza
        predicted_uncertainty = self.uncertainty_model.predict(X_test)
        
        # Calcular intervalos de confiança
        alpha = 1 - self.confidence_level
        z_score = 1.96  # Para 95% de confiança
        
        # Intervalo de confiança
        lower_bound = base_predictions - z_score * predicted_uncertainty
        upper_bound = base_predictions + z_score * predicted_uncertainty
        
        # Garantir que os limites não sejam negativos (contagem não pode ser negativa)
        lower_bound = np.maximum(lower_bound, 0)
        
        return {
            'predictions': base_predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'uncertainty': predicted_uncertainty
        }
    
    def evaluate_uncertainty(self, X_test, y_test):
        """Avaliar qualidade dos intervalos de confiança"""
        results = self.predict_with_uncertainty(X_test)
        
        # Calcular cobertura (quantos valores reais estão dentro do intervalo)
        within_interval = (
            (y_test >= results['lower_bound']) & 
            (y_test <= results['upper_bound'])
        )
        coverage = within_interval.mean()
        
        # Largura média do intervalo
        avg_interval_width = (results['upper_bound'] - results['lower_bound']).mean()
        
        print(f"📊 AVALIAÇÃO DO MODELO DE INCERTEZA:")
        print(f"   Cobertura: {coverage:.1%} (esperado: {self.confidence_level:.1%})")
        print(f"   Largura média do intervalo: {avg_interval_width:.2f}")
        
        return {
            'coverage': coverage,
            'avg_interval_width': avg_interval_width,
            'within_interval': within_interval,
            **results
        }