"""
Configurações globais do projeto UFRJ Storm
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

# Diretórios do projeto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

# Criar diretórios se não existirem
LOGS_DIR.mkdir(exist_ok=True)

@dataclass
class DataConfig:
    """Configurações relacionadas aos dados"""
    input_file: str = "tma_sp.csv"
    date_column: str = "data"
    target_column: str = "contagem_raios"
    binary_target_column: str = "sim_nao"
    
    # Divisão temporal dos dados
    train_start: str = "2000-01-01"
    train_end: str = "2014-12-31"
    test_start: str = "2015-01-01"
    test_end: str = "2019-12-31"
    
    # Colunas de features (excluindo data, target e binary target)
    feature_columns: List[str] = None
    
    def __post_init__(self):
        if self.feature_columns is None:
            # Todas as colunas exceto data, contagem_raios e sim_nao
            self.feature_columns = [
                'Showalterindex', 'Liftedindex', 'LIFTcomputedusingvirtualtemperature',
                'Kindex', 'Crosstotalsindex', 'Verticaltotalsindex', 'Totalstotalsindex',
                'ConvectiveAvailablePotentialEnergy', 'CAPEusingvirtualtemperature',
                'ConvectiveInhibition', 'CINSusingvirtualtemperature',
                'Temp[K]oftheLiftedCondensationLevel', 'Pres[hPa]oftheLiftedCondensationLevel',
                'Equivalentpotentialtemp[K]oftheLCL', 'Meanmixedlayerpotentialtemperature',
                'Meanmixedlayermixingratio', '1000hPato500hPathickness',
                'Precipitablewater[mm]forentiresounding', 'shear_index', 'BRNSH_index',
                'shear_0_15000', 'shear_6000_15000', 'BRNSH_500_15000', 'BRNSH_6000_15000',
                'EL', 'EL (Tv)', 'LFC', 'LFC (Tv)', 'Prof_cam_convec', 'Prof_cam_tv'
            ]

@dataclass
class ModelConfig:
    """Configurações dos modelos"""
    # Algoritmos para o modelo 1 (previsão de quantidade de raios)
    model1_algorithms: List[str] = None
    
    # Algoritmos para o modelo 2 (intervalo de confiança)
    model2_algorithms: List[str] = None
    
    # Parâmetros de validação cruzada
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    
    # Configurações de otimização de hiperparâmetros
    n_trials: int = 100
    
    def __post_init__(self):
        if self.model1_algorithms is None:
            self.model1_algorithms = [
                'random_forest',
                'xgboost',
                'lightgbm',
                'catboost',
                'linear_regression',
                'ridge',
                'lasso'
            ]
        
        if self.model2_algorithms is None:
            self.model2_algorithms = [
                'random_forest',
                'xgboost',
                'quantile_regression'
            ]

@dataclass
class Config:
    """Configuração principal do projeto"""
    data: DataConfig = None
    model: ModelConfig = None
    
    # Configurações de logging
    log_level: str = "INFO"
    
    # Configurações de validação
    confidence_interval: float = 0.95
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()