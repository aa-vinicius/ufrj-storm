#!/usr/bin/env python3
"""
Script de validação do pipeline RINDAT
Executa testes abrangentes antes do processamento principal
"""

import os
import sys
import pytest
import tempfile
import shutil
import logging
from datetime import datetime

def run_validation_tests():
    """Executa todos os testes de validação"""
    print("🔍 INICIANDO VALIDAÇÃO DO PIPELINE RINDAT")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configurar logging para capturar problemas
    logging.basicConfig(level=logging.INFO)
    
    # Executar testes
    test_file = "tests/test_full_pipeline.py"
    
    if not os.path.exists(test_file):
        print(f"❌ Arquivo de teste não encontrado: {test_file}")
        return False
    
    print("🧪 Executando testes de validação...")
    
    # Executar pytest com saída detalhada
    result = pytest.main([
        test_file,
        "-v",
        "--tb=short",
        "--durations=10"
    ])
    
    if result == 0:
        print()
        print("✅ VALIDAÇÃO CONCLUÍDA COM SUCESSO!")
        print("✅ Todos os testes passaram - Pipeline está pronto para execução")
        print()
        return True
    else:
        print()
        print("❌ FALHA NA VALIDAÇÃO!")
        print("❌ Alguns testes falharam - Corrija os problemas antes de executar o pipeline")
        print()
        return False

def validate_environment():
    """Valida o ambiente antes de executar testes"""
    print("🔧 Validando ambiente...")
    
    # Verificar se está no diretório correto
    if not os.path.exists("src/data_processor.py"):
        print("❌ Execute este script a partir do diretório raiz do projeto")
        return False
    
    # Verificar se o ambiente virtual está ativo
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Ambiente virtual não detectado - certifique-se de ativá-lo")
    
    # Verificar bibliotecas essenciais
    try:
        import pandas
        import numpy
        import matplotlib
        import folium
        print("✅ Bibliotecas essenciais encontradas")
    except ImportError as e:
        print(f"❌ Biblioteca necessária não encontrada: {e}")
        return False
    
    print("✅ Ambiente validado")
    return True

if __name__ == "__main__":
    try:
        # Validar ambiente
        if not validate_environment():
            sys.exit(1)
        
        # Executar testes de validação
        success = run_validation_tests()
        
        if success:
            print("🚀 Pipeline validado - pode executar 'python main.py' com segurança")
            sys.exit(0)
        else:
            print("🛑 Pipeline não validado - corrija os problemas antes de continuar")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️  Validação interrompida pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado durante validação: {e}")
        sys.exit(1)