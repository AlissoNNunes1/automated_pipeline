"""
GPU Manager - Utilitario centralizado para gerenciar configuracao de GPU/CUDA

Este modulo fornece funcionalidades centralizadas para:
- Detectar disponibilidade de GPU/CUDA
- Configurar dispositivos para modelos ML
- Log de status de GPU
- Fallback automatico para CPU quando necessario

Autor: Sistema de Deteccao de Furtos
"""

import torch
import logging
from typing import Optional, Dict, Any


class GPUManager:
    """
    Gerenciador centralizado de configuracao GPU/CUDA
    
    Fornece deteccao de GPU, configuracao de devices e logging
    de status para todos os componentes do pipeline.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        """Singleton pattern para garantir uma unica instancia"""
        if cls._instance is None:
            cls._instance = super(GPUManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inicializa gerenciador GPU apenas uma vez"""
        if not GPUManager._initialized:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.INFO)
            
            # Detectar configuracao GPU
            self._detect_gpu_config()
            
            # Log de inicializacao
            self._log_system_info()
            
            GPUManager._initialized = True
    
    def _detect_gpu_config(self):
        """Detecta disponibilidade e configuracao da GPU"""
        self.cuda_available = torch.cuda.is_available()
        
        if self.cuda_available:
            self.gpu_count = torch.cuda.device_count()
            self.gpu_name = torch.cuda.get_device_name(0) if self.gpu_count > 0 else "Unknown"
            self.cuda_version = torch.version.cuda
            self.device = torch.device('cuda:0')
            self.device_name = f"cuda:0 ({self.gpu_name})"
        else:
            self.gpu_count = 0
            self.gpu_name = None
            self.cuda_version = None
            self.device = torch.device('cpu')
            self.device_name = "cpu"
    
    def _log_system_info(self):
        """Registra informacoes do sistema no log"""
        self.logger.info("=" * 60)
        self.logger.info("GPU MANAGER - CONFIGURACAO DETECTADA")
        self.logger.info("=" * 60)
        
        # PyTorch info
        self.logger.info(f"PyTorch version: {torch.__version__}")
        
        # CUDA info
        if self.cuda_available:
            self.logger.info(f"CUDA disponivel: SIM")
            self.logger.info(f"CUDA version: {self.cuda_version}")
            self.logger.info(f"GPUs detectadas: {self.gpu_count}")
            self.logger.info(f"GPU principal: {self.gpu_name}")
            self.logger.info(f"Device selecionado: {self.device_name}")
            
            # Memoria GPU
            if self.gpu_count > 0:
                memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                
                self.logger.info(f"Memoria GPU total: {memory_total:.1f} GB")
                self.logger.info(f"Memoria reservada: {memory_reserved:.1f} GB")
                self.logger.info(f"Memoria alocada: {memory_allocated:.1f} GB")
        else:
            self.logger.info(f"CUDA disponivel: NAO")
            self.logger.info(f"Device selecionado: CPU")
            self.logger.warning("GPU nao detectada - pipeline rodara em CPU (mais lento)")
        
        self.logger.info("=" * 60)
    
    def get_device(self) -> torch.device:
        """
        Retorna device PyTorch configurado (CUDA ou CPU)
        
        Returns:
            torch.device: Device para usar com modelos PyTorch
        """
        return self.device
    
    def get_device_string(self) -> str:
        """
        Retorna string do device para usar com YOLO
        
        Returns:
            str: 'cuda:0' se GPU disponivel, 'cpu' caso contrario
        """
        return str(self.device)
    
    def is_cuda_available(self) -> bool:
        """
        Verifica se CUDA esta disponivel
        
        Returns:
            bool: True se CUDA disponivel, False caso contrario
        """
        return self.cuda_available
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Retorna informacoes completas da GPU
        
        Returns:
            Dict com informacoes da GPU/CUDA
        """
        info = {
            'cuda_available': self.cuda_available,
            'gpu_count': self.gpu_count,
            'device': str(self.device),
            'device_name': self.device_name,
            'pytorch_version': torch.__version__
        }
        
        if self.cuda_available:
            info.update({
                'gpu_name': self.gpu_name,
                'cuda_version': self.cuda_version
            })
            
            if self.gpu_count > 0:
                props = torch.cuda.get_device_properties(0)
                info.update({
                    'gpu_memory_total_gb': props.total_memory / (1024**3),
                    'gpu_memory_reserved_gb': torch.cuda.memory_reserved(0) / (1024**3),
                    'gpu_memory_allocated_gb': torch.cuda.memory_allocated(0) / (1024**3)
                })
        
        return info
    
    def log_component_init(self, component_name: str, model_type: str = "YOLO"):
        """
        Registra inicializacao de componente com info de GPU
        
        Args:
            component_name: Nome do componente (ex: "EventDetector")
            model_type: Tipo do modelo (ex: "YOLO", "torch")
        """
        device_info = "GPU (CUDA)" if self.cuda_available else "CPU"
        self.logger.info(
            f"[{component_name}] Inicializando {model_type} em {device_info} "
            f"(device: {self.device_name})"
        )
    
    def get_yolo_device_config(self) -> Optional[str]:
        """
        Retorna configuracao de device para modelos YOLO
        
        Returns:
            str: Device string para YOLO ou None para auto-deteccao
        """
        # YOLO auto-detecta GPU por padrao, mas podemos forcar
        return self.get_device_string() if self.cuda_available else None
    
    def clear_gpu_memory(self):
        """Limpa cache de memoria da GPU se disponivel"""
        if self.cuda_available and self.gpu_count > 0:
            torch.cuda.empty_cache()
            self.logger.debug("Cache de memoria GPU limpo")
    
    @staticmethod
    def get_instance() -> 'GPUManager':
        """Retorna instancia singleton do GPUManager"""
        return GPUManager()


# Funcoes de conveniencia para uso rapido
def get_device() -> torch.device:
    """Retorna device configurado (CUDA ou CPU)"""
    return GPUManager.get_instance().get_device()


def is_cuda_available() -> bool:
    """Verifica se CUDA esta disponivel"""
    return GPUManager.get_instance().is_cuda_available()


def log_gpu_status(component_name: str):
    """Log rapido de status GPU para componente"""
    GPUManager.get_instance().log_component_init(component_name)


def get_gpu_info() -> Dict[str, Any]:
    """Retorna informacoes completas da GPU"""
    return GPUManager.get_instance().get_gpu_info()


#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/