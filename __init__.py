r"""
"""
Automated Pipeline - Sistema automatizado para processamento de videos longos

Pipeline completo:
1. VideoChunker: Divide videos longos em chunks menores
2. ActivityFilter: Filtra chunks sem atividade relevante
3. EventDetector: Detecta eventos com tracking de pessoas
4. AutoLabeler: Gera propostas automaticas de anotacao

Produtividade esperada: 10-15x mais rapido que anotacao manual
"""

__version__ = '1.0.0'

# Imports principais
try:
    from .core.video_chunker import VideoChunker
    from .core.activity_filter import ActivityFilter
    from .core.event_detector import EventDetector
    from .core.auto_labeler import AutoLabeler
    
    __all__ = [
        'VideoChunker',
        'ActivityFilter',
        'EventDetector',
        'AutoLabeler'
    ]
except ImportError as e:
    # Durante desenvolvimento, alguns modulos podem nao estar disponiveis
    import warnings
    warnings.warn(f"Alguns componentes nao puderam ser importados: {e}")
    __all__ = []
"""

__version__ = "1.0.0"
__author__ = "Dataset Construction Team"

# Expor classes principais
from automated_pipeline.core.video_chunker import VideoChunker
from automated_pipeline.core.activity_filter import ActivityFilter
from automated_pipeline.core.event_detector import EventDetector
from automated_pipeline.core.auto_labeler import AutoLabeler

__all__ = [
    'VideoChunker',
    'ActivityFilter',
    'EventDetector',
    'AutoLabeler',
]
