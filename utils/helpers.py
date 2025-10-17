"""
Helpers: Funcoes utilitarias comuns do pipeline

Funcoes para validacao de arquivos, conversao de formatos,
manipulacao de paths, etc

   __  ____ ____ _  _
 / _\/ ___) ___) )( \
/    \___ \___ ) \/ (
\_/\_(____(____|____/
"""

import os
import logging
from pathlib import Path
from typing import List, Optional
import hashlib


logger = logging.getLogger(__name__)


def ensure_dir(directory: str) -> Path:
    """
    Garante que diretorio existe, criando se necessario
    
    Args:
        directory: Caminho do diretorio
        
    Returns:
        Path do diretorio
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def find_files(directory: str, extensions: List[str], recursive: bool = True) -> List[Path]:
    """
    Encontra arquivos com extensoes especificas
    
    Args:
        directory: Diretorio para buscar
        extensions: Lista de extensoes (ex: ['.dav', '.mp4'])
        recursive: Se True, busca em subdiretorios
        
    Returns:
        Lista de paths dos arquivos encontrados (sem duplicatas)
    """
    directory = Path(directory)
    
    if not directory.exists():
        logger.warning(f"Diretorio nao encontrado: {directory}")
        return []
    
    # Usar set para evitar duplicatas (ex: .dav e .DAV no mesmo arquivo)
    files_set = set()
    
    for ext in extensions:
        if not ext.startswith('.'):
            ext = f'.{ext}'
        
        if recursive:
            pattern = f"**/*{ext}"
        else:
            pattern = f"*{ext}"
        
        # Adicionar ao set (automaticamente remove duplicatas)
        for file in directory.glob(pattern):
            files_set.add(file)
    
    return sorted(list(files_set))


def get_file_size_mb(filepath: str) -> float:
    """
    Obtem tamanho de arquivo em MB
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        Tamanho em MB
    """
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except Exception as e:
        logger.error(f"Erro ao obter tamanho de {filepath}: {e}")
        return 0.0


def file_exists(filepath: str) -> bool:
    """
    Verifica se arquivo existe
    
    Args:
        filepath: Caminho do arquivo
        
    Returns:
        True se existe, False caso contrario
    """
    return Path(filepath).exists()


def get_file_hash(filepath: str, algorithm: str = 'md5') -> Optional[str]:
    """
    Calcula hash de um arquivo
    
    Args:
        filepath: Caminho do arquivo
        algorithm: Algoritmo de hash (md5, sha256, etc)
        
    Returns:
        Hash hexadecimal ou None se erro
    """
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
    
    except Exception as e:
        logger.error(f"Erro ao calcular hash de {filepath}: {e}")
        return None


def format_duration(seconds: float) -> str:
    """
    Formata duracao em formato legivel
    
    Args:
        seconds: Duracao em segundos
        
    Returns:
        String formatada (ex: "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)


def sanitize_filename(filename: str) -> str:
    """
    Remove caracteres invalidos de nome de arquivo
    
    Args:
        filename: Nome do arquivo original
        
    Returns:
        Nome sanitizado
    """
    # Caracteres invalidos no Windows
    invalid_chars = '<>:"/\\|?*'
    
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    return filename


def get_video_base_name(video_path: str) -> str:
    """
    Obtem nome base de um video sem extensao
    
    Args:
        video_path: Caminho do video
        
    Returns:
        Nome base (ex: "camera_loja_01" de "camera_loja_01.dav")
    """
    return Path(video_path).stem


def create_output_structure(base_dir: str) -> dict:
    """
    Cria estrutura completa de diretorios para output
    
    Args:
        base_dir: Diretorio base
        
    Returns:
        Dict com paths dos diretorios criados (como objetos Path)
    """
    from pathlib import Path
    
    base_path = Path(base_dir)
    
    dirs = {
        'base': base_path,
        'chunks': base_path / 'chunks',
        'active_chunks': base_path / 'active_chunks',
        'events': base_path / 'events',
        'proposals': base_path / 'proposals',
        'annotations': base_path / 'annotations'
    }
    
    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Estrutura de diretorios criada em: {base_dir}")
    return dirs


def count_files_in_dir(directory: str, extensions: Optional[List[str]] = None) -> int:
    """
    Conta arquivos em um diretorio
    
    Args:
        directory: Diretorio para contar
        extensions: Lista de extensoes para filtrar (opcional)
        
    Returns:
        Numero de arquivos
    """
    if not Path(directory).exists():
        return 0
    
    if extensions:
        files = find_files(directory, extensions, recursive=False)
        return len(files)
    else:
        return len([f for f in Path(directory).iterdir() if f.is_file()])


def validate_video_file(video_path: str) -> tuple:
    """
    Valida se arquivo de video e valido
    
    Args:
        video_path: Caminho do video
        
    Returns:
        (is_valid, error_message)
    """
    video_path = Path(video_path)
    
    # Verificar existencia
    if not video_path.exists():
        return False, f"Arquivo nao encontrado: {video_path}"
    
    # Verificar se e arquivo
    if not video_path.is_file():
        return False, f"Path nao e um arquivo: {video_path}"
    
    # Verificar tamanho minimo (1 MB)
    size_mb = get_file_size_mb(str(video_path))
    if size_mb < 1:
        return False, f"Arquivo muito pequeno ({size_mb:.2f} MB)"
    
    # Verificar extensao
    valid_extensions = ['.dav', '.mp4', '.avi', '.mov', '.mkv']
    if video_path.suffix.lower() not in valid_extensions:
        return False, f"Extensao invalida: {video_path.suffix}"
    
    return True, None


#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
