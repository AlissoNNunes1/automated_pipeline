"""
Script de teste para verificar qual encoder GPU esta sendo usado

Executa: python test_gpu_encoder.py
"""

import sys
import subprocess
from pathlib import Path

# Adicionar path dos utils
sys.path.insert(0, str(Path(__file__).parent))

from utils.convert_dav_videos import DAVConverter


def test_gpu_detection():
    """Testa deteccao de GPU e encoder"""
    print("\n" + "="*60)
    print("  TESTE DE DETECCAO DE GPU/ENCODER")
    print("="*60 + "\n")
    
    # Criar conversor
    print("Inicializando DAVConverter...")
    converter = DAVConverter()
    
    print("\n" + "-"*60)
    print("RESULTADOS DA DETECCAO:")
    print("-"*60)
    
    # FFmpeg disponivel?
    print(f"\nFFmpeg disponivel: {converter.ffmpeg_available}")
    
    if not converter.ffmpeg_available:
        print("\n❌ ERRO: FFmpeg nao encontrado!")
        print("   Instale FFmpeg: https://ffmpeg.org/download.html")
        return
    
    # Qual encoder?
    print(f"Encoder detectado: {converter.gpu_encoder}")
    
    # Decodificar tipo
    encoder_info = {
        'h264_nvenc': {
            'nome': 'NVIDIA NVENC',
            'tipo': 'GPU NVIDIA (GeForce/Quadro)',
            'velocidade': '5-10x CPU',
            'icone': '🟢'
        },
        'h264_amf': {
            'nome': 'AMD AMF',
            'tipo': 'GPU AMD (Radeon)',
            'velocidade': '4-8x CPU',
            'icone': '🔴'
        },
        'h264_qsv': {
            'nome': 'Intel Quick Sync',
            'tipo': 'GPU Intel Integrada',
            'velocidade': '3-5x CPU',
            'icone': '🔵'
        },
        'libx264': {
            'nome': 'CPU (x264)',
            'tipo': 'Software (CPU)',
            'velocidade': '1x (baseline)',
            'icone': '⚪'
        }
    }
    
    info = encoder_info.get(converter.gpu_encoder, {
        'nome': 'Desconhecido',
        'tipo': 'N/A',
        'velocidade': 'N/A',
        'icone': '❓'
    })
    
    print(f"\n{info['icone']} Encoder: {info['nome']}")
    print(f"   Tipo: {info['tipo']}")
    print(f"   Velocidade esperada: {info['velocidade']}")
    
    # Listar todos encoders disponiveis
    print("\n" + "-"*60)
    print("ENCODERS H.264 DISPONIVEIS NO FFMPEG:")
    print("-"*60)
    
    try:
        result = subprocess.run(
            ['ffmpeg', '-encoders'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Filtrar apenas h264
        h264_encoders = []
        for line in result.stdout.split('\n'):
            if 'h264' in line.lower() and line.strip().startswith('V'):
                h264_encoders.append(line.strip())
        
        if h264_encoders:
            for encoder in h264_encoders:
                print(f"  {encoder}")
        else:
            print("  Nenhum encoder H.264 encontrado")
            
    except Exception as e:
        print(f"  Erro ao listar encoders: {e}")
    
    # Recomendacoes
    print("\n" + "="*60)
    print("RECOMENDACOES:")
    print("="*60)
    
    if converter.gpu_encoder == 'libx264':
        print("\n⚠️  Usando CPU - conversao sera LENTA")
        print("\nPara acelerar com GPU:")
        print("  • NVIDIA: Instale drivers GeForce atualizados")
        print("  • AMD: Instale drivers Adrenalin atualizados")
        print("  • Intel: Habilite GPU integrada na BIOS")
        print("\nE certifique-se que FFmpeg tenha suporte GPU:")
        print("  https://github.com/BtbN/FFmpeg-Builds/releases")
        print("  Baixe versao 'full' ou 'shared'")
    else:
        print(f"\n✅ GPU configurada corretamente!")
        print(f"   Conversoes serao {info['velocidade']} mais rapidas")
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    test_gpu_detection()


#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
