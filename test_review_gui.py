"""
Script de teste para Review GUI
Gera propostas de exemplo (mockadas) para validar interface

Uso:
    python test_review_gui.py
    
   __  ____ ____ _  _
 / _\/ ___) ___) )( \
/    \___ \___ ) \/ (
\_/\_(____(____|____/
"""

import json
import os
from pathlib import Path
from datetime import datetime


def create_mock_proposals():
    """
    Cria propostas mockadas para teste da GUI
    """
    
    # Classes do modelo
    classes = [
        'comportamento_normal',
        'furto_discreto',
        'furto_bolsa_mochila',
        'furto_grupo_colaborativo',
        'acoes_ambiguas_suspeitas',
        'funcionario_reposicao'
    ]
    
    # Criar 10 propostas de exemplo
    proposals = []
    
    for i in range(10):
        # Alternar entre classes
        class_idx = i % len(classes)
        suggested_class = classes[class_idx]
        
        # Dados de exemplo
        proposal = {
            'event_id': f'chunk_000{i//3}_track_{i+1}',
            'images': [
                {
                    'id': 100 + (i * 30) + j,
                    'file_name': f'frame_{100 + (i * 30) + j:06d}.jpg',
                    'width': 1920,
                    'height': 1080
                }
                for j in range(20)  # 20 frames por evento
            ],
            'annotations': [
                {
                    'image_id': 100 + (i * 30) + j,
                    'track_id': i + 1,
                    'bbox': [
                        500 + (j * 5),  # x1 (movimento)
                        300,            # y1
                        680 + (j * 5),  # x2
                        680             # y2
                    ],
                    'category_id': class_idx,
                    'confidence': 0.75 + (i * 0.02),
                    'frame_number': 100 + (i * 30) + j
                }
                for j in range(20)
            ],
            'suggested_class': suggested_class,
            'confidence': 0.75 + (i * 0.02),
            'classification_confidence': 0.50 + (i * 0.03),
            'needs_review': i % 3 != 0,  # 2/3 precisam revisao
            'reasoning': _get_reasoning(suggested_class, i),
            'metadata': {
                'chunk_id': f'chunk_000{i//3}',
                'chunk_timestamp': datetime.now().isoformat(),
                'track_id': i + 1,
                'start_frame': 100 + (i * 30),
                'end_frame': 100 + (i * 30) + 19,
                'duration_seconds': 0.6 + (i * 0.3),
                'frame_count': 20,
                'confidence_avg': 0.75 + (i * 0.02),
                'event_type': 'person_detected'
            }
        }
        
        proposals.append(proposal)
    
    return proposals


def _get_reasoning(class_name: str, idx: int) -> str:
    """
    Gera raciocinio baseado na classe
    """
    reasoning_map = {
        'comportamento_normal': f"Duracao curta ({0.6 + idx*0.3:.1f}s) com 20 frames indica passagem rapida. Movimento fluido sem interacao prolongada com produtos.",
        'furto_discreto': f"Movimento rapido da mao detectado. Track {idx+1} permanece {0.6 + idx*0.3:.1f}s em area de produtos com gestos suspeitos de ocultacao.",
        'furto_bolsa_mochila': f"Pessoa com bolsa/mochila detectada por {0.6 + idx*0.3:.1f}s. Movimentos indicam possivel insercao de itens nao pagos.",
        'furto_grupo_colaborativo': f"Multiplos tracks detectados ({idx+1} e outros) com coordenacao temporal. Duracao de {0.6 + idx*0.3:.1f}s sugere acao planejada.",
        'acoes_ambiguas_suspeitas': f"Duracao de {0.6 + idx*0.3:.1f}s com 20 frames. Comportamento nao conclusivo mas merece atencao. Track {idx+1} apresenta hesitacao.",
        'funcionario_reposicao': f"Padrao de movimento compativel com reposicao de estoque. Track {idx+1} permanece {0.6 + idx*0.3:.1f}s organizando produtos."
    }
    
    return reasoning_map.get(class_name, "Comportamento observado requer classificacao manual.")


def save_mock_proposals(proposals: list, output_dir: str = 'test_data'):
    """
    Salva propostas mockadas no formato esperado pela GUI
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Salvar proposals_metadata.json
    metadata_file = output_path / 'proposals_metadata.json'
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump({
            'proposals': proposals,
            'metadata': {
                'total_proposals': len(proposals),
                'needs_review': sum(1 for p in proposals if p['needs_review']),
                'generated_at': datetime.now().isoformat(),
                'source': 'mock_data_for_testing'
            }
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Propostas mockadas criadas: {metadata_file}")
    print(f"📊 Total: {len(proposals)} propostas")
    print(f"⚠️  Precisam revisao: {sum(1 for p in proposals if p['needs_review'])}")
    
    return str(metadata_file)


def create_mock_chunks_dir(output_dir: str = 'test_data/chunks'):
    """
    Cria diretorio de chunks vazio (GUI vai mostrar mensagem se nao encontrar video)
    """
    chunks_path = Path(output_dir)
    chunks_path.mkdir(parents=True, exist_ok=True)
    
    # Criar arquivo README explicando
    readme = chunks_path / 'README.txt'
    readme.write_text("""
AVISO: Chunks de video nao disponiveis para teste

Este diretorio esta vazio propositalmente para teste da GUI.
A interface de revisao vai mostrar:
- Informacoes dos eventos (metadata)
- Mensagem "Video nao encontrado" no preview
- Controles de navegacao e classificacao funcionais

Para teste completo com video:
1. Execute o pipeline ate o estagio de labeling
2. Use os chunks reais gerados em data_processing/active_chunks/
    """, encoding='utf-8')
    
    print(f"📁 Diretorio de chunks criado: {chunks_path}")
    print(f"ℹ️  Chunks de video nao incluidos (GUI vai avisar)")
    
    return str(chunks_path)


def main():
    """
    Funcao principal de teste
    """
    print("\n" + "=" * 70)
    print("GERANDO DADOS MOCKADOS PARA TESTE DA GUI DE REVISAO")
    print("=" * 70 + "\n")
    
    # Criar propostas mockadas
    proposals = create_mock_proposals()
    
    # Salvar propostas
    proposals_file = save_mock_proposals(proposals)
    
    # Criar diretorio de chunks
    chunks_dir = create_mock_chunks_dir()
    
    # Instrucoes de uso
    print("\n" + "=" * 70)
    print("DADOS DE TESTE GERADOS COM SUCESSO!")
    print("=" * 70)
    
    print("\n🚀 Para testar a GUI, execute:\n")
    print(f"  python automated_pipeline/review_gui.py \\")
    print(f"    --proposals {proposals_file} \\")
    print(f"    --chunks {chunks_dir}")
    
    print("\n📝 FUNCIONALIDADES PARA TESTAR:")
    print("  - Navegacao com setas (← →)")
    print("  - Aprovar com tecla A")
    print("  - Rejeitar com tecla R")
    print("  - Corrigir classe com tecla C")
    print("  - Preview de frames (vai mostrar 'Video nao encontrado')")
    print("  - Exibicao de metadata dos eventos")
    print("  - Contador de progresso")
    print("  - Finalizacao e export para YOLO")
    
    print("\n⚠️  NOTA: Preview de video nao estara disponivel (chunks mockados)")
    print("   Para teste completo, use dados reais do pipeline.")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()

#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
