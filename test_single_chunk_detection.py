"""
Script de teste para analisar deteccao em um unico chunk
Mostra estatisticas detalhadas de filtragem
"""

import json
import logging
import sys
from pathlib import Path

# Setup logging para ver mensagens de debug
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Importar modulos
from core.event_detector import EventDetector
import yaml

# Carregar config
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Carregar active chunks report
with open('data_processing/1/active_chunks/active_chunks_report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

active_chunks = report['active_chunks']

if not active_chunks:
    print("Nenhum chunk ativo encontrado!")
    sys.exit(1)

# Testar apenas primeiro chunk
test_chunk = active_chunks[0]

# Corrigir caminho relativo para absoluto
from pathlib import Path
base_dir = Path(__file__).parent
chunk_path = base_dir / test_chunk['filepath']

print(f"\n{'='*80}")
print(f"TESTANDO CHUNK: {test_chunk['chunk_id']}")
print(f"Filepath: {chunk_path}")
print(f"Exists: {chunk_path.exists()}")
print(f"Duration: {test_chunk['duration_seconds']:.1f}s")
print(f"FPS: {test_chunk['fps']:.2f}")
print(f"Person frames: {test_chunk['person_frames']}")
print(f"{'='*80}\n")

# Criar detector com config atual
cfg = config['event_detector']
detector = EventDetector(
    detector_model=cfg['detector_model'],
    tracker_config=cfg['tracker'],
    confidence_threshold=cfg['conf_threshold'],
    iou_threshold=cfg.get('iou_threshold', 0.5),
    min_duration_seconds=cfg['min_event_duration_seconds'],
    sample_rate=cfg.get('sample_rate', 1),
    min_bbox_area=cfg.get('min_bbox_area', 1500),
    max_bbox_area=cfg.get('max_bbox_area', 500000),
    min_aspect_ratio=cfg.get('min_aspect_ratio', 0.25),
    max_aspect_ratio=cfg.get('max_aspect_ratio', 5.0),
    min_track_length=cfg.get('min_track_length', 10),
    min_track_confidence_avg=cfg.get('min_track_confidence_avg', 0.40),
    require_motion_for_event=cfg.get('require_motion_for_event', True),
    min_track_movement_pixels=cfg.get('min_track_movement_pixels', 8.0),
)

print("\nConfiguracoes do detector:")
print(f"  conf_threshold: {detector.confidence_threshold}")
print(f"  min_duration_seconds: {detector.min_duration_seconds}")
print(f"  sample_rate: {detector.sample_rate}")
print(f"  min_track_length: {detector.min_track_length}")
print(f"  min_track_confidence_avg: {detector.min_track_confidence_avg}")
print(f"  require_motion_for_event: {detector.require_motion_for_event}")
print(f"  min_track_movement_pixels: {detector.min_track_movement_pixels}")
print()

# Detectar eventos - passar caminho absoluto
events = detector.detect_events_in_chunk(str(chunk_path), test_chunk)

print(f"\n{'='*80}")
print(f"RESULTADO: {len(events)} eventos detectados")
print(f"{'='*80}\n")

if events:
    for event in events:
        print(f"Evento {event['event_id']}:")
        print(f"  Track ID: {event['track_id']}")
        print(f"  Duracao: {event['duration_seconds']:.2f}s")
        print(f"  Frames: {event['frame_count']}")
        print(f"  Confianca media: {event['confidence_avg']:.2f}")
        print(f"  Movimento: {event['movement_distance']:.1f}px")
        print()
else:
    print("Nenhum evento passou pelos filtros!")
    print("\nVerifique os logs acima para entender onde os tracks foram rejeitados.")
    print("Os contadores de rejeicao mostram:")
    print("  - rejected_track_length: tracks muito curtos")
    print("  - rejected_duration: duracao muito curta")
    print("  - rejected_confidence: confianca media muito baixa")
    print("  - rejected_movement: movimento insuficiente")

#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
