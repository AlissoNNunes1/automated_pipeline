"""
Teste direto do YOLO para comparar predict vs track
"""

import json
import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

# Carregar active chunks report
with open('data_processing/1/active_chunks/active_chunks_report.json', 'r', encoding='utf-8') as f:
    report = json.load(f)

active_chunks = report['active_chunks']

if not active_chunks:
    print("Nenhum chunk ativo encontrado!")
    exit(1)

# Pegar chunk ID da linha de comando ou usar um especifico
if len(sys.argv) > 1:
    chunk_id = sys.argv[1]
    test_chunk = next((c for c in active_chunks if c['chunk_id'] == chunk_id), None)
    if test_chunk is None:
        print(f"Chunk {chunk_id} nao encontrado!")
        exit(1)
else:
    # Procurar chunk_0037 que falhou antes
    test_chunk = next((c for c in active_chunks if c['chunk_id'] == 'chunk_0037'), None)
    if test_chunk is None:
        print("chunk_0037 nao encontrado, usando primeiro chunk")
        test_chunk = active_chunks[0]

# Corrigir caminho relativo para absoluto
base_dir = Path(__file__).parent
chunk_path = base_dir / test_chunk['filepath']

print(f"\n{'='*80}")
print(f"TESTANDO CHUNK: {test_chunk['chunk_id']}")
print(f"Filepath: {chunk_path}")
print(f"Exists: {chunk_path.exists()}")
print(f"Person frames (ActivityFilter): {test_chunk['person_frames']}")
print(f"{'='*80}\n")

# Carregar modelo
model = YOLO('yolo11n.pt')

# Teste 1: PREDICT (como ActivityFilter faz)
print("=== TESTE 1: YOLO PREDICT (como ActivityFilter) ===")
cap = cv2.VideoCapture(str(chunk_path))

predict_detections = 0
frames_checked = 0
sample_rate = 15  # Mesmo do ActivityFilter

while True:
    # Pular frames
    for _ in range(sample_rate - 1):
        cap.grab()
    
    ret, frame = cap.read()
    if not ret:
        break
    
    frames_checked += 1
    
    # Detectar pessoas com PREDICT
    results = model.predict(
        frame,
        verbose=False,
        conf=0.35,  # Mesma conf do EventDetector
        classes=[0]  # Apenas pessoa
    )
    
    for result in results:
        if result.boxes is not None and len(result.boxes) > 0:
            predict_detections += len(result.boxes)
            if frames_checked <= 3:
                print(f"  Frame {frames_checked}: {len(result.boxes)} pessoas detectadas com PREDICT")
            break

cap.release()

print(f"\nRESULTADO PREDICT:")
print(f"  Frames verificados: {frames_checked}")
print(f"  Total deteccoes: {predict_detections}")
print(f"  Frames com pessoas: {predict_detections > 0}")

# Teste 2: TRACK (como EventDetector faz)
print(f"\n{'='*80}")
print("=== TESTE 2: YOLO TRACK (como EventDetector) ===")

track_detections_with_ids = 0
track_detections_without_ids = 0
frames_processed = 0

results = model.track(
    source=str(chunk_path),
    persist=True,
    conf=0.35,
    iou=0.5,
    tracker='bytetrack.yaml',
    classes=[0],
    verbose=False,
    stream=True
)

for frame_idx, result in enumerate(results):
    # Sample rate
    if (frame_idx + 1) % 2 != 0:  # sample_rate=2
        continue
    
    frames_processed += 1
    
    # Verificar deteccoes COM IDs (ByteTrack funcionou)
    if result.boxes is not None and result.boxes.id is not None:
        num_with_ids = len(result.boxes.id)
        track_detections_with_ids += num_with_ids
        if frames_processed <= 3:
            print(f"  Frame {frames_processed}: {num_with_ids} pessoas COM IDs de tracking")
    
    # Verificar deteccoes SEM IDs (ByteTrack falhou)
    elif result.boxes is not None and len(result.boxes) > 0:
        num_without_ids = len(result.boxes)
        track_detections_without_ids += num_without_ids
        if frames_processed <= 3:
            print(f"  Frame {frames_processed}: {num_without_ids} pessoas SEM IDs de tracking")
    
    if frames_processed >= 100:  # Testar apenas primeiros 100 frames processados
        break

print(f"\nRESULTADO TRACK:")
print(f"  Frames processados: {frames_processed}")
print(f"  Deteccoes COM IDs: {track_detections_with_ids}")
print(f"  Deteccoes SEM IDs: {track_detections_without_ids}")

print(f"\n{'='*80}")
print("CONCLUSAO:")
if predict_detections > 0 and track_detections_with_ids == 0:
    print("  PROBLEMA: YOLO detecta com PREDICT mas ByteTrack nao atribui IDs!")
    print("  SOLUCAO: Ajustar configuracao do ByteTrack ou parametros do tracker")
elif predict_detections == 0:
    print("  PROBLEMA: YOLO nao detecta pessoas nem com PREDICT!")
    print("  SOLUCAO: Verificar conf_threshold ou modelo corrompido")
elif track_detections_with_ids > 0:
    print("  SUCESSO: ByteTrack esta funcionando corretamente!")
else:
    print("  INDEFINIDO: Resultado inesperado")
print(f"{'='*80}\n")

#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
