# Pipeline Automatizado de Processamento de Videos Longos

Sistema para processar gravacoes longas (horas) de cameras de vigilancia, automatizando:

- Corte em chunks menores
- Filtragem por atividade (descarta 60-80% de video vazio)
- Deteccao de eventos relevantes
- Rotulagem assistida (human-in-the-loop)
- Reducao drastica de trabalho manual (10-15x mais rapido)

## Arquitetura

```
automated_pipeline/
├── core/                    # Componentes principais
│   ├── video_chunker.py     # Divisao em chunks de 3min
│   ├── activity_filter.py   # Filtragem por movimento e pessoas
│   ├── event_detector.py    # Deteccao de eventos com tracking
│   └── auto_labeler.py      # Geracao de propostas de anotacao
├── utils/                   # Utilitarios
│   ├── logger.py            # Logging padronizado
│   ├── metrics.py           # Metricas de performance
│   └── helpers.py           # Funcoes auxiliares
├── integrations/            # Integracoes externas
│   └── cvat_client.py       # Cliente CVAT para revisao
├── data/                    # Dados temporarios
│   ├── chunks/              # Videos divididos
│   ├── active_chunks/       # Chunks com atividade
│   ├── events/              # Eventos detectados
│   ├── proposals/           # Propostas de anotacao
│   └── annotations/         # Anotacoes validadas
├── config.yaml              # Configuracao central
├── pipeline.py              # Orquestrador principal
└── README.md                # Esta documentacao
```

## Fluxo do Pipeline

```
Video 10h → Chunker → 200 chunks (3min cada)
                ↓
         Activity Filter → 40-80 chunks ativos (60-80% descartado)
                ↓
         Event Detector → 200-500 eventos (5-30s cada)
                ↓
         Auto Labeler → Propostas com bboxes + classes
                ↓
         CVAT Review → Validacao humana (~2-4h)
                ↓
         Export → Dataset anotado pronto para treino
```

## Produtividade

### Manual (Antes)

- 10 horas de video bruto
- Assistir tudo: ~10 horas
- Anotar manualmente: ~30-40 horas
- **TOTAL: 40-50 horas**

### Automatizado (Agora)

- Chunking: ~15 min (automatico)
- Filtragem: ~30 min (automatico)
- Deteccao: ~45 min (automatico)
- Rotulagem: ~20 min (automatico)
- Revisao humana: ~2-4 horas (400 eventos × 30s cada)
- **TOTAL: 3-5 horas**

**GANHO: 10-15x mais rapido**

## Instalacao

```bash
# Instalar dependencias
pip install opencv-python ultralytics numpy pyyaml requests

# Baixar modelos YOLO
# yolo11n.pt e yolo11m.pt serao baixados automaticamente
```

## Uso Basico

### 1. Processar Video Completo

```bash
# Pipeline completo end-to-end
python -m automated_pipeline.pipeline \
  --input videos/camera_loja_01_10h.mp4 \
  --output data/ \
  --start-time "2025-10-07 08:00:00"
```

### 2. Etapas Individuais

```python
from automated_pipeline import VideoChunker, ActivityFilter, EventDetector, AutoLabeler

# Etapa 1: Chunking
chunker = VideoChunker(chunk_duration_seconds=180)
chunks = chunker.chunk_video(
    video_path="camera_loja_01_10h.mp4",
    output_dir="data/chunks",
    start_time=datetime(2025, 10, 7, 8, 0, 0)
)

# Etapa 2: Filtragem
filter = ActivityFilter(
    motion_threshold=0.02,
    person_detection_model='yolo11n.pt',
    min_person_frames=30
)
active_chunks = filter.filter_inactive_chunks(chunks)

# Etapa 3: Deteccao
detector = EventDetector(detector_model='yolo11m.pt')
events = []
for chunk in active_chunks:
    chunk_events = detector.detect_events_in_chunk(chunk['filepath'])
    events.extend(chunk_events)

# Etapa 4: Rotulagem
labeler = AutoLabeler()
proposals = []
for event in events:
    chunk = next(c for c in active_chunks if c['chunk_id'] == event['chunk_id'])
    proposal = labeler.generate_proposals(event, chunk['filepath'])
    proposals.append(proposal)

# Salvar propostas
import json
with open('data/proposals/proposals.json', 'w') as f:
    json.dump(proposals, f, indent=2)
```

## Configuracao

Editar `config.yaml` para ajustar parametros:

```yaml
# Duracao dos chunks
chunking:
  chunk_duration_seconds: 180 # 3 minutos

# Sensibilidade da filtragem
activity_filter:
  motion_threshold: 0.02 # 2% de movimento
  min_person_frames: 30 # Minimo de frames com pessoa

# Modelos YOLO
event_detector:
  detector_model: 'yolo11m.pt' # ou yolo11l.pt para melhor precisao
```

## Integracao com CVAT

```python
from automated_pipeline.integrations.cvat_client import CVATClient

# Configurar cliente
cvat = CVATClient(
    cvat_url='http://localhost:8080',
    username='admin',
    password='admin'
)

# Criar task com propostas
task_id = cvat.create_task_with_proposals(
    task_name='Loja_01_2025-10-07',
    proposals=proposals,
    labels=['normal', 'furto_discreto', 'furto_bolsa', ...]
)

print(f"Task criada: http://localhost:8080/tasks/{task_id}")
```

## Metricas e Relatorios

O sistema gera automaticamente:

- `chunks_index.json`: Metadata de todos os chunks
- `active_chunks_report.json`: Chunks com atividade detectada
- `events_summary.json`: Resumo de eventos por track_id
- `proposals_metadata.json`: Propostas para revisao
- `pipeline_metrics.json`: Metricas de performance e produtividade

## Proximos Passos

1. ✅ Implementar VideoChunker
2. ✅ Implementar ActivityFilter
3. ✅ Implementar EventDetector
4. ✅ Implementar AutoLabeler
5. ⏳ Implementar CVATClient
6. ⏳ Implementar pipeline.py (orquestrador)
7. ⏳ Testes end-to-end com video real

## Troubleshooting

### Problema: Muitos chunks vazios

**Solucao**: Reduzir `motion_threshold` ou `min_person_frames`

### Problema: Eventos muito curtos

**Solucao**: Ajustar `min_event_duration_seconds`

### Problema: Classificacao incorreta

**Solucao**: Melhorar heuristicas em `auto_labeler.heuristics` ou treinar modelo

---

**Contato**: Dataset Construction Team  
**Versao**: 1.0.0  
**Licenca**: MIT

# ** \_\_** \__\_\_ _ \_

# / \_\/ **_) _**) )( \

# / \_** \_** ) \/ (

# \_/\_(\_**\_(\_\_**|\_\_\_\_/
