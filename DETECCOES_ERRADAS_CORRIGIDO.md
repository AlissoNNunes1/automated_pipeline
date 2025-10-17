# Correcao de Deteccoes Erradas - Analise e Solucoes

## 🔍 PROBLEMAS IDENTIFICADOS

### 1. **Confidence Thresholds Muito Baixos**
**Problema**: Sistema detectava praticamente tudo, incluindo objetos que nao eram pessoas

- `ActivityFilter`: Usava `conf=0.3` (30%) hardcoded
- `EventDetector`: Recebia `conf=0.25` (25%) como default
- `config.yaml`: Definia `conf_threshold: 0.35` mas codigo ignorava

**Impacto**: Falsos positivos em objetos, sombras, reflexos

---

### 2. **Falta de Validacao de Bounding Boxes**
**Problema**: Nenhum filtro de qualidade nas deteccoes

- Bboxes muito pequenas (ruido, pixels isolados)
- Bboxes muito grandes (fundo, paredes inteiras)
- Bboxes com aspect ratio estranho (objetos horizontais/verticais)

**Impacto**: Deteccoes de "pessoas" em:
- Mochilas esquecidas
- Cadeiras/mesas
- Reflexos em vidros
- Sombras no chao

---

### 3. **Tracks Muito Curtos Aceitos**
**Problema**: Eventos com apenas 2 deteccoes eram aceitos

- `min_track_length`: Nao existia
- Deteccoes esparsas viravam "eventos"

**Impacto**: Ruido e artefatos temporarios geravam propostas

---

### 4. **Duracao Minima Muito Baixa**
**Problema**: Eventos de 0.5s eram aceitos

- `min_event_duration_seconds: 0.5` (meio segundo)
- Piscar de olhos do detector virava evento

**Impacto**: Centenas de eventos de 1-2 frames (falsos positivos)

---

## ✅ SOLUCOES IMPLEMENTADAS

### 1. **Confidence Thresholds Aumentados**

#### Config.yaml (nova configuracao)
```yaml
activity_filter:
  person_conf_threshold: 0.5  # ANTES: 0.3 (hardcoded)

event_detector:
  conf_threshold: 0.5  # ANTES: 0.35 (ignorado)
```

#### ActivityFilter (novo codigo)
```python
def __init__(
    self,
    person_conf_threshold: float = 0.5,  # NOVO parametro
    ...
):
    self.person_conf_threshold = person_conf_threshold

# Linha 201 (antes):
# conf=0.3,  # HARDCODED

# Linha 201 (depois):
conf=self.person_conf_threshold,  # Configuravel via config
```

#### EventDetector (novo codigo)
```python
def __init__(
    self,
    confidence_threshold: float = 0.5,  # ANTES: 0.25
    ...
):
```

**Resultado**: Apenas deteccoes com 50%+ confidence sao aceitas

---

### 2. **Validacao de Bounding Boxes**

#### Config.yaml (novos parametros)
```yaml
activity_filter:
  min_bbox_area: 2000      # Min 2000 pixels (40x50px aprox)
  max_bbox_area: 500000    # Max 500k pixels (evita fundo inteiro)
  min_aspect_ratio: 0.3    # Min altura/largura
  max_aspect_ratio: 4.0    # Max altura/largura

event_detector:
  min_bbox_area: 2000
  max_bbox_area: 500000
  min_aspect_ratio: 0.3
  max_aspect_ratio: 4.0
```

#### ActivityFilter (nova validacao)
```python
# Linha 205+ (NOVO):
for box in boxes:
    xyxy = box.xyxy[0].cpu().numpy()
    x1, y1, x2, y2 = xyxy
    
    # Calcular area e aspect ratio
    width = x2 - x1
    height = y2 - y1
    area = width * height
    aspect_ratio = height / width if width > 0 else 0
    
    # Filtros de qualidade
    if (self.min_bbox_area <= area <= self.max_bbox_area and
        self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
        valid_detection = True
        break
```

#### EventDetector (nova validacao)
```python
# Linha 230+ (NOVO):
for track_id, conf, bbox in zip(track_ids, confidences, xyxy):
    x1, y1, x2, y2 = bbox
    
    # Validar qualidade da bbox
    width = x2 - x1
    height = y2 - y1
    area = width * height
    aspect_ratio = height / width if width > 0 else 0
    
    # Filtros de qualidade
    if not (self.min_bbox_area <= area <= self.max_bbox_area):
        continue  # Bbox muito pequena ou muito grande
    
    if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
        continue  # Aspect ratio invalido
```

**Resultado**: Bboxes suspeitas sao rejeitadas antes de criar eventos

---

### 3. **Minimo de Deteccoes por Track**

#### Config.yaml (novo parametro)
```yaml
event_detector:
  min_track_length: 15  # Min 15 deteccoes (0.5s @ 30fps)
```

#### EventDetector (nova validacao)
```python
# Linha 260 (antes):
# if len(detections) < 2:

# Linha 260 (depois):
if len(detections) < self.min_track_length:  # 15 deteccoes minimo
    continue
```

**Resultado**: Tracks esparsas sao filtradas

---

### 4. **Duracao Minima Aumentada**

#### Config.yaml
```yaml
event_detector:
  min_event_duration_seconds: 1.0  # ANTES: 0.8s
```

#### EventDetector
```python
def __init__(
    self,
    min_duration_seconds: float = 1.0,  # ANTES: 0.5
    ...
):
```

**Resultado**: Eventos muito curtos sao rejeitados

---

## 📊 IMPACTO ESPERADO

### Antes das Correcoes
- Confidence: 25-35%
- Validacao bbox: Nenhuma
- Min track length: 2 deteccoes
- Min duration: 0.5s

**Resultado**:
- ~1328 propostas
- ~60-70% falsos positivos
- Tempo de revisao: Horas

---

### Depois das Correcoes
- Confidence: 50%
- Validacao bbox: Area + aspect ratio
- Min track length: 15 deteccoes
- Min duration: 1.0s

**Resultado Esperado**:
- ~300-500 propostas (reducao 60-70%)
- ~10-20% falsos positivos
- Tempo de revisao: 30-60 min

---

## 🧪 COMO TESTAR

### 1. Reprocessar Videos

Delete cache antigo e reprocesse:

```powershell
# Deletar dados antigos
Remove-Item -Recurse -Force automated_pipeline\data_processing

# Reprocessar com novas configuracoes
python automated_pipeline\pipeline_new.py
```

### 2. Verificar Metricas

**Espere ver**:
- Reducao de 60-80% no numero de chunks ativos (ActivityFilter mais rigoroso)
- Reducao de 50-70% no numero de eventos (EventDetector mais seletivo)
- Reducao de 60-70% no numero de propostas (Menos eventos = menos propostas)

### 3. Revisar Qualidade

Execute review GUI:

```powershell
python automated_pipeline\review_gui.py
```

**Verifique**:
- Bboxes cobrem pessoas inteiras (nao objetos)
- Pessoas sao realmente pessoas (nao sombras/reflexos)
- Tracks sao continuas (nao pulam frames)

---

## 🔧 AJUSTES FINOS (Se Necessario)

### Se ainda houver MUITOS falsos positivos:

```yaml
# config.yaml
activity_filter:
  person_conf_threshold: 0.6  # Aumentar para 60%

event_detector:
  conf_threshold: 0.6
  min_track_length: 20  # Aumentar para 20 deteccoes
  min_bbox_area: 3000   # Aumentar area minima
```

### Se estiver perdendo deteccoes validas:

```yaml
# config.yaml
activity_filter:
  person_conf_threshold: 0.45  # Reduzir para 45%

event_detector:
  conf_threshold: 0.45
  min_track_length: 10  # Reduzir para 10 deteccoes
  min_bbox_area: 1500   # Reduzir area minima
```

---

## 📝 CHECKLIST DE VALIDACAO

- [ ] Reprocessar 1 video de teste
- [ ] Verificar reducao de propostas (esperado: 60-70%)
- [ ] Revisar 20-30 propostas no GUI
- [ ] Confirmar reducao de falsos positivos (esperado: <20%)
- [ ] Ajustar thresholds se necessario
- [ ] Reprocessar dataset completo
- [ ] Documentar thresholds finais escolhidos

---

## 🎯 THRESHOLDS RECOMENDADOS POR CENARIO

### Ambiente Controlado (loja pequena, boa iluminacao)
```yaml
person_conf_threshold: 0.6
conf_threshold: 0.6
min_track_length: 20
min_bbox_area: 3000
```

### Ambiente Normal (loja media, iluminacao variavel)
```yaml
person_conf_threshold: 0.5  # ATUAL
conf_threshold: 0.5         # ATUAL
min_track_length: 15        # ATUAL
min_bbox_area: 2000         # ATUAL
```

### Ambiente Dificil (loja grande, baixa iluminacao, muita oclusao)
```yaml
person_conf_threshold: 0.4
conf_threshold: 0.4
min_track_length: 10
min_bbox_area: 1500
```

---

## 🔄 PROXIMOS PASSOS

1. **Testar com 1 video** para validar impacto
2. **Revisar metricas** de reducao
3. **Ajustar thresholds** se necessario
4. **Reprocessar dataset completo**
5. **Atualizar documentacao** com thresholds finais

---

**Data**: 2025-10-17  
**Autor**: Pipeline Automatizado v2.0  
**Status**: ✅ Corrigido e pronto para teste

#    __  ____ ____ _  _ 
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
