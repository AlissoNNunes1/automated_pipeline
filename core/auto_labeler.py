r"""
Auto Labeler - Gera propostas de anotacao automaticas

Analisa eventos detectados e aplica heuristicas para classificacao inicial.
Gera propostas prontas para revisao humana no CVAT.

Tecnicas:
- Heuristicas baseadas em duracao e movimento
- Analise de bounding boxes
- Geracao de metadata COCO-compatible
- Classificacao inicial para acelerar revisao

Output: Propostas prontas para CVAT ou revisao manual
"""

import cv2
import json
import os
import logging
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class AutoLabeler:
    """
    Gera propostas de anotacao automaticas para eventos
    
    Processo:
    1. Analisar caracteristicas do evento (duracao, movimento, bbox)
    2. Aplicar heuristicas de classificacao
    3. Gerar propostas em formato COCO/YOLO
    4. Marcar nivel de confianca para revisao
    
    Classes comportamentais:
    0 - comportamento_normal
    1 - furto_discreto
    2 - furto_bolsa_mochila
    3 - furto_grupo_colaborativo
    4 - acoes_ambiguas_suspeitas
    5 - funcionario_reposicao
    
    Attributes:
        normal_max_duration: Duracao maxima para classificar como normal
        suspicious_min_duration: Duracao minima para suspeita
        suspicious_min_frames: Frames minimos para suspeita
        high_confidence_threshold: Threshold para alta confianca
    """
    
    def __init__(
        self,
        normal_max_duration: float = 2.0,
        suspicious_min_duration: float = 10.0,
        suspicious_min_frames: int = 150,
        high_confidence_threshold: float = 0.7
    ):
        """
        Inicializa AutoLabeler
        
        Args:
            normal_max_duration: Max duracao para comportamento normal (s)
            suspicious_min_duration: Min duracao para suspeita (s)
            suspicious_min_frames: Min frames para suspeita
            high_confidence_threshold: Threshold para confianca alta
        """
        self.normal_max_duration = normal_max_duration
        self.suspicious_min_duration = suspicious_min_duration
        self.suspicious_min_frames = suspicious_min_frames
        self.high_confidence_threshold = high_confidence_threshold
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Mapeamento de classes
        self.class_names = [
            'comportamento_normal',
            'furto_discreto',
            'furto_bolsa_mochila',
            'furto_grupo_colaborativo',
            'acoes_ambiguas_suspeitas',
            'funcionario_reposicao'
        ]
        
        self.class_map = {name: idx for idx, name in enumerate(self.class_names)}
    
    def generate_proposals_batch(
        self,
        events: List[Dict],
        output_dir: Optional[str] = None
    ) -> Tuple[List[Dict], Dict]:
        """
        Gera propostas para uma lista de eventos
        
        Args:
            events: Lista de eventos (do EventDetector)
            output_dir: Diretorio para salvar propostas (opcional)
        
        Returns:
            Tupla com:
            - Lista de propostas geradas
            - Dict com estatisticas do processamento
        """
        self.logger.info(f"Gerando propostas para {len(events)} eventos...")
        
        proposals = []
        stats = {
            'total_events': len(events),
            'proposals_by_class': defaultdict(int),
            'confidence_distribution': {
                'high': 0,    # > 0.7
                'medium': 0,  # 0.4-0.7
                'low': 0      # < 0.4
            },
            'processing_time_seconds': 0,
            'start_time': datetime.now().isoformat()
        }
        
        start_time = datetime.now()
        
        # Barra de progresso
        total_events = len(events)
        progress_width = 40
        
        for idx, event in enumerate(events):
            event_id = event.get('event_id', f'event_{idx:04d}')
            
            # Progresso a cada 10% ou multiplos de 50 eventos
            if (idx + 1) % max(1, total_events // 10) == 0 or (idx + 1) % 50 == 0:
                progress = (idx + 1) / total_events
                filled = int(progress_width * progress)
                bar = '█' * filled + '░' * (progress_width - filled)
                percent = progress * 100
                
                elapsed = (datetime.now() - start_time).total_seconds()
                avg_time = elapsed / (idx + 1)
                remaining = total_events - (idx + 1)
                eta_seconds = avg_time * remaining
                eta_minutes = eta_seconds / 60
                
                self.logger.info(
                    f"[{idx+1}/{total_events}] {bar} {percent:.1f}% | "
                    f"ETA: {eta_minutes:.1f} min"
                )
            
            # Gerar proposta
            proposal = self.generate_proposal(event)
            
            proposals.append(proposal)
            
            # Atualizar estatisticas
            suggested_class = proposal['suggested_class']
            stats['proposals_by_class'][suggested_class] += 1
            
            confidence = proposal['classification_confidence']
            if confidence > 0.7:
                stats['confidence_distribution']['high'] += 1
            elif confidence > 0.4:
                stats['confidence_distribution']['medium'] += 1
            else:
                stats['confidence_distribution']['low'] += 1
        
        end_time = datetime.now()
        stats['processing_time_seconds'] = (end_time - start_time).total_seconds()
        stats['end_time'] = end_time.isoformat()
        
        # Log resumo
        self.logger.info(
            f"\n=== RESUMO DAS PROPOSTAS ===\n"
            f"Eventos processados: {stats['total_events']}\n"
            f"Propostas por classe:\n" +
            '\n'.join(f"  - {cls}: {count}" for cls, count in stats['proposals_by_class'].items()) +
            f"\nConfianca:\n"
            f"  - Alta (>70%): {stats['confidence_distribution']['high']}\n"
            f"  - Media (40-70%): {stats['confidence_distribution']['medium']}\n"
            f"  - Baixa (<40%): {stats['confidence_distribution']['low']}\n"
            f"Tempo de processamento: {stats['processing_time_seconds']:.1f}s\n"
        )
        
        # Salvar propostas se output_dir fornecido
        if output_dir:
            self._save_proposals(proposals, stats, output_dir)
        
        return proposals, stats
    
    def generate_proposal(self, event: Dict) -> Dict:
        """
        Gera proposta de anotacao para um evento
        
        Args:
            event: Evento do EventDetector
        
        Returns:
            Dict com proposta completa:
            - event_id: ID do evento
            - suggested_class: Classe sugerida (nome)
            - category_id: ID da classe (0-5)
            - classification_confidence: Confianca na classificacao (0-1)
            - needs_review: Se precisa revisao humana
            - annotations: Lista de anotacoes por frame
            - reasoning: Explicacao da classificacao
        """
        # Extrair caracteristicas
        duration = event['duration_seconds']
        frame_count = event['frame_count']
        movement = event.get('movement_distance', 0)
        avg_conf = event['confidence_avg']
        
        # Aplicar heuristicas
        suggested_class, reasoning, confidence = self._classify_behavior_heuristic(
            duration, frame_count, movement, avg_conf
        )
        
        # Verificar se precisa revisao
        needs_review = (
            confidence < self.high_confidence_threshold or
            suggested_class == 'acoes_ambiguas_suspeitas'
        )
        
        # Criar proposta
        proposal = {
            'event_id': event.get('event_id', 'unknown'),
            'track_id': event.get('track_id', -1),
            'chunk_id': event.get('chunk_id', 'unknown'),
            'suggested_class': suggested_class,
            'category_id': self.class_map[suggested_class],
            'classification_confidence': confidence,
            'needs_review': needs_review,
            'reasoning': reasoning,
            'event_characteristics': {
                'duration_seconds': duration,
                'frame_count': frame_count,
                'movement_distance': movement,
                'confidence_avg': avg_conf
            },
            'bbox_sequence': event.get('bbox_sequence', []),
            'start_frame': event.get('start_frame', 0),
            'end_frame': event.get('end_frame', 0),
            'timestamp': datetime.now().isoformat()
        }
        
        return proposal
    
    def _classify_behavior_heuristic(
        self,
        duration: float,
        frame_count: int,
        movement: float,
        confidence: float
    ) -> Tuple[str, str, float]:
        """
        Classifica comportamento usando heuristicas simples
        
        Args:
            duration: Duracao do evento (segundos)
            frame_count: Numero de frames
            movement: Distancia de movimento (pixels)
            confidence: Confianca media das deteccoes
        
        Returns:
            Tupla (classe, reasoning, confidence)
        """
        # HEURISTICA 1: Duracao muito curta -> Normal
        if duration < self.normal_max_duration:
            return (
                'comportamento_normal',
                f'Duracao curta ({duration:.1f}s < {self.normal_max_duration}s) indica passagem rapida',
                0.7
            )
        
        # HEURISTICA 2: Duracao longa + muitos frames -> Suspeito
        if duration > self.suspicious_min_duration and frame_count > self.suspicious_min_frames:
            return (
                'acoes_ambiguas_suspeitas',
                f'Duracao longa ({duration:.1f}s) com {frame_count} frames indica permanencia prolongada',
                0.5
            )
        
        # HEURISTICA 3: Movimento muito pequeno -> Parado (pode ser suspeito)
        if movement < 50:  # Pixels
            return (
                'acoes_ambiguas_suspeitas',
                f'Pouco movimento ({movement:.1f}px) indica pessoa parada por {duration:.1f}s',
                0.4
            )
        
        # HEURISTICA 4: Confianca baixa nas deteccoes -> Revisar
        if confidence < 0.5:
            return (
                'acoes_ambiguas_suspeitas',
                f'Confianca baixa ({confidence:.2f}) nas deteccoes requer revisao',
                0.3
            )
        
        # DEFAULT: Comportamento normal
        return (
            'comportamento_normal',
            f'Comportamento padrao: {duration:.1f}s, {frame_count} frames, movimento {movement:.1f}px',
            0.6
        )
    
    def _save_proposals(
        self,
        proposals: List[Dict],
        stats: Dict,
        output_dir: str
    ):
        """
        Salva propostas em JSON
        
        Args:
            proposals: Lista de propostas
            stats: Estatisticas do processamento
            output_dir: Diretorio de saida
        """
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'statistics': stats,
            'proposals': proposals,
            'class_names': self.class_names,
            'labeler_config': {
                'normal_max_duration': self.normal_max_duration,
                'suspicious_min_duration': self.suspicious_min_duration,
                'suspicious_min_frames': self.suspicious_min_frames,
                'high_confidence_threshold': self.high_confidence_threshold
            }
        }
        
        report_path = os.path.join(output_dir, 'proposals_metadata.json')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Propostas salvas em: {report_path}")
        
        # Salvar tambem em formato YOLO (opcional)
        self._export_yolo_format(proposals, output_dir)
    
    def _export_yolo_format(self, proposals: List[Dict], output_dir: str):
        """
        Exporta propostas em formato YOLO
        
        Gera arquivos .txt para cada proposta com:
        class_id x_center y_center width height (normalizados)
        
        Args:
            proposals: Lista de propostas
            output_dir: Diretorio de saida
        """
        yolo_dir = os.path.join(output_dir, 'yolo_format')
        os.makedirs(yolo_dir, exist_ok=True)
        
        for proposal in proposals:
            event_id = proposal['event_id']
            category_id = proposal['category_id']
            bbox_sequence = proposal.get('bbox_sequence', [])
            
            if not bbox_sequence:
                continue
            
            # Para cada frame na sequencia, criar arquivo YOLO
            for frame_idx, bbox in enumerate(bbox_sequence):
                # bbox formato: [x1, y1, x2, y2]
                x1, y1, x2, y2 = bbox
                
                # Converter para YOLO format (assumindo 1920x1080 padrao)
                img_width = 1920  # TODO: pegar da metadata real
                img_height = 1080
                
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height
                
                # Criar arquivo YOLO
                frame_filename = f"{event_id}_frame_{frame_idx:04d}.txt"
                frame_path = os.path.join(yolo_dir, frame_filename)
                
                with open(frame_path, 'w') as f:
                    f.write(f"{category_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        self.logger.info(f"Formato YOLO exportado para: {yolo_dir}")
    
    @staticmethod
    def load_proposals(proposals_path: str) -> Tuple[List[Dict], Dict]:
        """
        Carrega propostas previamente geradas
        
        Args:
            proposals_path: Caminho do arquivo JSON
        
        Returns:
            Tupla (proposals, stats)
        """
        with open(proposals_path, 'r', encoding='utf-8') as f:
            report = json.load(f)
        
        return report['proposals'], report['statistics']


# Exemplo de uso
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Exemplo: Gerar propostas para eventos
    from event_detector import EventDetector
    
    # 1. Carregar eventos detectados
    events, _ = EventDetector.load_events('data/events/events_summary.json')
    
    # 2. Gerar propostas
    labeler = AutoLabeler(
        normal_max_duration=2.0,
        suspicious_min_duration=10.0,
        suspicious_min_frames=150
    )
    
    proposals, stats = labeler.generate_proposals_batch(
        events,
        output_dir='data/proposals'
    )
    
    # 3. Resultados
    print(f"\n=== RESULTADOS ===")
    print(f"Total de propostas: {len(proposals)}")
    print(f"Alta confianca: {stats['confidence_distribution']['high']}")
    print(f"Precisam revisao: {sum(1 for p in proposals if p['needs_review'])}")
    print(f"Tempo total: {stats['processing_time_seconds']:.1f}s")


#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
