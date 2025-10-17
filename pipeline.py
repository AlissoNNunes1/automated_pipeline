#!/usr/bin/env python
"""
Pipeline Orchestrator - Coordena todo o fluxo automatizado

Executa pipeline completo de processamento:
1. Chunking: Divide video longo em chunks
2. Filtering: Remove chunks sem atividade
3. Detection: Detecta eventos com tracking
4. Labeling: Gera propostas de anotacao
5. Export: Prepara para revisao CVAT

Uso:
    python pipeline.py --video camera.mp4 --output data/ --full
    python pipeline.py --video camera.mp4 --output data/ --stage chunk
"""

import argparse
import sys
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

# Importar componentes do pipeline
from core.video_chunker import VideoChunker
from core.activity_filter import ActivityFilter
from core.event_detector import EventDetector
from core.auto_labeler import AutoLabeler


class PipelineOrchestrator:
    """
    Orquestrador principal do pipeline automatizado
    
    Coordena execucao sequencial de todos os componentes
    com tracking de progresso e metricas
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa orquestrador
        
        Args:
            config_path: Path para arquivo YAML de configuracao (opcional)
        """
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        
        # Metricas globais
        self.metrics = {
            'pipeline_start': datetime.now().isoformat(),
            'stages_completed': [],
            'total_processing_time': 0
        }
    
    def run_full_pipeline(
        self,
        video_path: str,
        output_base_dir: str,
        start_timestamp: Optional[datetime] = None
    ):
        """
        Executa pipeline completo
        
        Args:
            video_path: Caminho do video de entrada
            output_base_dir: Diretorio base para outputs
            start_timestamp: Timestamp inicial do video (opcional)
        """
        self.logger.info("=" * 60)
        self.logger.info("INICIANDO PIPELINE AUTOMATIZADO")
        self.logger.info("=" * 60)
        self.logger.info(f"Video: {video_path}")
        self.logger.info(f"Output: {output_base_dir}")
        
        # Criar estrutura de diretorios
        self._create_output_structure(output_base_dir)
        
        # STAGE 1: Chunking
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STAGE 1: VIDEO CHUNKING")
        self.logger.info("=" * 60)
        
        chunks = self._run_chunking(
            video_path,
            os.path.join(output_base_dir, 'chunks'),
            start_timestamp
        )
        
        self.metrics['total_chunks'] = len(chunks)
        self.metrics['stages_completed'].append('chunking')
        
        # STAGE 2: Activity Filtering
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STAGE 2: ACTIVITY FILTERING")
        self.logger.info("=" * 60)
        
        active_chunks, filter_stats = self._run_filtering(
            chunks,
            os.path.join(output_base_dir, 'active_chunks')
        )
        
        self.metrics['active_chunks'] = len(active_chunks)
        self.metrics['filter_reduction'] = (1 - len(active_chunks)/len(chunks)) * 100
        self.metrics['stages_completed'].append('filtering')
        
        # STAGE 3: Event Detection
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STAGE 3: EVENT DETECTION")
        self.logger.info("=" * 60)
        
        events, detection_stats = self._run_detection(
            active_chunks,
            os.path.join(output_base_dir, 'events')
        )
        
        self.metrics['total_events'] = len(events)
        self.metrics['stages_completed'].append('detection')
        
        # STAGE 4: Auto Labeling
        self.logger.info("\n" + "=" * 60)
        self.logger.info("STAGE 4: AUTO LABELING")
        self.logger.info("=" * 60)
        
        proposals, labeling_stats = self._run_labeling(
            events,
            os.path.join(output_base_dir, 'proposals')
        )
        
        self.metrics['total_proposals'] = len(proposals)
        self.metrics['needs_review'] = sum(1 for p in proposals if p['needs_review'])
        self.metrics['stages_completed'].append('labeling')
        
        # Finalizar
        self.metrics['pipeline_end'] = datetime.now().isoformat()
        self._save_final_report(output_base_dir)
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("PIPELINE CONCLUIDO COM SUCESSO!")
        self.logger.info("=" * 60)
        self._print_summary()
    
    def run_single_stage(
        self,
        stage: str,
        input_path: str,
        output_dir: str,
        **kwargs
    ):
        """
        Executa apenas um estagio do pipeline
        
        Args:
            stage: Nome do estagio (chunk, filter, detect, label)
            input_path: Path de entrada
            output_dir: Diretorio de saida
            **kwargs: Argumentos adicionais especificos do estagio
        """
        self.logger.info(f"Executando estagio: {stage}")
        
        if stage == 'chunk':
            start_timestamp = kwargs.get('start_timestamp')
            chunks = self._run_chunking(input_path, output_dir, start_timestamp)
            self.logger.info(f"Resultado: {len(chunks)} chunks gerados")
        
        elif stage == 'filter':
            from video_chunker import VideoChunker
            chunker = VideoChunker()
            chunks = chunker.load_chunks_index(input_path)
            active_chunks, stats = self._run_filtering(chunks, output_dir)
            self.logger.info(f"Resultado: {len(active_chunks)} chunks ativos")
        
        elif stage == 'detect':
            active_chunks, _ = ActivityFilter.load_report(input_path)
            events, stats = self._run_detection(active_chunks, output_dir)
            self.logger.info(f"Resultado: {len(events)} eventos detectados")
        
        elif stage == 'label':
            events, _ = EventDetector.load_events(input_path)
            proposals, stats = self._run_labeling(events, output_dir)
            self.logger.info(f"Resultado: {len(proposals)} propostas geradas")
        
        else:
            self.logger.error(f"Estagio invalido: {stage}")
            sys.exit(1)
    
    def _run_chunking(
        self,
        video_path: str,
        output_dir: str,
        start_timestamp: Optional[datetime]
    ) -> list:
        """Executa stage de chunking"""
        chunker = VideoChunker(
            chunk_duration_seconds=self.config.get('chunk_duration', 180)
        )
        
        chunks = chunker.chunk_video(
            video_path=video_path,
            output_dir=output_dir,
            start_time=start_timestamp
        )
        
        return chunks
    
    def _run_filtering(self, chunks: list, output_dir: str) -> tuple:
        """Executa stage de filtering"""
        filter = ActivityFilter(
            motion_threshold=self.config.get('motion_threshold', 0.02),
            min_person_frames=self.config.get('min_person_frames', 30),
            person_detection_model=self.config.get('filter_model', 'yolo11n.pt'),
            motion_sample_rate=self.config.get('motion_sample_rate', 10),
            person_sample_rate=self.config.get('person_sample_rate', 15)
        )
        
        active_chunks, stats = filter.filter_inactive_chunks(
            chunks,
            output_dir=output_dir
        )
        
        return active_chunks, stats
    
    def _run_detection(self, active_chunks: list, output_dir: str) -> tuple:
        """Executa stage de detection"""
        detector = EventDetector(
            detector_model=self.config.get('detector_model', 'yolo11m.pt'),
            tracker_config=self.config.get('tracker_config', 'bytetrack.yaml'),
            confidence_threshold=self.config.get('confidence_threshold', 0.25),
            min_duration_seconds=self.config.get('min_duration', 0.5)
        )
        
        events, stats = detector.detect_events_batch(
            active_chunks,
            output_dir=output_dir
        )
        
        return events, stats
    
    def _run_labeling(self, events: list, output_dir: str) -> tuple:
        """Executa stage de labeling"""
        labeler = AutoLabeler(
            normal_max_duration=self.config.get('normal_max_duration', 2.0),
            suspicious_min_duration=self.config.get('suspicious_min_duration', 10.0),
            suspicious_min_frames=self.config.get('suspicious_min_frames', 150)
        )
        
        proposals, stats = labeler.generate_proposals_batch(
            events,
            output_dir=output_dir
        )
        
        return proposals, stats
    
    def _create_output_structure(self, base_dir: str):
        """Cria estrutura de diretorios para outputs"""
        dirs = [
            'chunks',
            'active_chunks',
            'events',
            'proposals',
            'annotations'
        ]
        
        for dir_name in dirs:
            os.makedirs(os.path.join(base_dir, dir_name), exist_ok=True)
        
        self.logger.info(f"Estrutura de diretorios criada em: {base_dir}")
    
    def _load_config(self, config_path: Optional[str]) -> dict:
        """Carrega configuracao do arquivo YAML"""
        if config_path and os.path.exists(config_path):
            import yaml
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        # Config padrao
        return {
            'chunk_duration': 180,
            'motion_threshold': 0.02,
            'min_person_frames': 30,
            'filter_model': 'yolo11n.pt',
            'detector_model': 'yolo11m.pt',
            'tracker_config': 'bytetrack.yaml',
            'confidence_threshold': 0.25,
            'min_duration': 0.5,
            'normal_max_duration': 2.0,
            'suspicious_min_duration': 10.0,
            'suspicious_min_frames': 150
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Configura sistema de logging"""
        logger = logging.getLogger('pipeline')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        
        return logger
    
    def _save_final_report(self, output_dir: str):
        """Salva relatorio final com todas as metricas"""
        import json
        
        report_path = os.path.join(output_dir, 'pipeline_report.json')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Relatorio final salvo em: {report_path}")
    
    def _print_summary(self):
        """Imprime resumo das metricas"""
        print("\n" + "=" * 60)
        print("RESUMO DO PIPELINE")
        print("=" * 60)
        print(f"Total de chunks: {self.metrics.get('total_chunks', 0)}")
        print(f"Chunks ativos: {self.metrics.get('active_chunks', 0)}")
        print(f"Reducao: {self.metrics.get('filter_reduction', 0):.1f}%")
        print(f"Eventos detectados: {self.metrics.get('total_events', 0)}")
        print(f"Propostas geradas: {self.metrics.get('total_proposals', 0)}")
        print(f"Requerem revisao: {self.metrics.get('needs_review', 0)}")
        print("=" * 60)


def main():
    """Funcao principal CLI"""
    parser = argparse.ArgumentParser(
        description='Pipeline automatizado para processamento de videos de vigilancia'
    )
    
    parser.add_argument(
        '--video',
        required=True,
        help='Caminho do video de entrada'
    )
    
    parser.add_argument(
        '--output',
        required=True,
        help='Diretorio base para outputs'
    )
    
    parser.add_argument(
        '--config',
        help='Caminho para arquivo de configuracao YAML (opcional)'
    )
    
    parser.add_argument(
        '--stage',
        choices=['chunk', 'filter', 'detect', 'label', 'full'],
        default='full',
        help='Estagio a executar (padrao: full)'
    )
    
    parser.add_argument(
        '--start-timestamp',
        help='Timestamp inicial do video (formato: YYYY-MM-DD HH:MM:SS)'
    )
    
    args = parser.parse_args()
    
    # Criar orquestrador
    orchestrator = PipelineOrchestrator(config_path=args.config)
    
    # Converter timestamp se fornecido
    start_timestamp = None
    if args.start_timestamp:
        try:
            start_timestamp = datetime.strptime(
                args.start_timestamp,
                '%Y-%m-%d %H:%M:%S'
            )
        except ValueError:
            print(f"Erro: Formato de timestamp invalido. Use: YYYY-MM-DD HH:MM:SS")
            sys.exit(1)
    
    # Executar pipeline
    if args.stage == 'full':
        orchestrator.run_full_pipeline(
            video_path=args.video,
            output_base_dir=args.output,
            start_timestamp=start_timestamp
        )
    else:
        orchestrator.run_single_stage(
            stage=args.stage,
            input_path=args.video,
            output_dir=args.output,
            start_timestamp=start_timestamp
        )


if __name__ == "__main__":
    main()


#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
