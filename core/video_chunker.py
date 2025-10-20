r"""
VideoChunker: Divide videos longos em chunks menores indexados por timestamp

Funcionalidades:
- Divide videos de horas em chunks de 3 minutos (configuravel)
- Indexa cada chunk com timestamp preciso
- Gera metadata JSON com informacoes de cada chunk
- Preserva qualidade do video original

   __  ____ ____ _  _
 / _\/ ___) ___) )( \
/    \___ \___ ) \/ (
\_/\_(____(____|____/
"""

import cv2
import os
import json
import logging
import subprocess
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional


class VideoChunker:
    """
    Divide videos longos em chunks menores indexados por timestamp
    """

    def __init__(self, chunk_duration_seconds: int = 180, use_gpu: bool = False):
        """
        Inicializa o VideoChunker

        Args:
            chunk_duration_seconds: Duracao de cada chunk em segundos (padrao: 180 = 3 min)
        """
        self.chunk_duration = chunk_duration_seconds
        self.logger = logging.getLogger(__name__)
        # Se True, tentaremos usar ffmpeg + NVENC para extracao (usa GPU)
        self.use_gpu = use_gpu
        # flag para indicar se ffmpeg esta disponivel no PATH
        self._ffmpeg_available = shutil.which('ffmpeg') is not None

    def chunk_video(
        self,
        video_path: str,
        output_dir: str,
        start_time: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Divide video em chunks

        Args:
            video_path: Caminho do video de entrada
            output_dir: Diretorio de saida para chunks
            start_time: Timestamp inicial do video (se None, usa datetime.now())

        Returns:
            Lista de dicts com metadata dos chunks:
            {
                'chunk_id': 'chunk_001',
                'chunk_index': 0,
                'filepath': 'output/chunk_001.mp4',
                'start_timestamp': '2025-10-07T08:00:00',
                'end_timestamp': '2025-10-07T08:03:00',
                'duration_seconds': 180.0,
                'frame_count': 5400,
                'start_frame': 0,
                'end_frame': 5400
            }
        """
        self.logger.info(f"Iniciando chunking de video: {video_path}")
        self.logger.info(f"Duracao configurada por chunk: {self.chunk_duration}s ({self.chunk_duration // 60}min {self.chunk_duration % 60}s)")

        # Abrir video
        self.logger.debug("Abrindo arquivo de video...")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Nao foi possivel abrir video: {video_path}")

        # Obter propriedades do video
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calcular duracao total do video
        duration_seconds = total_frames / fps
        duration_hours = int(duration_seconds // 3600)
        duration_minutes = int((duration_seconds % 3600) // 60)
        duration_secs = int(duration_seconds % 60)

        self.logger.info(f"  Propriedades do video:")
        self.logger.info(f"    Total de frames: {total_frames:,}")
        self.logger.info(f"    FPS: {fps:.2f}")
        self.logger.info(f"    Resolucao: {width}x{height}")
        self.logger.info(f"    Duracao: {duration_hours:02d}:{duration_minutes:02d}:{duration_secs:02d} ({duration_seconds:.0f}s)")

        # Calcular numero de chunks
        frames_per_chunk = int(fps * self.chunk_duration)
        num_chunks = (total_frames + frames_per_chunk - 1) // frames_per_chunk

        self.logger.info(f"  Configuracao de chunking:")
        self.logger.info(f"    Frames por chunk: {frames_per_chunk:,}")
        self.logger.info(f"    Total de chunks: {num_chunks}")
        self.logger.info(f"    Tamanho estimado: ~{self.chunk_duration}s por chunk")

        # Criar diretorio de saida
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"Diretorio de saida: {output_dir}")

        # Timestamp inicial
        if start_time is None:
            start_time = datetime.now()
        
        self.logger.info(f"  Timestamp inicial: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        chunks_metadata = []
        
        self.logger.info("\n  Iniciando extracao de chunks...")

        # Processar cada chunk com barra de progresso
        for chunk_idx in range(num_chunks):
            chunk_start_frame = chunk_idx * frames_per_chunk
            chunk_end_frame = min((chunk_idx + 1) * frames_per_chunk, total_frames)

            # Calcular timestamps
            chunk_start_time = start_time + timedelta(seconds=chunk_idx * self.chunk_duration)
            chunk_duration_actual = (chunk_end_frame - chunk_start_frame) / fps
            chunk_end_time = chunk_start_time + timedelta(seconds=chunk_duration_actual)

            # Nome do arquivo
            chunk_filename = f"chunk_{chunk_idx:04d}.mp4"
            chunk_filepath = os.path.join(output_dir, chunk_filename)

            # Calcular progresso
            progress = ((chunk_idx + 1) / num_chunks) * 100
            
            # Barra de progresso visual (40 caracteres)
            bar_length = 40
            filled = int(bar_length * (chunk_idx + 1) / num_chunks)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            # Exibir progresso
            print(f"\r  Progresso: [{bar}] {progress:.1f}% ({chunk_idx + 1}/{num_chunks})", end='', flush=True)

            # Extrair chunk
            self.logger.debug(f"\n  Chunk {chunk_idx + 1}/{num_chunks}:")
            self.logger.debug(f"    Frames: {chunk_start_frame} - {chunk_end_frame} ({chunk_end_frame - chunk_start_frame} frames)")
            self.logger.debug(f"    Duracao: {chunk_duration_actual:.1f}s")
            self.logger.debug(f"    Timestamp: {chunk_start_time.strftime('%H:%M:%S')} - {chunk_end_time.strftime('%H:%M:%S')}")
            self.logger.debug(f"    Arquivo: {chunk_filename}")

            # Se use_gpu foi ativado e ffmpeg esta disponivel, usar extracao via ffmpeg NVENC
            if getattr(self, 'use_gpu', False) and self._ffmpeg_available:
                start_seconds = chunk_start_frame / fps
                success = self._extract_chunk_gpu(
                    video_path,
                    start_seconds,
                    chunk_duration_actual,
                    chunk_filepath
                )
            else:
                success = self._extract_chunk(
                    cap,
                    chunk_start_frame,
                    chunk_end_frame,
                    chunk_filepath,
                    fps,
                    (width, height),
                    chunk_idx + 1,
                    num_chunks
                )

            if not success:
                self.logger.warning(f"Falha ao extrair chunk {chunk_idx}")
                continue

            # Metadata
            chunk_meta = {
                'chunk_id': f'chunk_{chunk_idx:04d}',
                'chunk_index': chunk_idx,
                'filepath': chunk_filepath,
                'start_timestamp': chunk_start_time.isoformat(),
                'end_timestamp': chunk_end_time.isoformat(),
                'duration_seconds': chunk_duration_actual,
                'frame_count': chunk_end_frame - chunk_start_frame,
                'start_frame': chunk_start_frame,
                'end_frame': chunk_end_frame,
                'fps': fps,
                'resolution': f"{width}x{height}"
            }
            chunks_metadata.append(chunk_meta)

        # Finalizar barra de progresso
        print()  # Nova linha apos a barra de progresso

        cap.release()

        # Salvar index JSON
        index_path = os.path.join(output_dir, 'chunks_index.json')
        self.logger.info(f"\n  Salvando index de chunks...")
        
        index_data = {
            'source_video': video_path,
            'total_chunks': len(chunks_metadata),
            'chunk_duration_seconds': self.chunk_duration,
            'start_time': start_time.isoformat(),
            'video_properties': {
                'total_frames': total_frames,
                'fps': fps,
                'resolution': f"{width}x{height}",
                'duration_seconds': duration_seconds
            },
            'chunks': chunks_metadata
        }
        
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, indent=2)

        self.logger.info(f"  Chunking concluido com sucesso!")
        self.logger.info(f"    Chunks gerados: {len(chunks_metadata)}")
        self.logger.info(f"    Index salvo: {index_path}")
        
        # Calcular estatisticas
        total_duration = sum(c['duration_seconds'] for c in chunks_metadata)
        avg_chunk_size = total_duration / len(chunks_metadata) if chunks_metadata else 0
        self.logger.info(f"    Duracao total dos chunks: {total_duration:.1f}s")
        self.logger.info(f"    Duracao media por chunk: {avg_chunk_size:.1f}s")

        return chunks_metadata

    def _extract_chunk(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        output_path: str,
        fps: float,
        resolution: tuple,
        chunk_num: int = 0,
        total_chunks: int = 0
    ) -> bool:
        """
        Extrai um chunk do video

        Args:
            cap: VideoCapture aberto
            start_frame: Frame inicial
            end_frame: Frame final
            output_path: Caminho de saida
            fps: FPS do video
            resolution: (width, height)
            chunk_num: Numero do chunk atual (para logs)
            total_chunks: Total de chunks (para logs)

        Returns:
            True se sucesso, False caso contrario
        """
        try:
            # Posicionar no frame inicial
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            # Criar VideoWriter
            width, height = resolution
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec H.264
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            if not writer.isOpened():
                self.logger.error(f"Nao foi possivel criar VideoWriter: {output_path}")
                return False

            # Escrever frames
            frames_written = 0
            expected_frames = end_frame - start_frame
            
            for frame_idx in range(start_frame, end_frame):
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning(f"Nao foi possivel ler frame {frame_idx}")
                    break
                
                writer.write(frame)
                frames_written += 1

            writer.release()

            # Verificar completude do chunk
            completion_percent = (frames_written / expected_frames) * 100 if expected_frames > 0 else 0
            
            if frames_written < expected_frames * 0.9:  # Menos de 90% dos frames
                self.logger.warning(f"Chunk {chunk_num}/{total_chunks} incompleto: {frames_written}/{expected_frames} frames ({completion_percent:.1f}%)")
            else:
                self.logger.debug(f"    Chunk extraido: {frames_written}/{expected_frames} frames ({completion_percent:.1f}%)")

            return True

        except Exception as e:
            self.logger.error(f"Erro ao extrair chunk {chunk_num}/{total_chunks}: {str(e)}")
            return False

    def _extract_chunk_gpu(
        self,
        video_path: str,
        start_seconds: float,
        duration_seconds: float,
        output_path: str
    ) -> bool:
        """
        Extrai um segmento usando ffmpeg + NVENC (quando disponivel)

        Args:
            video_path: arquivo de entrada
            start_seconds: tempo inicial em segundos
            duration_seconds: duracao do segmento em segundos
            output_path: caminho do arquivo de saida

        Returns:
            True se sucesso, False caso contrario
        """
        try:
            # Montar comando ffmpeg (-ss antes do -i para seek rapido)
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_seconds),
                '-i', str(video_path),
                '-t', str(duration_seconds),
                '-c:v', 'h264_nvenc',
                '-preset', 'p1',
                '-cq', '19',
                '-c:a', 'copy',
                str(output_path)
            ]

            self.logger.debug(f"Executando ffmpeg NVENC: {' '.join(cmd)}")
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if res.returncode != 0:
                # Logar uma parte do stderr para diagnostico
                stderr_preview = res.stderr[:1024].replace('\n', ' ')
                self.logger.warning(f"ffmpeg retornou codigo {res.returncode}: {stderr_preview}")
                return False

            return True

        except FileNotFoundError:
            self.logger.error('ffmpeg nao encontrado no PATH')
            return False
        except Exception as e:
            self.logger.error(f"Erro ao extrair com ffmpeg: {e}")
            return False

    def load_chunks_index(self, index_path: str) -> Dict:
        """
        Carrega index de chunks gerados anteriormente

        Args:
            index_path: Caminho do arquivo chunks_index.json

        Returns:
            Dicionario com metadata dos chunks
        """
        with open(index_path, 'r', encoding='utf-8') as f:
            return json.load(f)


#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/
