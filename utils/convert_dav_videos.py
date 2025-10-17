r"""
DAVConverter: Converte videos .dav para MP4

Integrado com StateManager para evitar reconversao
Trabalha com pasta videos_full/ automaticamente

   __  ____ ____ _  _
 / _\/ ___) ___) )( \
/    \___ \___ ) \/ (
\_/\_(____(____|____/
"""

import os
import subprocess
import sys
import logging
import re
import threading
from pathlib import Path
from typing import Optional, Tuple
import yaml


class DAVConverter:
    """
    Conversor de videos .dav para MP4
    Integrado com StateManager
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Inicializa conversor com configuracoes do YAML
        
        Args:
            config_path: Caminho do arquivo de configuracao
        """
        self.logger = logging.getLogger(__name__)
        self.config = self._load_config(config_path)
        
        # Diretorios
        self.videos_dir = Path(self.config['directories']['videos_full'])
        self.output_dir = Path(self.config['directories']['videos_converted'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Verificar se ffmpeg esta disponivel
        self.ffmpeg_available = self._check_ffmpeg()
        
        # Detectar encoder GPU disponivel
        self.gpu_encoder = self._detect_gpu_encoder() if self.ffmpeg_available else 'libx264'
        
    def _load_config(self, config_path: str) -> dict:
        """Carrega configuracao do YAML"""
        try:
            config_file = Path(__file__).parent.parent / config_path
            with open(config_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            self.logger.warning(f"Erro ao carregar config: {e}, usando padroes")
            return {
                'directories': {
                    'videos_full': 'videos_full',
                    'videos_converted': 'videos_converted'
                },
                'conversion': {
                    'use_ffmpeg': True,
                    'fallback_opencv': True,
                    'codec': 'libx264',
                    'preset': 'fast',
                    'crf': 23,
                    'audio_codec': 'aac',
                    'audio_bitrate': '128k'
                }
            }

    def _check_ffmpeg(self) -> bool:
        """Verifica se ffmpeg esta instalado"""
        try:
            result = subprocess.run(['ffmpeg', '-version'],
                                  capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return False

    def _detect_gpu_encoder(self) -> str:
        """
        Detecta encoder de GPU disponivel
        
        Returns:
            Nome do encoder (h264_nvenc, h264_amf, h264_qsv) ou 'libx264' se nenhum
        """
        if not self.ffmpeg_available:
            return 'libx264'
        
        try:
            # Listar encoders disponiveis
            result = subprocess.run(['ffmpeg', '-encoders'],
                                  capture_output=True, text=True, timeout=10)
            
            encoders_output = result.stdout
            
            # Prioridade: NVIDIA > AMD > Intel > CPU
            if 'h264_nvenc' in encoders_output:
                self.logger.info("GPU NVIDIA detectada - usando h264_nvenc")
                return 'h264_nvenc'
            elif 'h264_amf' in encoders_output:
                self.logger.info("GPU AMD detectada - usando h264_amf")
                return 'h264_amf'
            elif 'h264_qsv' in encoders_output:
                self.logger.info("Intel Quick Sync detectado - usando h264_qsv")
                return 'h264_qsv'
            else:
                self.logger.info("Nenhuma GPU detectada - usando CPU (libx264)")
                return 'libx264'
                
        except Exception as e:
            self.logger.warning(f"Erro ao detectar GPU: {e}, usando CPU")
            return 'libx264'

    def _convert_with_ffmpeg(self, input_path: Path, output_path: Path, use_cpu_fallback: bool = False) -> Tuple[bool, Optional[str]]:
        """
        Converte video usando ffmpeg com aceleracao GPU
        
        Args:
            input_path: Caminho do video .dav
            output_path: Caminho de saida .mp4
            use_cpu_fallback: Se True, ignora GPU e usa CPU
            
        Returns:
            (sucesso, mensagem_erro)
        """
        try:
            conv_cfg = self.config['conversion']
            
            # Usar encoder GPU ou CPU (forcar CPU se fallback ativado)
            encoder = 'libx264' if use_cpu_fallback else self.gpu_encoder
            
            # Construir comando base
            cmd = ['ffmpeg', '-i', str(input_path)]
            
            # Adicionar parametros especificos do encoder
            if encoder == 'h264_nvenc' and not use_cpu_fallback:
                # NVIDIA GPU
                cmd.extend([
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p4',  # p1 (rapido) a p7 (lento), p4 e balanceado
                    '-cq', str(conv_cfg.get('crf', 23)),  # Qualidade constante (CQ mode)
                    '-b:v', '0',  # Bitrate 0 para modo CQ
                ])
            elif encoder == 'h264_amf':
                # AMD GPU
                cmd.extend([
                    '-c:v', 'h264_amf',
                    '-quality', 'balanced',  # speed, balanced, quality
                    '-rc', 'cqp',  # Constant QP mode
                    '-qp_i', str(conv_cfg.get('crf', 23)),
                    '-qp_p', str(conv_cfg.get('crf', 23)),
                ])
            elif encoder == 'h264_qsv':
                # Intel Quick Sync
                cmd.extend([
                    '-c:v', 'h264_qsv',
                    '-preset', 'medium',
                    '-global_quality', str(conv_cfg.get('crf', 23)),
                ])
            else:
                # CPU fallback (libx264)
                cmd.extend([
                    '-c:v', 'libx264',
                    '-preset', conv_cfg.get('preset', 'fast'),
                    '-crf', str(conv_cfg.get('crf', 23)),
                ])
            
            # Audio e parametros finais (comum para todos)
            cmd.extend([
                '-c:a', conv_cfg.get('audio_codec', 'aac'),
                '-b:a', conv_cfg.get('audio_bitrate', '128k'),
                '-movflags', '+faststart',
                '-progress', 'pipe:1',  # Progresso para stdout
                '-y',
                str(output_path)
            ])

            self.logger.debug(f"Comando FFmpeg: {' '.join(cmd)}")
            self.logger.info(f"  Usando encoder: {encoder}")
            
            # Processar com progresso
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )

            # Obter duracao total do video do stderr
            duration_seconds = None
            stderr_lines = []
            
            def read_stderr():
                nonlocal duration_seconds, stderr_lines
                if process.stderr:
                    for line in process.stderr:
                        stderr_lines.append(line)
                        if 'Duration:' in line:
                            # Extrair duracao: Duration: 00:10:30.45
                            match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})', line)
                            if match:
                                h, m, s = match.groups()
                                duration_seconds = int(h) * 3600 + int(m) * 60 + float(s)
                                print(f"  Duracao total: {int(h):02d}:{int(m):02d}:{int(float(s)):02d}")
            
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()

            # Ler stdout para progresso
            last_progress = -1
            if process.stdout:
                for line in process.stdout:
                    if 'out_time_ms=' in line:
                        # Extrair tempo processado em microsegundos
                        match = re.search(r'out_time_ms=(\d+)', line)
                        if match and duration_seconds:
                            time_ms = int(match.group(1))
                            time_seconds = time_ms / 1000000
                            progress = min(100, int((time_seconds / duration_seconds) * 100))
                            
                            # Atualizar a cada 5% ou no final
                            if progress >= last_progress + 5 or progress == 100:
                                bar_length = 40
                                filled = int(bar_length * progress / 100)
                                bar = '█' * filled + '░' * (bar_length - filled)
                                print(f"\r  Progresso: [{bar}] {progress}%", end='', flush=True)
                                last_progress = progress

            # Aguardar conclusao
            process.wait(timeout=600)
            print()  # Nova linha apos progresso

            if process.returncode == 0:
                return True, None
            else:
                error_msg = ''.join(stderr_lines[-20:]) if stderr_lines else "Erro desconhecido"
                return False, f"FFmpeg erro: {error_msg}"

        except subprocess.TimeoutExpired:
            return False, "Conversao timeout (10 minutos)"
        except Exception as e:
            return False, f"Erro inesperado: {str(e)}"

    def _convert_with_opencv(self, input_path: Path, output_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Converte video usando OpenCV (fallback)
        
        Args:
            input_path: Caminho do video .dav
            output_path: Caminho de saida .mp4
            
        Returns:
            (sucesso, mensagem_erro)
        """
        try:
            import cv2

            cap = cv2.VideoCapture(str(input_path))
            if not cap.isOpened():
                return False, "Nao foi possivel abrir o video .dav"

            # Propriedades do video
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if fps == 0:
                fps = 30  # Default FPS

            duration_seconds = total_frames / fps
            minutes = int(duration_seconds // 60)
            seconds = int(duration_seconds % 60)
            
            print(f"  Video: {total_frames} frames @ {fps}fps ({width}x{height})")
            print(f"  Duracao: {minutes:02d}:{seconds:02d}")

            # Criar writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

            if not out.isOpened():
                return False, "Nao foi possivel criar VideoWriter"

            frame_count = 0
            last_progress = -1

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                out.write(frame)
                frame_count += 1

                # Atualizar barra de progresso
                if total_frames > 0:
                    progress = int((frame_count / total_frames) * 100)
                    
                    if progress >= last_progress + 5 or progress == 100:
                        bar_length = 40
                        filled = int(bar_length * progress / 100)
                        bar = '█' * filled + '░' * (bar_length - filled)
                        print(f"\r  Progresso: [{bar}] {progress}% ({frame_count}/{total_frames})", 
                              end='', flush=True)
                        last_progress = progress

            print()  # Nova linha apos progresso
            cap.release()
            out.release()

            if frame_count > 0:
                self.logger.info(f"  Convertido: {frame_count} frames")
                return True, None
            else:
                return False, "Nenhum frame foi processado"

        except Exception as e:
            return False, f"Erro OpenCV: {str(e)}"

    def convert_video(self, input_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Converte um video .dav para MP4
        
        Args:
            input_path: Caminho do video .dav
            
        Returns:
            (sucesso, caminho_output | mensagem_erro)
        """
        output_filename = f"{input_path.stem}_converted.mp4"
        output_path = self.output_dir / output_filename

        # Verificar se ja existe
        if output_path.exists():
            self.logger.info(f"Video ja convertido: {output_filename}")
            return True, str(output_path)

        self.logger.info(f"Convertendo: {input_path.name}")

        # Tentar ffmpeg primeiro
        conv_cfg = self.config['conversion']
        
        if self.ffmpeg_available and conv_cfg.get('use_ffmpeg', True):
            self.logger.info("  Usando FFmpeg...")
            success, error = self._convert_with_ffmpeg(input_path, output_path)
            
            # Se falhou com GPU, tentar novamente com CPU
            if not success and self.gpu_encoder != 'libx264':
                if 'nvcuda.dll' in str(error) or 'nvenc' in str(error).lower() or \
                   'amf' in str(error).lower() or 'qsv' in str(error).lower():
                    self.logger.warning("  GPU encoder falhou, tentando com CPU...")
                    success, error = self._convert_with_ffmpeg(input_path, output_path, use_cpu_fallback=True)
                    
        elif conv_cfg.get('fallback_opencv', True):
            self.logger.info("  FFmpeg nao disponivel, usando OpenCV...")
            success, error = self._convert_with_opencv(input_path, output_path)
        else:
            return False, "Nem FFmpeg nem OpenCV estao disponiveis"

        if success:
            self.logger.info(f"Convertido com sucesso: {output_filename}")
            return True, str(output_path)
        else:
            self.logger.error(f"Falha na conversao: {error}")
            # Limpar arquivo parcial se existir
            if output_path.exists():
                output_path.unlink()
            return False, error

    def find_dav_files(self) -> list:
        """
        Encontra todos os arquivos .dav na pasta videos_full
        
        Returns:
            Lista de paths dos arquivos .dav
        """
        if not self.videos_dir.exists():
            self.logger.warning(f"Pasta {self.videos_dir} nao encontrada")
            return []

        dav_files = list(self.videos_dir.glob("**/*.dav"))
        dav_files.extend(list(self.videos_dir.glob("**/*.DAV")))  # Case insensitive

        return sorted(set(dav_files))  # Remove duplicatas



#    __  ____ ____ _  _
#  / _\/ ___) ___) )( \
# /    \___ \___ ) \/ (
# \_/\_(____(____|____/