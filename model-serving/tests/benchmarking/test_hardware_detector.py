"""
Unit tests for hardware detector.
"""

import pytest
from unittest.mock import Mock, patch

from src.benchmarking.hardware_detector import (
    HardwareDetector, CPUInfo, GPUInfo, MemoryInfo, HardwareInfo
)


class TestCPUInfo:
    """Test CPUInfo dataclass."""
    
    def test_cpu_info_creation(self):
        """Test creating CPUInfo."""
        cpu_info = CPUInfo(
            model="Intel Core i7-9700K",
            cores=8,
            threads=16,
            frequency_mhz=3600.0,
            architecture="x86_64",
            features=["avx", "avx2", "sse4"],
            cache_size_mb=16.0,
            avx_support=True,
            sse_support=True,
            intel_mkl_available=True
        )
        
        assert cpu_info.model == "Intel Core i7-9700K"
        assert cpu_info.cores == 8
        assert cpu_info.threads == 16
        assert cpu_info.frequency_mhz == 3600.0
        assert cpu_info.architecture == "x86_64"
        assert "avx" in cpu_info.features
        assert cpu_info.cache_size_mb == 16.0
        assert cpu_info.avx_support is True
        assert cpu_info.sse_support is True
        assert cpu_info.intel_mkl_available is True


class TestGPUInfo:
    """Test GPUInfo dataclass."""
    
    def test_gpu_info_creation(self):
        """Test creating GPUInfo."""
        gpu_info = GPUInfo(
            model="NVIDIA GeForce RTX 3080",
            memory_gb=10.0,
            compute_capability="8.6",
            cuda_available=True,
            tensorrt_available=True,
            driver_version="11.4",
            cuda_version="11.4"
        )
        
        assert gpu_info.model == "NVIDIA GeForce RTX 3080"
        assert gpu_info.memory_gb == 10.0
        assert gpu_info.compute_capability == "8.6"
        assert gpu_info.cuda_available is True
        assert gpu_info.tensorrt_available is True
        assert gpu_info.driver_version == "11.4"
        assert gpu_info.cuda_version == "11.4"


class TestMemoryInfo:
    """Test MemoryInfo dataclass."""
    
    def test_memory_info_creation(self):
        """Test creating MemoryInfo."""
        memory_info = MemoryInfo(
            total_gb=32.0,
            available_gb=16.0,
            swap_gb=8.0,
            memory_type="DDR4"
        )
        
        assert memory_info.total_gb == 32.0
        assert memory_info.available_gb == 16.0
        assert memory_info.swap_gb == 8.0
        assert memory_info.memory_type == "DDR4"


class TestHardwareInfo:
    """Test HardwareInfo dataclass."""
    
    def test_hardware_info_creation(self):
        """Test creating HardwareInfo."""
        cpu_info = CPUInfo(
            model="Intel Core i7-9700K", cores=8, threads=16,
            frequency_mhz=3600.0, architecture="x86_64", features=["avx"],
            cache_size_mb=16.0, avx_support=True, sse_support=True,
            intel_mkl_available=True
        )
        
        gpu_info = GPUInfo(
            model="NVIDIA GeForce RTX 3080", memory_gb=10.0,
            compute_capability="8.6", cuda_available=True,
            tensorrt_available=True, driver_version="11.4", cuda_version="11.4"
        )
        
        memory_info = MemoryInfo(
            total_gb=32.0, available_gb=16.0, swap_gb=8.0, memory_type="DDR4"
        )
        
        hardware_info = HardwareInfo(
            cpu=cpu_info,
            gpu=gpu_info,
            memory=memory_info,
            platform="Linux-5.4.0-74-generic-x86_64-with-glibc2.29",
            python_version="3.9.7",
            torch_version="1.12.0",
            optimization_recommendations=["Use GPU for inference"]
        )
        
        assert hardware_info.cpu == cpu_info
        assert hardware_info.gpu == gpu_info
        assert hardware_info.memory == memory_info
        assert hardware_info.platform.startswith("Linux")
        assert hardware_info.python_version == "3.9.7"
        assert hardware_info.torch_version == "1.12.0"
        assert len(hardware_info.optimization_recommendations) > 0


class TestHardwareDetector:
    """Test HardwareDetector class."""
    
    def test_detector_initialization(self):
        """Test hardware detector initialization."""
        detector = HardwareDetector()
        
        assert detector._cpu_info is None
        assert detector._gpu_info is None
        assert detector._memory_info is None
    
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_freq')
    @patch('platform.processor')
    def test_detect_cpu(self, mock_processor, mock_cpu_freq, mock_cpu_count):
        """Test CPU detection."""
        # Mock psutil and platform calls
        mock_cpu_count.return_value = 8
        mock_cpu_freq.return_value = Mock(max=3600.0)
        mock_processor.return_value = "Intel Core i7-9700K"
        
        detector = HardwareDetector()
        cpu_info = detector.detect_cpu()
        
        assert isinstance(cpu_info, CPUInfo)
        assert cpu_info.cores == 8
        assert cpu_info.frequency_mhz == 3600.0
        assert cpu_info.model == "Intel Core i7-9700K"
        assert detector._cpu_info == cpu_info
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_name')
    @patch('torch.cuda.get_device_properties')
    @patch('torch.version.cuda')
    def test_detect_gpu_with_cuda(self, mock_cuda_version, mock_device_props, 
                                  mock_device_name, mock_cuda_available):
        """Test GPU detection with CUDA available."""
        mock_cuda_available.return_value = True
        mock_device_name.return_value = "NVIDIA GeForce RTX 3080"
        mock_device_props.return_value = Mock(total_memory=10 * 1024**3)  # 10GB
        mock_cuda_version.return_value = "11.4"
        
        with patch('torch.cuda.get_device_capability', return_value=(8, 6)):
            detector = HardwareDetector()
            gpu_info = detector.detect_gpu()
            
            assert isinstance(gpu_info, GPUInfo)
            assert gpu_info.model == "NVIDIA GeForce RTX 3080"
            assert gpu_info.memory_gb == 10.0
            assert gpu_info.compute_capability == "8.6"
            assert gpu_info.cuda_available is True
            assert detector._gpu_info == gpu_info
    
    @patch('torch.cuda.is_available')
    def test_detect_gpu_without_cuda(self, mock_cuda_available):
        """Test GPU detection without CUDA."""
        mock_cuda_available.return_value = False
        
        detector = HardwareDetector()
        gpu_info = detector.detect_gpu()
        
        assert gpu_info is None
        assert detector._gpu_info is None
    
    @patch('psutil.virtual_memory')
    @patch('psutil.swap_memory')
    def test_detect_memory(self, mock_swap, mock_virtual):
        """Test memory detection."""
        mock_virtual.return_value = Mock(
            total=32 * 1024**3,  # 32GB
            available=16 * 1024**3  # 16GB
        )
        mock_swap.return_value = Mock(total=8 * 1024**3)  # 8GB
        
        detector = HardwareDetector()
        memory_info = detector.detect_memory()
        
        assert isinstance(memory_info, MemoryInfo)
        assert memory_info.total_gb == 32.0
        assert memory_info.available_gb == 16.0
        assert memory_info.swap_gb == 8.0
        assert detector._memory_info == memory_info
    
    @patch('psutil.cpu_count')
    @patch('psutil.cpu_freq')
    @patch('platform.processor')
    @patch('psutil.virtual_memory')
    @patch('psutil.swap_memory')
    @patch('torch.cuda.is_available')
    def test_get_hardware_info(self, mock_cuda_available, mock_swap, mock_virtual,
                               mock_processor, mock_cpu_freq, mock_cpu_count):
        """Test complete hardware information gathering."""
        # Mock all the dependencies
        mock_cpu_count.return_value = 8
        mock_cpu_freq.return_value = Mock(max=3600.0)
        mock_processor.return_value = "Intel Core i7-9700K"
        mock_virtual.return_value = Mock(total=32 * 1024**3, available=16 * 1024**3)
        mock_swap.return_value = Mock(total=8 * 1024**3)
        mock_cuda_available.return_value = False
        
        detector = HardwareDetector()
        hardware_info = detector.get_hardware_info()
        
        assert isinstance(hardware_info, HardwareInfo)
        assert isinstance(hardware_info.cpu, CPUInfo)
        assert hardware_info.gpu is None  # No CUDA
        assert isinstance(hardware_info.memory, MemoryInfo)
        assert hardware_info.platform is not None
        assert hardware_info.python_version is not None
        assert hardware_info.torch_version is not None
        assert isinstance(hardware_info.optimization_recommendations, list)
    
    def test_get_optimal_batch_size(self):
        """Test optimal batch size recommendation."""
        detector = HardwareDetector()
        
        # Test with GPU
        cpu_info = CPUInfo(
            model="Intel Core i7-9700K", cores=8, threads=16,
            frequency_mhz=3600.0, architecture="x86_64", features=["avx"],
            cache_size_mb=16.0, avx_support=True, sse_support=True,
            intel_mkl_available=True
        )
        
        gpu_info = GPUInfo(
            model="NVIDIA GeForce RTX 3080", memory_gb=10.0,
            compute_capability="8.6", cuda_available=True,
            tensorrt_available=True, driver_version="11.4", cuda_version="11.4"
        )
        
        memory_info = MemoryInfo(
            total_gb=32.0, available_gb=16.0, swap_gb=8.0, memory_type="DDR4"
        )
        
        hardware_info = HardwareInfo(
            cpu=cpu_info, gpu=gpu_info, memory=memory_info,
            platform="Linux", python_version="3.9.7", torch_version="1.12.0",
            optimization_recommendations=[]
        )
        
        batch_size = detector.get_optimal_batch_size(hardware_info)
        assert batch_size == 32  # GPU with 10GB memory
    
    def test_get_optimal_thread_count(self):
        """Test optimal thread count recommendation."""
        detector = HardwareDetector()
        
        cpu_info = CPUInfo(
            model="Intel Core i7-9700K", cores=8, threads=16,
            frequency_mhz=3600.0, architecture="x86_64", features=["avx"],
            cache_size_mb=16.0, avx_support=True, sse_support=True,
            intel_mkl_available=True
        )
        
        memory_info = MemoryInfo(
            total_gb=32.0, available_gb=16.0, swap_gb=8.0, memory_type="DDR4"
        )
        
        hardware_info = HardwareInfo(
            cpu=cpu_info, gpu=None, memory=memory_info,
            platform="Linux", python_version="3.9.7", torch_version="1.12.0",
            optimization_recommendations=[]
        )
        
        thread_count = detector.get_optimal_thread_count(hardware_info)
        assert thread_count == 8  # Capped at 8 for inference
    
    def test_should_use_gpu(self):
        """Test GPU usage recommendation."""
        detector = HardwareDetector()
        
        # Test with good GPU
        gpu_info = GPUInfo(
            model="NVIDIA GeForce RTX 3080", memory_gb=10.0,
            compute_capability="8.6", cuda_available=True,
            tensorrt_available=True, driver_version="11.4", cuda_version="11.4"
        )
        
        memory_info = MemoryInfo(
            total_gb=32.0, available_gb=16.0, swap_gb=8.0, memory_type="DDR4"
        )
        
        hardware_info = HardwareInfo(
            cpu=None, gpu=gpu_info, memory=memory_info,
            platform="Linux", python_version="3.9.7", torch_version="1.12.0",
            optimization_recommendations=[]
        )
        
        should_use_gpu = detector.should_use_gpu(hardware_info)
        assert should_use_gpu is True
        
        # Test without GPU
        hardware_info_no_gpu = HardwareInfo(
            cpu=None, gpu=None, memory=memory_info,
            platform="Linux", python_version="3.9.7", torch_version="1.12.0",
            optimization_recommendations=[]
        )
        
        should_use_gpu = detector.should_use_gpu(hardware_info_no_gpu)
        assert should_use_gpu is False
