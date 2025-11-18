"""
Realistic Sensor Simulation
Add real-world imperfections to Isaac Sim sensors to make them indistinguishable from real hardware
"""

import numpy as np
import cv2
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RealisticCamera:
    """
    Camera sensor with real-world imperfections
    Makes Isaac Sim's perfect renders look like real camera data
    """

    def __init__(
        self,
        noise_std: float = 0.01,
        motion_blur_enabled: bool = True,
        compression_artifacts: bool = True,
        jpeg_quality: int = 85,
        auto_exposure: bool = True
    ):
        self.noise_std = noise_std
        self.motion_blur_enabled = motion_blur_enabled
        self.compression_artifacts = compression_artifacts
        self.jpeg_quality = jpeg_quality
        self.auto_exposure = auto_exposure

        self.robot_velocity = 0.0  # Updated externally

    def capture(self, isaac_rgb: np.ndarray, robot_velocity: float = 0.0) -> np.ndarray:
        """
        Add realistic imperfections to perfect Isaac Sim render

        Args:
            isaac_rgb: Perfect RGB image from Isaac Sim (H, W, 3) float32 [0, 1]
            robot_velocity: Robot end-effector velocity for motion blur

        Returns:
            Realistic RGB image (H, W, 3) uint8 [0, 255]
        """
        self.robot_velocity = robot_velocity

        # Work in float32 [0, 1] space
        rgb = isaac_rgb.copy()

        # 1. Add Gaussian noise (sensor noise)
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, rgb.shape)
            rgb = np.clip(rgb + noise, 0, 1)

        # 2. Add motion blur if robot is moving
        if self.motion_blur_enabled and robot_velocity > 0.1:
            rgb = self._add_motion_blur(rgb, velocity=robot_velocity)

        # 3. Simulate auto-exposure variations
        if self.auto_exposure:
            rgb = self._apply_auto_exposure(rgb)

        # Convert to uint8
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        # 4. Add compression artifacts (JPEG)
        if self.compression_artifacts:
            rgb_uint8 = self._add_compression_artifacts(rgb_uint8)

        return rgb_uint8

    def _add_motion_blur(self, image: np.ndarray, velocity: float) -> np.ndarray:
        """
        Add motion blur based on robot velocity

        Args:
            image: Input image [0, 1]
            velocity: Robot velocity (m/s)

        Returns:
            Blurred image
        """
        # Blur amount proportional to velocity
        kernel_size = int(np.clip(velocity * 10, 1, 15))
        if kernel_size % 2 == 0:
            kernel_size += 1  # Must be odd

        # Horizontal blur (assuming horizontal motion)
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1.0 / kernel_size

        blurred = cv2.filter2D(image, -1, kernel)
        return blurred

    def _apply_auto_exposure(self, image: np.ndarray) -> np.ndarray:
        """
        Simulate auto-exposure brightness adjustment

        Args:
            image: Input image [0, 1]

        Returns:
            Exposure-adjusted image
        """
        # Random exposure variation ±10%
        exposure_factor = 1.0 + np.random.uniform(-0.1, 0.1)
        adjusted = image * exposure_factor
        return np.clip(adjusted, 0, 1)

    def _add_compression_artifacts(self, image: np.ndarray) -> np.ndarray:
        """
        Add JPEG compression artifacts

        Args:
            image: Input image uint8

        Returns:
            Compressed image
        """
        # Encode to JPEG and decode
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return decoded


class RealisticProprioception:
    """
    Joint sensor with real-world characteristics
    Add noise and latency to perfect Isaac Sim measurements
    """

    def __init__(
        self,
        position_noise_std: float = 0.001,  # radians
        velocity_noise_std: float = 0.01,   # rad/s
        latency_ms: float = 1.0,            # milliseconds
        quantization_bits: int = 16         # ADC resolution
    ):
        self.position_noise_std = position_noise_std
        self.velocity_noise_std = velocity_noise_std
        self.latency_ms = latency_ms
        self.quantization_bits = quantization_bits

        # Circular buffer for latency simulation
        self.history = []
        self.max_history = int(np.ceil(latency_ms / 0.1))  # Assuming 10Hz sensing

    def sense(
        self,
        true_position: np.ndarray,
        true_velocity: np.ndarray
    ) -> tuple:
        """
        Add realistic imperfections to joint measurements

        Args:
            true_position: Perfect joint positions (rad)
            true_velocity: Perfect joint velocities (rad/s)

        Returns:
            Tuple of (measured_position, measured_velocity)
        """
        # Add noise
        measured_pos = true_position + np.random.normal(
            0, self.position_noise_std, true_position.shape
        )
        measured_vel = true_velocity + np.random.normal(
            0, self.velocity_noise_std, true_velocity.shape
        )

        # Quantize (simulate ADC)
        measured_pos = self._quantize(measured_pos, bits=self.quantization_bits)
        measured_vel = self._quantize(measured_vel, bits=self.quantization_bits)

        # Add latency (return delayed measurement)
        self.history.append((measured_pos, measured_vel))
        if len(self.history) > self.max_history:
            return self.history.pop(0)  # Return delayed measurement
        else:
            return measured_pos, measured_vel  # Not enough history yet

    def _quantize(self, values: np.ndarray, bits: int) -> np.ndarray:
        """
        Quantize values to simulate ADC resolution

        Args:
            values: Continuous values
            bits: Number of bits

        Returns:
            Quantized values
        """
        # Assume range [-pi, pi] for joints
        levels = 2 ** bits
        quantized = np.round(values * levels / (2 * np.pi)) * (2 * np.pi) / levels
        return quantized


class SensorSimulator:
    """
    Combined sensor simulator for complete realism
    """

    def __init__(
        self,
        camera_noise: float = 0.01,
        proprio_noise: float = 0.001,
        motion_blur: bool = True,
        jpeg_quality: int = 85
    ):
        self.camera = RealisticCamera(
            noise_std=camera_noise,
            motion_blur_enabled=motion_blur,
            jpeg_quality=jpeg_quality
        )

        self.proprioception = RealisticProprioception(
            position_noise_std=proprio_noise
        )

        logger.info("Sensor simulator initialized with realistic imperfections")

    def process_observation(
        self,
        isaac_rgb: np.ndarray,
        isaac_joint_pos: np.ndarray,
        isaac_joint_vel: np.ndarray,
        robot_velocity: float = 0.0
    ) -> dict:
        """
        Process perfect Isaac Sim observation to realistic sensor data

        Args:
            isaac_rgb: Perfect RGB from Isaac Sim
            isaac_joint_pos: Perfect joint positions
            isaac_joint_vel: Perfect joint velocities
            robot_velocity: End-effector velocity

        Returns:
            Dictionary with realistic sensor data
        """
        # Process camera
        realistic_rgb = self.camera.capture(isaac_rgb, robot_velocity)

        # Process proprioception
        measured_pos, measured_vel = self.proprioception.sense(
            isaac_joint_pos, isaac_joint_vel
        )

        return {
            "rgb": realistic_rgb,
            "joint_positions": measured_pos,
            "joint_velocities": measured_vel
        }
