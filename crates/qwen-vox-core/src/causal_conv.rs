//! Causal convolution with fixed-size ring buffer state management.
//!
//! No runtime dynamic reallocation — the ring buffer is pre-allocated
//! at initialization and reused across all inference steps.

use crate::error::{VoxError, VoxResult};
use candle_core::Tensor;

/// Fixed-size ring buffer for causal convolution hidden state.
///
/// Capacity is set at construction time and never changes.
pub struct RingBuffer {
    /// Internal storage: `[capacity, channels]`.
    data: Vec<f32>,
    /// Number of channels per frame.
    channels: usize,
    /// Maximum number of frames.
    capacity: usize,
    /// Current write position (wraps around).
    write_pos: usize,
    /// Number of valid frames currently stored.
    len: usize,
}

impl RingBuffer {
    /// Create a new ring buffer with fixed capacity.
    pub fn new(capacity: usize, channels: usize) -> Self {
        Self {
            data: vec![0.0; capacity * channels],
            channels,
            capacity,
            write_pos: 0,
            len: 0,
        }
    }

    /// Push a single frame into the buffer.
    ///
    /// Returns error if buffer is full (should not happen with proper
    /// causal conv sizing, but enforced for safety).
    pub fn push(&mut self, frame: &[f32]) -> VoxResult<()> {
        if frame.len() != self.channels {
            return Err(VoxError::ShapeMismatch {
                expected: vec![self.channels],
                actual: vec![frame.len()],
            });
        }
        if self.len >= self.capacity {
            return Err(VoxError::RingBufferOverflow {
                capacity: self.capacity,
            });
        }

        let offset = self.write_pos * self.channels;
        self.data[offset..offset + self.channels].copy_from_slice(frame);
        self.write_pos = (self.write_pos + 1) % self.capacity;
        self.len += 1;
        Ok(())
    }

    /// Read the last `n` frames in chronological order.
    ///
    /// Returns a flat `Vec<f32>` of length `n * channels`.
    pub fn read_last(&self, n: usize) -> VoxResult<Vec<f32>> {
        if n > self.len {
            return Err(VoxError::Other(format!(
                "requested {n} frames but only {} available",
                self.len
            )));
        }

        let mut result = Vec::with_capacity(n * self.channels);
        for i in 0..n {
            let idx = if self.len < self.capacity {
                // Buffer hasn't wrapped yet
                self.len - n + i
            } else {
                // Buffer has wrapped
                (self.write_pos + self.capacity - n + i) % self.capacity
            };
            let offset = idx * self.channels;
            result.extend_from_slice(&self.data[offset..offset + self.channels]);
        }
        Ok(result)
    }

    /// Reset the buffer to empty state (zero-fill, reset position).
    pub fn reset(&mut self) {
        self.data.fill(0.0);
        self.write_pos = 0;
        self.len = 0;
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Causal 1-D convolution layer with streaming state.
///
/// Maintains a ring buffer of past input frames to compute
/// causal (no future look-ahead) convolution.
pub struct CausalConv1d {
    /// Convolution weight tensor: `[out_channels, in_channels, kernel_size]`.
    weight: Tensor,
    /// Optional bias: `[out_channels]`.
    _bias: Option<Tensor>,
    /// Ring buffer for input history.
    state: RingBuffer,
    /// Kernel size.
    kernel_size: usize,
    /// Input channels.
    in_channels: usize,
    /// Output channels.
    out_channels: usize,
}

impl CausalConv1d {
    /// Create from preloaded weights.
    pub fn from_weights(
        weight: Tensor,
        bias: Option<Tensor>,
        ring_buffer_capacity: usize,
    ) -> VoxResult<Self> {
        let dims = weight
            .dims3()
            .map_err(|e| VoxError::WeightLoad(format!("conv weight must be 3-D: {e}")))?;
        let out_channels = dims.0;
        let in_channels = dims.1;
        let kernel_size = dims.2;

        Ok(Self {
            weight,
            _bias: bias,
            state: RingBuffer::new(ring_buffer_capacity, in_channels),
            kernel_size,
            in_channels,
            out_channels,
        })
    }

    /// Process a single input frame through the causal convolution.
    ///
    /// Pushes the frame into the ring buffer, then computes the
    /// convolution over the last `kernel_size` frames.
    pub fn forward(&mut self, input: &[f32]) -> VoxResult<Vec<f32>> {
        self.state.push(input)?;

        // Need at least kernel_size frames for a valid output
        if self.state.len() < self.kernel_size {
            // Pad with zeros for initial frames
            return Ok(vec![0.0; self.out_channels]);
        }

        let context = self.state.read_last(self.kernel_size)?;
        let device = self.weight.device().clone();

        // Reshape context to [1, in_channels, kernel_size] for conv1d
        let context_tensor =
            Tensor::from_slice(&context, (1, self.in_channels, self.kernel_size), &device)?;

        // Apply convolution (simplified — actual impl would use candle conv1d)
        let output = context_tensor
            .conv1d(&self.weight, 1, 0, 1, 1)?
            .squeeze(0)?;

        let output_vec: Vec<f32> = output.flatten_all()?.to_vec1()?;
        Ok(output_vec)
    }

    /// Reset the convolution state.
    pub fn reset(&mut self) {
        self.state.reset();
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }
}
