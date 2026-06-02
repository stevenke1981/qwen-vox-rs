//! Async streaming interface for real-time TTS output.
//!
//! Uses tokio channels to pipe audio chunks from the decoder
//! to the consumer (file writer, audio device, network stream).

use crate::error::VoxResult;
use tokio::sync::mpsc;

/// A single audio chunk produced by the streaming decoder.
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// PCM samples (mono, f32, normalized).
    pub samples: Vec<f32>,
    /// Sample rate of the audio.
    pub sample_rate: u32,
    /// Sequence number (monotonically increasing).
    pub seq: u64,
    /// Whether this is the final chunk in the stream.
    pub is_final: bool,
}

/// Producer side — sends audio chunks from the decoder.
pub struct AudioSender {
    inner: mpsc::Sender<AudioChunk>,
    seq: u64,
}

impl AudioSender {
    /// Send an audio chunk.
    pub async fn send(&mut self, samples: Vec<f32>, sample_rate: u32) -> VoxResult<()> {
        let chunk = AudioChunk {
            samples,
            sample_rate,
            seq: self.seq,
            is_final: false,
        };
        self.seq += 1;
        self.inner
            .send(chunk)
            .await
            .map_err(|e| crate::error::VoxError::Stream(format!("send failed: {e}")))?;
        Ok(())
    }

    /// Send the final chunk and close the stream.
    pub async fn finish(&mut self, sample_rate: u32) -> VoxResult<()> {
        let chunk = AudioChunk {
            samples: Vec::new(),
            sample_rate,
            seq: self.seq,
            is_final: true,
        };
        self.inner
            .send(chunk)
            .await
            .map_err(|e| crate::error::VoxError::Stream(format!("finish failed: {e}")))?;
        Ok(())
    }
}

/// Consumer side — receives audio chunks.
pub struct AudioReceiver {
    inner: mpsc::Receiver<AudioChunk>,
}

impl AudioReceiver {
    /// Receive the next audio chunk.
    ///
    /// Returns `None` when the stream is closed.
    pub async fn recv(&mut self) -> Option<AudioChunk> {
        self.inner.recv().await
    }
}

/// Create a bounded audio channel with the given buffer size.
pub fn audio_channel(buffer_size: usize) -> (AudioSender, AudioReceiver) {
    let (tx, rx) = mpsc::channel(buffer_size);
    (
        AudioSender { inner: tx, seq: 0 },
        AudioReceiver { inner: rx },
    )
}
