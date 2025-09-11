use thiserror::Error;

#[derive(Error, Debug)]
pub enum PipeError {
    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("API error: {message}")]
    Api { message: String, status: u16 },

    #[error("Authentication failed")]
    AuthError,

    #[error("Insufficient tokens: need {required} PIPE, have {available}")]
    InsufficientTokens { required: f64, available: f64 },

    #[error("Encryption error: {0}")]
    Crypto(String),

    #[error("File not found: {0}")]
    FileNotFound(String),

    #[error("Invalid response: {0}")]
    InvalidResponse(String),

    #[error("Session not found: {0}")]
    SessionNotFound(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, PipeError>;

impl From<serde_json::Error> for PipeError {
    fn from(err: serde_json::Error) -> Self {
        PipeError::InvalidResponse(err.to_string())
    }
}
