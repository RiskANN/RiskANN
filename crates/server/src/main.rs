//! RiskANN server binary: configuration, logging, startup loading and
//! graceful shutdown. Routes live in the `server` library (`build_router`).

use engine::DBInfo;
use server::logging::{self, LogFormat};
use server::{AppState, ServerConfig, build_router};
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    let format = match LogFormat::from_env() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    logging::init(format);
    logging::install_panic_hook();
    if let Err(e) = run(format).await {
        tracing::error!(error = %e, "server failed");
        std::process::exit(1);
    }
}

async fn run(format: LogFormat) -> Result<(), Box<dyn std::error::Error>> {
    let config = ServerConfig::from_env()?;
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        host = %config.host,
        port = config.port,
        data_dir = %config.data_loc.display(),
        ingest_dir = ?config.ingest_root.as_ref().map(|p| p.display().to_string()),
        max_body_bytes = config.max_body_bytes,
        slow_query_ms = config.slow_query_ms,
        log_format = format.name(),
        "starting RiskANN server"
    );

    let started = Instant::now();
    let data_loc = config.data_loc.clone();
    let (db, report) = tokio::task::spawn_blocking(move || DBInfo::open(data_loc)).await??;
    let failed: Vec<&str> = report.failed.iter().map(|(n, _)| n.as_str()).collect();
    tracing::info!(
        loaded = ?report.loaded,
        failed = ?failed,
        skipped = ?report.skipped,
        elapsed_ms = started.elapsed().as_millis() as u64,
        "startup loading finished"
    );
    if config.ingest_root.is_none() {
        tracing::warn!(
            "RISKANN_INGEST_DIR is not set: ingestion accepts any absolute file path readable by the server process"
        );
    }

    let address = format!("{}:{}", config.host, config.port);
    let state = Arc::new(AppState::new(db, config, report));
    let app = build_router(Arc::clone(&state));
    let listener = TcpListener::bind(&address).await?;
    tracing::info!(address = %address, "RiskANN server listening");
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal(Arc::clone(&state)))
        .await?;
    tracing::info!("HTTP server stopped; waiting for running ingestion jobs to finish");
    Ok(())
}

/// Resolves on Ctrl-C or SIGTERM and withdraws readiness first.
async fn shutdown_signal(state: Arc<AppState>) {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    state.set_ready(false);
    tracing::info!("shutdown signal received; readiness withdrawn, draining in-flight requests");
}
