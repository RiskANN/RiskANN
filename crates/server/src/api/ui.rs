//! `GET /` and `GET /ui`: the RiskANN Console, a single self-contained HTML
//! page compiled into the binary. It talks to this server through the same
//! public API as any other client (same origin, so no CORS is needed).

use axum::response::Html;

pub const CONSOLE_HTML: &str = include_str!("../../ui/index.html");

pub async fn console() -> Html<&'static str> {
    Html(CONSOLE_HTML)
}
