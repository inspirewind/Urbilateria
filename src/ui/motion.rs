use ratatui::style::{Color, Style};
use ratatui::text::Span;
use std::time::Duration;

pub fn spinner(elapsed: Duration) -> &'static str {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    FRAMES[(elapsed.as_millis() / 100 % FRAMES.len() as u128) as usize]
}

pub fn generating(elapsed: Duration) -> Vec<Span<'static>> {
    const LABEL: &str = "Generating";
    const BASE: [u8; 3] = [139, 198, 191];
    const HIGHLIGHT: [u8; 3] = [225, 255, 250];
    const HALF_WIDTH: f64 = 3.0;

    // Start after 600 ms, sweep for one second, then rest until the next four-second cycle.
    let phase = elapsed
        .saturating_sub(Duration::from_millis(600))
        .as_secs_f64()
        % 4.0;
    let position = phase.min(1.0) * (LABEL.len() as f64 + 2.0 * HALF_WIDTH) - HALF_WIDTH;
    // This label is fixed ASCII, so each character occupies exactly one terminal column.
    LABEL
        .chars()
        .enumerate()
        .map(|(index, ch)| {
            let distance = ((index as f64 + 0.5 - position).abs() / HALF_WIDTH).min(1.0);
            let intensity = 0.5 * (1.0 + (std::f64::consts::PI * distance).cos());
            let [r, g, b] = std::array::from_fn(|channel| {
                (f64::from(BASE[channel])
                    + f64::from(HIGHLIGHT[channel] - BASE[channel]) * intensity)
                    .round() as u8
            });
            Span::styled(ch.to_string(), Style::new().fg(Color::Rgb(r, g, b)).bold())
        })
        .collect()
}
