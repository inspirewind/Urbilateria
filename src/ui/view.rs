use super::app::App;
use super::commands::clean_text;
use super::report::Kind;
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

const ACCENT: Color = Color::Cyan;
const MUTED: Color = Color::DarkGray;

pub fn draw(frame: &mut Frame, app: &mut App) {
    let area = frame.area();
    if area.width < 36 || area.height < 10 {
        frame.render_widget(
            Paragraph::new("URB\nEnlarge to 36 x 10 or more.\nCtrl+C to quit.")
                .wrap(Wrap { trim: false }),
            area,
        );
        return;
    }
    let completions = app.completions();
    let completion_height = (completions.len() as u16).min(area.height.saturating_sub(8));
    let input_height = (app.editor.lines().len() as u16).clamp(2, 5) + 2;
    let [header, body, suggestions, input, footer] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(2),
        Constraint::Length(completion_height),
        Constraint::Length(input_height.min(area.height.saturating_sub(5 + completion_height))),
        Constraint::Length(1),
    ])
    .areas(area);

    let model = app.model_family.as_deref().unwrap_or("no model inspected");
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" URB ", Style::new().fg(Color::Black).bg(ACCENT).bold()),
            Span::styled("  Urbilateria", Style::new().bold()),
            Span::styled(format!("  /  {model}"), Style::new().fg(MUTED)),
        ])),
        header,
    );

    let mut lines = Vec::new();
    for entry in &app.entries {
        let (marker, color) = match entry.kind {
            Kind::Input => ("›", ACCENT),
            Kind::Info => ("•", ACCENT),
            Kind::Success => ("✓", Color::Green),
            Kind::Warning => ("!", Color::Yellow),
            Kind::Error => ("!", Color::Red),
        };
        lines.push(Line::from(Span::styled(
            format!(" {marker} {}", entry.title),
            Style::new().fg(color).bold(),
        )));
        for detail in &entry.details {
            for (index, text) in detail.text.lines().enumerate() {
                let mut spans = vec![Span::raw("   ")];
                if detail.warning {
                    spans.push(Span::styled("warning: ", Style::new().fg(Color::Yellow)));
                }
                if index == 0 {
                    if let Some(label) = detail.label {
                        spans.push(Span::styled(format!("{label}  "), Style::new().fg(MUTED)));
                    }
                }
                spans.push(Span::styled(
                    text.to_owned(),
                    Style::new().fg(if detail.warning {
                        Color::Yellow
                    } else {
                        Color::Reset
                    }),
                ));
                lines.push(Line::from(spans));
            }
        }
        lines.push(Line::default());
    }
    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            " Type / to get started.",
            Style::new().fg(MUTED),
        )));
    }
    let transcript = Paragraph::new(lines).wrap(Wrap { trim: false });
    app.page_size = usize::from(body.height.saturating_sub(1).max(1));
    app.max_scroll = transcript
        .line_count(body.width)
        .saturating_sub(usize::from(body.height));
    if let Some(row) = &mut app.scroll {
        *row = (*row).min(app.max_scroll);
    }
    let row = app.scroll.unwrap_or(app.max_scroll).min(u16::MAX as usize) as u16;
    frame.render_widget(transcript.scroll((row, 0)), body);

    let selected = app.completion_index % completions.len().max(1);
    let first_choice = selected.saturating_sub(usize::from(completion_height).saturating_sub(1));
    let choices: Vec<_> = completions
        .iter()
        .enumerate()
        .skip(first_choice)
        .take(completion_height.into())
        .map(|(index, (name, description))| {
            let style = if index == selected {
                Style::new().fg(ACCENT).bold()
            } else {
                Style::new().fg(MUTED)
            };
            Line::from(Span::styled(
                format!(
                    " {} {name:<10} {description}",
                    if index == selected { "›" } else { " " }
                ),
                style,
            ))
        })
        .collect();
    frame.render_widget(Paragraph::new(choices), suggestions);

    let title = match &app.pending {
        Some(pending) => format!(" Command · {} running ", pending.task.command()),
        None => " Command ".into(),
    };
    app.editor.set_block(
        Block::default()
            .borders(Borders::ALL)
            .border_type(ratatui::widgets::BorderType::Rounded)
            .border_style(Style::new().fg(ACCENT))
            .title(title),
    );
    app.editor.set_style(Style::new());
    app.editor.set_cursor_line_style(Style::new());
    app.editor
        .set_cursor_style(Style::new().add_modifier(Modifier::REVERSED));
    app.editor.set_placeholder_style(Style::new().fg(MUTED));
    frame.render_widget(&app.editor, input);

    let status = match &app.pending {
        Some(pending) => {
            let elapsed = pending.started.elapsed();
            let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                [(elapsed.as_millis() / 100 % 10) as usize];
            format!(
                " {spinner} {} · {:.1}s",
                pending.task.label(),
                elapsed.as_secs_f64()
            )
        }
        None => " ● Ready".into(),
    };
    let scroll = if app.scroll.is_some() {
        " · scrolled"
    } else {
        ""
    };
    let hints = if area.width >= 90 {
        "Enter run · Ctrl+J newline · Tab complete · PgUp/PgDn scroll · Ctrl+C quit"
    } else if area.width >= 60 {
        "Enter run · /help keys · Ctrl+C quit"
    } else {
        "Ctrl+C quit"
    };
    // All path/model text is sanitized before it reaches terminal cells.
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(
                clean_text(&status, 128),
                Style::new().fg(if app.pending.is_some() {
                    ACCENT
                } else {
                    Color::Green
                }),
            ),
            Span::styled(format!("{scroll}  {hints}"), Style::new().fg(MUTED)),
        ])),
        footer,
    );
}

#[cfg(test)]
mod tests {
    use super::super::report::Entry;
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    #[test]
    fn completion_menu_keeps_the_last_command_visible_in_a_small_terminal() {
        let mut app = App::new(None);
        app.paste("/");
        app.key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Up,
            crossterm::event::KeyModifiers::NONE,
        ));
        let mut terminal = Terminal::new(TestBackend::new(36, 10)).unwrap();
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let display: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(display.contains("/quit"));
        app.key(crossterm::event::KeyEvent::new(
            crossterm::event::KeyCode::Tab,
            crossterm::event::KeyModifiers::NONE,
        ));
        assert_eq!(app.input(), "/quit");
    }

    #[test]
    fn renders_unicode_and_clamps_scrolling_after_resize() {
        let mut app = App::new(None);
        app.push(Entry::message(
            Kind::Info,
            "模型检查",
            "中文🙂 and a long path /".repeat(80),
        ));
        app.paste("/ins");
        let mut terminal = Terminal::new(TestBackend::new(60, 18)).unwrap();
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        assert!(app.max_scroll > 0);
        let display: String = terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect();
        assert!(display.contains("/inspect"));
        assert!(terminal
            .backend()
            .buffer()
            .content
            .iter()
            .any(|cell| cell.symbol() == "中"));
        assert!(terminal
            .backend()
            .buffer()
            .content
            .iter()
            .any(|cell| cell.symbol() == "文"));
        app.scroll = Some(usize::MAX);
        for (width, height) in [(120, 40), (36, 10), (8, 3), (0, 0)] {
            terminal.backend_mut().resize(width, height);
            terminal
                .resize(ratatui::layout::Rect::new(0, 0, width, height))
                .unwrap();
            terminal.draw(|frame| draw(frame, &mut app)).unwrap();
            assert!(app.scroll.unwrap() <= app.max_scroll);
        }
    }
}
