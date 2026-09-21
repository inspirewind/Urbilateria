use super::app::App;
use super::commands::clean_text;
use super::report::Kind;
use super::worker::Task;
use ratatui::layout::{Alignment, Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Clear, Paragraph, Wrap};
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
    let show_metrics = app.runtime.is_some()
        && app
            .pending
            .as_ref()
            .is_none_or(|pending| matches!(pending.task, Task::Generate(_)));
    let footer_height = if show_metrics && area.width < 110 {
        2
    } else {
        1
    };
    let completions = app.completions();
    let completion_height =
        (completions.len() as u16).min(area.height.saturating_sub(7 + footer_height));
    let input_height = (app.editor.lines().len() as u16).clamp(2, 5) + 2;
    let [header, body, suggestions, input, footer] = Layout::vertical([
        Constraint::Length(2),
        Constraint::Min(2),
        Constraint::Length(completion_height),
        Constraint::Length(
            input_height.min(
                area.height
                    .saturating_sub(4 + footer_height + completion_height),
            ),
        ),
        Constraint::Length(footer_height),
    ])
    .areas(area);

    let sidebar = (app.model_summary.is_some() || app.runtime.is_some())
        && area.width >= 110
        && body.height >= 12;
    let model = app.model_family.as_deref().unwrap_or("no model inspected");
    frame.render_widget(
        Paragraph::new(Line::from(vec![
            Span::styled(" URB ", Style::new().fg(Color::Black).bg(ACCENT).bold()),
            Span::styled("  Urbilateria", Style::new().bold()),
            Span::styled(format!("  /  {model}"), Style::new().fg(MUTED)),
        ])),
        header,
    );
    let body = if sidebar {
        let [transcript, card] = Layout::horizontal([
            Constraint::Min(60),
            Constraint::Length((area.width / 3).clamp(38, 48)),
        ])
        .spacing(1)
        .areas(body);
        if app.runtime.is_some() {
            if app.model_summary.is_some() {
                let [model, runtime] = Layout::vertical([
                    Constraint::Length(model_height(app, card.width).min(card.height / 2)),
                    Constraint::Min(6),
                ])
                .spacing(1)
                .areas(card);
                draw_model(frame, app, model);
                draw_runtime(frame, app, runtime, false);
            } else {
                draw_runtime(frame, app, card, false);
            }
        } else {
            draw_model(frame, app, card);
        }
        transcript
    } else {
        if let Some(model) = &app.model_summary {
            // Preserve the model identity without squeezing the conversation on narrow screens.
            frame.render_widget(
                Paragraph::new(format!("{} · {}", model.name, model.family))
                    .alignment(Alignment::Right)
                    .style(Style::new().fg(ACCENT)),
                Rect::new(header.x, header.y + 1, header.width, 1),
            );
        }
        body
    };

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
            " Type a message to chat, or / for commands.",
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
        Some(pending) => format!(" Message · {} running ", pending.task.command()),
        None => " Message ".into(),
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

    let mut status = match &app.pending {
        Some(pending) => {
            let elapsed = pending.started.elapsed();
            let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
                [(elapsed.as_millis() / 100 % 10) as usize];
            format!(
                " {spinner} {} · {:.1}s",
                if pending.cancelling {
                    "Stopping"
                } else {
                    pending.task.label()
                },
                elapsed.as_secs_f64()
            )
        }
        None => match &app.runtime {
            Some(runtime) => format!(" ● {}", runtime.status),
            None => " ● Ready".into(),
        },
    };
    let metrics = app
        .runtime
        .as_ref()
        .filter(|_| show_metrics)
        .map(|runtime| {
            let text = metrics_text(&runtime.metrics, false);
            if Line::from(text.as_str()).width() + 2 > usize::from(area.width) {
                metrics_text(&runtime.metrics, true)
            } else {
                text
            }
        });
    if footer_height == 1 {
        if let Some(metrics) = &metrics {
            status.push_str(" · ");
            status.push_str(metrics);
        }
    }
    let scroll = if app.scroll.is_some() {
        " · scrolled"
    } else {
        ""
    };
    let generating = app
        .pending
        .as_ref()
        .is_some_and(|pending| matches!(pending.task, Task::Generate(_)));
    let hints = if app.runtime_open {
        "F2/Esc close · PgUp/PgDn scroll"
    } else if generating && area.width >= 140 {
        "Esc stop · F2 runtime · Ctrl+C quit"
    } else if generating {
        "Esc stop · F2 runtime"
    } else if app.runtime.is_some() {
        "F2 runtime · Ctrl+C quit"
    } else if area.width >= 90 {
        "Enter run · Ctrl+J newline · Tab complete · PgUp/PgDn scroll · Ctrl+C quit"
    } else if area.width >= 60 {
        "Enter run · /help keys · Ctrl+C quit"
    } else {
        "Ctrl+C quit"
    };
    // All path/model text is sanitized before it reaches terminal cells.
    let mut footer_lines = vec![Line::from(vec![
        Span::styled(
            clean_text(&status, 256),
            Style::new().fg(if app.pending.is_some() {
                ACCENT
            } else {
                Color::Green
            }),
        ),
        Span::styled(format!("{scroll}  {hints}"), Style::new().fg(MUTED)),
    ])];
    if footer_height == 2 {
        footer_lines.push(Line::from(Span::styled(
            format!(" {}", metrics.as_deref().unwrap_or("")),
            Style::new().fg(ACCENT),
        )));
    }
    frame.render_widget(Paragraph::new(footer_lines), footer);
    if app.runtime_open {
        let expanded = Rect {
            height: area.height.saturating_sub(footer.height),
            ..area
        };
        frame.render_widget(Clear, expanded);
        draw_runtime(frame, app, expanded, true);
    }
}

fn metrics_text(metrics: &crate::progress::Snapshot, compact: bool) -> String {
    let count = |value: Option<usize>| {
        value
            .map(|value| {
                if compact {
                    crate::human_count(value as u64)
                } else {
                    value.to_string()
                }
            })
            .unwrap_or_else(|| "—".into())
    };
    let rate = metrics
        .decode_tokens_per_second
        .map(|value| {
            if compact && value >= 1000.0 {
                format!("{:.1}k", value / 1000.0)
            } else {
                format!("{value:.1}")
            }
        })
        .unwrap_or_else(|| "—".into());
    let ttft = metrics
        .ttft_seconds
        .map(|value| format!("{value:.2}s"))
        .unwrap_or_else(|| "—".into());
    if compact {
        format!("{}tok {rate}t/s TTFT {ttft}", count(metrics.total_tokens))
    } else {
        format!(
            "out {} · total {} tok · {rate} tok/s · TTFT {ttft}",
            metrics.generated_tokens,
            count(metrics.total_tokens)
        )
    }
}

fn model_content(app: &App) -> Paragraph<'_> {
    let model = app.model_summary.as_ref().expect("inspected model");
    let mut lines = vec![
        Line::from(Span::styled(&model.name, Style::new().bold())),
        Line::from(Span::styled(&model.family, Style::new().fg(ACCENT))),
        Line::from(Span::styled(
            format!(
                "{} turns · {} new tokens",
                app.conversation.turns.len(),
                app.settings.max_new_tokens
            ),
            Style::new().fg(MUTED),
        )),
        Line::default(),
    ];
    for field in &model.fields {
        lines.push(Line::from(vec![
            Span::styled(
                format!("{}  ", field.label.unwrap_or("")),
                Style::new().fg(MUTED),
            ),
            Span::raw(&field.text),
        ]));
    }
    lines.push(Line::default());
    lines.push(Line::from(Span::styled(
        &model.path,
        Style::new().fg(MUTED),
    )));
    Paragraph::new(lines).wrap(Wrap { trim: false })
}

fn model_height(app: &App, width: u16) -> u16 {
    (model_content(app).line_count(width.saturating_sub(4)) + 2).min(u16::MAX.into()) as u16
}

fn draw_model(frame: &mut Frame, app: &App, area: Rect) {
    let height = model_height(app, area.width).min(area.height);
    let block = Block::bordered()
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(Style::new().fg(MUTED))
        .padding(ratatui::widgets::Padding::horizontal(1))
        .title(Span::styled(" Model ", Style::new().fg(ACCENT)))
        .title_bottom(" /inspect for details ");
    frame.render_widget(model_content(app).block(block), Rect { height, ..area });
}

fn draw_runtime(frame: &mut Frame, app: &mut App, area: Rect, expanded: bool) {
    let block = Block::bordered()
        .border_type(ratatui::widgets::BorderType::Rounded)
        .border_style(Style::new().fg(MUTED))
        .padding(ratatui::widgets::Padding::horizontal(1))
        .title(Span::styled(
            if expanded {
                " Runtime · F2/Esc close · PgUp/PgDn scroll "
            } else {
                " Runtime "
            },
            Style::new().fg(ACCENT),
        ))
        .title_bottom(if expanded {
            " Ctrl+Home top · Ctrl+End latest "
        } else {
            " F2 expand "
        });
    let inner = block.inner(area);
    frame.render_widget(block, area);
    let Some(runtime) = &app.runtime else {
        frame.render_widget(Paragraph::new("No generation yet."), inner);
        return;
    };
    let summary = Paragraph::new(vec![
        Line::from(Span::styled(runtime.status, Style::new().fg(ACCENT).bold())),
        Line::from(Span::styled(&runtime.path, Style::new().fg(MUTED))),
    ])
    .wrap(Wrap { trim: false });
    let summary_height = summary
        .line_count(inner.width)
        .min(4)
        .min(inner.height.saturating_sub(2).into()) as u16;
    let [header, log] = Layout::vertical([Constraint::Length(summary_height), Constraint::Min(1)])
        .spacing(1)
        .areas(inner);
    frame.render_widget(summary, header);
    let text = runtime.text();
    let logs = Paragraph::new(text).wrap(Wrap { trim: false });
    let max_scroll = logs.line_count(log.width).saturating_sub(log.height.into());
    let scroll = if expanded {
        app.runtime_max_scroll = max_scroll;
        app.runtime_page_size = usize::from(log.height.saturating_sub(1).max(1));
        if let Some(row) = &mut app.runtime_scroll {
            *row = (*row).min(max_scroll);
        }
        app.runtime_scroll.unwrap_or(max_scroll)
    } else {
        max_scroll
    };
    frame.render_widget(logs.scroll((scroll.min(u16::MAX.into()) as u16, 0)), log);
}

#[cfg(test)]
mod tests {
    use super::super::report::Entry;
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    fn display(terminal: &Terminal<TestBackend>) -> String {
        terminal
            .backend()
            .buffer()
            .content
            .iter()
            .map(|cell| cell.symbol())
            .collect()
    }

    #[test]
    fn runtime_stays_out_of_chat_and_metrics_survive_narrow_windows() {
        use super::super::generate::{Event, Outcome, Stream};
        let mut app = App::new(Some("/model".into()));
        app.model_summary = Some(super::super::report::ModelSummary {
            name: "Inspected model".into(),
            family: "GLM-5.2".into(),
            path: "/model".into(),
            fields: vec![],
        });
        app.paste("/generate hello --ram-gib 2 --allow-large-model");
        let request = app.submit().unwrap();
        app.generation_event(
            request.id,
            Event::Output(Stream::Text, "answer only".into()),
        );
        app.generation_event(
            request.id,
            Event::Output(Stream::Log, "runtime diagnostic".into()),
        );
        app.generation_event(
            request.id,
            Event::Progress(crate::progress::Snapshot {
                prompt_tokens: Some(6),
                generated_tokens: 3,
                total_tokens: Some(9),
                ttft_seconds: Some(2.0),
                elapsed_seconds: 4.0,
                decode_tokens_per_second: Some(1.0),
                ..Default::default()
            }),
        );
        let mut terminal = Terminal::new(TestBackend::new(140, 35)).unwrap();
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let left: String = (2..30)
            .flat_map(|y| {
                let buffer = terminal.backend().buffer();
                (0..90).map(move |x| buffer[(x, y)].symbol())
            })
            .collect();
        assert!(left.contains("answer only"));
        assert!(!left.contains("runtime diagnostic"));
        assert!(display(&terminal).contains("Inspected model"));
        assert!(display(&terminal).contains("runtime diagnostic"));
        assert!(display(&terminal).contains("total 9 tok"));
        assert!(display(&terminal).contains("1.0 tok/s"));
        assert!(display(&terminal).contains("TTFT 2.00s"));
        for (width, height) in [(60, 18), (36, 10)] {
            terminal.backend_mut().resize(width, height);
            terminal.resize(Rect::new(0, 0, width, height)).unwrap();
            terminal.draw(|frame| draw(frame, &mut app)).unwrap();
            let rendered = display(&terminal);
            assert!(rendered.contains("TTFT 2.00s"), "{width}: {rendered}");
            assert!(!rendered.contains("runtime diagnostic"));
        }
        let metrics = &mut app.runtime.as_mut().unwrap().metrics;
        metrics.prompt_tokens = Some(900_000);
        metrics.generated_tokens = 148_576;
        metrics.total_tokens = Some(1_048_576);
        metrics.elapsed_seconds = 180.0;
        metrics.ttft_seconds = Some(120.0);
        metrics.decode_tokens_per_second = Some(123.4);
        for (width, height) in [(60, 18), (36, 10)] {
            terminal.backend_mut().resize(width, height);
            terminal.resize(Rect::new(0, 0, width, height)).unwrap();
            terminal.draw(|frame| draw(frame, &mut app)).unwrap();
            let rendered = display(&terminal);
            assert!(rendered.contains("TTFT 120.00s"), "{width}: {rendered}");
            assert!(rendered.contains("123.4t/s"));
        }
        app.runtime_open = true;
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        assert!(display(&terminal).contains("runtime diagnostic"));
        app.generation_event(request.id, Event::Finished(Outcome::Complete));
        assert!(app.runtime.as_ref().unwrap().metrics.ttft_seconds.is_some());
        assert!(!app
            .entries
            .iter()
            .flat_map(|entry| &entry.details)
            .any(|detail| detail.label == Some("Runtime")));
    }

    #[test]
    fn model_card_is_fixed_at_the_right_and_survives_clear_scroll_and_resize() {
        let mut app = App::new(Some("/models/DeepSeek".into()));
        app.model_summary = Some(super::super::report::ModelSummary {
            name: "DeepSeek-V4".into(),
            family: "DeepSeek-V4".into(),
            path: "/models/DeepSeek".into(),
            fields: vec![super::super::report::Detail {
                label: Some("Max context"),
                text: "1048576".into(),
                warning: false,
            }],
        });
        app.push(Entry::message(
            Kind::Info,
            "Long history",
            "line\n".repeat(100),
        ));
        let mut terminal = Terminal::new(TestBackend::new(140, 35)).unwrap();
        for scroll in [None, Some(0)] {
            app.scroll = scroll;
            terminal.draw(|frame| draw(frame, &mut app)).unwrap();
            let right_top: String = (2..8)
                .flat_map(|y| {
                    let buffer = terminal.backend().buffer();
                    (95..140).map(move |x| buffer[(x, y)].symbol())
                })
                .collect();
            assert!(right_top.contains("DeepSeek-V4"), "{right_top}");
            assert!(display(&terminal).contains("1048576"));
        }
        app.paste("/clear");
        app.submit();
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        assert!(display(&terminal).contains("DeepSeek-V4"));
        assert!(display(&terminal).contains("Type a message"));
        terminal.backend_mut().resize(60, 18);
        terminal.resize(Rect::new(0, 0, 60, 18)).unwrap();
        terminal.draw(|frame| draw(frame, &mut app)).unwrap();
        let header: String = (0..60)
            .map(|x| terminal.backend().buffer()[(x, 1)].symbol())
            .collect();
        assert!(header.contains("DeepSeek-V4"));
        assert!(!display(&terminal).contains("/inspect for details"));
    }

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
