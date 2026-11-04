//! Structured runtime details; original diagnostics remain available in the expanded view.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use urbilateria::runtime::telemetry::{Event, LayerCache, Weights};
use urbilateria::runtime::ExpertTelemetry;

#[derive(Default)]
pub struct Details {
    context: Option<(usize, usize, usize, usize)>,
    kv: Option<(usize, usize, usize)>,
    kv_state: Option<(usize, usize)>,
    weights: Option<Weights>,
    experts: Option<(ExpertTelemetry, Vec<LayerCache>)>,
    stop: Option<String>,
}

impl Details {
    pub fn update(&mut self, event: Event) {
        match event {
            Event::Context {
                history,
                dropped,
                prompt,
                limit,
            } => self.context = Some((history, dropped, prompt, limit)),
            Event::Kv {
                reused,
                prefill,
                capacity,
            } => {
                self.kv = Some((reused, prefill, capacity));
                // A new plan is emitted after restoring/resetting the state to this prefix.
                self.kv_state = Some((reused, capacity));
            }
            Event::KvState { position, capacity } => self.kv_state = Some((position, capacity)),
            Event::Weights(weights) => self.weights = Some(weights),
            Event::Experts { totals, layers } => self.experts = Some((totals, layers)),
            Event::Stop { reason } => self.stop = Some(super::commands::clean_text(&reason, 128)),
        }
    }

    pub fn available(&self) -> bool {
        self.context.is_some()
            || self.kv.is_some()
            || self.kv_state.is_some()
            || self.weights.is_some()
            || self.experts.is_some()
    }

    pub fn lines(&self, expanded: bool, live: bool, width: u16) -> Vec<Line<'static>> {
        let mut lines = Vec::new();
        if let Some(w) = &self.weights {
            lines.push(heading(if live {
                "Weight residency"
            } else {
                "Last weight snapshot"
            }));
            lines.push(field(
                "Backbone",
                format!(
                    "{}/{} layers · ~{}",
                    w.loaded_layers,
                    w.total_layers,
                    crate::human_bytes(w.resident_bytes)
                ),
            ));
            if w.loading {
                lines.push(field(
                    "Loading",
                    format!("{}/{} planned layers", w.loaded_layers, w.planned_layers),
                ));
            }
            lines.push(field(
                "LM head",
                if w.lm_head_loaded {
                    "resident"
                } else if w.loading && w.lm_head_planned {
                    "pending"
                } else {
                    "streamed"
                },
            ));
            if w.row_streamed_embeddings {
                lines.push(field("Embedding/PLE", "rows on demand"));
            }
        }
        if let Some((totals, _)) = &self.experts {
            let used = crate::human_bytes(totals.resident_bytes);
            lines.push(field(
                "Expert cache",
                match &self.weights {
                    Some(w) => format!(
                        "{used} / {} budget",
                        crate::human_bytes(w.expert_budget_bytes)
                    ),
                    None => used,
                },
            ));
            let hit_rate = totals
                .hit_rate()
                .map(|rate| format!("{:.1}%", rate * 100.0))
                .unwrap_or_else(|| "—".into());
            lines.push(field(
                "Experts",
                format!("{} cached · {hit_rate} hit", totals.resident_experts),
            ));
            if expanded {
                lines.push(field("Payload read", crate::human_bytes(totals.bytes_read)));
                lines.push(field(
                    "Accesses",
                    format!(
                        "{} hits · {} misses · {} evictions",
                        totals.hits, totals.misses, totals.evictions
                    ),
                ));
                lines.push(muted("Counters accumulate with this expert cache; payload reads can hit the OS page cache."));
            }
        }
        if let Some(w) = &self.weights {
            lines.push(Line::default());
            // A cell describes non-routed backbone weights, never all experts in that layer.
            let digits = w.total_layers.saturating_sub(1).to_string().len().max(2);
            let columns = if usize::from(width) >= 2 * digits + 4 + 34 {
                16
            } else {
                8
            };
            for start in (0..w.total_layers.min(64)).step_by(columns) {
                let end = (start + columns).min(w.total_layers);
                let mut row = vec![Span::styled(
                    format!("L{start:0digits$}-{:0digits$}  ", end - 1),
                    Style::new().fg(Color::DarkGray),
                )];
                for layer in start..end {
                    if layer != start {
                        row.push(Span::raw(if (layer - start) % 4 == 0 { "  " } else { " " }));
                    }
                    let (symbol, color) = if layer < w.loaded_layers {
                        ("■", Color::Cyan)
                    } else if layer < w.planned_layers {
                        ("□", Color::Yellow)
                    } else {
                        ("·", Color::DarkGray)
                    };
                    row.push(Span::styled(symbol, Style::new().fg(color)));
                }
                lines.push(Line::from(row));
            }
            lines.push(Line::from(vec![
                Span::styled("■", Style::new().fg(Color::Cyan)),
                Span::styled(" cached  ", Style::new().fg(Color::DarkGray)),
                Span::styled("□", Style::new().fg(Color::Yellow)),
                Span::styled(" pending  · streamed", Style::new().fg(Color::DarkGray)),
            ]));
            if w.total_layers > 64 {
                lines.push(muted("F2: all layer IDs"));
            }
        }
        if self.context.is_some() || self.kv.is_some() || self.kv_state.is_some() {
            lines.push(Line::default());
            lines.push(heading(if self.kv_state.is_some() {
                if live {
                    "KV cache"
                } else {
                    "KV cache (last)"
                }
            } else {
                "Context"
            }));
            let prompt = self
                .kv
                .map(|(reused, prefill, _)| reused.saturating_add(prefill))
                .or_else(|| self.context.map(|(_, _, prompt, _)| prompt));
            if let Some(prompt) = prompt {
                lines.push(field("Prompt", format!("{prompt} tok")));
            }
            if let Some((reused, prefill, capacity)) = self.kv {
                let fraction = percentage(reused, reused.saturating_add(prefill))
                    .map(|ratio| format!("{ratio:.1}%"))
                    .unwrap_or_else(|| "—".into());
                lines.push(field("Reused", format!("{reused} tok · {fraction}")));
                let done = self
                    .kv_state
                    .filter(|(_, cap)| *cap == capacity)
                    .map(|(position, _)| position.saturating_sub(reused).min(prefill).to_string())
                    .unwrap_or_else(|| "—".into());
                lines.push(field("Prefill", format!("{done} / {prefill} tok")));
            }
            if let Some((position, capacity)) = self.kv_state {
                lines.push(field("Cached", format!("{position} / {capacity} tok")));
                lines.push(usage_bar(position, capacity, width));
                lines.push(field(
                    "Free",
                    format!("{} tok", capacity.saturating_sub(position)),
                ));
                if expanded {
                    lines.push(muted("Reuse is a fraction of this prompt. Cached counts fully forwarded positions, including hybrid/recurrent state; it is not a byte count. The final sampled token has not been cached yet."));
                    lines.push(muted("Capacity is the current runtime reservation, not the model's maximum context."));
                }
            }
            if expanded {
                if let Some((history, dropped, _, limit)) = self.context {
                    lines.push(field(
                        "History",
                        format!("{history} turns · {dropped} dropped"),
                    ));
                    lines.push(field("Model limit", format!("{limit} tok")));
                }
            }
        }
        if let Some(reason) = &self.stop {
            lines.push(field("Stop", reason));
        }
        if expanded {
            if let Some(w) = &self.weights {
                lines.push(Line::default());
                lines.push(heading("Layers · zero-based IDs"));
                lines.push(muted("Backbone excludes routed experts. Payload estimates exclude KV/state, scratch and allocator overhead; they are not RSS."));
                lines.push(muted("Completed/cancelled runs show the last observed snapshot, not a live memory measurement."));
                if w.experts_per_layer > 0 {
                    lines.push(field(
                        "Expert pool",
                        format!("{} per layer", w.experts_per_layer),
                    ));
                }
                for layer in 0..w.total_layers {
                    let backbone = if layer < w.loaded_layers {
                        "cached"
                    } else if layer < w.planned_layers {
                        "pending"
                    } else {
                        "streamed"
                    };
                    let cache = self
                        .experts
                        .as_ref()
                        .and_then(|(_, layers)| layers.iter().find(|l| l.layer == layer));
                    lines.push(field(
                        &format!("L{layer:02}"),
                        match cache {
                            Some(c) => format!(
                                "{backbone} · experts {}/{} · {}",
                                c.experts,
                                c.capacity,
                                crate::human_bytes(c.bytes)
                            ),
                            None => format!("{backbone} · experts —"),
                        },
                    ));
                }
            }
        }
        lines
    }
}

fn field(label: &str, value: impl Into<String>) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label}  "), Style::new().fg(Color::DarkGray)),
        Span::raw(value.into()),
    ])
}
fn heading(text: &str) -> Line<'static> {
    Line::styled(text.to_owned(), Style::new().fg(Color::Cyan))
}
fn muted(text: &str) -> Line<'static> {
    Line::styled(text.to_owned(), Style::new().fg(Color::DarkGray))
}

fn percentage(value: usize, capacity: usize) -> Option<f64> {
    (capacity != 0).then(|| value as f64 / capacity as f64 * 100.0)
}

fn usage_bar(position: usize, capacity: usize, width: u16) -> Line<'static> {
    let Some(percent) = percentage(position, capacity) else {
        return muted("Usage  —");
    };
    let cells = usize::from(width.saturating_sub(9)).min(32);
    // Round nonzero occupancy up to one cell; the exact value is shown alongside the bar.
    let filled =
        ((position.min(capacity) as u128 * cells as u128).div_ceil(capacity as u128)) as usize;
    Line::from(vec![
        Span::styled("[", Style::new().fg(Color::DarkGray)),
        Span::styled("━".repeat(filled), Style::new().fg(Color::Cyan)),
        Span::styled("·".repeat(cells - filled), Style::new().fg(Color::DarkGray)),
        Span::styled("]", Style::new().fg(Color::DarkGray)),
        Span::raw(format!(" {percent:>5.1}%")),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn layer_map_distinguishes_loaded_pending_and_streamed_weights_from_experts() {
        let mut details = Details::default();
        let weights = Weights {
            total_layers: 4,
            loaded_layers: 1,
            planned_layers: 2,
            resident_bytes: 1024,
            lm_head_loaded: true,
            lm_head_planned: true,
            loading: true,
            expert_budget_bytes: 4096,
            experts_per_layer: 512,
            row_streamed_embeddings: true,
        };
        let wire = serde_json::to_string(&Event::Weights(weights.clone())).unwrap();
        let event = serde_json::from_str::<Event>(&wire).unwrap();
        assert!(event.is_valid());
        details.update(event);
        details.update(Event::Experts {
            totals: ExpertTelemetry {
                hits: 3,
                misses: 1,
                resident_bytes: 1024,
                resident_experts: 1,
                ..Default::default()
            },
            layers: vec![LayerCache {
                layer: 0,
                experts: 1,
                capacity: 2,
                bytes: 1024,
            }],
        });
        let compact = details
            .lines(false, true, 44)
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(compact.contains("■ □ · ·"), "{compact}");
        assert!(compact.contains("1/4 layers"));
        assert!(compact.contains("1.00 KiB / 4.00 KiB budget"));
        assert!(compact.contains("75.0% hit"));
        assert!(!compact.contains("experts 1/2"));
        let expanded = details
            .lines(true, false, 44)
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n");
        assert!(expanded.contains("Last weight snapshot"));
        assert!(expanded.contains("L00  cached · experts 1/2 · 1.00 KiB"));
        assert!(expanded.contains("L02  streamed · experts —"));
        assert!(expanded.contains("not RSS"));
        assert!(!Event::Weights(Weights {
            loaded_layers: 3,
            ..weights
        })
        .is_valid());
    }

    #[test]
    fn layer_cells_stay_separate_and_keep_their_states_when_the_grid_reflows() {
        let mut details = Details::default();
        details.update(Event::Weights(Weights {
            total_layers: 48,
            loaded_layers: 10,
            planned_layers: 24,
            resident_bytes: 1024,
            lm_head_loaded: true,
            lm_head_planned: true,
            loading: true,
            expert_budget_bytes: 4096,
            experts_per_layer: 512,
            row_streamed_embeddings: true,
        }));
        for (width, rows) in [(32, 6), (42, 3)] {
            let lines = details.lines(false, true, width);
            let grid: Vec<_> = lines
                .iter()
                .filter(|line| {
                    let Some(first) = line.spans.first() else {
                        return false;
                    };
                    let label = first.content.as_ref();
                    label.starts_with('L')
                        && label.as_bytes().get(1).is_some_and(|b| b.is_ascii_digit())
                })
                .collect();
            assert_eq!(grid.len(), rows);
            let text = grid
                .iter()
                .map(|line| line.to_string())
                .collect::<Vec<_>>()
                .join("\n");
            assert_eq!(text.matches('■').count(), 10);
            assert_eq!(text.matches('□').count(), 14);
            assert_eq!(text.matches('·').count(), 24);
            assert!(!text.contains("■■"));
            assert!(grid.iter().all(|line| line.width() <= usize::from(width)));
            let legend = lines
                .iter()
                .find(|line| line.to_string().contains("cached  □ pending"))
                .unwrap();
            assert_eq!(legend.spans[0].style.fg, Some(Color::Cyan));
            assert_eq!(legend.spans[2].style.fg, Some(Color::Yellow));
        }
    }

    #[test]
    fn kv_view_separates_prompt_reuse_prefill_progress_and_runtime_occupancy() {
        let mut details = Details::default();
        details.update(Event::Context {
            history: 1,
            dropped: 0,
            prompt: 192,
            limit: 262_144,
        });
        details.update(Event::Kv {
            reused: 176,
            prefill: 16,
            capacity: 2048,
        });
        let render = |details: &Details, live| {
            details
                .lines(false, live, 32)
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("\n")
        };
        let text = render(&details, true);
        assert!(text.contains("Reused  176 tok · 91.7%"));
        assert!(text.contains("Prefill  0 / 16 tok"));
        details.update(Event::KvState {
            position: 184,
            capacity: 2048,
        });
        assert!(render(&details, true).contains("Prefill  8 / 16 tok"));
        details.update(Event::KvState {
            position: 220,
            capacity: 2048,
        });
        let text = render(&details, false);
        assert!(text.contains("KV cache (last)"));
        assert!(text.contains("Prefill  16 / 16 tok"));
        assert!(text.contains("Cached  220 / 2048 tok"));
        assert!(text.contains("10.7%"));
        assert!(text.contains("Free  1828 tok"));
        details.update(Event::Kv {
            reused: 0,
            prefill: 16,
            capacity: 4096,
        });
        assert!(render(&details, true).contains("Cached  0 / 4096 tok"));
        details.update(Event::Kv {
            reused: 0,
            prefill: 0,
            capacity: 0,
        });
        assert!(!render(&details, true).contains("NaN"));
        for width in [24, 32, 42] {
            assert!(usage_bar(220, 2048, width).width() <= usize::from(width));
            assert!(usage_bar(2048, 2048, width).to_string().contains("100.0%"));
        }
    }
}
