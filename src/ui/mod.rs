//! Interactive presentation layer. Workers return data; only this module owns terminal output.

mod app;
mod chat;
mod commands;
mod generate;
mod report;
mod terminal;
mod view;
mod worker;

use app::App;
use crossterm::event::{self, Event};
use std::error::Error;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use terminal::Session;
use worker::{Event as WorkerEvent, Finished, Worker};

pub fn run_args(args: Vec<String>) -> Result<(), Box<dyn Error>> {
    if matches!(args.as_slice(), [flag] if flag == "--help" || flag == "-h") {
        println!("Usage: urb ui [MODEL_DIR]\n\nInteractive checkpoint explorer. Requires a terminal on stdin and stdout.");
        let help = report::help_entry();
        for detail in help.details {
            if let Some(label) = detail.label {
                println!("{label}  {}", detail.text);
            } else {
                println!("{}", detail.text);
            }
        }
        return Ok(());
    }
    let model = match args.as_slice() {
        [] => None,
        [path] if !path.starts_with('-') => Some(commands::expand_home(path)),
        _ => return Err("usage: urb ui [MODEL_DIR] (see urb ui --help)".into()),
    };
    let mut session = Session::enter()?;
    let mut worker = Worker::start()?;
    let mut app = App::new(model);
    // Register input/resize handlers before the first visible frame.
    let _ = event::poll(Duration::from_millis(1))?;
    let mut dirty = true;
    let mut last_draw = Instant::now() - Duration::from_secs(1);
    let mut last_size = crossterm::terminal::size()?;
    let mut last_size_check = Instant::now();
    while !app.quit && !session.stopping.load(Ordering::Relaxed) {
        // Resize notifications can be coalesced/lost alongside input. An idle terminal still
        // needs to redraw after a size change; query at 4 Hz without painting unchanged frames.
        if last_size_check.elapsed() >= Duration::from_millis(250) {
            let size = crossterm::terminal::size()?;
            dirty |= size != last_size;
            last_size = size;
            last_size_check = Instant::now();
        }
        // Bound work per frame so continuous generation/log output cannot starve input.
        for _ in 0..64 {
            match worker.poll()? {
                Some(WorkerEvent::Analysis(event)) => app.finished(event),
                Some(WorkerEvent::Generation { id, event }) => app.generation_event(id, event),
                None => break,
            }
            dirty = true;
        }
        if (dirty || app.pending.is_some()) && last_draw.elapsed() >= Duration::from_millis(50) {
            session.terminal.draw(|frame| view::draw(frame, &mut app))?;
            dirty = false;
            last_draw = Instant::now();
        }
        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                Event::Key(key) => {
                    if let Some(request) = app.key(key) {
                        let id = request.id;
                        if let Err(error) = worker.submit(request) {
                            app.finished(Finished {
                                id,
                                result: Err(error),
                            });
                        }
                    }
                    if std::mem::take(&mut app.cancel_requested) {
                        if let Err(error) = worker.cancel_generation() {
                            if let Some(pending) = &mut app.pending {
                                pending.cancelling = false;
                            }
                            app.error(error);
                        }
                    }
                }
                Event::Paste(text) => app.paste(&text),
                Event::Resize(_, _) => {}
                _ => continue,
            }
            dirty = true;
        }
    }
    Ok(())
}
