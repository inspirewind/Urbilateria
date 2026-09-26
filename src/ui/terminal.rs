use crossterm::{
    cursor::{Hide, Show},
    event::{DisableBracketedPaste, EnableBracketedPaste},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::io::{self, IsTerminal, Stdout};
use std::panic::{self, PanicHookInfo};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

type PanicHook = Arc<dyn Fn(&PanicHookInfo<'_>) + Send + Sync>;

pub struct Session {
    pub terminal: Terminal<CrosstermBackend<Stdout>>,
    pub stopping: Arc<AtomicBool>,
    _guard: Guard,
}

struct Guard {
    old_hook: PanicHook,
    #[cfg(unix)]
    signals: Vec<signal_hook::SigId>,
}

impl Session {
    pub fn enter() -> io::Result<Self> {
        if !io::stdin().is_terminal() || !io::stdout().is_terminal() {
            return Err(io::Error::other("urb ui needs an interactive terminal on stdin and stdout; use urb inspect MODEL_DIR [--json] for redirected output"));
        }
        let stopping = Arc::new(AtomicBool::new(false));
        let old_hook: PanicHook = Arc::from(panic::take_hook());
        let hook = Arc::clone(&old_hook);
        let stop = Arc::clone(&stopping);
        panic::set_hook(Box::new(move |info| {
            stop.store(true, Ordering::Relaxed);
            restore();
            hook(info);
        }));
        // Install cleanup before the first terminal mutation, including partial startup errors.
        let mut guard = Guard {
            old_hook,
            #[cfg(unix)]
            signals: Vec::new(),
        };
        #[cfg(unix)]
        for signal in [
            signal_hook::consts::SIGTERM,
            signal_hook::consts::SIGHUP,
            signal_hook::consts::SIGINT,
        ] {
            guard
                .signals
                .push(signal_hook::flag::register(signal, Arc::clone(&stopping))?);
        }
        enable_raw_mode()?;
        execute!(
            io::stdout(),
            EnterAlternateScreen,
            EnableBracketedPaste,
            Hide
        )?;
        let terminal = Terminal::new(CrosstermBackend::new(io::stdout()))?;
        Ok(Self {
            terminal,
            stopping,
            _guard: guard,
        })
    }
}

fn restore() {
    // Attempt every cleanup operation even when one fails (e.g. a disconnected terminal).
    let _ = execute!(io::stdout(), DisableBracketedPaste);
    let _ = execute!(io::stdout(), LeaveAlternateScreen);
    let _ = execute!(io::stdout(), Show);
    let _ = disable_raw_mode();
}

impl Drop for Guard {
    fn drop(&mut self) {
        restore();
        #[cfg(unix)]
        for signal in self.signals.drain(..) {
            signal_hook::low_level::unregister(signal);
        }
        if !std::thread::panicking() {
            let hook = Arc::clone(&self.old_hook);
            panic::set_hook(Box::new(move |info| hook(info)));
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(unix)]
    #[test]
    #[ignore = "requires a controlling PTY; exercised by tests/tui_smoke.py"]
    fn reads_queued_resize_and_input() {
        use crossterm::event::{self, Event, KeyCode};
        use std::io::{Read, Write};
        use std::time::{Duration, Instant};

        let _session = super::Session::enter().unwrap();
        let _ = event::poll(Duration::from_millis(1)).unwrap();
        println!("event-reader-ready");
        std::io::stdout().flush().unwrap();

        // The PTY driver queues a resize signal and keyboard input, in both orders.
        // A FIFO handshake holds off polling until both sources are ready, without
        // relying on scheduler delays or adding another keypress to wake the reader.
        let path = std::env::var_os("URB_TUI_EVENT_GATE").expect("PTY event gate is required");
        let mut gate = std::fs::File::open(path).unwrap();
        let mut step = [0];
        gate.read_exact(&mut step).unwrap();
        println!("event-reader-queued");
        std::io::stdout().flush().unwrap();
        gate.read_exact(&mut step).unwrap();

        let deadline = Instant::now() + Duration::from_secs(2);
        let mut resized = false;
        let mut input = String::new();
        let mut entered = false;
        while Instant::now() < deadline && !(resized && entered) {
            if event::poll(Duration::from_millis(50)).unwrap() {
                match event::read().unwrap() {
                    Event::Resize(_, _) => resized = true,
                    Event::Key(key) => match key.code {
                        KeyCode::Char(ch) => input.push(ch),
                        KeyCode::Enter => entered = true,
                        _ => {}
                    },
                    _ => {}
                }
            }
        }
        assert!(resized, "queued resize event was lost");
        assert_eq!(input, "/quit", "queued keyboard input was lost");
        assert!(entered, "queued Enter key was lost");
    }

    #[test]
    #[ignore = "requires a controlling PTY; exercised by tests/tui_smoke.py"]
    fn restores_terminal_on_panic() {
        let result = std::panic::catch_unwind(|| {
            let _session = super::Session::enter().unwrap();
            panic!("intentional terminal cleanup test");
        });
        assert!(result.is_err());
        assert!(!crossterm::terminal::is_raw_mode_enabled().unwrap());
    }
}
