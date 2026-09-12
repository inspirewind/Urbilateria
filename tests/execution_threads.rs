use std::sync::{Arc, Barrier};

use urbilateria::execution::{configure_threads, worker_threads};

#[test]
fn concurrent_identical_configuration_is_idempotent() {
    let callers = 8;
    let barrier = Arc::new(Barrier::new(callers));
    let handles = (0..callers)
        .map(|_| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                barrier.wait();
                configure_threads(4)
            })
        })
        .collect::<Vec<_>>();
    for handle in handles {
        handle.join().unwrap().unwrap();
    }
    assert_eq!(worker_threads(), 4);
    let error = configure_threads(3).unwrap_err().to_string();
    assert!(error.contains("already has 4 workers"));
    assert!(error.contains("cannot be resized to 3"));
}
