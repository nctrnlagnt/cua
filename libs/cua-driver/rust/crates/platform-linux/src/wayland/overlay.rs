//! Native-Wayland agent-cursor overlay via `zwlr_layer_shell_v1`.
//!
//! Replaces the X11-only `overlay.rs` render loop on wlroots compositors
//! (sway, labwc, kwin 5.27+, hyprland) by creating a full-screen,
//! click-through, always-on-top `wl_surface` anchored to the first output
//! via `zwlr_layer_shell_v1`. The surface renders the same gradient-arrow
//! cursor as the X11 path by sharing `cursor_overlay::RenderStateCore` —
//! bloom, click-pulse, idle-fade, and motion all work identically.
//!
//! GNOME mutter does not expose `zwlr_layer_shell_v1` — those sessions
//! either fall through to the X11 path (XWayland) or the nested-compositor
//! mode that spawns labwc internally.
//!
//! Architecture mirrors the existing `wayland/persistent_vptr.rs`: one
//! owner thread (`cua-overlay-wl`) holds the wayland Connection +
//! EventQueue + layer surface; commands flow in over a `crossbeam-channel`.
//! The render core is ticked at ~60Hz via a calloop timer so motion +
//! spring physics + click pulse advance smoothly even when no new
//! Position command has arrived.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::thread;
use std::time::Instant;

use crossbeam_channel::{bounded, Receiver, Sender};
use cursor_overlay::{CursorConfig, OverlayCommand, OverlayMsg, RenderStateCore};
use wayland_client::{
    protocol::{
        wl_buffer::WlBuffer,
        wl_compositor::WlCompositor,
        wl_output::WlOutput,
        wl_region::WlRegion,
        wl_registry,
        wl_shm::{self, WlShm},
        wl_shm_pool::WlShmPool,
        wl_surface::WlSurface,
    },
    Connection, Dispatch, Proxy, QueueHandle,
};
use wayland_protocols::wp::fractional_scale::v1::client::{
    wp_fractional_scale_manager_v1::WpFractionalScaleManagerV1,
    wp_fractional_scale_v1::{self, WpFractionalScaleV1},
};
use wayland_protocols::wp::viewporter::client::{
    wp_viewport::WpViewport, wp_viewporter::WpViewporter,
};
use wayland_protocols_wlr::layer_shell::v1::client::{
    zwlr_layer_shell_v1::{Layer, ZwlrLayerShellV1},
    zwlr_layer_surface_v1::{self, Anchor, KeyboardInteractivity, ZwlrLayerSurfaceV1},
};

/// Commands the overlay owner thread accepts. The richer commands the
/// cross-platform [`RenderStateCore`] understands (MoveTo, ClickPulse,
/// SetPressed) are forwarded as-is so the layer-shell overlay matches the
/// X11 visual: bloom + animated arrow + click pulse + press ring.
enum WlOverlayCmd {
    Cmd { cmd: OverlayCommand },
    Remove,
    Shutdown,
}

static TX: OnceLock<Sender<WlOverlayCmd>> = OnceLock::new();

fn tx() -> Option<&'static Sender<WlOverlayCmd>> {
    TX.get()
}

/// Lazily start the owner thread. Idempotent — safe to call from every
/// MCP tool invocation; subsequent calls are no-ops.
pub fn ensure_started() {
    TX.get_or_init(|| {
        // Larger than the old 64: a burst of SetEnabled + MoveTo + ClickPulse
        // across multi-cursor tools can fill a small queue, and try_send used
        // to drop those silently — cursor never appears.
        let (tx, rx) = bounded::<WlOverlayCmd>(256);
        thread::Builder::new()
            .name("cua-overlay-wl".into())
            .spawn(move || {
                if let Err(e) = owner_thread(rx) {
                    tracing::warn!("cua-overlay-wl thread exited with error: {e}");
                }
            })
            .expect("spawn cua-overlay-wl thread");
        tx
    });
}

/// Enqueue a command for the owner thread. Prefer non-blocking try_send; if
/// the channel is full, block briefly rather than drop — a dropped MoveTo /
/// ClickPulse leaves the cursor invisible for the whole action.
fn enqueue(tx: &Sender<WlOverlayCmd>, cmd: WlOverlayCmd) {
    match tx.try_send(cmd) {
        Ok(()) => {}
        Err(crossbeam_channel::TrySendError::Full(cmd)) => {
            if let Err(e) = tx.send_timeout(cmd, std::time::Duration::from_millis(100)) {
                tracing::warn!("cua-overlay-wl command dropped (channel full/disconnected): {e}");
            }
        }
        Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
            tracing::warn!("cua-overlay-wl command dropped: owner thread gone");
        }
    }
}

/// Translate a generic [`OverlayMsg`] (the cross-platform command shape)
/// to the layer-shell owner thread. The owner-thread render core consumes
/// every variant the X11 path handles; only `ShowFocusRect` (macOS-only)
/// is silently dropped here.
pub fn forward(msg: &OverlayMsg) -> bool {
    // Lazy startup: spawning the layer-shell owner thread + connecting to
    // the Wayland compositor takes 100-300ms. Doing that at cua-driver mcp
    // boot (the old eager-init path) was tipping the borderline CI
    // cursor-click-gif test over its 20s budget. ensure_started is
    // idempotent so calling it on every forward is fine — the OnceLock
    // bypasses the spawn after the first call.
    ensure_started();
    let Some(tx) = tx() else { return false };
    match msg {
        OverlayMsg::Remove(k) => {
            let _ = k;
            enqueue(tx, WlOverlayCmd::Remove);
            true
        }
        OverlayMsg::Cmd(kc) => {
            if matches!(&kc.cmd, OverlayCommand::ShowFocusRect(_)) {
                return false;
            }
            enqueue(
                tx,
                WlOverlayCmd::Cmd {
                    cmd: kc.cmd.clone(),
                },
            );
            true
        }
    }
}

/// Cleanly stop the owner thread. Tests use this; production code typically
/// lets the thread die at process exit.
pub fn shutdown() {
    if let Some(tx) = tx() {
        let _ = tx.send(WlOverlayCmd::Shutdown);
    }
}

// ── owner thread ─────────────────────────────────────────────────────────

struct OverlayState {
    compositor: Option<WlCompositor>,
    shm: Option<WlShm>,
    layer_shell: Option<ZwlrLayerShellV1>,
    output: Option<WlOutput>,
    /// Logical (surface-local) dimensions from the layer-surface `configure`
    /// event. The wl_shm buffer is allocated at `surface_w × scale` physical
    /// pixels (see `output_scale`); a `wp_viewport` maps that buffer back to
    /// this logical size so the compositor places it correctly on a HiDPI
    /// output. Populated only by `ZwlrLayerSurfaceV1::Configure` when the
    /// compositor sends a non-zero size. Anchored full-screen surfaces often
    /// get Configure(0,0) — then `mode_w/h ÷ scale` is the fallback.
    surface_w: u32,
    surface_h: u32,
    /// Physical screen pixels from `wl_output::Mode`. Used only as a fallback
    /// when Configure leaves surface dims at 0 (anchor-all + set_size(0,0)).
    /// Kept separate from `surface_w/h` so a later Mode event cannot overwrite
    /// a real logical Configure size.
    mode_w: u32,
    mode_h: u32,
    /// Effective output scale (logical→physical). Driven by
    /// `wp_fractional_scale_v1::preferred_scale` when the compositor supports
    /// it (covers 1.5/1.75 fractional), otherwise by `wl_output.scale`
    /// (integer only). Stays `1.0` on unscaled outputs.
    output_scale: f64,
    /// Integer `wl_output.scale`, kept separately as a fallback for the
    /// `set_buffer_scale` path on compositors without `wp_viewporter`.
    output_scale_int: i32,
    viewporter: Option<WpViewporter>,
    fractional_mgr: Option<WpFractionalScaleManagerV1>,
    viewport: Option<WpViewport>,
    fractional: Option<WpFractionalScaleV1>,
    surface: Option<WlSurface>,
    layer_surface: Option<ZwlrLayerSurfaceV1>,
    configured: bool,
    /// Cross-platform render core: position, animation, gradient arrow,
    /// bloom, click pulse, idle-fade. Shared verbatim with the X11 path.
    core: RenderStateCore,
    /// In-flight wl_shm buffers awaiting `wl_buffer.release` from the
    /// compositor. Keyed by `WlBuffer` object id; value is the
    /// `(mmap ptr, mmap size, memfd fd)` triple that must be unmapped +
    /// closed once the compositor signals it's done with the buffer.
    /// Replaces the per-redraw `mem::forget` leak: the previous frame's
    /// memory is reclaimed as soon as the compositor releases it.
    pending_buffers: HashMap<u32, (*mut libc::c_void, usize, i32)>,
    /// Object id of the buffer most recently attached+committed. On
    /// `wl_buffer.release` of this id we must re-attach (dirty-only redraw
    /// would otherwise leave the surface with no buffer → blank cursor).
    current_buffer_id: Option<u32>,
    /// Set when the compositor released `current_buffer_id` and we have not
    /// yet committed a replacement. Forces a redraw on the next loop tick.
    needs_reattach: bool,
}

// SAFETY: the raw pointers in pending_buffers point at mmap regions owned
// exclusively by this thread (the owner thread). OverlayState is never
// shared across threads — wayland-client's EventQueue<State> is !Send so
// it stays pinned to the owner thread. The Send/Sync bounds wayland-client
// requires for State types apply to the struct as a whole, hence the
// explicit assertion.
unsafe impl Send for OverlayState {}

impl Default for OverlayState {
    fn default() -> Self {
        Self {
            compositor: None,
            shm: None,
            layer_shell: None,
            output: None,
            surface_w: 0,
            surface_h: 0,
            mode_w: 0,
            mode_h: 0,
            output_scale: 1.0,
            output_scale_int: 1,
            viewporter: None,
            fractional_mgr: None,
            viewport: None,
            fractional: None,
            surface: None,
            layer_surface: None,
            configured: false,
            core: RenderStateCore::new(CursorConfig::default()),
            pending_buffers: HashMap::new(),
            current_buffer_id: None,
            needs_reattach: false,
        }
    }
}

fn dbg(msg: &str) {
    if std::env::var_os("CUA_OVERLAY_DEBUG").is_some() {
        eprintln!("[cua-overlay-wl] {msg}");
    }
}

/// Resolve the logical surface-local size used for buffer sizing and the
/// viewport destination.
///
/// Prefer a non-zero layer-surface Configure size. When the compositor sends
/// Configure(0,0) for an anchor-all surface (`set_size(0,0)`), fall back to
/// `wl_output::Mode` physical pixels divided by the effective output scale.
fn logical_surface_size(state: &OverlayState) -> (u32, u32) {
    if state.surface_w > 0 && state.surface_h > 0 {
        return (state.surface_w, state.surface_h);
    }
    if state.mode_w == 0 || state.mode_h == 0 {
        return (0, 0);
    }
    let scale = state.output_scale.max(1.0);
    let lw = ((state.mode_w as f64) / scale).round().max(1.0) as u32;
    let lh = ((state.mode_h as f64) / scale).round().max(1.0) as u32;
    (lw, lh)
}

fn owner_thread(rx: Receiver<WlOverlayCmd>) -> anyhow::Result<()> {
    let conn = Connection::connect_to_env()?;
    let mut queue = conn.new_event_queue::<OverlayState>();
    let qh = queue.handle();
    let _registry = conn.display().get_registry(&qh, ());

    let mut state = OverlayState::default();
    queue.roundtrip(&mut state)?;
    for _ in 0..3 {
        queue.roundtrip(&mut state)?;
    }

    let compositor = state
        .compositor
        .clone()
        .ok_or_else(|| anyhow::anyhow!("compositor does not expose wl_compositor"))?;
    let shm = state
        .shm
        .clone()
        .ok_or_else(|| anyhow::anyhow!("compositor does not expose wl_shm"))?;
    let layer_shell = state
        .layer_shell
        .clone()
        .ok_or_else(|| anyhow::anyhow!("compositor does not expose zwlr_layer_shell_v1"))?;
    let output = state
        .output
        .clone()
        .ok_or_else(|| anyhow::anyhow!("compositor exposed no wl_output"))?;

    // Build the layer surface: fullscreen, overlay layer, click-through.
    let surface = compositor.create_surface(&qh, ());

    // Wire up HiDPI scaling for the surface. `wp_viewporter` lets us map a
    // physical-pixel buffer back onto the logical surface-local rectangle
    // (required for fractional scales); `wp_fractional_scale_v1` tells us the
    // compositor's preferred fractional scale via `preferred_scale` events.
    // Both are optional — compositors without them fall back to integer
    // `wl_output.scale` + `set_buffer_scale`.
    let viewport = state.viewporter.as_ref().map(|vp| vp.get_viewport(&surface, &qh, ()));
    let fractional = state
        .fractional_mgr
        .as_ref()
        .map(|fm| fm.get_fractional_scale(&surface, &qh, ()));

    let layer_surface = layer_shell.get_layer_surface(
        &surface,
        Some(&output),
        Layer::Overlay,
        "cua-agent-cursor".to_string(),
        &qh,
        (),
    );
    // Anchor to all four edges = full screen.
    layer_surface.set_anchor(Anchor::Top | Anchor::Bottom | Anchor::Left | Anchor::Right);
    layer_surface.set_size(0, 0);
    layer_surface.set_exclusive_zone(-1);
    layer_surface.set_keyboard_interactivity(KeyboardInteractivity::None);

    // Click-through: empty input region.
    let region: WlRegion = compositor.create_region(&qh, ());
    surface.set_input_region(Some(&region));
    region.destroy();

    state.surface = Some(surface);
    state.layer_surface = Some(layer_surface);
    state.viewport = viewport;
    state.fractional = fractional;

    // First commit kicks off the configure handshake.
    if let Some(s) = state.surface.as_ref() {
        s.commit();
    }

    // Wait for the first configure event so we know the output dimensions
    // before drawing. Anchored full-screen surfaces often get Configure(0,0);
    // in that case Mode physical dims are enough to proceed.
    for _ in 0..10 {
        queue.roundtrip(&mut state)?;
        let has_size = (state.surface_w > 0 && state.surface_h > 0)
            || (state.mode_w > 0 && state.mode_h > 0);
        if state.configured && has_size {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }

    if !state.configured {
        anyhow::bail!("layer surface never received configure event");
    }
    let (lw, lh) = logical_surface_size(&state);
    if lw == 0 || lh == 0 {
        anyhow::bail!(
            "layer surface configured but no usable size (configure={}/{} mode={}/{})",
            state.surface_w,
            state.surface_h,
            state.mode_w,
            state.mode_h
        );
    }
    dbg(&format!(
        "configured: logical={lw}x{lh} surface={}/{} mode={}/{} scale={:.3}",
        state.surface_w,
        state.surface_h,
        state.mode_w,
        state.mode_h,
        state.output_scale
    ));

    // Main loop. Tick the render core at ~60Hz so motion + spring physics
    // + click pulse animate smoothly; redraw every tick when the cursor is
    // visible. Commands arriving via the channel update the render core
    // first, then the next tick paints the result.
    redraw(&mut state, &shm, &qh)?;
    queue.roundtrip(&mut state)?;

    let frame_dur = std::time::Duration::from_millis(16);
    let mut last_tick = Instant::now();
    loop {
        // Drain all pending commands without blocking.
        let mut shutdown = false;
        let mut had_cmd = false;
        loop {
            match rx.try_recv() {
                Ok(WlOverlayCmd::Shutdown) => {
                    shutdown = true;
                    break;
                }
                Ok(WlOverlayCmd::Cmd { cmd }) => {
                    had_cmd = true;
                    // Seed: if the cursor is still at the off-screen sentinel
                    // `(-200, -200)` from `RenderStateCore::new`, snap to a
                    // point near the MoveTo / SnapTo target so the spring
                    // animation starts on-screen. Mirrors X11 overlay.rs's
                    // `seed_start_if_sentinel` helper — without it, the
                    // spring oscillates around the sentinel and the cursor
                    // never reaches the screen.
                    let seed_target = match &cmd {
                        OverlayCommand::MoveTo { x, y, .. }
                        | OverlayCommand::SnapTo { x, y, .. }
                        | OverlayCommand::ClickPulse { x, y } => Some((*x, *y)),
                        _ => None,
                    };
                    if let Some((tx, ty)) = seed_target {
                        if state.core.pos.0 < -50.0 {
                            const SEED_OFFSET: f64 = 16.0;
                            let sx = (tx - SEED_OFFSET).max(2.0);
                            let sy = (ty - SEED_OFFSET).max(2.0);
                            state.core.pos = (sx, sy);
                        }
                    }
                    // apply_command_base consumes every variant the X11
                    // path handles. `move_to_snap_sentinel` / `click_pulse
                    // _sentinel_only` are both `false` here — same as X11.
                    let _ = state.core.apply_command_base(cmd, false, false);
                }
                Ok(WlOverlayCmd::Remove) => {
                    // Single-cursor overlay: removing the active cursor
                    // hides it. Multi-cursor wlroots support can layer on
                    // top of this in a follow-up if needed.
                    had_cmd = true;
                    state.core.visible = false;
                }
                Err(crossbeam_channel::TryRecvError::Empty) => break,
                Err(crossbeam_channel::TryRecvError::Disconnected) => {
                    shutdown = true;
                    break;
                }
            }
        }
        if shutdown {
            break;
        }

        // Tick animation forward and repaint only when pixels can change.
        // Unconditional 60Hz full-screen shm commits exhaust memfds (EMFILE)
        // and kill the owner thread under sustained idle.
        let now = Instant::now();
        let dt = now.duration_since(last_tick).as_secs_f64().min(0.05);
        last_tick = now;
        let arrived = state.core.tick_motion(dt);
        // needs_reattach: compositor released the buffer we last committed
        // while we were idle (dirty-only). Without a forced redraw the
        // surface has no buffer and the cursor vanishes.
        let needs_frame = arrived
            || state.core.path.is_some()
            || state.core.spring.is_some()
            || state.core.click_t.is_some()
            || state.core.idle_alpha < 0.999
            || had_cmd
            || state.needs_reattach;
        if state.configured && needs_frame {
            // Cap in-flight shm buffers so a slow compositor cannot pin
            // unbounded memfds open (EMFILE on the owner thread).
            const MAX_PENDING: usize = 3;
            while state.pending_buffers.len() >= MAX_PENDING {
                // Drain events (including buffer releases) non-blocking so
                // the compositor can free pending shm slots. prepare_read +
                // read reads from the socket without a sync round-trip.
                let _ = conn.flush();
                if let Some(guard) = queue.prepare_read() {
                    let _ = guard.read();
                }
                if let Err(e) = queue.dispatch_pending(&mut state) {
                    tracing::warn!("cua-overlay-wl dispatch_pending: {e}");
                    break;
                }
                if state.pending_buffers.len() >= MAX_PENDING {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                    let _ = conn.flush();
                    if let Some(guard) = queue.prepare_read() {
                        let _ = guard.read();
                    }
                    if let Err(e) = queue.dispatch_pending(&mut state) {
                        tracing::warn!("cua-overlay-wl dispatch_pending: {e}");
                        break;
                    }
                    if state.pending_buffers.len() >= MAX_PENDING {
                        break;
                    }
                }
            }
            // Prefer to skip a frame when the compositor is still holding too
            // many buffers, but never skip the frame that carries a new
            // command — that is the one that makes the cursor appear.
            if state.pending_buffers.len() < MAX_PENDING || had_cmd {
                // redraw failures (shm OOM, etc.) must not exit the owner
                // thread — log and keep serving later commands.
                if let Err(e) = redraw(&mut state, &shm, &qh) {
                    tracing::warn!("cua-overlay-wl redraw failed: {e}");
                }
            }
        }
        // CRITICAL: dispatch_pending alone does NOT read from the Wayland
        // socket — it only dispatches events already read into the queue.
        // Without reading the socket, buffer releases, reconfigure events,
        // and frame callbacks pile up in the kernel socket buffer unprocessed.
        // KWin withholds compositing until a reconfigure ack arrives, and
        // the pending buffer cap chokes the redraw path.
        //
        // roundtrip() fixes the socket-read problem but BLOCKS on a
        // wl_display.sync round-trip — if the compositor is slow to reply
        // (e.g. busy compositing a fullscreen layer surface), the loop
        // stalls and channel commands pile up, freezing the cursor.
        //
        // The correct pattern is prepare_read + read: this reads any
        // events sitting in the socket buffer WITHOUT blocking or sending
        // a sync request. Combined with conn.flush() to push our own
        // requests out, this keeps the event queue drained at 60Hz without
        // ever blocking the animation loop.
        let _ = conn.flush();
        if let Some(guard) = queue.prepare_read() {
            match guard.read() {
                Ok(_) => {
                    if let Err(e) = queue.dispatch_pending(&mut state) {
                        tracing::warn!("cua-overlay-wl dispatch_pending: {e}");
                    }
                }
                Err(e) => {
                    // WouldBlock = no events on socket (normal at 60Hz);
                    // anything else is a protocol fault worth logging.
                    let msg = e.to_string();
                    if !msg.contains("WouldBlock") {
                        tracing::warn!("cua-overlay-wl read: {msg}");
                    }
                }
            }
        }

        // Sleep for the remainder of the frame budget so the loop doesn't
        // spin. Channel-driven wakeups would be lower-latency, but layer
        // overlays only need to keep up with display refresh.
        let elapsed = last_tick.elapsed();
        if elapsed < frame_dur {
            std::thread::sleep(frame_dur - elapsed);
        }
    }

    if let Some(ls) = state.layer_surface.take() {
        ls.destroy();
    }
    if let Some(s) = state.surface.take() {
        s.destroy();
    }
    queue.roundtrip(&mut state)?;
    Ok(())
}

/// Render one cursor frame into a fresh wl_shm ARGB8888 buffer and attach
/// it to the layer surface.
///
/// Pipeline:
/// 1. Allocate a memfd-backed wl_shm pool sized at the output's *physical*
///    pixel dimensions (`logical × output_scale`).
/// 2. Paint the cross-platform cursor (bloom + click pulse + gradient
///    arrow) into a `tiny_skia::Pixmap` at that physical resolution,
///    passing `output_scale` as `backing_scale` so `core.pos` (logical) is
///    mapped to the correct physical pixel — this is the crux of the HiDPI
///    fix: previously the buffer was logical-sized with `backing_scale=1.0`,
///    so a cursor at logical `(x,y)` landed at half the screen on a 2× output.
/// 3. Channel-swap RGBA → BGRA into the wl_shm buffer (wl_shm Argb8888
///    is little-endian BGRA in memory). This is the inverse of the swap
///    in `ext_screencopy::encode_buffer_to_png`.
/// 4. If `output_scale != 1.0`, use `wp_viewport` to map the physical
///    buffer onto the logical surface-local rectangle (destination =
///    `surface_w × surface_h`), as required by `wp-fractional-scale-v1` and
///    correct for any scale. Without a viewport, fall back to integer
///    `set_buffer_scale`.
/// 5. Attach + damage + commit on the layer surface.
///
/// When the cursor is hidden (`core.visible == false`, idle-faded, or
/// off-screen sentinel) the pixmap is all zeros — the surface remains
/// transparent and click-through.
fn redraw(
    state: &mut OverlayState,
    shm: &WlShm,
    qh: &QueueHandle<OverlayState>,
) -> anyhow::Result<()> {
    let Some(surface) = state.surface.as_ref() else {
        return Ok(());
    };
    // Logical surface-local size: Configure wins; Mode÷scale is the fallback
    // for anchor-all surfaces that get Configure(0,0).
    let (lw, lh) = logical_surface_size(state);
    if lw == 0 || lh == 0 {
        return Ok(());
    }
    // Effective scale: fractional preferred_scale wins, else integer
    // wl_output.scale, else 1.0.
    let scale = state.output_scale.max(1.0);
    // Physical buffer dimensions. paint_cursor expects a backing_scale-sized
    // pixmap so the logical cursor position maps to the right physical pixel.
    let w = (lw as f64 * scale).round().max(1.0) as u32;
    let h = (lh as f64 * scale).round().max(1.0) as u32;
    let stride = w as i32 * 4;
    let size = (stride as usize) * (h as usize);

    // Reuses the same anon_shm pattern as the screencopy path in mod.rs.
    let (fd, ptr) =
        super::anon_shm(size).map_err(|e| anyhow::anyhow!("overlay shm allocation failed: {e}"))?;

    // SAFETY: ptr came from mmap of `size` bytes, lifetime bounded to this
    // function.
    let pixels: &mut [u8] = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, size) };

    // Paint the cursor into a tiny_skia pixmap. paint_cursor early-returns
    // when the cursor is hidden / off-screen / idle-faded, so the pixmap
    // is left fully transparent in those cases (which is also what we want
    // for the click-through layer surface).
    let pm_result = tiny_skia::Pixmap::new(w, h);
    let mut pm = match pm_result {
        Some(p) => p,
        // tiny_skia::Pixmap::new only fails on OOM at sizes that fit u32.
        // We refuse to fall back to a 1x1 pixmap because the subsequent
        // RGBA → BGRA loop indexes `src[i+3]` over the full `size` range —
        // a 1x1 source would crash. Surface the allocation failure
        // properly instead.
        None => anyhow::bail!(
            "tiny_skia::Pixmap::new({w}, {h}) failed — out of memory for the overlay buffer"
        ),
    };
    // `core.pos` is in logical screen points; `paint_cursor` multiplies it
    // by `backing_scale` to land at the right physical pixel in the pixmap.
    cursor_overlay::paint_cursor(&mut pm, &state.core, 0.0, 0.0, None, scale as f32);

    // CUA_OVERLAY_DEBUG=1 paints a 60x60 magenta square at the cursor's
    // current pos on top of the gradient arrow. Useful when validating
    // layer-shell visibility on a new compositor — the gradient arrow is
    // small at native scale and easy to miss in a screenshot, while the
    // magenta block is impossible to miss. Position is logical→physical.
    if std::env::var_os("CUA_OVERLAY_DEBUG").is_some() {
        let cx = (state.core.pos.0 * scale) as i32;
        let cy = (state.core.pos.1 * scale) as i32;
        let half = 30i32;
        for dy in -half..half {
            for dx in -half..half {
                let px = cx + dx;
                let py = cy + dy;
                if px < 0 || py < 0 || px >= w as i32 || py >= h as i32 {
                    continue;
                }
                let off = ((py as usize) * (w as usize) + (px as usize)) * 4;
                pm.data_mut()[off] = 0xFF; // R
                pm.data_mut()[off + 1] = 0x00; // G
                pm.data_mut()[off + 2] = 0xFF; // B
                pm.data_mut()[off + 3] = 0xFF; // A
            }
        }
    }

    // RGBA → BGRA channel swap. tiny_skia stores pixels as RGBA8888
    // (premultiplied); wl_shm Argb8888 is little-endian = BGRA in memory.
    // Mirrors the inverse swap in ext_screencopy::encode_buffer_to_png.
    let src = pm.data();
    for i in (0..size).step_by(4) {
        // pm.data() is already RGBA premultiplied; just swap R↔B.
        pixels[i] = src[i + 2]; // B ← R
        pixels[i + 1] = src[i + 1]; // G
        pixels[i + 2] = src[i]; // R ← B
        pixels[i + 3] = src[i + 3]; // A
    }

    use std::os::fd::AsFd as _;
    let pool_fd = unsafe { super::borrowed_fd(fd) };
    let pool: WlShmPool = shm.create_pool(pool_fd.as_fd(), size as i32, qh, ());
    let buffer: WlBuffer = pool.create_buffer(
        0,
        w as i32,
        h as i32,
        stride,
        wl_shm::Format::Argb8888,
        qh,
        (),
    );

    // Track the (mmap, fd) by buffer object id so the wl_buffer.release
    // event Dispatch handler can clean up exactly when the compositor is
    // done with the underlying memory — no leak, no use-after-free.
    let buffer_id = buffer.id().protocol_id();
    state.pending_buffers.insert(buffer_id, (ptr, size, fd));
    // Remember which buffer is currently on the surface so a later
    // wl_buffer.release of it can force a reattach (dirty-only redraw
    // would otherwise leave the surface buffer-less → blank cursor).
    state.current_buffer_id = Some(buffer_id);
    state.needs_reattach = false;

    dbg(&format!(
        "redraw logical={lw}x{lh} scale={scale:.3} phys={w}x{h} stride={stride} buf_id={buffer_id} pos=({:.1},{:.1}) visible={}",
        state.core.pos.0, state.core.pos.1, state.core.visible
    ));

    // Map the physical buffer onto the logical surface-local rectangle. The
    // viewport destination is in surface-local (logical) coords; the source
    // covers the full physical buffer. This is the only correct way to
    // submit a fractional-scaled buffer (`wp-fractional-scale-v1` requires
    // `wp_viewporter`), and it works for integer scales too. Without a
    // viewport, fall back to integer `set_buffer_scale`.
    if scale != 1.0 {
        if let Some(vp) = state.viewport.as_ref() {
            vp.set_source(0.0, 0.0, w as f64, h as f64);
            vp.set_destination(lw as i32, lh as i32);
        } else {
            surface.set_buffer_scale(state.output_scale_int.max(1));
        }
    } else {
        surface.set_buffer_scale(1);
    }

    surface.attach(Some(&buffer), 0, 0);
    surface.damage_buffer(0, 0, w as i32, h as i32);
    surface.commit();
    pool.destroy();
    Ok(())
}

// ── Wayland Dispatch impls ───────────────────────────────────────────────

impl Dispatch<wl_registry::WlRegistry, ()> for OverlayState {
    fn event(
        state: &mut Self,
        registry: &wl_registry::WlRegistry,
        event: wl_registry::Event,
        _data: &(),
        _conn: &Connection,
        qh: &QueueHandle<Self>,
    ) {
        if let wl_registry::Event::Global {
            name,
            interface,
            version,
        } = event
        {
            match interface.as_str() {
                "wl_compositor" => {
                    state.compositor =
                        Some(registry.bind::<WlCompositor, _, _>(name, version.min(6), qh, ()));
                }
                "wl_shm" => {
                    state.shm = Some(registry.bind::<WlShm, _, _>(name, version.min(1), qh, ()));
                }
                "wl_output" => {
                    if state.output.is_none() {
                        state.output =
                            Some(registry.bind::<WlOutput, _, _>(name, version.min(4), qh, ()));
                    }
                }
                "wp_viewporter" => {
                    if state.viewporter.is_none() {
                        state.viewporter = Some(
                            registry.bind::<WpViewporter, _, _>(name, version.min(1), qh, ()),
                        );
                    }
                }
                "wp_fractional_scale_manager_v1" => {
                    if state.fractional_mgr.is_none() {
                        state.fractional_mgr = Some(
                            registry.bind::<WpFractionalScaleManagerV1, _, _>(
                                name,
                                version.min(1),
                                qh,
                                (),
                            ),
                        );
                    }
                }
                "zwlr_layer_shell_v1" => {
                    state.layer_shell =
                        Some(registry.bind::<ZwlrLayerShellV1, _, _>(name, version.min(4), qh, ()));
                }
                _ => {}
            }
        }
    }
}

impl Dispatch<WlCompositor, ()> for OverlayState {
    fn event(
        _state: &mut Self,
        _: &WlCompositor,
        _: <WlCompositor as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlShm, ()> for OverlayState {
    fn event(
        _state: &mut Self,
        _: &WlShm,
        _: <WlShm as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlOutput, ()> for OverlayState {
    fn event(
        state: &mut Self,
        _: &WlOutput,
        event: <WlOutput as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        use wayland_client::protocol::wl_output;
        match event {
            wl_output::Event::Mode { width, height, .. } => {
                // Physical screen pixels. Kept separate from surface_w/h so a
                // later Mode event cannot overwrite a real logical Configure
                // size. Used only when Configure leaves surface dims at 0.
                if width > 0 && height > 0 {
                    state.mode_w = width as u32;
                    state.mode_h = height as u32;
                    dbg(&format!("wl_output.mode = {width}x{height}"));
                }
            }
            wl_output::Event::Scale { factor } => {
                // Integer compositor scale (e.g. 2 on a retina panel). Used
                // directly when no fractional-scale object exists, and as the
                // `set_buffer_scale` fallback for compositors without
                // `wp_viewporter`.
                if factor > 0 {
                    state.output_scale_int = factor;
                    if state.fractional.is_none() {
                        state.output_scale = factor as f64;
                    }
                    dbg(&format!("wl_output.scale = {factor}"));
                }
            }
            _ => {}
        }
    }
}

impl Dispatch<ZwlrLayerShellV1, ()> for OverlayState {
    fn event(
        _state: &mut Self,
        _: &ZwlrLayerShellV1,
        _: <ZwlrLayerShellV1 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<ZwlrLayerSurfaceV1, ()> for OverlayState {
    fn event(
        state: &mut Self,
        layer: &ZwlrLayerSurfaceV1,
        event: <ZwlrLayerSurfaceV1 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let zwlr_layer_surface_v1::Event::Configure {
            serial,
            width,
            height,
        } = event
        {
            layer.ack_configure(serial);
            if width > 0 {
                state.surface_w = width;
            }
            if height > 0 {
                state.surface_h = height;
            }
            state.configured = true;
        }
    }
}

impl Dispatch<WlSurface, ()> for OverlayState {
    fn event(
        _: &mut Self,
        _: &WlSurface,
        _: <WlSurface as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlShmPool, ()> for OverlayState {
    fn event(
        _: &mut Self,
        _: &WlShmPool,
        _: <WlShmPool as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WlBuffer, ()> for OverlayState {
    fn event(
        state: &mut Self,
        buffer: &WlBuffer,
        event: <WlBuffer as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        use wayland_client::protocol::wl_buffer;
        if matches!(event, wl_buffer::Event::Release) {
            // Compositor is done with the underlying mmap. Free it +
            // close the memfd + destroy the wayland object.
            let id = buffer.id().protocol_id();
            if let Some((ptr, size, fd)) = state.pending_buffers.remove(&id) {
                super::cleanup_mmap(ptr, size, fd);
            }
            // If this was the buffer currently on the surface, the surface
            // is now buffer-less until the next commit. Force a redraw so
            // dirty-only idle does not leave a blank cursor.
            if state.current_buffer_id == Some(id) {
                state.current_buffer_id = None;
                state.needs_reattach = true;
            }
            buffer.destroy();
        }
    }
}

impl Dispatch<WpViewporter, ()> for OverlayState {
    fn event(
        _: &mut Self,
        _: &WpViewporter,
        _: <WpViewporter as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WpViewport, ()> for OverlayState {
    fn event(
        _: &mut Self,
        _: &WpViewport,
        _: <WpViewport as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WpFractionalScaleManagerV1, ()> for OverlayState {
    fn event(
        _: &mut Self,
        _: &WpFractionalScaleManagerV1,
        _: <WpFractionalScaleManagerV1 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}

impl Dispatch<WpFractionalScaleV1, ()> for OverlayState {
    fn event(
        state: &mut Self,
        _: &WpFractionalScaleV1,
        event: <WpFractionalScaleV1 as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
        if let wp_fractional_scale_v1::Event::PreferredScale { scale } = event {
            // `scale` is in 1/120 of a unit (120 == 1.0, 180 == 1.5, 210 ==
            // 1.75). This is the authoritative fractional scale for the
            // surface and overrides the integer `wl_output.scale`.
            if scale > 0 {
                let factor = scale as f64 / 120.0;
                if factor > 0.0 {
                    state.output_scale = factor;
                    dbg(&format!(
                        "wp_fractional_scale.preferred_scale = {scale} ({factor:.3})"
                    ));
                }
            }
        }
    }
}

impl Dispatch<WlRegion, ()> for OverlayState {
    fn event(
        _: &mut Self,
        _: &WlRegion,
        _: <WlRegion as wayland_client::Proxy>::Event,
        _: &(),
        _: &Connection,
        _: &QueueHandle<Self>,
    ) {
    }
}
