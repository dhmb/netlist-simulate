mod app;
mod graph;
mod graph_viewer;
mod sky130;

use crate::app::App;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0]),
        ..Default::default()
    };

    eframe::run_native(
        "netlist-simulate",
        native_options,
        Box::new(|cx| Ok(Box::new(App::new(cx)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn get_canvas_element() -> Option<web_sys::HtmlCanvasElement> {
    use eframe::wasm_bindgen::JsCast;

    let document = web_sys::window()?.document()?;
    let canvas = document.get_element_by_id("netlist_simulate")?;
    canvas.dyn_into::<web_sys::HtmlCanvasElement>().ok()
}

#[cfg(target_arch = "wasm32")]
fn main() {
    let canvas = get_canvas_element().expect("Failed to find canvas with id 'netlist_simulate'");

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async move {
        eframe::WebRunner::new()
            .start(
                canvas.clone(),
                web_options,
                Box::new(|cx| Ok(Box::new(App::new(cx)))),
            )
            .await
            .expect("failed to start eframe");

        canvas.set_attribute("data-loaded", "true").ok();
        if let Some(document) = web_sys::window().and_then(|w| w.document())
            && let Some(loading_text) = document.get_element_by_id("loading_text")
        {
            loading_text.remove();
        }
    });
}
