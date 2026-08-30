use egui::{Color32, Rect, Ui, emath::TSTransform};

use egui_snarl::{
    InPin, NodeId, OutPin, Snarl,
    ui::{AnyPins, PinInfo, SnarlViewer},
};
use serde::{Deserialize, Serialize};

use crate::app::{STYLE_MAX_SCALE, STYLE_MIN_SCALE, label};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CellNode {
    name: String,
    input_names: Vec<String>,
    output_names: Vec<String>,
}

impl CellNode {
    pub fn new(name: String, input_names: Vec<String>, output_names: Vec<String>) -> Self {
        CellNode {
            name,
            input_names,
            output_names,
        }
    }
}

/// Default for [`TransformState::viewport_rect`] — see its doc comment for
/// why this is skipped rather than persisted.
fn rect_nothing() -> Rect {
    Rect::NOTHING
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransformState {
    /// Whether `current` has been pushed into the live Snarl widget yet in
    /// this process. Deliberately not persisted (`current` itself is the
    /// thing that gets saved/restored) — it only exists so that, once per
    /// app run, the transform restored from storage gets applied to the
    /// widget instead of it starting at identity.
    #[serde(skip)]
    initial_applied: bool,
    current: TSTransform,
    /// The last UI rect this viewer was shown in. Purely transient (it's
    /// overwritten every frame by [`GraphViewer::set_viewport_rect`]) and
    /// its unset value, [`Rect::NOTHING`], is made of infinities — which
    /// `serde_json` can't round-trip (it serializes to `null`, then fails
    /// to deserialize back into an `f32`). A viewer that's never actually
    /// rendered (e.g. a graph action's, before its output is shown) would
    /// otherwise save that `null` and fail to load on the next run.
    #[serde(skip, default = "rect_nothing")]
    viewport_rect: Rect,
}

impl Default for TransformState {
    fn default() -> Self {
        TransformState {
            initial_applied: false,
            current: TSTransform::IDENTITY,
            viewport_rect: Rect::NOTHING,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum ZoomToFitState {
    #[default]
    NoZoomToFit,
    CalculateBoundingRect(Rect),
    FinishedCalculateBoundingRect(Rect),
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct GraphViewer {
    pub zoom_to_fit_state: ZoomToFitState,
    pub transform: TransformState,
    //pub style: SnarlStyle,
}

impl GraphViewer {
    #[inline]
    pub fn logic(&mut self) {
        if let ZoomToFitState::CalculateBoundingRect(bounding_rect) = self.zoom_to_fit_state {
            self.zoom_to_fit_state = ZoomToFitState::FinishedCalculateBoundingRect(bounding_rect)
        }
    }

    pub fn request_zoom_to_fit(&mut self) {
        self.zoom_to_fit_state = ZoomToFitState::CalculateBoundingRect(Rect::NOTHING)
    }

    /// Updates the viewport rect used to fit the transform when zooming to
    /// fit. Called every frame with the current available UI rect, since
    /// that can change (e.g. window resize).
    pub fn set_viewport_rect(&mut self, viewport_rect: Rect) {
        self.transform.viewport_rect = viewport_rect;
    }
}

const CELL_COLOR: Color32 = Color32::from_rgb(0x40, 0x50, 0x60);

impl SnarlViewer<CellNode> for GraphViewer {
    #[inline]
    fn current_transform(&mut self, transform: &mut TSTransform, _snarl: &mut Snarl<CellNode>) {
        // Apply the persisted transform once per app run, so a restored
        // session doesn't snap back to identity on startup.
        if !self.transform.initial_applied {
            self.transform.initial_applied = true;
            *transform = self.transform.current;
        }

        // Set Zoom-to-fit state if requested.
        if let ZoomToFitState::FinishedCalculateBoundingRect(bounding_rect) = self.zoom_to_fit_state
        {
            self.zoom_to_fit_state = ZoomToFitState::NoZoomToFit;

            if let Some(fit) = fit_transform(
                bounding_rect,
                self.transform.viewport_rect,
                STYLE_MIN_SCALE,
                STYLE_MAX_SCALE,
            ) {
                *transform = fit;
            }
        }

        // Copy transform for persistence
        self.transform.current = *transform;
    }

    #[inline]
    fn final_node_rect(
        &mut self,
        node: NodeId,
        rect: Rect,
        ui: &mut Ui,
        snarl: &mut Snarl<CellNode>,
    ) {
        let _ = (node, ui, snarl);

        if let ZoomToFitState::CalculateBoundingRect(bounding_rect) = self.zoom_to_fit_state {
            self.zoom_to_fit_state =
                ZoomToFitState::CalculateBoundingRect(bounding_rect.union(rect))
        }
    }

    #[inline]
    fn connect(&mut self, _from: &OutPin, _to: &InPin, _snarl: &mut Snarl<CellNode>) {
        return;
    }

    #[inline]
    fn disconnect(&mut self, _from: &OutPin, _to: &InPin, _snarl: &mut Snarl<CellNode>) {
        // Not allowed to disconnect edges manually
    }

    #[inline]
    fn drop_inputs(&mut self, _pin: &InPin, _snarl: &mut Snarl<CellNode>) {
        // Not allowed to disconnect edges manually
    }

    #[inline]
    fn drop_outputs(&mut self, _pin: &OutPin, _snarl: &mut Snarl<CellNode>) {
        // Not allowed to disconnect edges manually
    }

    fn title(&mut self, node: &CellNode) -> String {
        node.name.clone()
    }

    fn inputs(&mut self, node: &CellNode) -> usize {
        node.input_names.len()
    }

    fn outputs(&mut self, node: &CellNode) -> usize {
        node.output_names.len()
    }

    #[allow(clippy::too_many_lines)]
    #[allow(refining_impl_trait)]
    fn show_input(&mut self, pin: &InPin, ui: &mut Ui, snarl: &mut Snarl<CellNode>) -> PinInfo {
        let node = &snarl[pin.id.node];
        label(ui, &node.input_names[pin.id.input]);
        PinInfo::circle().with_fill(CELL_COLOR)
    }

    #[allow(refining_impl_trait)]
    fn show_output(&mut self, pin: &OutPin, ui: &mut Ui, snarl: &mut Snarl<CellNode>) -> PinInfo {
        let node = &snarl[pin.id.node];
        label(ui, &node.output_names[pin.id.output]);
        PinInfo::circle().with_fill(CELL_COLOR)
    }

    fn has_graph_menu(&mut self, _pos: egui::Pos2, _snarl: &mut Snarl<CellNode>) -> bool {
        false
    }

    fn show_graph_menu(&mut self, _pos: egui::Pos2, _ui: &mut Ui, _snarl: &mut Snarl<CellNode>) {
        // Empty
    }

    fn has_dropped_wire_menu(&mut self, _src_pins: AnyPins, _snarl: &mut Snarl<CellNode>) -> bool {
        false
    }

    fn show_dropped_wire_menu(
        &mut self,
        _pos: egui::Pos2,
        _ui: &mut Ui,
        _src_pins: AnyPins,
        _snarl: &mut Snarl<CellNode>,
    ) {
        // Empty
    }

    fn has_node_menu(&mut self, _node: &CellNode) -> bool {
        false
    }

    fn show_node_menu(
        &mut self,
        _node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        _ui: &mut Ui,
        _snarl: &mut Snarl<CellNode>,
    ) {
        // Empty
    }

    fn has_on_hover_popup(&mut self, _: &CellNode) -> bool {
        true
    }

    fn show_on_hover_popup(
        &mut self,
        node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        ui: &mut Ui,
        snarl: &mut Snarl<CellNode>,
    ) {
        label(
            ui,
            format!("Netlist cell loaded from graph.json: {}", snarl[node].name),
        );
    }

    fn header_frame(
        &mut self,
        frame: egui::Frame,
        _node: NodeId,
        _inputs: &[InPin],
        _outputs: &[OutPin],
        _snarl: &Snarl<CellNode>,
    ) -> egui::Frame {
        frame
    }
}

fn fit_transform(
    view: Rect,
    viewport: Rect,
    min_scale: f32,
    max_scale: f32,
) -> Option<TSTransform> {
    if !view.is_finite() || !viewport.is_finite() {
        return None;
    }

    let view = view.expand(100.0);
    let scaling = (viewport.size() / view.size())
        .min_elem()
        .clamp(min_scale, max_scale);

    Some(TSTransform {
        scaling,
        translation: viewport.center().to_vec2() - view.center().to_vec2() * scaling,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `GraphViewer` that has never had a Snarl widget render it (as
    /// happens for a graph action's viewer before its output is shown)
    /// keeps `TransformState::viewport_rect` at its infinity-filled
    /// `Rect::NOTHING` default. `serde_json` can't round-trip infinities
    /// (it serializes them as `null`, which then fails to deserialize back
    /// into an `f32`), so that field must be skipped rather than saved.
    #[test]
    fn default_graph_viewer_round_trips_through_json() {
        let viewer = GraphViewer::default();

        let json = serde_json::to_string(&viewer).unwrap();
        let restored: GraphViewer = serde_json::from_str(&json).unwrap();

        assert_eq!(restored.transform.viewport_rect, Rect::NOTHING);
    }
}
