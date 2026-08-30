use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[cfg(target_arch = "wasm32")]
use crate::sky130::parse_gds;
#[cfg(not(target_arch = "wasm32"))]
use crate::sky130::read_gds;
use crate::{
    graph::{
        AmbiguousRemoval, BusConstraint, Cell, CellType, CnfEncoder, Graph, Simulation, Simulator,
        SystemSolution, split_bus_pin,
    },
    graph_viewer::{CellNode, GraphViewer},
    sky130::{DrivenPin, DummyInput, Layout, Sky130Options},
};
use batsat::{BasicSolver, Lit, SolverInterface, Var, lbool};
use eframe::CreationContext;
use egui::{Id, pos2};
use egui_icons::icons::{
    ICON_ADD, ICON_ARROW_DOWNWARD, ICON_ARROW_UPWARD, ICON_CLOSE, ICON_REFRESH,
};
use egui_snarl::{
    InPinId, OutPinId, Snarl,
    ui::{NodeLayout, PinPlacement, SnarlStyle, SnarlWidget},
};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum LayoutMode {
    #[default]
    Grid,
    Centroid,
}

const fn theme_fills(theme: egui::Theme) -> (egui::Color32, egui::Color32) {
    match theme {
        egui::Theme::Dark => (egui::Color32::from_gray(30), egui::Color32::from_gray(40)),
        egui::Theme::Light => (egui::Color32::from_gray(250), egui::Color32::from_gray(235)),
    }
}

fn load_json<T: serde::de::DeserializeOwned>(
    storage: Option<&dyn eframe::Storage>,
    key: &str,
    startup_error: &mut Option<String>,
) -> Option<T> {
    let text = storage?.get_string(key)?;
    match serde_json::from_str(&text) {
        Ok(value) => Some(value),
        Err(err) => {
            let message = format!("Failed to load saved {key}: {err}");
            *startup_error = Some(match startup_error.take() {
                Some(existing) => format!("{existing}\n{message}"),
                None => message,
            });
            None
        }
    }
}

pub const STYLE_MIN_SCALE: f32 = 0.005;
pub const STYLE_MAX_SCALE: f32 = 1.0;

pub fn label(ui: &mut egui::Ui, text: impl Into<egui::WidgetText>) {
    ui.add(egui::Label::new(text).selectable(false));
}

/// A left-aligned label occupying exactly `width`, truncating rather than
/// growing past it (see [`editor_field_width`] for why anything in the
/// right panel that can widen its column is a problem). The full text is
/// still readable on hover, for whatever the truncation cut off.
fn fixed_width_label(ui: &mut egui::Ui, text: &str, width: f32) {
    ui.allocate_ui_with_layout(
        egui::vec2(width, ui.spacing().interact_size.y),
        egui::Layout::left_to_right(egui::Align::Center),
        |ui| {
            ui.set_min_width(width);
            ui.add(egui::Label::new(text).truncate().selectable(false))
                .on_hover_text(text);
        },
    );
}

/// A row title that truncates to whatever width is left for it, with the
/// full text on hover. Used for the action list's row headers, where a
/// long title (`"Merge boolean functions by cone of influence"`) would
/// otherwise run underneath the row's trailing icon buttons.
fn truncating_label(ui: &mut egui::Ui, text: &str) {
    ui.add(egui::Label::new(text).truncate().selectable(false))
        .on_hover_text(text);
}

/// A left-aligned column heading of exactly `width`, for labelling the
/// text field below it: it shares the field's left edge, and truncates
/// rather than widening its column (see [`editor_field_width`]).
fn column_heading(ui: &mut egui::Ui, text: &str, width: f32) {
    fixed_width_label(ui, text, width);
}

/// How the Graph information panel breaks the graph's cells down: one
/// row per distinct cell name (the default, and the finer view), or one
/// row per [`CellCategory`], each the sum of the cell-name rows that fall
/// into it. An app-wide viewing preference rather than a per-project one:
/// it's about how the reader likes the summary, not about any graph.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum GraphInformationGrouping {
    #[default]
    CellType,
    Category,
}

impl GraphInformationGrouping {
    const ALL: [GraphInformationGrouping; 2] = [
        GraphInformationGrouping::CellType,
        GraphInformationGrouping::Category,
    ];

    fn label(self) -> &'static str {
        match self {
            GraphInformationGrouping::CellType => "Cell type",
            GraphInformationGrouping::Category => "Category",
        }
    }
}

/// The panel's bottom section: how big the graph being viewed is (its
/// cell and edge counts) and what it's made of — per `grouping`, either
/// every distinct cell name with its category and how many cells carry
/// it, or just the categories and their totals; most frequent first
/// either way. Purely informational — nothing here edits the graph;
/// `grouping` is the one thing it changes, and only at the reader's ask.
fn show_graph_information(
    ui: &mut egui::Ui,
    graph: &Graph,
    grouping: &mut GraphInformationGrouping,
) {
    ui.heading("Graph information");
    ui.separator();

    label(ui, format!("Nodes: {}", graph.cells.len()));
    label(ui, format!("Edges: {}", graph.edge_count()));
    ui.separator();

    ui.horizontal(|ui| {
        label(ui, "Group by:");
        for option in GraphInformationGrouping::ALL {
            ui.selectable_value(grouping, option, option.label());
        }
    });

    /// Room for the count column: wide enough for the biggest counts
    /// these graphs reach, so the cell name column takes what's left.
    const COUNT_WIDTH: f32 = 48.0;
    /// Room for the category column: wide enough for the longest
    /// category label ("Boolean").
    const CATEGORY_WIDTH: f32 = 60.0;
    /// `TextEdit`'s own floor, reused as this table's (see
    /// [`editor_field_width`]).
    const MIN_NAME_WIDTH: f32 = 24.0;

    // The scrollbar below is reserved for as well as the count column:
    // rows wider than the panel would widen the panel itself on the next
    // frame, exactly as an over-wide text field does. Grouped by
    // category there's no name column, so the category column takes
    // its room instead.
    let reserved = CATEGORY_WIDTH
        + COUNT_WIDTH
        + 2.0 * ui.spacing().item_spacing.x
        + ui.spacing().scroll.allocated_width();
    let name_width = (ui.available_width() - reserved).max(MIN_NAME_WIDTH);
    let grouped_category_width = name_width + ui.spacing().item_spacing.x + CATEGORY_WIDTH;

    // Takes whatever height the panel has left (`auto_shrink` off in both
    // directions), scrolling within it — so the table fills the space
    // below the editor above instead of stopping at its content, and a
    // graph with many distinct cell names still can't push the section
    // past the bottom of the window.
    egui::ScrollArea::vertical()
        .id_salt("graph_information_cell_types")
        .auto_shrink([false, false])
        .show(ui, |ui| match *grouping {
            GraphInformationGrouping::CellType => {
                let counts = graph.cell_type_counts();
                egui::Grid::new("graph_information_cell_type_counts")
                    .num_columns(3)
                    .striped(true)
                    .show(ui, |ui| {
                        column_heading(ui, "Cell type", name_width);
                        column_heading(ui, "Category", CATEGORY_WIDTH);
                        column_heading(ui, "Count", COUNT_WIDTH);
                        ui.end_row();

                        for entry in &counts {
                            fixed_width_label(ui, &entry.name, name_width);
                            fixed_width_label(ui, entry.category.label(), CATEGORY_WIDTH);
                            fixed_width_label(ui, &entry.count.to_string(), COUNT_WIDTH);
                            ui.end_row();
                        }
                    });
            }
            GraphInformationGrouping::Category => {
                let counts = graph.cell_category_counts();
                egui::Grid::new("graph_information_cell_category_counts")
                    .num_columns(2)
                    .striped(true)
                    .show(ui, |ui| {
                        column_heading(ui, "Category", grouped_category_width);
                        column_heading(ui, "Count", COUNT_WIDTH);
                        ui.end_row();

                        for entry in &counts {
                            fixed_width_label(ui, entry.category.label(), grouped_category_width);
                            fixed_width_label(ui, &entry.count.to_string(), COUNT_WIDTH);
                            ui.end_row();
                        }
                    });
            }
        });
}

/// The right panel's notice for a loaded layout: every input the reader
/// had to invent, and what each one drives — see [`DummyInput`] for why
/// an invented input is a finding about the layout rather than a fix.
/// Each driven pin comes with its cell's centroid, so the dangling wire
/// can be found in the layout. Says so explicitly when there are none,
/// since silence would read the same as not having checked.
fn show_dummy_inputs(ui: &mut egui::Ui, dummy_inputs: &[DummyInput]) {
    if dummy_inputs.is_empty() {
        label(
            ui,
            egui::RichText::new("Every cell input is driven; no dummy inputs were needed.").weak(),
        );
        return;
    }

    let warning = egui::Color32::from_rgb(220, 140, 40);
    ui.colored_label(
        warning,
        format!(
            "{} net{} drive cell inputs but are driven by nothing in the layout. Each was given \
             an input of its own:",
            dummy_inputs.len(),
            if dummy_inputs.len() == 1 { "" } else { "s" }
        ),
    );
    egui::ScrollArea::vertical()
        .id_salt("dummy_inputs")
        .auto_shrink([false, true])
        .max_height(160.0)
        .show(ui, |ui| {
            for dummy in dummy_inputs {
                egui::CollapsingHeader::new(
                    egui::RichText::new(format!(
                        "{} — drives {} pin{}",
                        dummy.label,
                        dummy.drives.len(),
                        if dummy.drives.len() == 1 { "" } else { "s" }
                    ))
                    .color(warning),
                )
                .default_open(dummy_inputs.len() == 1)
                .show(ui, |ui| {
                    for DrivenPin { pin, centroid } in &dummy.drives {
                        ui.horizontal(|ui| {
                            ui.add_space(12.0);
                            label(ui, pin);
                            label(
                                ui,
                                egui::RichText::new(format!(
                                    "({:.3}, {:.3}) µm",
                                    centroid.0, centroid.1
                                ))
                                .weak(),
                            );
                        });
                    }
                });
            }
        });
    ui.separator();
}

/// The width to give each of `columns` text fields on one row of the
/// right-side editor panel, reserving room for a trailing remove button.
///
/// A `TextEdit` asks for `spacing.text_edit_width` (280) by default,
/// which two-per-row overflows the panel — and [`egui::Panel`] takes its
/// next frame's width from what its contents actually occupied, so the
/// overflow shows up as the panel widening itself on redraw.
fn editor_field_width(ui: &egui::Ui, columns: usize) -> f32 {
    /// Generous upper bound on the frameless [`ICON_CLOSE`] button ending
    /// each row. Reserving *less* than the button really takes would leave
    /// every row a little wider than the panel, and so re-trigger the
    /// self-widening this reserve exists to prevent.
    const REMOVE_BUTTON_WIDTH: f32 = 32.0;
    /// `TextEdit`'s own floor; going below it doesn't buy any more room.
    const MIN_FIELD_WIDTH: f32 = 24.0;

    let spacing = ui.spacing().item_spacing.x * columns as f32;
    ((ui.available_width() - REMOVE_BUTTON_WIDTH - spacing) / columns as f32).max(MIN_FIELD_WIDTH)
}

fn default_style(theme: egui::Theme) -> SnarlStyle {
    let (node_fill, bg_fill) = theme_fills(theme);
    SnarlStyle {
        node_layout: Some(NodeLayout::coil()),
        pin_placement: Some(PinPlacement::Edge),
        pin_size: Some(7.0),
        min_scale: Some(STYLE_MIN_SCALE),
        max_scale: Some(STYLE_MAX_SCALE),
        node_frame: Some(egui::Frame {
            inner_margin: egui::Margin::same(8),
            outer_margin: egui::Margin {
                left: 0,
                right: 0,
                top: 0,
                bottom: 4,
            },
            corner_radius: egui::CornerRadius::same(8),
            fill: node_fill,
            stroke: egui::Stroke::NONE,
            shadow: egui::Shadow::NONE,
        }),
        bg_frame: Some(egui::Frame {
            inner_margin: egui::Margin::ZERO,
            outer_margin: egui::Margin::same(2),
            corner_radius: egui::CornerRadius::ZERO,
            fill: bg_fill,
            stroke: egui::Stroke::NONE,
            shadow: egui::Shadow::NONE,
        }),
        ..SnarlStyle::new()
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct AppState {
    projects: Vec<Project>,
    selected_graph: Option<usize>,
    /// How the Graph information panel breaks cells down. Defaults to
    /// per cell type, for saves from before it could be chosen.
    #[serde(default)]
    graph_information_grouping: GraphInformationGrouping,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Project {
    id: Uuid,
    loaded_graph: LoadGraph,
    graph_actions: Vec<GraphAction>,
    /// Which row in the sidebar's action list is currently shown in the
    /// central panel: `None` for the "Load GDS file" row, `Some(i)` for
    /// `graph_actions[i]`. Defaults to `None` (falls back to the loaded
    /// graph) both for new projects and for saves from before this field
    /// existed.
    #[serde(default)]
    selected_action: Option<usize>,
    /// The camera/zoom shared by every `GraphData` in this project — the
    /// loaded graph and every action all get viewed through the same
    /// projection, rather than each remembering its own.
    #[serde(default)]
    graph_viewer: GraphViewer,
}

/// Resolves whichever `GraphData` is currently selected in a project — an
/// action's, if `selected_action` names a valid index, otherwise the loaded
/// graph. Takes `loaded_graph`/`graph_actions` as separate parameters
/// (rather than `&mut Project`) so callers that also need
/// `Project::graph_viewer` at the same time can still borrow that sibling
/// field independently — a `&mut self` method here would make the borrow
/// checker treat the whole `Project` as borrowed.
fn resolve_graph_data_mut<'a>(
    loaded_graph: &'a mut LoadGraph,
    graph_actions: &'a mut [GraphAction],
    selected_action: Option<usize>,
) -> &'a mut GraphData {
    match selected_action {
        Some(index) if index < graph_actions.len() => graph_actions[index].graph_data_mut(),
        _ => &mut loaded_graph.graph_data,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct GraphData {
    graph: Graph,
    snarl: Snarl<CellNode>,
    layout_mode: LayoutMode,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RemoveClockBufferCells {
    id: Uuid,
    cells_to_remove: HashSet<String>,
    graph_data: GraphData,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MergeMuxedResetableFlipflops {
    id: Uuid,
    graph_data: GraphData,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MergeShiftRegisters {
    id: Uuid,
    graph_data: GraphData,
}

/// A boundary pin's value in a [`MergeBooleanFunctions`] action's right-side
/// panel: left blank by the user, typed in by the user, or (after Solve)
/// filled in from the model the solver found. Rendered in red only in the `Solved`
/// case, to set the solver's guesses apart from the user's own input.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum PinValue {
    #[default]
    Unset,
    User(bool),
    Solved(bool),
}

impl PinValue {
    fn bool(self) -> Option<bool> {
        match self {
            PinValue::Unset => None,
            PinValue::User(b) | PinValue::Solved(b) => Some(b),
        }
    }
}

/// Which grouping a [`MergeBooleanFunctions`] action ran the merge with —
/// raw connectivity, per feedback loop, or per cone of influence. All
/// three produce the same kind of `"BooleanFunction"` cells, differing
/// only in how the gates are split across them, so everything downstream
/// (the pin panel, Solve) works identically for any of them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum BooleanFunctionGrouping {
    /// [`Graph::merge_boolean_functions`].
    #[default]
    ConnectedGates,
    /// [`Graph::merge_boolean_functions_by_register_scc`].
    RegisterScc,
    /// [`Graph::merge_boolean_functions_by_cone_of_influence`].
    ConeOfInfluence,
}

/// Which tab of a [`MergeBooleanFunctions`] action's editor panel is
/// showing. Each asks something different of the same graph (see
/// [`show_system_solve_section`]) and each wants the whole panel height
/// for its results, so they take turns rather than share it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum BooleanFunctionTab {
    /// [`show_boolean_function_editor`]: the cone pin list and the
    /// one-instant solve over every function's pins.
    #[default]
    Functions,
    /// [`show_system_solve_section`]: the unrolled-in-time solve.
    System,
    /// [`show_simulator_section`]: the graph run forward, clock edge by
    /// clock edge, on inputs set by hand.
    Simulator,
}

impl BooleanFunctionTab {
    const ALL: [BooleanFunctionTab; 3] = [
        BooleanFunctionTab::Functions,
        BooleanFunctionTab::System,
        BooleanFunctionTab::Simulator,
    ];

    fn label(self) -> &'static str {
        match self {
            BooleanFunctionTab::Functions => "Solve boolean functions",
            BooleanFunctionTab::System => "Solve system",
            BooleanFunctionTab::Simulator => "Simulate",
        }
    }
}

/// How the simulator tab shows a multi-bit output — an indexed bus like
/// `O[7]..O[0]` — in its table.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
enum OutputFormat {
    /// One character per bit, highest index first.
    #[default]
    Bits,
    /// The bus as one unsigned number, in hex.
    Hex,
    /// The bus as one unsigned number, in decimal.
    Decimal,
    /// The bus as the ASCII character its value encodes, where that is a
    /// printable one; any other value falls back to hex.
    Ascii,
}

impl OutputFormat {
    const ALL: [OutputFormat; 4] = [
        OutputFormat::Bits,
        OutputFormat::Hex,
        OutputFormat::Decimal,
        OutputFormat::Ascii,
    ];

    fn label(self) -> &'static str {
        match self {
            OutputFormat::Bits => "Bits",
            OutputFormat::Hex => "Hex",
            OutputFormat::Decimal => "Decimal",
            OutputFormat::Ascii => "ASCII",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MergeBooleanFunctions {
    id: Uuid,
    graph_data: GraphData,
    /// Which tab of the editor panel is showing. Saved with the rest of
    /// the panel's state so reopening the app lands on the solve that was
    /// being worked on.
    #[serde(default)]
    editor_tab: BooleanFunctionTab,
    /// Which grouping this action ran the merge with. `None` only in app
    /// state saved before this field existed, where `by_register_scc`
    /// below still carries it — so read it through
    /// [`MergeBooleanFunctions::grouping`] rather than directly.
    #[serde(default)]
    grouping: Option<BooleanFunctionGrouping>,
    /// Superseded by `grouping`; still read so an action saved by an
    /// earlier build comes back as the register-SCC merge it was.
    #[serde(default)]
    by_register_scc: bool,
    /// Every boundary input/output net's current value across every
    /// `"BooleanFunction"` cell in `graph_data.graph`, shown and edited in
    /// the right-side panel. A net missing here is unset.
    #[serde(default)]
    pin_values: HashMap<u32, PinValue>,
    /// The last Solve attempt's outcome, if it didn't find a satisfying
    /// assignment (e.g. `"unsat"`, or that there was nothing to solve).
    #[serde(default)]
    solve_error: Option<String>,
    /// Which input groups (by cell label, e.g. `"ShiftRegister#123"`) are
    /// currently shown as one decimal number instead of per-bit toggles.
    #[serde(default)]
    decimal_groups: HashSet<String>,
    /// `Output` pin names narrowing a `ConeOfInfluence` merge to the union
    /// of those pins' cones, everything outside pruned away. Empty — the
    /// default, and all this ever is for the other two groupings — merges
    /// the whole graph. Fed to
    /// [`Graph::merge_boolean_functions_by_cone_of_influence`].
    #[serde(default)]
    cone_output_pins: Vec<String>,
    /// The pin list as shown in the right-side editor panel, editable
    /// freely (including blank rows mid-edit). Only committed into
    /// `cone_output_pins` — and the graph recomputed from it — when Apply
    /// is clicked, so edits in progress don't touch the graph. Mirrors
    /// [`PropagatePinValues::pending_entries`].
    #[serde(default)]
    pending_cone_output_pins: Vec<String>,
    /// The Solve system tab's condition, bounds and result. Boxed to keep
    /// this variant of [`GraphAction`] from growing the whole enum, and
    /// flattened so its fields keep the top-level names they were saved
    /// under before they were grouped.
    #[serde(flatten)]
    system: Box<SystemSolveState>,
    /// The Simulate tab's settings and trace. Boxed for the same reason.
    /// Absent from state saved before the tab existed, where it defaults
    /// to a fresh, unrun one.
    #[serde(default)]
    simulator: Box<SimulatorState>,
}

/// Everything the Solve system tab holds for a [`MergeBooleanFunctions`]
/// action. Edited in place — there is no Apply, since unlike the cone
/// pins this changes nothing about the graph, only what the next solve
/// asks of it.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct SystemSolveState {
    /// The condition to solve for: each row a pin name anywhere in this
    /// action's graph (`"success"`) and the value the found sequence has
    /// to leave it holding.
    #[serde(default, rename = "system_targets")]
    targets: Vec<(String, bool)>,
    /// Further conditions on output buses, as typed: `O ≠ 'A'`. Parsed
    /// into [`BusConstraint`]s when Solve is pressed (see
    /// [`BusConstraintRow::parse`]).
    #[serde(default, rename = "system_bus_constraints")]
    bus_constraints: Vec<BusConstraintRow>,
    /// The fewest clock cycles a solve may answer with — where the length
    /// search starts, for finding the *next* sequence after the shortest.
    #[serde(default, rename = "system_min_cycles")]
    min_cycles: usize,
    /// How many clock cycles the length search may go to before giving
    /// up. Read through [`SystemSolveState::max_cycles`], which supplies
    /// the default for state saved before this field existed.
    #[serde(default, rename = "system_max_cycles")]
    max_cycles: Option<usize>,
    /// The last solve's sequence, kept so it survives switching between
    /// actions (and app restarts) rather than having to be solved for
    /// again to be read. Boxed: a whole waveform is far bigger than
    /// anything else here.
    #[serde(default, rename = "system_solution")]
    solution: Option<Box<SystemSolution>>,
    /// The last solve's failure, if it had one — no sequence within the
    /// cycle bounds, an unknown pin name, nothing asked for.
    #[serde(default, rename = "system_error")]
    error: Option<String>,
}

impl SystemSolveState {
    /// How many clock cycles the solve searches to, falling back to
    /// [`DEFAULT_SYSTEM_MAX_CYCLES`] for state saved before the field
    /// existed.
    fn max_cycles(&mut self) -> &mut usize {
        self.max_cycles.get_or_insert(DEFAULT_SYSTEM_MAX_CYCLES)
    }
}

/// Everything the Simulate tab holds for a [`MergeBooleanFunctions`]
/// action: what each input does over the run, how long a run of constant
/// inputs is, how the table reads, and the trace itself.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SimulatorState {
    /// What each design input does over the run, by pin name, as typed:
    /// `"0"` or `"1"` to hold it there for every step, or a bitstring
    /// with one character per step, left to right (see
    /// [`expand_input_patterns`]). A pin missing here is `"0"`. Kept by
    /// name rather than position so the settings survive the graph being
    /// recomputed around them.
    #[serde(default)]
    input_patterns: BTreeMap<String, String>,
    /// How many steps the run is when no pattern is a bitstring — when
    /// one is, its length says instead.
    #[serde(default = "default_simulator_steps")]
    steps: usize,
    /// How the table shows multi-bit outputs.
    #[serde(default)]
    output_format: OutputFormat,
    /// The last run — kept, like the system solve's sequence, so it
    /// survives switching actions and restarts. `None` until the tab is
    /// first shown (which runs it), and again whenever the graph under
    /// it changes.
    #[serde(default)]
    simulation: Option<Box<Simulation>>,
    /// Why the last run could not happen, if it couldn't: a pattern that
    /// isn't a bitstring, bitstrings of different lengths, a graph the
    /// simulator can't be built over.
    #[serde(default)]
    error: Option<String>,
}

/// How many steps a run of constant inputs is, until changed.
fn default_simulator_steps() -> usize {
    10
}

impl Default for SimulatorState {
    fn default() -> Self {
        SimulatorState {
            input_patterns: BTreeMap::new(),
            steps: default_simulator_steps(),
            output_format: OutputFormat::default(),
            simulation: None,
            error: None,
        }
    }
}

/// One row of the Solve system tab's bus conditions, as typed: the bus
/// name, whether it should equal or differ from the value, and the value
/// — a single character (`A`), or a number (`65`, `0x41`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
struct BusConstraintRow {
    bus: String,
    equal: bool,
    value: String,
    /// The frame to apply it in, as typed; blank for the last frame.
    #[serde(default)]
    cycle: String,
}

impl BusConstraintRow {
    /// A fresh row: `O ≠ ' '`, blank value.
    fn new() -> Self {
        BusConstraintRow {
            bus: "O".to_string(),
            equal: false,
            value: String::new(),
            cycle: String::new(),
        }
    }

    /// The constraint this row asks for; `None` for a row with no bus
    /// name (skipped, like a blank target row). A value that is neither
    /// one character nor a number, or a frame that isn't a number, is an
    /// error naming the row.
    fn parse(&self) -> Result<Option<BusConstraint>, String> {
        let bus = self.bus.trim();
        if bus.is_empty() {
            return Ok(None);
        }
        // Not trimmed: a single space is a legitimate character.
        let text = self.value.as_str();
        let mut chars = text.chars();
        let value = match (chars.next(), chars.next()) {
            (Some(only), None) => u64::from(u32::from(only)),
            _ => {
                let trimmed = text.trim();
                let parsed = match trimmed
                    .strip_prefix("0x")
                    .or_else(|| trimmed.strip_prefix("0X"))
                {
                    Some(hex) => u64::from_str_radix(hex, 16),
                    None => trimmed.parse::<u64>(),
                };
                parsed.map_err(|_| {
                    format!("{bus}: give the value as one character (A), or a number (65, 0x41).")
                })?
            }
        };
        let cycle = match self.cycle.trim() {
            "" => None,
            text => Some(text.parse::<usize>().map_err(|_| {
                format!("{bus}: the frame must be a number, or blank for the last frame.")
            })?),
        };
        Ok(Some(BusConstraint {
            bus: bus.to_string(),
            equal: self.equal,
            value,
            cycle,
        }))
    }
}

/// The cycle bound a [`MergeBooleanFunctions`] system solve starts with:
/// far enough for a design that counts through a sequence (a real design
/// can take over a hundred cycles to reach `success`), and small enough
/// that an unreachable condition gives up rather than grinding.
const DEFAULT_SYSTEM_MAX_CYCLES: usize = 256;

impl MergeBooleanFunctions {
    /// This action's grouping, falling back to what the superseded
    /// `by_register_scc` flag says for state saved before `grouping`
    /// existed.
    fn grouping(&self) -> BooleanFunctionGrouping {
        self.grouping.unwrap_or(if self.by_register_scc {
            BooleanFunctionGrouping::RegisterScc
        } else {
            BooleanFunctionGrouping::ConnectedGates
        })
    }

    /// Drops whatever the last system solve found and the simulator's
    /// trace. Called wherever the graph underneath them changes, since a
    /// sequence solved — or run — against the old graph says nothing
    /// about the new one. The simulator's input settings stay: they are
    /// by pin name, and the pins are most likely still there.
    fn clear_system_solution(&mut self) {
        self.system.solution = None;
        self.system.error = None;
        self.simulator.simulation = None;
        self.simulator.error = None;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PropagatePinValues {
    id: Uuid,
    /// Each `(input_pin_label, target_pin_label)` entry fed to
    /// [`Graph::propagate_pin_values`].
    entries: Vec<(String, String)>,
    graph_data: GraphData,
    /// The entry list as shown in the right-side editor panel: one
    /// `(input pin label, target pin label)` row per entry, editable
    /// freely (including blank/duplicate rows mid-edit). Only committed
    /// into `entries` — and `graph_data` recomputed from it — when Apply
    /// is clicked, so edits in progress don't touch the graph.
    #[serde(default)]
    pending_entries: Vec<(String, String)>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum GraphAction {
    RemoveClockBufferCells(RemoveClockBufferCells),
    MergeMuxedResetableFlipflops(MergeMuxedResetableFlipflops),
    MergeShiftRegisters(MergeShiftRegisters),
    MergeBooleanFunctions(MergeBooleanFunctions),
    PropagatePinValues(PropagatePinValues),
}

impl GraphAction {
    fn graph_data(&self) -> &GraphData {
        match self {
            GraphAction::RemoveClockBufferCells(action) => &action.graph_data,
            GraphAction::MergeMuxedResetableFlipflops(action) => &action.graph_data,
            GraphAction::MergeShiftRegisters(action) => &action.graph_data,
            GraphAction::MergeBooleanFunctions(action) => &action.graph_data,
            GraphAction::PropagatePinValues(action) => &action.graph_data,
        }
    }

    fn graph_data_mut(&mut self) -> &mut GraphData {
        match self {
            GraphAction::RemoveClockBufferCells(action) => &mut action.graph_data,
            GraphAction::MergeMuxedResetableFlipflops(action) => &mut action.graph_data,
            GraphAction::MergeShiftRegisters(action) => &mut action.graph_data,
            GraphAction::MergeBooleanFunctions(action) => &mut action.graph_data,
            GraphAction::PropagatePinValues(action) => &mut action.graph_data,
        }
    }

    fn set_graph_data(&mut self, graph_data: GraphData) {
        match self {
            GraphAction::RemoveClockBufferCells(action) => action.graph_data = graph_data,
            GraphAction::MergeMuxedResetableFlipflops(action) => action.graph_data = graph_data,
            GraphAction::MergeShiftRegisters(action) => action.graph_data = graph_data,
            GraphAction::MergeBooleanFunctions(action) => {
                action.graph_data = graph_data;
                // This is every path by which the graph under an action
                // changes — an upstream edit, a refresh, a narrowed cone —
                // and a sequence solved against the graph it replaces
                // describes a machine that is no longer there.
                action.clear_system_solution();
            }
            GraphAction::PropagatePinValues(action) => action.graph_data = graph_data,
        }
    }

    /// Recomputes this action's `graph_data` from scratch against
    /// `input_graph` — the same transform it was created with, applied
    /// again (e.g. because an earlier action in the chain changed).
    fn recompute_graph_data(
        &self,
        input_graph: &Graph,
        layout_mode: LayoutMode,
    ) -> Result<GraphData, AmbiguousRemoval> {
        match self {
            GraphAction::RemoveClockBufferCells(action) => {
                compute_graph_data_after_removal(input_graph, &action.cells_to_remove, layout_mode)
            }
            GraphAction::MergeMuxedResetableFlipflops(_) => {
                Ok(compute_graph_data_after_merge(input_graph, layout_mode))
            }
            GraphAction::MergeShiftRegisters(_) => Ok(
                compute_graph_data_after_shift_register_merge(input_graph, layout_mode),
            ),
            GraphAction::MergeBooleanFunctions(action) => {
                Ok(compute_graph_data_after_boolean_function_merge(
                    input_graph,
                    action.grouping(),
                    &action.cone_output_pins,
                    layout_mode,
                ))
            }
            GraphAction::PropagatePinValues(action) => {
                Ok(compute_graph_data_after_pin_value_propagation(
                    input_graph,
                    &action.entries,
                    layout_mode,
                ))
            }
        }
    }

    fn kind(&self) -> GraphActionKind {
        match self {
            GraphAction::RemoveClockBufferCells(_) => GraphActionKind::RemoveClockBufferCells,
            GraphAction::MergeMuxedResetableFlipflops(_) => {
                GraphActionKind::MergeMuxedResetableFlipflops
            }
            GraphAction::MergeShiftRegisters(_) => GraphActionKind::MergeShiftRegisters,
            GraphAction::MergeBooleanFunctions(action) => match action.grouping() {
                BooleanFunctionGrouping::ConnectedGates => GraphActionKind::MergeBooleanFunctions,
                BooleanFunctionGrouping::RegisterScc => {
                    GraphActionKind::MergeBooleanFunctionsByRegisterScc
                }
                BooleanFunctionGrouping::ConeOfInfluence => {
                    GraphActionKind::MergeBooleanFunctionsByConeOfInfluence
                }
            },
            GraphAction::PropagatePinValues(_) => GraphActionKind::PropagatePinValues,
        }
    }
}

/// A reordering or removal requested from a project's action list, applied
/// once the caller is done iterating over the project list (see
/// [`show_project_list_item`]).
#[derive(Clone, Copy, Debug)]
enum PendingGraphActionEdit {
    Remove(usize),
    MoveUp(usize),
    MoveDown(usize),
}

/// The kind of [`GraphAction`] that can be created from the project's "+"
/// menu. `RemoveClockBufferCells` knows which standard-cell names it strips
/// via [`Graph::without_cell_names`]. `MergeMuxedResetableFlipflops`,
/// `MergeShiftRegisters` and the three `MergeBooleanFunctions*` kinds
/// instead run [`Graph::merge_muxed_resetable_flipflops`],
/// [`Graph::merge_shift_registers`] and the merge for their
/// [`BooleanFunctionGrouping`] respectively, none of which takes a cell
/// list at all. `PropagatePinValues` runs
/// [`Graph::propagate_pin_values`], whose list is entries of `(input pin
/// label, target pin label)` pairs rather than cell names, filled in
/// afterward via the right-side editor panel.
#[derive(Clone, Copy, Debug)]
enum GraphActionKind {
    RemoveClockBufferCells,
    MergeMuxedResetableFlipflops,
    MergeShiftRegisters,
    MergeBooleanFunctions,
    MergeBooleanFunctionsByRegisterScc,
    MergeBooleanFunctionsByConeOfInfluence,
    PropagatePinValues,
}

impl GraphActionKind {
    fn label(self) -> &'static str {
        match self {
            GraphActionKind::RemoveClockBufferCells => "Remove (clock) buffer cells",
            GraphActionKind::MergeMuxedResetableFlipflops => "Merge muxed resetable flipflops",
            GraphActionKind::MergeShiftRegisters => "Merge shift registers",
            GraphActionKind::MergeBooleanFunctions => "Merge boolean functions",
            GraphActionKind::MergeBooleanFunctionsByRegisterScc => {
                "Merge boolean functions by register SCC"
            }
            GraphActionKind::MergeBooleanFunctionsByConeOfInfluence => {
                "Merge boolean functions by cone of influence"
            }
            GraphActionKind::PropagatePinValues => "Propagate pin values",
        }
    }

    fn cells_to_remove(self) -> HashSet<String> {
        match self {
            GraphActionKind::RemoveClockBufferCells => HashSet::from([
                "sky130_fd_sc_hd__buf_2".to_string(),
                "sky130_fd_sc_hd__clkbuf_4".to_string(),
                "sky130_fd_sc_hd__clkbuf_8".to_string(),
                "sky130_fd_sc_hd__clkbuf_16".to_string(),
            ]),
            GraphActionKind::MergeMuxedResetableFlipflops
            | GraphActionKind::MergeShiftRegisters
            | GraphActionKind::MergeBooleanFunctions
            | GraphActionKind::MergeBooleanFunctionsByRegisterScc
            | GraphActionKind::MergeBooleanFunctionsByConeOfInfluence
            | GraphActionKind::PropagatePinValues => HashSet::new(),
        }
    }

    /// Runs this kind against `input_graph`, building the resulting
    /// [`GraphAction`].
    fn create(
        self,
        id: Uuid,
        input_graph: &Graph,
        layout_mode: LayoutMode,
    ) -> Result<GraphAction, AmbiguousRemoval> {
        match self {
            GraphActionKind::RemoveClockBufferCells => {
                let cells_to_remove = self.cells_to_remove();
                let graph_data =
                    compute_graph_data_after_removal(input_graph, &cells_to_remove, layout_mode)?;
                Ok(GraphAction::RemoveClockBufferCells(
                    RemoveClockBufferCells {
                        id,
                        cells_to_remove,
                        graph_data,
                    },
                ))
            }
            GraphActionKind::MergeMuxedResetableFlipflops => {
                let graph_data = compute_graph_data_after_merge(input_graph, layout_mode);
                Ok(GraphAction::MergeMuxedResetableFlipflops(
                    MergeMuxedResetableFlipflops { id, graph_data },
                ))
            }
            GraphActionKind::MergeShiftRegisters => {
                let graph_data =
                    compute_graph_data_after_shift_register_merge(input_graph, layout_mode);
                Ok(GraphAction::MergeShiftRegisters(MergeShiftRegisters {
                    id,
                    graph_data,
                }))
            }
            GraphActionKind::MergeBooleanFunctions
            | GraphActionKind::MergeBooleanFunctionsByRegisterScc
            | GraphActionKind::MergeBooleanFunctionsByConeOfInfluence => {
                let grouping = match self {
                    GraphActionKind::MergeBooleanFunctionsByRegisterScc => {
                        BooleanFunctionGrouping::RegisterScc
                    }
                    GraphActionKind::MergeBooleanFunctionsByConeOfInfluence => {
                        BooleanFunctionGrouping::ConeOfInfluence
                    }
                    _ => BooleanFunctionGrouping::ConnectedGates,
                };
                let graph_data = compute_graph_data_after_boolean_function_merge(
                    input_graph,
                    grouping,
                    &[],
                    layout_mode,
                );
                Ok(GraphAction::MergeBooleanFunctions(MergeBooleanFunctions {
                    id,
                    graph_data,
                    editor_tab: BooleanFunctionTab::default(),
                    grouping: Some(grouping),
                    by_register_scc: grouping == BooleanFunctionGrouping::RegisterScc,
                    pin_values: HashMap::new(),
                    solve_error: None,
                    decimal_groups: HashSet::new(),
                    cone_output_pins: Vec::new(),
                    pending_cone_output_pins: Vec::new(),
                    system: Box::default(),
                    simulator: Box::default(),
                }))
            }
            GraphActionKind::PropagatePinValues => {
                let entries = Vec::new();
                let graph_data = compute_graph_data_after_pin_value_propagation(
                    input_graph,
                    &entries,
                    layout_mode,
                );
                Ok(GraphAction::PropagatePinValues(PropagatePinValues {
                    id,
                    entries,
                    graph_data,
                    pending_entries: Vec::new(),
                }))
            }
        }
    }
}

/// Filters `cells_to_remove` out of `input_graph` and packages the result
/// (along with a freshly laid-out snarl) into a [`GraphData`].
fn compute_graph_data_after_removal(
    input_graph: &Graph,
    cells_to_remove: &HashSet<String>,
    layout_mode: LayoutMode,
) -> Result<GraphData, AmbiguousRemoval> {
    let excluded_cell_names = cells_to_remove.iter().map(String::as_str).collect();
    let graph = input_graph.without_cell_names(&excluded_cell_names)?;
    Ok(GraphData {
        snarl: create_snarl(layout_mode, &graph),
        graph,
        layout_mode,
    })
}

/// Runs [`Graph::merge_muxed_resetable_flipflops`] over `input_graph` and
/// packages the result (along with a freshly laid-out snarl) into a
/// [`GraphData`]. Unlike [`compute_graph_data_after_removal`], this can't
/// fail.
fn compute_graph_data_after_merge(input_graph: &Graph, layout_mode: LayoutMode) -> GraphData {
    let graph = input_graph.merge_muxed_resetable_flipflops();
    GraphData {
        snarl: create_snarl(layout_mode, &graph),
        graph,
        layout_mode,
    }
}

/// Runs [`Graph::merge_shift_registers`] over `input_graph` and packages
/// the result (along with a freshly laid-out snarl) into a [`GraphData`].
/// Also can't fail.
fn compute_graph_data_after_shift_register_merge(
    input_graph: &Graph,
    layout_mode: LayoutMode,
) -> GraphData {
    let graph = input_graph.merge_shift_registers();
    GraphData {
        snarl: create_snarl(layout_mode, &graph),
        graph,
        layout_mode,
    }
}

/// Runs whichever boolean-function merge `grouping` names over
/// `input_graph` and packages the result (along with a freshly laid-out
/// snarl) into a [`GraphData`]. Also can't fail.
///
/// `cone_output_pins` only means anything for
/// [`BooleanFunctionGrouping::ConeOfInfluence`], where a non-blank name in
/// it narrows the merge to that `Output` pin's cone; the other two
/// groupings have no cone to narrow and ignore it.
fn compute_graph_data_after_boolean_function_merge(
    input_graph: &Graph,
    grouping: BooleanFunctionGrouping,
    cone_output_pins: &[String],
    layout_mode: LayoutMode,
) -> GraphData {
    let graph = match grouping {
        BooleanFunctionGrouping::ConnectedGates => input_graph.merge_boolean_functions(),
        BooleanFunctionGrouping::RegisterScc => {
            input_graph.merge_boolean_functions_by_register_scc()
        }
        BooleanFunctionGrouping::ConeOfInfluence => {
            input_graph.merge_boolean_functions_by_cone_of_influence(cone_output_pins)
        }
    };
    GraphData {
        snarl: create_snarl(layout_mode, &graph),
        graph,
        layout_mode,
    }
}

/// Runs [`Graph::propagate_pin_values`] over `input_graph` with `entries`
/// and packages the result (along with a freshly laid-out snarl) into a
/// [`GraphData`]. Also can't fail.
///
/// The `GraphData::graph` kept here — what later actions are recomputed
/// from, and what the solver and simulator encode — is the fully wired
/// one, so the propagation only ever renames pins as far as they're
/// concerned. The edges those labels make redundant are hidden by
/// [`create_snarl`], as they are for every action's view from here on.
fn compute_graph_data_after_pin_value_propagation(
    input_graph: &Graph,
    entries: &[(String, String)],
    layout_mode: LayoutMode,
) -> GraphData {
    let graph = input_graph.propagate_pin_values(entries);
    GraphData {
        snarl: create_snarl(layout_mode, &graph),
        graph,
        layout_mode,
    }
}

/// Every `"BooleanFunction"` `MergeCell` in `graph`.
fn boolean_function_cells(graph: &Graph) -> impl Iterator<Item = &Cell> {
    graph.cells.iter().filter(
        |cell| matches!(cell, Cell::MergeCell { cell_name, .. } if cell_name == "BooleanFunction"),
    )
}

/// Renders one editable row per pin in `pins` (a `BooleanFunction` cell's
/// `inputs` or `outputs`), grouped by the ancestor cell each pin came from
/// — its `Pin` name is `"<cell_name>#<cell_id>.<pin_name>"`, as built in
/// [`crate::graph::Graph::merge_boolean_functions`]. Each row shows the
/// pin's current [`PinValue`] (a solver-filled value in red) with buttons
/// to set it to 0/1 or clear it back to unset.
/// Groups `pins` (a `BooleanFunction` cell's `inputs` or `outputs`) by the
/// ancestor cell each pin came from — its `Pin` name is
/// `"<cell_name>#<cell_id>.<pin_name>"`, as built in
/// [`crate::graph::Graph::merge_boolean_functions`] (or, for a
/// `BooleanFunction`'s own numbered outputs like `"X0"`, has no `.` at all
/// and falls into one unlabeled group).
fn group_boolean_pins(
    pins: &[crate::graph::Pin],
) -> std::collections::BTreeMap<&str, Vec<(u32, &str)>> {
    let mut groups: std::collections::BTreeMap<&str, Vec<(u32, &str)>> =
        std::collections::BTreeMap::new();
    for (net, name) in pins {
        let (cell_label, pin_label) = name.rsplit_once('.').unwrap_or(("", name.as_str()));
        groups
            .entry(cell_label)
            .or_default()
            .push((*net, pin_label));
    }
    groups
}

/// One pin's row: its name, its current [`PinValue`] (a solver-filled value
/// in red) with buttons to set it to 0/1 or clear it back to unset.
fn show_pin_row(
    ui: &mut egui::Ui,
    net: u32,
    pin_label: &str,
    pin_values: &mut HashMap<u32, PinValue>,
) {
    ui.horizontal(|ui| {
        ui.add_space(12.0);
        label(ui, pin_label);

        let value = pin_values.entry(net).or_insert(PinValue::Unset);
        let solved_color = egui::Color32::from_rgb(220, 80, 80);

        let zero_text = if matches!(*value, PinValue::Solved(false)) {
            egui::RichText::new("0").color(solved_color)
        } else {
            egui::RichText::new("0")
        };
        if ui
            .selectable_label(value.bool() == Some(false), zero_text)
            .clicked()
        {
            *value = PinValue::User(false);
        }

        let one_text = if matches!(*value, PinValue::Solved(true)) {
            egui::RichText::new("1").color(solved_color)
        } else {
            egui::RichText::new("1")
        };
        if ui
            .selectable_label(value.bool() == Some(true), one_text)
            .clicked()
        {
            *value = PinValue::User(true);
        }

        if ui.small_button("×").clicked() {
            *value = PinValue::Unset;
        }
    });
}

/// Renders `pins` (a `BooleanFunction` cell's `outputs`) as one row per pin,
/// grouped by originating ancestor cell.
fn show_boolean_pin_rows(
    ui: &mut egui::Ui,
    pins: &[crate::graph::Pin],
    pin_values: &mut HashMap<u32, PinValue>,
) {
    for (cell_label, pins) in group_boolean_pins(pins) {
        label(ui, cell_label);
        for (net, pin_label) in pins {
            show_pin_row(ui, net, pin_label, pin_values);
        }
    }
}

/// Renders `pins` (a `BooleanFunction` cell's `inputs`) grouped by
/// originating ancestor cell, same as [`show_boolean_pin_rows`], except a
/// group with more than one bit (e.g. a whole shift register's `Q0`..`Qn`)
/// gets a "Decimal" checkbox: switches that one group from per-bit 0/1
/// toggles to a single decimal number covering every bit at once, `Q0`
/// being the least significant. `decimal_groups` remembers which cell
/// labels are currently shown that way.
fn show_boolean_input_rows(
    ui: &mut egui::Ui,
    pins: &[crate::graph::Pin],
    pin_values: &mut HashMap<u32, PinValue>,
    decimal_groups: &mut HashSet<String>,
) {
    for (cell_label, pins) in group_boolean_pins(pins) {
        ui.horizontal(|ui| {
            label(ui, cell_label);
            if pins.len() > 1 {
                let mut decimal = decimal_groups.contains(cell_label);
                if ui.checkbox(&mut decimal, "Decimal").changed() {
                    if decimal {
                        decimal_groups.insert(cell_label.to_string());
                    } else {
                        decimal_groups.remove(cell_label);
                    }
                }
            }
        });

        if pins.len() > 1 && decimal_groups.contains(cell_label) {
            show_decimal_pin_row(ui, &pins, pin_values);
        } else {
            for &(net, pin_label) in &pins {
                show_pin_row(ui, net, pin_label, pin_values);
            }
        }
    }
}

/// Renders one group's bits as a single decimal number (`Q0` = bit 0, the
/// least significant), editable via a drag/type box that fans back out to
/// each bit's [`PinValue::User`] when changed. Any bit currently unset is
/// treated as 0 for display; any bit the solver filled in is flagged with
/// a small red note next to the box, since the box itself can't mix colors
/// per digit.
fn show_decimal_pin_row(
    ui: &mut egui::Ui,
    pins: &[(u32, &str)],
    pin_values: &mut HashMap<u32, PinValue>,
) {
    let mut bit_nets: Vec<(u32, u32)> = pins
        .iter()
        .filter_map(|&(net, name)| {
            let digits_at = name
                .rfind(|c: char| !c.is_ascii_digit())
                .map_or(0, |i| i + 1);
            let bit_index: u32 = name[digits_at..].parse().ok()?;
            Some((bit_index, net))
        })
        .collect();
    bit_nets.sort_unstable_by_key(|&(bit_index, _)| bit_index);

    let mut value: u64 = 0;
    let mut any_solved = false;
    for &(bit_index, net) in &bit_nets {
        let pin_value = pin_values.get(&net).copied().unwrap_or_default();
        any_solved |= matches!(pin_value, PinValue::Solved(_));
        if pin_value.bool() == Some(true) {
            value |= 1u64 << bit_index;
        }
    }

    ui.horizontal(|ui| {
        ui.add_space(12.0);
        let max = if bit_nets.len() >= u64::BITS as usize {
            u64::MAX
        } else {
            (1u64 << bit_nets.len()) - 1
        };
        let mut edit_value = value;
        if ui
            .add(egui::DragValue::new(&mut edit_value).range(0..=max))
            .changed()
        {
            for &(bit_index, net) in &bit_nets {
                pin_values.insert(net, PinValue::User((edit_value >> bit_index) & 1 == 1));
            }
        }
        if any_solved {
            ui.colored_label(
                egui::Color32::from_rgb(220, 80, 80),
                "(includes solved bits)",
            );
        }
    });
}

/// Solves every `"BooleanFunction"` cell in `action`'s graph combined:
/// asserts each cell's composed [`BoolExpr`] for its output nets, asserts
/// every user-fixed pin's value (`PinValue::User`) as an extra constraint,
/// and — if satisfiable — fills in every other boundary pin from the model
/// found (marked `PinValue::Solved`, and rendered in red in the panel to
/// set it apart from the user's own input). Records `unsat` in
/// `action.solve_error`.
///
/// The problem is purely propositional, so this Tseitin-encodes it to CNF
/// (see [`CnfEncoder`]) and runs the in-process `batsat` SAT solver rather
/// than shelling out to an SMT solver — which also means it works on the
/// `wasm32` build, where there are no subprocesses.
fn solve_boolean_functions(action: &mut MergeBooleanFunctions) {
    let mut encoder = CnfEncoder::new(BasicSolver::default());
    // The boundary pins the panel shows; only these are read back into
    // `pin_values`, so nets that exist only inside an expression don't
    // leak into the action's saved state.
    let mut boundary: std::collections::BTreeSet<u32> = std::collections::BTreeSet::new();

    for cell in boolean_function_cells(&action.graph_data.graph) {
        let Cell::MergeCell {
            inputs,
            outputs,
            boolean_outputs,
            ..
        } = cell
        else {
            continue;
        };
        // Boundary pins are what the panel shows and fills in, so they get
        // solver variables even when no expression happens to mention them.
        for &(net, _) in inputs.iter().chain(outputs.iter()) {
            encoder.net_var(net);
            boundary.insert(net);
        }
        for (net, expr) in boolean_outputs {
            encoder.assert_net_eq(*net, expr);
        }
    }

    if boundary.is_empty() {
        action.solve_error = Some("No boolean-function ports to solve.".to_string());
        return;
    }

    for (&net, value) in &action.pin_values {
        if let PinValue::User(b) = *value {
            encoder.assert_net_value(net, b);
        }
    }

    if encoder.solver_mut().solve_limited(&[]) != lbool::TRUE {
        action.solve_error = Some("No solution matches the given values (unsat).".to_string());
        return;
    }

    let nets: Vec<(u32, Var)> = encoder
        .nets()
        .filter(|(net, _)| boundary.contains(net))
        .collect();
    let solver = encoder.solver_mut();
    for (net, var) in nets {
        let value = solver.value_lit(Lit::new(var, true)) == lbool::TRUE;
        let entry = action.pin_values.entry(net).or_insert(PinValue::Unset);
        if !matches!(entry, PinValue::User(_)) {
            *entry = PinValue::Solved(value);
        }
    }
    action.solve_error = None;
}

/// The right panel's Solve system tab: the condition to reach, the cycle
/// bound to search to, and — once solved — the input sequence that reaches
/// it.
///
/// Distinct from the boolean-function Solve on the other tab in what it
/// treats the graph as. That one asks a question about one combinational instant:
/// every `"BooleanFunction"` cell at once, with the registers between them
/// left as free variables that need not agree with each other over time.
/// This one asks a question about the machine: the whole graph unrolled in
/// time, flip-flops carrying state from cycle to cycle from the power-up
/// value their set/reset pins give them, and the answer a *sequence* of
/// inputs rather than a single assignment. See [`Graph::solve_system`].
fn show_system_solve_section(ui: &mut egui::Ui, action: &mut MergeBooleanFunctions) {
    // One bounded scroll region for the whole tab, for the same reason
    // [`show_boolean_function_editor`] has one: the target rows can grow
    // past the height the Graph information panel leaves, and a plain
    // `Ui` would paint the overflow straight over it.
    egui::ScrollArea::vertical()
        .id_salt("system_solve")
        .auto_shrink([false, false])
        .show(ui, |ui| show_system_solve_contents(ui, action));
}

/// The body of [`show_system_solve_section`], inside its scroll region.
fn show_system_solve_contents(ui: &mut egui::Ui, action: &mut MergeBooleanFunctions) {
    let state = &mut *action.system;
    let graph = &action.graph_data.graph;
    label(
        ui,
        "Solves this whole graph as a synchronous machine — one frame per clock edge, every \
         flip-flop starting from what its set/reset pin says — for the shortest input sequence \
         that leaves every pin below holding the value next to it.",
    );

    // Narrower than a plain editor row by the two value toggles that
    // follow the name field, so a target row still fits the panel without
    // widening it (see [`editor_field_width`]).
    let field_width = (editor_field_width(ui, 1) - 56.0).max(24.0);
    let mut remove_row = None;
    let mut solve = false;
    for (row_index, (pin_name, value)) in state.targets.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(pin_name).desired_width(field_width));
            if ui.selectable_label(!*value, "0").clicked() {
                *value = false;
            }
            if ui.selectable_label(*value, "1").clicked() {
                *value = true;
            }
            if egui_icons::icon_button(ui, ICON_CLOSE).clicked() {
                remove_row = Some(row_index);
            }
        });
    }
    if let Some(row_index) = remove_row {
        state.targets.remove(row_index);
    }
    if ui.button(ICON_ADD).clicked() {
        state.targets.push((String::new(), true));
    }

    ui.separator();
    label(
        ui,
        "Conditions on an output bus (the pins O[0], O[1], ...), read as a number: equal or \
         not equal to a character or a number, in the last frame or in the frame given after \
         the @.",
    );
    // Bus name, the =/≠ toggle, the value and the frame share the row:
    // a quarter each to the name and the frame, the value the rest.
    let bus_width = (field_width / 4.0).max(24.0);
    let cycle_width = bus_width;
    let value_width =
        (field_width - bus_width - cycle_width - 3.0 * ui.spacing().item_spacing.x).max(24.0);
    let mut remove_row = None;
    for (row_index, row) in state.bus_constraints.iter_mut().enumerate() {
        ui.horizontal(|ui| {
            ui.add(egui::TextEdit::singleline(&mut row.bus).desired_width(bus_width));
            if ui.selectable_label(row.equal, "=").clicked() {
                row.equal = true;
            }
            if ui.selectable_label(!row.equal, "≠").clicked() {
                row.equal = false;
            }
            ui.add(
                egui::TextEdit::singleline(&mut row.value)
                    .desired_width(value_width)
                    .hint_text("A or 0x41"),
            );
            label(ui, "@");
            ui.add(
                egui::TextEdit::singleline(&mut row.cycle)
                    .desired_width(cycle_width)
                    .hint_text("last"),
            )
            .on_hover_text("The frame this holds in; blank for the last frame of the sequence.");
            if egui_icons::icon_button(ui, ICON_CLOSE).clicked() {
                remove_row = Some(row_index);
            }
        });
    }
    if let Some(row_index) = remove_row {
        state.bus_constraints.remove(row_index);
    }
    if ui.button(ICON_ADD).clicked() {
        state.bus_constraints.push(BusConstraintRow::new());
    }

    ui.separator();
    ui.horizontal(|ui| {
        label(ui, "Search from");
        ui.add(egui::DragValue::new(&mut state.min_cycles).range(0..=100_000));
        label(ui, "up to");
        ui.add(
            egui::DragValue::new(state.max_cycles())
                .range(0..=100_000)
                .suffix(" cycles"),
        );
    });

    // Above the results, for the same reason Solve sits above the pin list
    // on the other tab: a waveform is long, and the button that produced
    // it shouldn't be scrolled out of view by it.
    if ui.button("Solve").clicked() {
        solve = true;
    }
    if let Some(err) = &state.error {
        ui.colored_label(egui::Color32::from_rgb(220, 80, 80), err);
    }

    if solve {
        let max_cycles = *state.max_cycles();
        let bus_constraints: Result<Vec<BusConstraint>, String> = state
            .bus_constraints
            .iter()
            .filter_map(|row| row.parse().transpose())
            .collect();
        let result = bus_constraints.and_then(|bus_constraints| {
            graph.solve_system(
                &state.targets,
                &bus_constraints,
                state.min_cycles,
                max_cycles,
            )
        });
        match result {
            Ok(solution) => {
                state.solution = Some(Box::new(solution));
                state.error = None;
            }
            Err(err) => {
                state.solution = None;
                state.error = Some(err);
            }
        }
    }

    let Some(solution) = &state.solution else {
        return;
    };

    ui.separator();
    label(
        ui,
        format!(
            "{} clock cycle{} ({} frames), found in {} solver call{} ({}).",
            solution.cycles,
            if solution.cycles == 1 { "" } else { "s" },
            solution.frames.len(),
            solution.probes.len(),
            if solution.probes.len() == 1 { "" } else { "s" },
            // The lengths tried, in the order the search tried them, so the
            // doubling and the binary search back over the gap are visible.
            solution
                .probes
                .iter()
                .map(|(cycles, sat)| format!("{cycles}{}", if *sat { "v" } else { "x" }))
                .collect::<Vec<_>>()
                .join(", "),
        ),
    );
    for note in &solution.notes {
        label(ui, egui::RichText::new(note).italics().weak());
    }

    // One row per signal and one *column* per frame, rather than the other
    // way round: a sequence is far longer than it is wide (a real design
    // can run well over a hundred frames over a handful of inputs), and a
    // row of bits reads as the waveform it is. Scrolls sideways here, since
    // no panel is that wide; vertically it just adds to the tab's own
    // scroll region.
    egui::ScrollArea::horizontal()
        .id_salt("system_solve_waveform")
        .auto_shrink([false, true])
        .show(ui, |ui| {
            let clock_inputs: HashSet<&str> =
                solution.clock_inputs.iter().map(String::as_str).collect();

            label(ui, "Inputs");
            for (index, name) in solution.input_labels.iter().enumerate() {
                // The clock is what the frames *are*, so it carries no
                // waveform of its own to show here.
                let waveform = if clock_inputs.contains(name.as_str()) {
                    "(one frame per edge)".to_string()
                } else {
                    bit_string(&solution.frames, index)
                };
                show_waveform_row(ui, name, &waveform);
            }

            ui.separator();
            label(ui, "Targets");
            for (index, name) in solution.target_labels.iter().enumerate() {
                show_waveform_row(ui, name, &bit_string(&solution.target_frames, index));
            }
            for (index, name) in solution.bus_labels.iter().enumerate() {
                show_bus_row(
                    ui,
                    name,
                    &solution.bus_frames,
                    index,
                    solution.bus_widths[index],
                );
            }

            if !solution.output_bus_labels.is_empty() {
                ui.separator();
                label(ui, "Outputs");
                for (index, name) in solution.output_bus_labels.iter().enumerate() {
                    show_bus_row(
                        ui,
                        name,
                        &solution.output_bus_frames,
                        index,
                        solution.output_bus_widths[index],
                    );
                }
            }

            if !solution.state_labels.is_empty() {
                ui.separator();
                egui::CollapsingHeader::new(format!(
                    "State ({} bits)",
                    solution.state_labels.len()
                ))
                .default_open(false)
                .show(ui, |ui| {
                    for (index, name) in solution.state_labels.iter().enumerate() {
                        show_waveform_row(ui, name, &bit_string(&solution.states, index));
                    }
                });
            }
        });
}

/// Column `index` of a frame-major table of bits, as one `"0"`/`"1"`
/// character per frame.
fn bit_string(frames: &[Vec<bool>], index: usize) -> String {
    frames
        .iter()
        .map(|frame| {
            if frame.get(index).copied().unwrap_or(false) {
                '1'
            } else {
                '0'
            }
        })
        .collect()
}

/// A bus's value per frame as the waveform shows it: three characters a
/// frame, the ASCII character (padded) where the value is a printable
/// one, its hex where it isn't — so `00 00 (  *     T  W  O` reads as the
/// message it is, zeros and all. Three wide rather than one because hex
/// needs two, and every frame must take the same room for the columns to
/// stay frames.
fn bus_text(frames: &[Vec<u64>], index: usize, width_bits: usize) -> String {
    let hex_width = width_bits.div_ceil(4);
    let cell = hex_width.max(2);
    let mut text = String::new();
    for frame in frames {
        let value = frame.get(index).copied().unwrap_or(0);
        match u8::try_from(value) {
            Ok(byte) if byte.is_ascii_graphic() || byte == b' ' => {
                text.push(char::from(byte));
                text.extend(std::iter::repeat_n(' ', cell - 1));
            }
            _ => text.push_str(&format!("{value:0cell$x}")),
        }
        text.push(' ');
    }
    text
}

/// One bus's row of the waveform: its name, then [`bus_text`] — with
/// the plain hex of every frame on hover, for when the characters are
/// what's in the way.
fn show_bus_row(
    ui: &mut egui::Ui,
    name: &str,
    frames: &[Vec<u64>],
    index: usize,
    width_bits: usize,
) {
    let hex: Vec<String> = frames
        .iter()
        .map(|frame| {
            format!(
                "{:0width$x}",
                frame.get(index).copied().unwrap_or(0),
                width = width_bits.div_ceil(4)
            )
        })
        .collect();
    ui.horizontal(|ui| {
        ui.add_space(12.0);
        ui.add(egui::Label::new(name).truncate().selectable(false))
            .on_hover_text(name);
        ui.label(egui::RichText::new(bus_text(frames, index, width_bits)).monospace())
            .on_hover_text(hex.join(" "));
    });
}

/// One signal's row of the waveform: its name, then its value in each
/// frame left to right. Monospaced so the columns of every row line up as
/// frames, and selectable so a sequence can be copied out.
fn show_waveform_row(ui: &mut egui::Ui, name: &str, waveform: &str) {
    ui.horizontal(|ui| {
        ui.add_space(12.0);
        ui.add(egui::Label::new(name).truncate().selectable(false))
            .on_hover_text(name);
        ui.label(egui::RichText::new(waveform).monospace());
    });
}

/// The step-by-step input values a run is made of, from the patterns as
/// typed: `rows[step][i]` is what input `labels[i]` holds in that step.
///
/// A pattern of one character holds its pin at that value for every
/// step; one of several is a bitstring, one character per step, left to
/// right. Every bitstring has to be the same length, and that length is
/// the number of steps; with no bitstring at all it is `steps`. A clock
/// input (`clock_inputs`) has no pattern — each step *is* one edge of it
/// — and reads as 0. Fails, naming the pin, on anything but 0s and 1s,
/// on an empty pattern, and on bitstrings that disagree in length.
fn expand_input_patterns(
    labels: &[String],
    clock_inputs: &[String],
    patterns: &BTreeMap<String, String>,
    steps: usize,
) -> Result<Vec<Vec<bool>>, String> {
    // Per input: its bits, and whether they were given per step.
    let mut parsed: Vec<(Vec<bool>, bool)> = Vec::with_capacity(labels.len());
    let mut bitstring_length: Option<(&str, usize)> = None;
    for label in labels {
        if clock_inputs.contains(label) {
            parsed.push((vec![false], false));
            continue;
        }
        let pattern = patterns
            .get(label)
            .map(String::as_str)
            .unwrap_or("0")
            .trim();
        if pattern.is_empty() {
            return Err(format!(
                "{label}: enter 0 or 1 to hold it, or a bitstring with one bit per step."
            ));
        }
        let mut bits = Vec::with_capacity(pattern.len());
        for character in pattern.chars() {
            match character {
                '0' => bits.push(false),
                '1' => bits.push(true),
                _ => {
                    return Err(format!(
                        "{label}: only 0 and 1 are allowed, not '{character}'."
                    ));
                }
            }
        }
        if bits.len() > 1 {
            match bitstring_length {
                Some((other, length)) if length != bits.len() => {
                    return Err(format!(
                        "{other} has {length} bits but {label} has {}: every bitstring must be \
                         the same length, one bit per step.",
                        bits.len()
                    ));
                }
                Some(_) => {}
                None => bitstring_length = Some((label, bits.len())),
            }
        }
        let per_step = bits.len() > 1;
        parsed.push((bits, per_step));
    }

    let steps = bitstring_length.map_or(steps, |(_, length)| length);
    Ok((0..steps)
        .map(|step| {
            parsed
                .iter()
                .map(|(bits, per_step)| if *per_step { bits[step] } else { bits[0] })
                .collect()
        })
        .collect())
}

/// Builds a [`Simulator`] over `graph` and runs it on `state`'s input
/// patterns, replacing the trace — or, if the patterns don't make a run
/// or the simulator can't be built, leaving the last trace and recording
/// why in [`SimulatorState::error`]. The simulator is rebuilt each run
/// rather than kept: building one is a pass over the graph, cheap next
/// to the run itself, and keeping one would mean another thing to
/// invalidate whenever the graph changes.
fn run_simulation(state: &mut SimulatorState, graph: &Graph) {
    let result = Simulator::new(graph).and_then(|simulator| {
        let labels: Vec<String> = simulator.input_labels().map(str::to_string).collect();
        let rows = expand_input_patterns(
            &labels,
            simulator.clock_inputs(),
            &state.input_patterns,
            state.steps,
        )?;
        Ok(simulator.run(rows.iter().map(Vec::as_slice)))
    });
    match result {
        Ok(simulation) => {
            state.simulation = Some(Box::new(simulation));
            state.error = None;
        }
        Err(err) => state.error = Some(err),
    }
}

/// One column of the simulator's output table: a lone output pin, or
/// every pin of an indexed bus (`O[0]`..`O[7]`) read together as one
/// number.
struct OutputColumn {
    /// The pin name, or the bus name without its index.
    name: String,
    /// Indices into a frame's `outputs`, most significant first — highest
    /// bus index down to lowest. One entry for a lone pin.
    bits: Vec<usize>,
    /// Whether this came from indexed pins, and so is shown as a number
    /// under [`OutputFormat::Hex`] and [`OutputFormat::Decimal`]. A lone
    /// pin is a bit in every format.
    bus: bool,
}

/// Groups `labels` (a [`Simulation`]'s output labels) into table columns:
/// indexed pins of one name become a bus, in the order the bus first
/// appears among the pins, everything else a column of its own.
fn output_columns(labels: &[String]) -> Vec<OutputColumn> {
    let mut columns: Vec<OutputColumn> = Vec::new();
    // Per column, its bus pins as `(bus index, output position)`, sorted
    // into `bits` once every pin has been seen.
    let mut bus_bits: Vec<Vec<(usize, usize)>> = Vec::new();
    for (position, label) in labels.iter().enumerate() {
        match split_bus_pin(label) {
            Some((name, index)) => {
                let column = columns
                    .iter()
                    .position(|column| column.bus && column.name == name)
                    .unwrap_or_else(|| {
                        columns.push(OutputColumn {
                            name: name.to_string(),
                            bits: Vec::new(),
                            bus: true,
                        });
                        bus_bits.push(Vec::new());
                        columns.len() - 1
                    });
                bus_bits[column].push((index, position));
            }
            None => {
                columns.push(OutputColumn {
                    name: label.clone(),
                    bits: vec![position],
                    bus: false,
                });
                bus_bits.push(Vec::new());
            }
        }
    }
    for (column, mut bits) in columns.iter_mut().zip(bus_bits) {
        if column.bus {
            bits.sort_unstable_by_key(|&(index, _)| std::cmp::Reverse(index));
            column.bits = bits.into_iter().map(|(_, position)| position).collect();
        }
    }
    columns
}

impl OutputColumn {
    /// Whether this column is shown as a number in `format`: a bus, in a
    /// numeric format, narrow enough for the `u128` it's read into. A
    /// lone pin — or a bus wider than that — is bits in every format.
    fn numeric(&self, format: OutputFormat) -> bool {
        self.bus && self.bits.len() <= 128 && format != OutputFormat::Bits
    }

    /// How many characters a value of this column takes in `format`, so
    /// the table can size the column before laying any row out.
    fn width(&self, format: OutputFormat) -> usize {
        let bits = self.bits.len();
        if !self.numeric(format) {
            return bits;
        }
        match format {
            OutputFormat::Bits => bits,
            // ASCII is one character, but its hex fallback is not.
            OutputFormat::Hex | OutputFormat::Ascii => bits.div_ceil(4),
            OutputFormat::Decimal => (u128::MAX >> (128 - bits)).to_string().len(),
        }
    }

    /// This column's value in `frame_outputs` (a frame's `outputs`), as
    /// the table shows it.
    fn value(&self, frame_outputs: &[bool], format: OutputFormat) -> String {
        let bit = |index: &usize| frame_outputs.get(*index).copied().unwrap_or(false);
        if !self.numeric(format) {
            return self
                .bits
                .iter()
                .map(|index| if bit(index) { '1' } else { '0' })
                .collect();
        }
        let value = self
            .bits
            .iter()
            .fold(0u128, |value, index| (value << 1) | u128::from(bit(index)));
        let hex = || format!("{value:0width$x}", width = self.width(format));
        match format {
            OutputFormat::Hex => hex(),
            OutputFormat::Decimal | OutputFormat::Bits => value.to_string(),
            // Printable ASCII only — space through tilde — so a control
            // code or a value past 7 bits shows as hex rather than as
            // nothing or a mojibake glyph. Space itself is shown as its
            // hex too, being invisible in a column of characters.
            OutputFormat::Ascii => match u8::try_from(value) {
                Ok(byte) if byte.is_ascii_graphic() => char::from(byte).to_string(),
                _ => hex(),
            },
        }
    }
}

/// The right panel's Simulate tab: every design input as a pattern —
/// `0`/`1` to hold it, or a bitstring with one bit per step — and the
/// graph run from power-up on them as a table: one row per step, one
/// column per input and output, an indexed output bus shown as bits, hex
/// or decimal.
///
/// The concrete counterpart of the Solve system tab: same reading of the
/// graph as a synchronous machine (see [`Simulator`]), but the inputs
/// are given rather than found — in the same frame-by-frame convention,
/// so a solved sequence's rows can be typed straight in as bitstrings.
/// The run is live: it reruns whenever a pattern or the step count
/// changes and still makes a valid run, and shows why when it doesn't.
fn show_simulator_section(ui: &mut egui::Ui, action: &mut MergeBooleanFunctions) {
    let state = &mut *action.simulator;
    let graph = &action.graph_data.graph;

    // The tab always has a run to show: the first time it's opened,
    // that's the defaults — every input held at 0. Not retried while it
    // is failing, or it would rebuild every frame.
    if state.simulation.is_none() && state.error.is_none() {
        run_simulation(state, graph);
    }

    let mut rerun = false;

    // The controls, bounded so they scroll rather than crowd the table
    // out on a design with many inputs; they otherwise take only what
    // they need. Never more than half the height, and never so much that
    // the fixed strip between them and the table — two separators, the
    // format row, the header — would be pushed past the bottom on a
    // window too short for even that.
    let spacing = ui.spacing();
    let strip_height = spacing.interact_size.y
        + ui.text_style_height(&egui::TextStyle::Monospace)
        + 2.0 * 6.0
        + 5.0 * spacing.item_spacing.y;
    let controls_height = (ui.available_height() * 0.5)
        .min(ui.available_height() - strip_height)
        .max(0.0);
    egui::ScrollArea::vertical()
        .id_salt("simulator_controls")
        .auto_shrink([false, true])
        .min_scrolled_height(0.0)
        .max_height(controls_height)
        .show(ui, |ui| {
            label(
                ui,
                "Runs this graph from power-up, one step per clock edge. Give each input 0 or 1 \
                 to hold it there, or a bitstring with one bit per step, left to right; every \
                 bitstring must be the same length, which is then the number of steps.",
            );

            let Some(simulation) = state.simulation.as_deref() else {
                if let Some(err) = &state.error {
                    ui.colored_label(egui::Color32::from_rgb(220, 80, 80), err);
                }
                return;
            };

            ui.separator();
            label(ui, "Inputs");
            /// Room for the pin name beside its pattern field. Long names
            /// truncate (with the full name on hover) rather than
            /// squeezing the field, which is where the typing happens.
            const NAME_WIDTH: f32 = 72.0;
            let field_width =
                (ui.available_width() - NAME_WIDTH - ui.spacing().item_spacing.x).max(24.0);
            let clock_inputs: HashSet<&str> =
                simulation.clock_inputs.iter().map(String::as_str).collect();
            let mut any_bitstring = false;
            for name in &simulation.input_labels {
                if clock_inputs.contains(name.as_str()) {
                    continue;
                }
                let pattern = state
                    .input_patterns
                    .entry(name.clone())
                    .or_insert_with(|| "0".to_string());
                any_bitstring |= pattern.trim().chars().count() > 1;
                ui.horizontal(|ui| {
                    fixed_width_label(ui, name, NAME_WIDTH);
                    if ui
                        .add(
                            egui::TextEdit::singleline(pattern)
                                .font(egui::TextStyle::Monospace)
                                .desired_width(field_width),
                        )
                        .changed()
                    {
                        rerun = true;
                    }
                });
            }

            ui.horizontal(|ui| {
                if any_bitstring {
                    label(ui, "Steps: as many as the bitstrings have bits.");
                } else {
                    label(ui, "Steps");
                    if ui
                        .add(egui::DragValue::new(&mut state.steps).range(1..=10_000))
                        .changed()
                    {
                        rerun = true;
                    }
                }
            });
            if let Some(err) = &state.error {
                ui.colored_label(egui::Color32::from_rgb(220, 80, 80), err);
            }
            for note in &simulation.notes {
                label(ui, egui::RichText::new(note).italics().weak());
            }
        });

    if rerun {
        run_simulation(state, graph);
    }

    let Some(simulation) = state.simulation.as_deref() else {
        return;
    };

    // Outside the scrolling controls, so the table never loses its
    // format switch or its length to a design with many inputs.
    ui.separator();
    ui.horizontal(|ui| {
        label(
            ui,
            format!(
                "{} step{}",
                simulation.frames.len(),
                if simulation.frames.len() == 1 {
                    ""
                } else {
                    "s"
                }
            ),
        );
        if output_columns(&simulation.output_labels)
            .iter()
            .any(|column| column.bus)
        {
            ui.separator();
            label(ui, "Buses as");
            for format in OutputFormat::ALL {
                ui.selectable_value(&mut state.output_format, format, format.label());
            }
        }
    });
    show_simulation_table(ui, simulation, state.output_format);
}

/// `simulation`'s trace as a table in whatever space is left: a header
/// row, then one row per frame — its index, every output column (see
/// [`output_columns`]) in `format`, then every non-clock input. Outputs
/// before inputs because they are what the table is read for, and the
/// panel is rarely wide enough for both without scrolling.
///
/// Rows are single monospaced strings with the columns padded to fixed
/// widths, rather than a grid of widgets: a trace can run to thousands
/// of cycles, and this way each row is one widget and the columns line
/// up by construction. The body is virtualized — only the rows in view
/// are laid out — with the header pinned above it and scrolled sideways
/// in step.
fn show_simulation_table(ui: &mut egui::Ui, simulation: &Simulation, format: OutputFormat) {
    let clock_inputs: HashSet<&str> = simulation.clock_inputs.iter().map(String::as_str).collect();
    let inputs: Vec<(usize, &str)> = simulation
        .input_labels
        .iter()
        .enumerate()
        .map(|(index, name)| (index, name.as_str()))
        .filter(|(_, name)| !clock_inputs.contains(name))
        .collect();
    let columns = output_columns(&simulation.output_labels);

    // Column widths in characters: each wide enough for its heading and
    // its widest value.
    let step_width = "Step".len().max(simulation.frames.len().to_string().len());
    let input_widths: Vec<usize> = inputs
        .iter()
        .map(|(_, name)| name.chars().count())
        .collect();
    let output_widths: Vec<usize> = columns
        .iter()
        .map(|column| column.name.chars().count().max(column.width(format)))
        .collect();

    let mut header = format!("{:>step_width$}", "Step");
    for (column, width) in columns.iter().zip(&output_widths) {
        header.push_str(&format!("  {:>width$}", column.name));
    }
    header.push_str("  │");
    for ((_, name), width) in inputs.iter().zip(&input_widths) {
        header.push_str(&format!("  {name:>width$}"));
    }
    let row_text = |frame_index: usize| {
        let frame = &simulation.frames[frame_index];
        let mut text = format!("{frame_index:>step_width$}");
        for (column, width) in columns.iter().zip(&output_widths) {
            text.push_str(&format!(
                "  {:>width$}",
                column.value(&frame.outputs, format)
            ));
        }
        text.push_str("  │");
        for ((index, _), width) in inputs.iter().zip(&input_widths) {
            let bit = u8::from(frame.inputs.get(*index).copied().unwrap_or(false));
            text.push_str(&format!("  {bit:>width$}"));
        }
        text
    };

    let row_height = ui.text_style_height(&egui::TextStyle::Monospace);
    let monospace = |text: String| {
        egui::Label::new(egui::RichText::new(text).monospace())
            .wrap_mode(egui::TextWrapMode::Extend)
    };

    // The header scrolls sideways with the body: it's given the body's
    // offset from the frame before, which is a frame behind but never
    // visibly so.
    let offset_id = ui.id().with("simulation_table_offset");
    let offset_x: f32 = ui.data(|data| data.get_temp(offset_id)).unwrap_or(0.0);
    egui::ScrollArea::horizontal()
        .id_salt("simulation_table_header")
        .auto_shrink([false, true])
        .scroll_bar_visibility(egui::scroll_area::ScrollBarVisibility::AlwaysHidden)
        .horizontal_scroll_offset(offset_x)
        .show(ui, |ui| {
            ui.add(monospace(header).selectable(false));
        });
    ui.separator();

    // No floor on the body's height: a scroll area otherwise insists on
    // 64px once it has to scroll, and on a short window that is more than
    // the panel below leaves, so it would paint over that panel.
    let output = egui::ScrollArea::both()
        .id_salt("simulation_table_body")
        .auto_shrink([false, false])
        .min_scrolled_height(0.0)
        .show_rows(ui, row_height, simulation.frames.len(), |ui, range| {
            for frame_index in range {
                ui.add(monospace(row_text(frame_index)));
            }
        });
    ui.data_mut(|data| data.insert_temp(offset_id, output.state.offset.x));
}

/// The right panel's Solve boolean functions tab for a
/// [`MergeBooleanFunctions`] action: the cone pin list (for the cone
/// grouping), then the per-function Solve with every function's pins. `functions` is the action's
/// `"BooleanFunction"` cells, passed in so the caller can keep the borrow
/// of `action` short.
///
/// Returns whether Apply was clicked on the cone pin list — the recompute
/// it asks for needs the whole project, which this only has one action of.
fn show_boolean_function_editor(
    ui: &mut egui::Ui,
    action: &mut MergeBooleanFunctions,
    functions: &[Cell],
) -> bool {
    let mut apply_cone_pins = false;

    // One scroll region for the whole editor, bounded to
    // whatever height the Graph information panel leaves
    // it (`auto_shrink` off): a plain `Ui` never clips, so
    // content taller than that space would otherwise be
    // painted straight over that panel underneath. The
    // cone editor alone can be that tall on a short
    // window.
    egui::ScrollArea::vertical()
        .id_salt("boolean_function_editor")
        .auto_shrink([false, false])
        .show(ui, |ui| {
            // Only the cone grouping has a cone to narrow;
            // the other two merge the whole graph
            // unconditionally, so the list would do
            // nothing there.
            if action.grouping() == BooleanFunctionGrouping::ConeOfInfluence {
                label(
                    ui,
                    "Output pins to keep the cone of influence of. Leave \
                     empty to merge the whole graph; with any pin named, \
                     every cell outside the union of those pins' cones is \
                     pruned.",
                );

                let field_width = editor_field_width(ui, 1);
                let mut remove_row = None;
                for (row_index, pin_name) in action.pending_cone_output_pins.iter_mut().enumerate()
                {
                    ui.horizontal(|ui| {
                        ui.add(egui::TextEdit::singleline(pin_name).desired_width(field_width));
                        if egui_icons::icon_button(ui, ICON_CLOSE).clicked() {
                            remove_row = Some(row_index);
                        }
                    });
                }
                if let Some(row_index) = remove_row {
                    action.pending_cone_output_pins.remove(row_index);
                }
                if ui.button(ICON_ADD).clicked() {
                    action.pending_cone_output_pins.push(String::new());
                }
                if ui.button("Apply").clicked() {
                    apply_cone_pins = true;
                }
                ui.separator();
            }

            label(
                ui,
                "Solves every boolean function below at once, for one \
                 combinational instant: fill in whichever pins you already \
                 know and the rest come back filled in red. Nothing here \
                 steps through time — the Solve system tab does that.",
            );

            // Solve (and any error from the last attempt)
            // sits above the pin list: it is the thing to
            // reach for, and the list below it can run to
            // hundreds of functions.
            if ui.button("Solve").clicked() {
                solve_boolean_functions(action);
            }
            if let Some(err) = &action.solve_error {
                ui.colored_label(egui::Color32::from_rgb(220, 80, 80), err);
            }
            ui.separator();

            for (index, cell) in functions.iter().enumerate() {
                let Cell::MergeCell {
                    inputs,
                    outputs,
                    ancestor_cells,
                    ..
                } = cell
                else {
                    continue;
                };
                egui::CollapsingHeader::new(format!(
                    "BooleanFunction #{index} ({} gates)",
                    ancestor_cells.len()
                ))
                .default_open(functions.len() == 1)
                .show(ui, |ui| {
                    label(ui, "Inputs");
                    show_boolean_input_rows(
                        ui,
                        inputs,
                        &mut action.pin_values,
                        &mut action.decimal_groups,
                    );
                    ui.separator();
                    label(ui, "Outputs");
                    show_boolean_pin_rows(ui, outputs, &mut action.pin_values);
                });
            }
        });

    apply_cone_pins
}

/// Runs `kind` against `project`'s current graph (the previous action's
/// output, or the loaded graph if `project.graph_actions` is still empty)
/// and appends the result to `project.graph_actions`.
fn append_graph_action(
    project: &mut Project,
    kind: GraphActionKind,
    last_error: &Mutex<Option<String>>,
) {
    let input_graph = match project.graph_actions.last() {
        Some(action) => &action.graph_data().graph,
        None => &project.loaded_graph.graph_data.graph,
    };
    let layout_mode = project.loaded_graph.graph_data.layout_mode;

    match kind.create(Uuid::new_v4(), input_graph, layout_mode) {
        Ok(action) => {
            project.graph_actions.push(action);
            // Select the action just created so its output is what gets shown.
            project.selected_action = Some(project.graph_actions.len() - 1);
            *last_error.lock().unwrap() = None;
        }
        Err(err) => *last_error.lock().unwrap() = Some(err.to_string()),
    }
}

/// Recomputes `graph_data` for every action at or after `start`, chaining
/// each one off the previous action's output (or the loaded graph, for
/// index 0). Needed after an action is removed or reordered, since that
/// shifts the input graph every later action builds on. The camera is
/// unaffected by any of this: it lives on `Project`, shared across every
/// action, not on the `GraphData` this recomputes.
fn recompute_graph_actions_from(
    project: &mut Project,
    start: usize,
) -> Result<(), AmbiguousRemoval> {
    let layout_mode = project.loaded_graph.graph_data.layout_mode;
    for i in start..project.graph_actions.len() {
        let input_graph = if i == 0 {
            &project.loaded_graph.graph_data.graph
        } else {
            &project.graph_actions[i - 1].graph_data().graph
        };
        let graph_data = project.graph_actions[i].recompute_graph_data(input_graph, layout_mode)?;
        project.graph_actions[i].set_graph_data(graph_data);
    }
    Ok(())
}

/// Applies a close/move-up/move-down edit from the action list, then
/// recomputes every action from the earliest affected index onward.
fn apply_graph_action_edit(
    project: &mut Project,
    edit: PendingGraphActionEdit,
    last_error: &Mutex<Option<String>>,
) {
    let start = match edit {
        PendingGraphActionEdit::Remove(action_index) => {
            project.graph_actions.remove(action_index);
            project.selected_action = match project.selected_action {
                Some(selected) if selected == action_index => None,
                Some(selected) if selected > action_index => Some(selected - 1),
                other => other,
            };
            action_index
        }
        PendingGraphActionEdit::MoveUp(action_index) => {
            project.graph_actions.swap(action_index, action_index - 1);
            project.selected_action = match project.selected_action {
                Some(selected) if selected == action_index => Some(action_index - 1),
                Some(selected) if selected == action_index - 1 => Some(action_index),
                other => other,
            };
            action_index - 1
        }
        PendingGraphActionEdit::MoveDown(action_index) => {
            project.graph_actions.swap(action_index, action_index + 1);
            project.selected_action = match project.selected_action {
                Some(selected) if selected == action_index => Some(action_index + 1),
                Some(selected) if selected == action_index + 1 => Some(action_index),
                other => other,
            };
            action_index
        }
    };

    match recompute_graph_actions_from(project, start) {
        Ok(()) => *last_error.lock().unwrap() = None,
        Err(err) => *last_error.lock().unwrap() = Some(err.to_string()),
    }
}

/// Commits a `PropagatePinValues` action's `pending_entries` (the
/// right-side panel's editable rows) into its `entries`, then recomputes
/// its `graph_data` — and everything downstream of it — to match. Does
/// nothing if `action_index` isn't a `PropagatePinValues` action.
fn apply_pin_value_propagation_edit(
    project: &mut Project,
    action_index: usize,
    last_error: &Mutex<Option<String>>,
) {
    if let Some(GraphAction::PropagatePinValues(action)) =
        project.graph_actions.get_mut(action_index)
    {
        action.entries = action
            .pending_entries
            .iter()
            .map(|(input, target)| (input.trim().to_string(), target.trim().to_string()))
            .filter(|(input, target)| !input.is_empty() && !target.is_empty())
            .collect();
    }

    match recompute_graph_actions_from(project, action_index) {
        Ok(()) => *last_error.lock().unwrap() = None,
        Err(err) => *last_error.lock().unwrap() = Some(err.to_string()),
    }
}

/// Regenerates a project's graphs from one row of the action list down,
/// as its refresh button asks for.
///
/// `action_index` of `None` is the "Load GDS file" row: the loaded graph
/// itself is re-normalized ([`Graph::normalize`]) and its snarl rebuilt,
/// then every action is recomputed on top of the result. That is the only
/// thing that revisits a loaded graph — a project restored from saved app
/// state carries the `GraphData` it was built with and never passes
/// through [`parse_graph`] again — so it is what picks up a normalization
/// added since the project was first loaded, without having to re-open the
/// file.
///
/// `Some(index)` regenerates that action from its own input graph and, as
/// every edit here does, everything after it.
fn refresh_graphs_from(
    project: &mut Project,
    action_index: Option<usize>,
    last_error: &Mutex<Option<String>>,
) {
    let start = match action_index {
        None => {
            let layout_mode = project.loaded_graph.graph_data.layout_mode;
            let graph_data = &mut project.loaded_graph.graph_data;
            graph_data.graph.normalize();
            graph_data.snarl = create_snarl(layout_mode, &graph_data.graph);
            0
        }
        Some(index) => index,
    };

    match recompute_graph_actions_from(project, start) {
        Ok(()) => *last_error.lock().unwrap() = None,
        Err(err) => *last_error.lock().unwrap() = Some(err.to_string()),
    }
}

/// Commits a [`MergeBooleanFunctions`] action's edited cone output pin
/// list into `cone_output_pins` (trimmed, with blank rows dropped) and
/// recomputes this action and everything after it. Mirrors
/// [`apply_pin_value_propagation_edit`].
fn apply_cone_output_pins_edit(
    project: &mut Project,
    action_index: usize,
    last_error: &Mutex<Option<String>>,
) {
    if let Some(GraphAction::MergeBooleanFunctions(action)) =
        project.graph_actions.get_mut(action_index)
    {
        action.cone_output_pins = action
            .pending_cone_output_pins
            .iter()
            .map(|name| name.trim().to_string())
            .filter(|name| !name.is_empty())
            .collect();
        // A merge narrowed to a different cone is a different graph, so
        // the pin values solved against the old one no longer describe it.
        action.pin_values.clear();
        action.solve_error = None;
    }

    match recompute_graph_actions_from(project, action_index) {
        Ok(()) => *last_error.lock().unwrap() = None,
        Err(err) => *last_error.lock().unwrap() = Some(err.to_string()),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct LoadGraph {
    orig_path: PathBuf,
    graph_data: GraphData,
    /// The inputs the layout reader had to invent for nets nothing
    /// drives — shown in the right panel so they are never a surprise.
    /// Empty for a project restored from state saved before layouts were
    /// what got loaded.
    #[serde(default)]
    dummy_inputs: Vec<DummyInput>,
}

impl LoadGraph {
    fn orig_filename(&self) -> String {
        self.orig_path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .to_string()
    }
}

/// Renders one project's row in the left sidebar (its header with the
/// close/add icons, and the "Load GDS file" preview group below it).
/// Selecting, removing or queuing a [`GraphActionKind`] for `item` is
/// reported back through the `&mut` out-params rather than applied
/// directly, since the caller is mid-iteration over the project list.
fn show_project_list_item(
    ui: &mut egui::Ui,
    index: usize,
    item: &Project,
    selected_graph: &mut Option<usize>,
    remove_index: &mut Option<usize>,
    pending_action: &mut Option<(usize, GraphActionKind)>,
    pending_action_edit: &mut Option<(usize, PendingGraphActionEdit)>,
    pending_select: &mut Option<(usize, Option<usize>)>,
    pending_refresh: &mut Option<(usize, Option<usize>)>,
) {
    let mut remove = false;

    let mut frame = egui::Frame::group(ui.style());
    if *selected_graph == Some(index) {
        frame.stroke.width *= 3.0;
    }

    let frame_inner_response =
        ui.scope_builder(egui::UiBuilder::new().sense(egui::Sense::click()), |ui| {
            frame.show(ui, |ui| {
                ui.horizontal(|ui| {
                    label(ui, item.loaded_graph.orig_filename());

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if egui_icons::icon_button(ui, ICON_CLOSE).clicked() {
                            remove = true;
                        }

                        let add_response = egui_icons::icon_button(ui, ICON_ADD);
                        egui::Popup::menu(&add_response).show(|ui| {
                            for kind in [
                                GraphActionKind::RemoveClockBufferCells,
                                GraphActionKind::MergeMuxedResetableFlipflops,
                                GraphActionKind::MergeShiftRegisters,
                                GraphActionKind::MergeBooleanFunctions,
                                GraphActionKind::MergeBooleanFunctionsByRegisterScc,
                                GraphActionKind::MergeBooleanFunctionsByConeOfInfluence,
                                GraphActionKind::PropagatePinValues,
                            ] {
                                if ui.button(kind.label()).clicked() {
                                    *pending_action = Some((index, kind));
                                    ui.close();
                                }
                            }
                        });
                    });
                });

                ui.separator();

                let mut load_graph_frame = egui::Frame::group(ui.style());
                if item.selected_action.is_none() {
                    load_graph_frame.stroke.width *= 3.0;
                }
                // Set when this row's refresh icon was clicked, so the
                // row-click fallback below doesn't also move the selection.
                let mut load_graph_refreshed = false;
                let load_graph_response =
                    ui.scope_builder(egui::UiBuilder::new().sense(egui::Sense::click()), |ui| {
                        load_graph_frame.show(ui, |ui| {
                            // The buttons are laid out first, right to left,
                            // and the title takes what's left — so a title too
                            // long for the row truncates instead of running
                            // underneath them.
                            ui.horizontal(|ui| {
                                ui.with_layout(
                                    egui::Layout::right_to_left(egui::Align::Center),
                                    |ui| {
                                        ui.add_enabled(false, |ui: &mut egui::Ui| {
                                            egui_icons::icon_button(ui, ICON_CLOSE)
                                        });
                                        ui.add_enabled(false, |ui: &mut egui::Ui| {
                                            egui_icons::icon_button(ui, ICON_ARROW_UPWARD)
                                        });
                                        ui.add_enabled(false, |ui: &mut egui::Ui| {
                                            egui_icons::icon_button(ui, ICON_ARROW_DOWNWARD)
                                        });
                                        // Added last, so this right-to-left
                                        // layout puts it left of the arrows.
                                        // Unlike them it *is* enabled here:
                                        // this is the only way to re-run the
                                        // load-time normalization over a
                                        // project restored from saved state.
                                        if egui_icons::icon_button(ui, ICON_REFRESH).clicked() {
                                            *pending_refresh = Some((index, None));
                                            load_graph_refreshed = true;
                                        }
                                        ui.with_layout(
                                            egui::Layout::left_to_right(egui::Align::Center),
                                            |ui| truncating_label(ui, "Load GDS file"),
                                        );
                                    },
                                );
                            });
                            ui.separator();

                            ui.horizontal(|ui| {
                                label(
                                    ui,
                                    format!("Load GDS file {}", item.loaded_graph.orig_filename()),
                                );
                            });
                        });
                    });
                if !load_graph_refreshed
                    && ui.input(|i| i.pointer.any_click())
                    && ui.rect_contains_pointer(load_graph_response.response.rect)
                {
                    *pending_select = Some((index, None));
                }

                let action_count = item.graph_actions.len();
                for (action_index, action) in item.graph_actions.iter().enumerate() {
                    let mut action_frame = egui::Frame::group(ui.style());
                    if item.selected_action == Some(action_index) {
                        action_frame.stroke.width *= 3.0;
                    }
                    // Set when a close/reorder/refresh icon was clicked, so the
                    // row-click fallback below doesn't override the selection that
                    // `apply_graph_action_edit` works out for a reorder (it keeps
                    // whichever action was selected pinned to that same action,
                    // not to whatever index it now occupies).
                    let mut action_edited = false;
                    let action_response = ui.scope_builder(
                        egui::UiBuilder::new().sense(egui::Sense::click()),
                        |ui| {
                            action_frame.show(ui, |ui| {
                                // As on the "Load GDS file" row above: buttons
                                // first, title last, so a long action name
                                // truncates rather than running under them.
                                ui.horizontal(|ui| {
                                    ui.with_layout(
                                        egui::Layout::right_to_left(egui::Align::Center),
                                        |ui| {
                                            if egui_icons::icon_button(ui, ICON_CLOSE).clicked() {
                                                *pending_action_edit = Some((
                                                    index,
                                                    PendingGraphActionEdit::Remove(action_index),
                                                ));
                                                action_edited = true;
                                            }
                                            let can_move_up = action_index > 0;
                                            if ui
                                                .add_enabled(can_move_up, |ui: &mut egui::Ui| {
                                                    egui_icons::icon_button(ui, ICON_ARROW_UPWARD)
                                                })
                                                .clicked()
                                            {
                                                *pending_action_edit = Some((
                                                    index,
                                                    PendingGraphActionEdit::MoveUp(action_index),
                                                ));
                                                action_edited = true;
                                            }
                                            let can_move_down = action_index + 1 < action_count;
                                            if ui
                                                .add_enabled(can_move_down, |ui: &mut egui::Ui| {
                                                    egui_icons::icon_button(ui, ICON_ARROW_DOWNWARD)
                                                })
                                                .clicked()
                                            {
                                                *pending_action_edit = Some((
                                                    index,
                                                    PendingGraphActionEdit::MoveDown(action_index),
                                                ));
                                                action_edited = true;
                                            }
                                            // Added last, so this right-to-left
                                            // layout puts it left of the arrows.
                                            if egui_icons::icon_button(ui, ICON_REFRESH).clicked() {
                                                *pending_refresh =
                                                    Some((index, Some(action_index)));
                                                action_edited = true;
                                            }
                                            ui.with_layout(
                                                egui::Layout::left_to_right(egui::Align::Center),
                                                |ui| truncating_label(ui, action.kind().label()),
                                            );
                                        },
                                    );
                                });
                                ui.separator();

                                ui.horizontal(|ui| {
                                    label(ui, action.kind().label());
                                });
                            });
                        },
                    );
                    if !action_edited
                        && ui.input(|i| i.pointer.any_click())
                        && ui.rect_contains_pointer(action_response.response.rect)
                    {
                        *pending_select = Some((index, Some(action_index)));
                    }
                }
            })
        });

    if remove {
        *remove_index = Some(index);
    } else if ui.input(|i| i.pointer.any_click())
        && ui.rect_contains_pointer(frame_inner_response.response.rect)
    {
        *selected_graph = Some(index);
    }
}

pub struct App {
    pending_new_project: Arc<Mutex<Option<Project>>>,
    last_error: Arc<Mutex<Option<String>>>,
    app_state: AppState,
}

impl App {
    pub fn new(cx: &CreationContext) -> Self {
        egui_icons::initialize(&cx.egui_ctx);

        let mut startup_error = None;

        let app_state: AppState =
            load_json(cx.storage, "app_state", &mut startup_error).unwrap_or_default();

        App {
            pending_new_project: Default::default(),
            last_error: Arc::new(Mutex::new(startup_error)),
            app_state: app_state,
        }
    }

    /// Opens the native file picker on a background thread — it blocks
    /// until the user picks a netlist or cancels — and hands whatever it
    /// loads back through [`publish_loaded_graph`].
    #[cfg(not(target_arch = "wasm32"))]
    fn pick_graph_data_file(&self, ctx: egui::Context) {
        let pending_new_project = Arc::clone(&self.pending_new_project);
        let last_error = Arc::clone(&self.last_error);

        std::thread::spawn(move || {
            let Some(file_path) = rfd::FileDialog::new()
                .add_filter("GDSII layouts", &["gds", "gds2", "gdsii"])
                .pick_file()
            else {
                return; // Dialog cancelled; nothing to report.
            };

            let loaded = read_gds(&file_path, &Sky130Options::default())
                .map(|layout| (file_path, layout))
                .map_err(|err| err.to_string());
            publish_loaded_graph(loaded, &pending_new_project, &last_error, &ctx);
        });
    }

    /// Opens the browser's file picker (an `<input type="file">` rfd puts
    /// up for us) and reads the upload's bytes, both asynchronously on the
    /// single browser thread. The page only ever gets a file's name and
    /// contents — never a path it could re-read later — so the bare name
    /// stands in as the project's `orig_path`.
    #[cfg(target_arch = "wasm32")]
    fn pick_graph_data_file(&self, ctx: egui::Context) {
        let pending_new_project = Arc::clone(&self.pending_new_project);
        let last_error = Arc::clone(&self.last_error);

        wasm_bindgen_futures::spawn_local(async move {
            let Some(file) = rfd::AsyncFileDialog::new()
                .add_filter("GDSII layouts", &["gds", "gds2", "gdsii"])
                .pick_file()
                .await
            else {
                return; // Dialog cancelled; nothing to report.
            };

            let file_name = PathBuf::from(file.file_name());
            let loaded = parse_gds(&file.read().await, &Sky130Options::default())
                .map(|layout| (file_name, layout))
                .map_err(|err| err.to_string());
            publish_loaded_graph(loaded, &pending_new_project, &last_error, &ctx);
        });
    }
}

/// Hands a freshly picked netlist (or the message of whatever stopped it
/// loading) back to the UI from the thread or browser task the file dialog
/// ran on, and wakes egui: neither of those mutations is otherwise visible
/// to it, so without the repaint request the project would only appear on
/// the next unrelated frame.
fn publish_loaded_graph(
    loaded: Result<(PathBuf, Layout), String>,
    pending_new_project: &Mutex<Option<Project>>,
    last_error: &Mutex<Option<String>>,
    ctx: &egui::Context,
) {
    match loaded {
        Ok((
            orig_path,
            Layout {
                graph,
                dummy_inputs,
            },
        )) => {
            let layout_mode = LayoutMode::default();
            let snarl = create_snarl(layout_mode, &graph);
            *pending_new_project.lock().unwrap() = Some(Project {
                id: Uuid::new_v4(),
                loaded_graph: LoadGraph {
                    orig_path,
                    graph_data: GraphData {
                        graph,
                        layout_mode,
                        snarl,
                    },
                    dummy_inputs,
                },
                graph_actions: Default::default(),
                selected_action: None,
                graph_viewer: Default::default(),
            });
            *last_error.lock().unwrap() = None;
        }
        Err(message) => *last_error.lock().unwrap() = Some(message),
    }

    ctx.request_repaint();
}

impl eframe::App for App {
    fn logic(&mut self, _ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // First handle logic for existing projects, then add new project.
        for project in &mut self.app_state.projects {
            project.graph_viewer.logic();
        }

        if let Some(mut project) = self.pending_new_project.lock().unwrap().take() {
            project.graph_viewer.request_zoom_to_fit();
            self.app_state.projects.push(project);
            // Select newly loaded project
            self.app_state.selected_graph = Some(self.app_state.projects.len() - 1);
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("top_panel").show(ui, |ui| {
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("File", |ui| {
                    if ui.button("Load GDS file").clicked() {
                        self.pick_graph_data_file(ui.ctx().clone());
                    }

                    // The web build runs inside a tab it can't close itself.
                    #[cfg(not(target_arch = "wasm32"))]
                    {
                        ui.separator();

                        if ui.button("Quit").clicked() {
                            ui.send_viewport_cmd(egui::ViewportCommand::Close);
                        }
                    }
                });
                ui.add_space(16.0);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    egui::widgets::global_theme_preference_switch(ui);
                });
            });
        });

        egui::Panel::left("left-side-bar").show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                let mut remove_index = None;
                let mut pending_action = None;
                let mut pending_action_edit = None;
                let mut pending_select = None;
                let mut pending_refresh = None;
                ui.vertical(|ui| {
                    for (index, item) in self.app_state.projects.iter().enumerate() {
                        show_project_list_item(
                            ui,
                            index,
                            item,
                            &mut self.app_state.selected_graph,
                            &mut remove_index,
                            &mut pending_action,
                            &mut pending_action_edit,
                            &mut pending_select,
                            &mut pending_refresh,
                        );
                    }
                });

                if let Some(index) = remove_index {
                    self.app_state.projects.remove(index);
                    self.app_state.selected_graph = match self.app_state.selected_graph {
                        Some(selected) if selected == index => None,
                        Some(selected) if selected > index => Some(selected - 1),
                        other => other,
                    };
                }

                if let Some((index, kind)) = pending_action {
                    append_graph_action(
                        &mut self.app_state.projects[index],
                        kind,
                        &self.last_error,
                    );
                }

                if let Some((index, edit)) = pending_action_edit {
                    apply_graph_action_edit(
                        &mut self.app_state.projects[index],
                        edit,
                        &self.last_error,
                    );
                }

                if let Some((index, action_index)) = pending_refresh {
                    refresh_graphs_from(
                        &mut self.app_state.projects[index],
                        action_index,
                        &self.last_error,
                    );
                }

                if let Some((index, selection)) = pending_select {
                    self.app_state.projects[index].selected_action = selection;
                }
            });
        });

        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.horizontal(|ui| {
                if let Some(index) = self.app_state.selected_graph {
                    let selected_project = &mut self.app_state.projects[index];
                    let previous_layout_mode = selected_project.loaded_graph.graph_data.layout_mode;
                    egui::ComboBox::from_id_salt("layout_mode_combo")
                        .width(125.0)
                        .selected_text(match selected_project.loaded_graph.graph_data.layout_mode {
                            LayoutMode::Grid => "Grid layout",
                            LayoutMode::Centroid => "Physical layout",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut selected_project.loaded_graph.graph_data.layout_mode,
                                LayoutMode::Grid,
                                "Grid layout",
                            );
                            ui.selectable_value(
                                &mut selected_project.loaded_graph.graph_data.layout_mode,
                                LayoutMode::Centroid,
                                "Physical layout",
                            );
                        });
                    // Layout mode is a project-wide rendering setting, so changing it
                    // re-lays-out the loaded graph *and* every action's graph, keeping
                    // the whole chain visually consistent.
                    if selected_project.loaded_graph.graph_data.layout_mode != previous_layout_mode
                    {
                        selected_project.loaded_graph.graph_data.snarl = create_snarl(
                            selected_project.loaded_graph.graph_data.layout_mode,
                            &selected_project.loaded_graph.graph_data.graph,
                        );
                        if let Err(err) = recompute_graph_actions_from(selected_project, 0) {
                            *self.last_error.lock().unwrap() = Some(err.to_string());
                        } else {
                            *self.last_error.lock().unwrap() = None;
                        }
                        selected_project.graph_viewer.request_zoom_to_fit();
                    }

                    if ui
                        .button(egui_icons::icons::MDI_FIT_TO_PAGE_OUTLINE)
                        .clicked()
                    {
                        selected_project.graph_viewer.request_zoom_to_fit();
                    }
                }
            });
        });

        egui::Panel::bottom("status_bar").show(ui, |ui| {
            ui.horizontal(|ui| {
                let mut last_error = self.last_error.lock().unwrap();
                let mut dismissed = false;
                if let Some(err) = &*last_error {
                    let response = ui
                        .colored_label(egui::Color32::from_rgb(220, 80, 80), err)
                        .interact(egui::Sense::click())
                        .on_hover_text("Click to dismiss")
                        .on_hover_cursor(egui::CursorIcon::PointingHand);
                    dismissed = response.clicked();
                } else {
                    label(ui, "Ready");
                }
                if dismissed {
                    *last_error = None;
                }
            });
        });

        // Persistently visible — one panel for whatever's selected, rather
        // than appearing/disappearing (and shifting the rest of the
        // layout) as the selection changes. Its title is always the
        // selected action's own name; its body depends on which kind of
        // action that is (empty for a kind with nothing to edit here).
        egui::Panel::right("action_editor").show(ui, |ui| {
            let Some(project_index) = self.app_state.selected_graph else {
                ui.heading("No project selected");
                return;
            };

            let project = &mut self.app_state.projects[project_index];
            // A stale index falls back to the loaded graph, like the graph
            // view does.
            let action_index = project
                .selected_action
                .filter(|index| *index < project.graph_actions.len());

            match action_index.map(|index| &project.graph_actions[index]) {
                Some(action) => {
                    ui.heading(action.kind().label());
                    ui.separator();
                }
                None => {
                    ui.heading("Load GDS file");
                    ui.separator();
                }
            }

            // Below whatever the selection above offers to edit, and
            // always describing the same graph the central view is showing
            // — the selected action's result, or the loaded graph when
            // nothing downstream of it is selected.
            //
            // Anchored as its own bottom panel rather than laid out after
            // the editor: an editor's ScrollArea eagerly claims every
            // remaining pixel of panel height once its content is long
            // (a cone-of-influence merge produces hundreds of boolean
            // functions), which would push this section out of sight
            // entirely. Reserving the space up front is what keeps it
            // visible for every action; it stays drag-resizable if a
            // particular graph wants more or less of it.
            let graph = match action_index {
                Some(index) => &project.graph_actions[index].graph_data().graph,
                None => &project.loaded_graph.graph_data.graph,
            };
            egui::Panel::bottom("graph_information")
                .resizable(true)
                .default_size(260.0)
                .min_size(80.0)
                .show(ui, |ui| {
                    show_graph_information(
                        ui,
                        graph,
                        &mut self.app_state.graph_information_grouping,
                    )
                });

            // The loaded layout has nothing to edit, but it may have
            // something to confess: inputs the reader invented for nets
            // nothing drives. Those go here, above the graph information,
            // where they can't be missed.
            if action_index.is_none() {
                show_dummy_inputs(ui, &project.loaded_graph.dummy_inputs);
            }

            if let Some(action_index) = action_index {
                match project.graph_actions.get_mut(action_index) {
                    Some(GraphAction::MergeBooleanFunctions(action)) => {
                        // One tab each: the two solves and the simulator
                        // ask different questions of the graph (see
                        // `show_system_solve_section`), and each wants
                        // the whole height for its results — hundreds of
                        // functions, a waveform, a trace. Only a boolean
                        // function merge offers the system solve and the
                        // simulator: it's that action's graph, already
                        // narrowed to whatever cone was asked for, that
                        // gets unrolled or run.
                        ui.horizontal(|ui| {
                            for tab in BooleanFunctionTab::ALL {
                                ui.selectable_value(&mut action.editor_tab, tab, tab.label());
                            }
                        });
                        ui.separator();

                        let mut apply_cone_pins = false;
                        match action.editor_tab {
                            BooleanFunctionTab::Functions => {
                                let functions: Vec<Cell> =
                                    boolean_function_cells(&action.graph_data.graph)
                                        .cloned()
                                        .collect();
                                apply_cone_pins =
                                    show_boolean_function_editor(ui, action, &functions);
                            }
                            BooleanFunctionTab::System => show_system_solve_section(ui, action),
                            BooleanFunctionTab::Simulator => show_simulator_section(ui, action),
                        }

                        // Deferred to here: `action` borrows `project`, and
                        // the recompute needs `project` itself.
                        if apply_cone_pins {
                            apply_cone_output_pins_edit(project, action_index, &self.last_error);
                        }
                    }
                    Some(GraphAction::PropagatePinValues(action)) => {
                        label(
                            ui,
                            "Each row seeds an Input pin's own label as a value at the pins with \
                             the target label that this input is directly wired to, then floods it \
                             forward along the graph's edges.",
                        );
                        ui.separator();

                        let field_width = editor_field_width(ui, 2);

                        // Headings and fields share one grid so the two
                        // columns line up by construction, instead of by
                        // hand-tuned spacing over fields whose width depends
                        // on how wide the panel currently is.
                        let mut remove_row = None;
                        egui::ScrollArea::vertical()
                            .auto_shrink([false, true])
                            .show(ui, |ui| {
                                egui::Grid::new("propagate_pin_value_entries")
                                    .num_columns(3)
                                    .show(ui, |ui| {
                                        column_heading(ui, "Input pin label", field_width);
                                        column_heading(ui, "Target pin label", field_width);
                                        ui.end_row();

                                        for (row_index, (input_label, target_label)) in
                                            action.pending_entries.iter_mut().enumerate()
                                        {
                                            ui.add(
                                                egui::TextEdit::singleline(input_label)
                                                    .desired_width(field_width),
                                            );
                                            ui.add(
                                                egui::TextEdit::singleline(target_label)
                                                    .desired_width(field_width),
                                            );
                                            if egui_icons::icon_button(ui, ICON_CLOSE).clicked() {
                                                remove_row = Some(row_index);
                                            }
                                            ui.end_row();
                                        }
                                    });
                            });
                        if let Some(row_index) = remove_row {
                            action.pending_entries.remove(row_index);
                        }

                        if ui.button(ICON_ADD).clicked() {
                            action.pending_entries.push((String::new(), String::new()));
                        }

                        ui.separator();
                        if ui.button("Apply").clicked() {
                            apply_pin_value_propagation_edit(
                                project,
                                action_index,
                                &self.last_error,
                            );
                        }
                    }
                    // Every other action kind has nothing to edit here — just
                    // the title above.
                    Some(
                        GraphAction::RemoveClockBufferCells(_)
                        | GraphAction::MergeMuxedResetableFlipflops(_)
                        | GraphAction::MergeShiftRegisters(_),
                    )
                    | None => {}
                }
            }
        });

        egui::CentralPanel::default().show(ui, |ui| {
            if let Some(index) = self.app_state.selected_graph {
                let selected_project = &mut self.app_state.projects[index];

                // One widget id per *project*, not per action: SnarlWidget
                // persists the live pan/zoom transform in egui memory keyed by
                // this id, and every action in a project should share that same
                // projection rather than each getting its own.
                let widget_id = Id::new(selected_project.id);

                selected_project
                    .graph_viewer
                    .set_viewport_rect(ui.available_rect_before_wrap());

                let graph_data = resolve_graph_data_mut(
                    &mut selected_project.loaded_graph,
                    &mut selected_project.graph_actions,
                    selected_project.selected_action,
                );

                SnarlWidget::new()
                    .id(widget_id)
                    .style(default_style(ui.ctx().theme()))
                    .show(
                        &mut graph_data.snarl,
                        &mut selected_project.graph_viewer,
                        ui,
                    );
            }
        });
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        let app_state = serde_json::to_string(&self.app_state).unwrap();
        storage.set_string("app_state", app_state);
    }
}

/// Lays `graph` out as the snarl the viewer draws. Every cell and pin is
/// there, but an edge whose two ends both carry the same value a
/// `PropagatePinValues` action propagated onto them is left out (see
/// [`Graph::without_propagated_edges`]): the matching labels already show
/// it. Since that is decided from the labels, which every later action
/// carries through, the hiding sticks for the rest of the chain — while
/// `graph` itself, being what those actions, the solver and the simulator
/// work from, stays fully wired.
fn create_snarl(layout_mode: LayoutMode, graph: &Graph) -> Snarl<CellNode> {
    let graph = &graph.without_propagated_edges();
    let mut snarl = Snarl::new();

    let mut output_pins: HashMap<u32, OutPinId> = HashMap::new();
    let mut input_pins: HashMap<u32, InPinId> = HashMap::new();

    let mut cells = graph.cells.clone();
    cells.sort_by(|a, b| {
        let (ax, ay) = a.centroid().unwrap_or((0.0, 0.0));
        let (bx, by) = b.centroid().unwrap_or((0.0, 0.0));
        ay.total_cmp(&by).then_with(|| ax.total_cmp(&bx))
    });

    const GRID_CELL_W: f32 = 280.0;
    const GRID_CELL_H: f32 = 200.0;
    const CENTROID_SCALE: f32 = 100.0;
    const CENTROID_IO_SPACING: f32 = 100.0;

    const CENTROID_IO_MARGIN: f32 = 300.0;
    let standard_cell_count = cells
        .iter()
        .filter(|cell| cell.cell_type() == CellType::Sky130Standard)
        .count();
    let grid_cols = (standard_cell_count as f32).sqrt().ceil().max(1.0) as usize;

    let (min_x, max_x, min_y) = cells.iter().filter_map(Cell::centroid).fold(
        (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY),
        |(min_x, max_x, min_y), (x, y)| (min_x.min(x), max_x.max(x), min_y.min(y)),
    );
    let (min_x, max_x, min_y) = if min_x.is_finite() {
        (min_x, max_x, min_y)
    } else {
        (0.0, 0.0, 0.0)
    };
    let input_x = min_x as f32 * CENTROID_SCALE - CENTROID_IO_MARGIN;
    let output_x = max_x as f32 * CENTROID_SCALE + CENTROID_IO_MARGIN;
    let io_y_start = min_y as f32 * CENTROID_SCALE;

    let mut next_grid_index = 0usize;
    let mut next_input_y = 0.0;
    let mut next_output_y = 0.0;
    let mut next_centroid_input_y = io_y_start;
    let mut next_centroid_output_y = io_y_start;

    for cell in cells {
        let name = match &cell {
            Cell::Sky130Standard {
                cell_name, cell_id, ..
            } => format!("{cell_name}#{cell_id}"),
            Cell::Input { .. } | Cell::Output { .. } => format!("{:?}", cell.cell_type()),
            Cell::MergeCell { cell_name, .. } => format!("{cell_name}"),
        };

        let pos = match (layout_mode, cell.cell_type()) {
            (LayoutMode::Grid, CellType::Sky130Standard) => {
                let col = next_grid_index % grid_cols;
                let row = next_grid_index / grid_cols;
                next_grid_index += 1;
                pos2(col as f32 * GRID_CELL_W, row as f32 * GRID_CELL_H)
            }
            (LayoutMode::Grid, CellType::Input) => {
                let y = next_input_y;
                next_input_y += GRID_CELL_H;
                pos2(-GRID_CELL_W, y)
            }
            (LayoutMode::Grid, CellType::MergeCell) => {
                let col = next_grid_index % grid_cols;
                let row = next_grid_index / grid_cols;
                next_grid_index += 1;
                pos2(col as f32 * GRID_CELL_W, row as f32 * GRID_CELL_H)
            }
            (LayoutMode::Grid, CellType::Output) => {
                let y = next_output_y;
                next_output_y += GRID_CELL_H;
                pos2(grid_cols as f32 * GRID_CELL_W, y)
            }
            (LayoutMode::Centroid, CellType::Sky130Standard) => {
                let (x, y) = cell.centroid().unwrap_or((0.0, 0.0));
                pos2(x as f32 * CENTROID_SCALE, y as f32 * CENTROID_SCALE)
            }
            (LayoutMode::Centroid, CellType::MergeCell) => {
                let (x, y) = cell.centroid().unwrap_or((0.0, 0.0));
                pos2(x as f32 * CENTROID_SCALE, y as f32 * CENTROID_SCALE)
            }
            (LayoutMode::Centroid, CellType::Input) => {
                let y = next_centroid_input_y;
                next_centroid_input_y += CENTROID_IO_SPACING;
                pos2(input_x, y)
            }
            (LayoutMode::Centroid, CellType::Output) => {
                let y = next_centroid_output_y;
                next_centroid_output_y += CENTROID_IO_SPACING;
                pos2(output_x, y)
            }
        };

        let input_names = cell.inputs().iter().map(|(_, name)| name.clone()).collect();
        let output_names = cell
            .outputs()
            .iter()
            .map(|(_, name)| name.clone())
            .collect();

        let node_id = snarl.insert_node(pos, CellNode::new(name, input_names, output_names));

        for (input, &(net_id, _)) in cell.inputs().iter().enumerate() {
            input_pins.insert(
                net_id,
                InPinId {
                    node: node_id,
                    input,
                },
            );
        }
        for (output, &(net_id, _)) in cell.outputs().iter().enumerate() {
            output_pins.insert(
                net_id,
                OutPinId {
                    node: node_id,
                    output,
                },
            );
        }
    }

    for conns in &graph.connections {
        for (src_id, dst_ids) in conns {
            let Some(&out_pin) = output_pins.get(src_id) else {
                continue;
            };
            for dst_id in dst_ids {
                if let Some(&in_pin) = input_pins.get(dst_id) {
                    assert!(snarl.connect(out_pin, in_pin));
                }
            }
        }
    }

    snarl
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-character pattern holds its pin for the whole run, a longer
    /// one is a bitstring read left to right and sets the run's length,
    /// a clock is skipped and reads as 0, and anything else — a stray
    /// character, an empty field, bitstrings of two lengths — is refused
    /// with the pin named.
    #[test]
    fn input_patterns_expand_to_one_row_per_step() {
        let labels: Vec<String> = ["I", "clk", "rst_n"]
            .iter()
            .map(|label| label.to_string())
            .collect();
        let clocks = vec!["clk".to_string()];
        let mut patterns = BTreeMap::new();

        // Nothing set: every input 0, the step count from `steps`.
        let rows = expand_input_patterns(&labels, &clocks, &patterns, 3).unwrap();
        assert_eq!(rows, vec![vec![false; 3]; 3]);

        patterns.insert("I".to_string(), "0110".to_string());
        patterns.insert("rst_n".to_string(), " 1 ".to_string());
        let rows = expand_input_patterns(&labels, &clocks, &patterns, 3).unwrap();
        assert_eq!(
            rows,
            vec![
                vec![false, false, true],
                vec![true, false, true],
                vec![true, false, true],
                vec![false, false, true],
            ]
        );

        patterns.insert("rst_n".to_string(), "101".to_string());
        let err = expand_input_patterns(&labels, &clocks, &patterns, 3).unwrap_err();
        assert!(err.contains("I has 4 bits but rst_n has 3"), "{err}");

        patterns.insert("rst_n".to_string(), "01x1".to_string());
        let err = expand_input_patterns(&labels, &clocks, &patterns, 3).unwrap_err();
        assert!(err.starts_with("rst_n:") && err.contains("'x'"), "{err}");

        patterns.insert("rst_n".to_string(), String::new());
        let err = expand_input_patterns(&labels, &clocks, &patterns, 3).unwrap_err();
        assert!(err.starts_with("rst_n:"), "{err}");

        // The clock's own pattern, if someone saved one, is ignored.
        patterns.insert("rst_n".to_string(), "1".to_string());
        patterns.insert("clk".to_string(), "zzz".to_string());
        assert!(expand_input_patterns(&labels, &clocks, &patterns, 2).is_ok());
    }

    /// A bus row is three characters a frame: the character where it is
    /// printable ASCII (space included), hex where it isn't, so the
    /// columns stay frames whichever a frame is.
    #[test]
    fn bus_text_mixes_characters_and_hex_in_fixed_cells() {
        let frames: Vec<Vec<u64>> = [0x00, 0x28, 0x20, 0x54, 0x7f, 0xc3]
            .iter()
            .map(|&v| vec![v])
            .collect();
        assert_eq!(bus_text(&frames, 0, 8), "00 (     T  7f c3 ");
        // A wider bus pads its hex to its width, and the characters with it.
        assert_eq!(bus_text(&[vec![0x41], vec![0x1ff]], 0, 12), "A   1ff ");
        // A narrow bus still gets two-character cells for its hex.
        assert_eq!(bus_text(&[vec![3], vec![0x31]], 0, 6), "03 1  ");
    }

    /// A bus condition row takes its value as one character or as a
    /// number, skips itself when it has no bus name, and refuses anything
    /// else with the bus named.
    #[test]
    fn bus_constraint_rows_parse_characters_and_numbers() {
        let row = |bus: &str, equal: bool, value: &str| BusConstraintRow {
            bus: bus.to_string(),
            equal,
            value: value.to_string(),
            cycle: String::new(),
        };
        let parsed = |bus: &str, equal: bool, value: &str| row(bus, equal, value).parse();

        assert_eq!(
            parsed("O", false, "A").unwrap(),
            Some(BusConstraint {
                bus: "O".to_string(),
                equal: false,
                value: 0x41,
                cycle: None,
            })
        );
        let mut pinned = row("O", true, "(");
        pinned.cycle = " 122 ".to_string();
        assert_eq!(pinned.parse().unwrap().unwrap().cycle, Some(122));
        pinned.cycle = "last".to_string();
        assert!(
            pinned
                .parse()
                .unwrap_err()
                .contains("frame must be a number")
        );
        assert_eq!(parsed("O", true, " ").unwrap().unwrap().value, 0x20);
        assert_eq!(parsed(" O ", true, "65").unwrap().unwrap().value, 65);
        assert_eq!(parsed("O", true, "0x41").unwrap().unwrap().value, 0x41);
        assert_eq!(parsed("O", true, " 0X2a ").unwrap().unwrap().value, 0x2a);
        assert_eq!(parsed("O", true, "0x41").unwrap().unwrap().bus, "O");
        assert_eq!(parsed("", true, "A").unwrap(), None);
        assert_eq!(parsed("  ", true, "A").unwrap(), None);
        assert!(parsed("O", true, "").unwrap_err().starts_with("O:"));
        assert!(parsed("O", true, "AB").is_err());
        assert!(parsed("O", true, "0xZZ").is_err());

        assert_eq!(
            BusConstraintRow::new().parse().unwrap_err(),
            "O: give the value as one character (A), or a number (65, 0x41)."
        );
    }

    /// Indexed output pins group into one bus column read highest index
    /// first, whatever order the pins come in; a lone pin stays a bit in
    /// every format; and every column knows how wide its values get.
    #[test]
    fn output_columns_group_buses_and_format_them() {
        let labels: Vec<String> = ["success", "O[0]", "O[2]", "O[1]", "O[3]", "P[0]"]
            .iter()
            .map(|label| label.to_string())
            .collect();
        let columns = output_columns(&labels);
        let names: Vec<&str> = columns.iter().map(|column| column.name.as_str()).collect();
        assert_eq!(names, ["success", "O", "P"]);
        assert_eq!(columns[1].bits, [4, 2, 3, 1]);

        // success = 1, O = 0b1100 (O[3] = 1, O[2] = 1), P = 1.
        let outputs = [true, false, true, false, true, true];
        for format in OutputFormat::ALL {
            assert_eq!(columns[0].value(&outputs, format), "1");
            assert_eq!(columns[0].width(format), 1);
        }
        assert_eq!(columns[1].value(&outputs, OutputFormat::Bits), "1100");
        assert_eq!(columns[1].value(&outputs, OutputFormat::Hex), "c");
        assert_eq!(columns[1].value(&outputs, OutputFormat::Decimal), "12");
        assert_eq!(columns[1].width(OutputFormat::Bits), 4);
        assert_eq!(columns[1].width(OutputFormat::Hex), 1);
        assert_eq!(columns[1].width(OutputFormat::Decimal), 2);
        assert_eq!(columns[2].value(&outputs, OutputFormat::Hex), "1");

        // A full byte pads hex to its width.
        let byte: Vec<String> = (0..8).map(|i| format!("O[{i}]")).collect();
        let byte_columns = output_columns(&byte);
        let outputs = [true, false, false, false, false, false, false, false];
        assert_eq!(byte_columns[0].value(&outputs, OutputFormat::Hex), "01");
        assert_eq!(byte_columns[0].width(OutputFormat::Decimal), 3);

        // ASCII where the value is a printable character, hex otherwise
        // — control codes, space, and anything past 7 bits.
        let bits_of = |value: u8| -> Vec<bool> { (0..8).map(|i| value & (1 << i) != 0).collect() };
        assert_eq!(
            byte_columns[0].value(&bits_of(b'A'), OutputFormat::Ascii),
            "A"
        );
        assert_eq!(
            byte_columns[0].value(&bits_of(b'~'), OutputFormat::Ascii),
            "~"
        );
        assert_eq!(
            byte_columns[0].value(&bits_of(0x01), OutputFormat::Ascii),
            "01"
        );
        assert_eq!(
            byte_columns[0].value(&bits_of(b' '), OutputFormat::Ascii),
            "20"
        );
        assert_eq!(
            byte_columns[0].value(&bits_of(0xc3), OutputFormat::Ascii),
            "c3"
        );
        assert_eq!(byte_columns[0].width(OutputFormat::Ascii), 2);
        // A lone pin is still a bit.
        assert_eq!(columns[0].value(&[true], OutputFormat::Ascii), "1");
    }
}
