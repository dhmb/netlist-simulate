//! Reading a `sky130` GDSII layout back into a [`Graph`].
//!
//! A GDS file holds geometry, not a netlist: rectangles of metal on
//! numbered layers, cell instances placed at coordinates, and text labels
//! floating on top of them. Recovering the circuit means working out which
//! shapes are electrically one wire, and which cell pin each wire lands on.
//!
//! The recovery runs in three stages:
//!
//! 1. **Flatten** ([`flatten`]). The layout is one top-level structure
//!    holding instance references to standard cells. Each reference's
//!    geometry is transformed into top-level coordinates, so everything
//!    afterwards works in one space. Every flattened piece of geometry is
//!    an *element*, numbered from zero in placement order.
//! 2. **Connect** ([`connected_components`]). Two elements are the same
//!    net when their geometry touches *and* their layers conduct — which
//!    is what [`CONDUCT_LIST`] says. A wire on `met1` and one on `met2`
//!    crossing at the same spot are not connected; they only join through
//!    a `via`. The connected components of that relation are the nets.
//! 3. **Assemble** ([`build_graph`]). Each standard cell instance's pin
//!    labels say which net reaches which pin, and the pin's name says
//!    whether it is an input or an output. The design's own ports come
//!    from labels on the top-level structure.
//!
//! Parsing is [`gds21`], which reads from a byte slice rather than a path,
//! so this works on the `wasm32` build where there is no filesystem.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use gds21::{GdsElement, GdsLibrary, GdsPoint, GdsStrans, GdsStruct};
use serde::{Deserialize, Serialize};

use crate::graph::{Cell, Graph, Pin};

/// One GDS layer: its layer number and datatype. Together these name a
/// mask layer — `(68, 20)` is `met1` drawing, `(68, 16)` its pin
/// annotations, `(68, 5)` its labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Layer {
    pub layer: i16,
    pub datatype: i16,
}

impl Layer {
    pub const fn new(layer: i16, datatype: i16) -> Self {
        Layer { layer, datatype }
    }
}

/// `li1` — the local interconnect layer above the transistors.
pub const LI1: Layer = Layer::new(67, 20);
pub const LI1_PIN: Layer = Layer::new(67, 16);
pub const LI1_LBL: Layer = Layer::new(67, 5);
/// `mcon` — the contact from `li1` up to `met1`.
pub const MCON: Layer = Layer::new(67, 44);
pub const MET1: Layer = Layer::new(68, 20);
pub const MET1_PIN: Layer = Layer::new(68, 16);
pub const MET1_LBL: Layer = Layer::new(68, 5);
pub const VIA: Layer = Layer::new(68, 44);
pub const MET2: Layer = Layer::new(69, 20);
pub const MET2_PIN: Layer = Layer::new(69, 16);
pub const MET2_LBL: Layer = Layer::new(69, 5);
pub const VIA2: Layer = Layer::new(69, 44);
pub const MET3: Layer = Layer::new(70, 20);
pub const MET3_PIN: Layer = Layer::new(70, 16);
pub const MET3_LBL: Layer = Layer::new(70, 5);
pub const VIA3: Layer = Layer::new(70, 44);
pub const MET4: Layer = Layer::new(71, 20);
pub const MET4_PIN: Layer = Layer::new(71, 16);
pub const MET4_LBL: Layer = Layer::new(71, 5);
pub const VIA4: Layer = Layer::new(71, 44);
pub const MET5: Layer = Layer::new(72, 20);
pub const MET5_PIN: Layer = Layer::new(72, 16);
pub const MET5_LBL: Layer = Layer::new(72, 5);
/// The layer carrying each standard cell's outline box, which is where
/// [`Graph`] cell centroids come from.
pub const CELL_OUTLINE: Layer = Layer::new(236, 0);
/// Free-text annotations. These are labels like any other, but they name
/// the *design*, not a pin, so pin recovery skips them.
pub const TEXT: Layer = Layer::new(83, 44);

/// Which layers conduct together — the heart of net recovery.
///
/// Each group is a set of layers whose geometry, where it touches, is one
/// electrical node. Layers not sharing a group never join: `met1` and
/// `met2` cross over each other constantly and are only connected where a
/// `via` bridges them, so `(MET1, VIA)` and `(VIA, MET2)` are groups while
/// `(MET1, MET2)` is not.
///
/// A layer appearing in several groups (`MET1` is in three) is tested in
/// each of them; the union-find in [`connected_components`] makes the
/// repetition harmless. Pin and label layers ride along with their drawing
/// layer, which is what attaches a text label to the wire beneath it.
pub const CONDUCT_LIST: &[&[Layer]] = &[
    &[LI1, MCON],
    &[LI1, LI1_PIN, LI1_LBL],
    &[MCON, MET1],
    &[MET1, MET1_PIN, MET1_LBL],
    &[MET1, VIA],
    &[VIA, MET2],
    &[MET2, MET2_PIN, MET2_LBL],
    &[MET2, VIA2],
    &[VIA2, MET3],
    &[MET3, MET3_PIN, MET3_LBL],
    &[MET3, VIA3],
    &[VIA3, MET4],
    &[MET4, MET4_PIN, MET4_LBL],
    &[MET4, VIA4],
    &[VIA4, MET5],
    &[MET5, MET5_PIN, MET5_LBL],
];

/// Pin names that are cell *outputs* in the `sky130_fd_sc_hd` library.
///
/// The library is regular enough that this short list decides the
/// direction of every pin: `X` and `Y` are the combinational outputs
/// (inverting cells use `Y`), `Q`/`Q_N` a flip-flop's state, `HI`/`LO` the
/// tie cell's constants, and `CO`/`COUT`/`SUM` the adders'. Every other
/// pin name is an input. Checked against all 66 cell types appearing in
/// real designs: no name is an output on one cell and an input on
/// another.
const OUTPUT_PIN_NAMES: [&str; 9] = ["X", "Y", "Q", "Q_N", "HI", "LO", "CO", "COUT", "SUM"];

/// Pin labels carrying power rather than signal: the supply, the ground,
/// and the two well taps. Every cell has them, none of them is part of the
/// circuit's logic, so they are skipped rather than turned into pins.
const POWER_PIN_NAMES: [&str; 4] = ["VGND", "VPWR", "VPB", "VNB"];

/// Cell types that hold no logic and are dropped rather than becoming
/// [`Graph`] cells: filler capacitance, well taps, and the antenna diode
/// (a single pin tied to a net for fab antenna-effect protection).
const IGNORED_CELL_TYPES: [&str; 3] = [
    "sky130_fd_sc_hd__decap_3",
    "sky130_fd_sc_hd__tapvpwrvgnd_1",
    "sky130_fd_sc_hd__diode_2",
];

/// The little a caller can still decide about reading a layout.
///
/// Everything that matters — which cells are placed, what they are wired
/// to, which nets are the design's ports and which way those ports face —
/// is recovered from the file itself, so a layout can be opened without
/// knowing anything about it in advance.
#[derive(Clone, Debug, Default)]
pub struct Sky130Options {
    /// The structure to read as the design. `None` picks the file's only
    /// top-level structure, and fails if there is more than one.
    pub top_cell: Option<String>,
}

/// An input this reader invented, because the layout has a net driving
/// cell inputs that nothing drives in turn.
///
/// A real chip has no such thing: every wire is driven by a cell output or
/// by a pad. One here means the layout is telling an incomplete story —
/// a port with no label, a cell this reader dropped, or a net recovered
/// wrongly — so rather than leave those pins dangling, each undriven net
/// gets an input of its own and the fact is reported for a human to judge.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DummyInput {
    /// `dummy1`, `dummy2`, ... numbered from one in net order.
    pub label: String,
    /// The pins this net reaches, which are the pins that would otherwise
    /// have had nothing driving them.
    pub drives: Vec<DrivenPin>,
}

/// One pin an invented input drives, with where its cell sits so the
/// dangling wire can be found in the layout.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DrivenPin {
    /// `"<cell>#<id>.<pin>"`, as the graph names it.
    pub pin: String,
    /// The centroid of the cell that owns the pin, in microns.
    pub centroid: (f64, f64),
}

/// A layout, read.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Layout {
    pub graph: Graph,
    /// The inputs that had to be invented — empty for a layout that
    /// explains itself. See [`DummyInput`].
    pub dummy_inputs: Vec<DummyInput>,
}

/// Why reading a layout failed.
#[derive(Clone, Debug)]
pub enum Sky130Error {
    /// The bytes are not a GDSII file this parser can read.
    Parse(String),
    /// No single top-level structure, and `top_cell` didn't name one.
    TopCell(String),
    /// A placement this reader can't represent exactly — see
    /// [`transform`].
    Transform(String),
    /// A cell instance references another cell. These layouts are one
    /// level deep, and nothing here flattens a deeper hierarchy.
    NestedReference(String),
    /// A label naming a design port matched no top-level label, or matched
    /// more than one.
    Port(String),
    /// Geometry whose shape this reader doesn't handle — a round-ended
    /// path, a polygon that isn't rectilinear.
    Geometry(String),
    /// Two cell pins carrying the same label landed on different nets, so
    /// the layout disagrees with itself.
    PinConflict(String),
    /// Two cell outputs drive one net. Nothing in this library is
    /// tri-state, so that can only be a recovery error.
    MultipleDrivers(String),
    #[cfg(not(target_arch = "wasm32"))]
    Io(String),
}

impl std::fmt::Display for Sky130Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (kind, message) = match self {
            Sky130Error::Parse(m) => ("could not parse GDSII", m),
            Sky130Error::TopCell(m) => ("no usable top-level cell", m),
            Sky130Error::Transform(m) => ("unsupported placement", m),
            Sky130Error::NestedReference(m) => ("nested cell reference", m),
            Sky130Error::Port(m) => ("design port not found", m),
            Sky130Error::Geometry(m) => ("unsupported geometry", m),
            Sky130Error::PinConflict(m) => ("conflicting pin labels", m),
            Sky130Error::MultipleDrivers(m) => ("net with several drivers", m),
            #[cfg(not(target_arch = "wasm32"))]
            Sky130Error::Io(m) => ("could not read file", m),
        };
        write!(f, "{kind}: {message}")
    }
}

impl std::error::Error for Sky130Error {}

/// An axis-aligned rectangle in half-database units, inclusive on every
/// side.
///
/// Everything in these layouts is rectilinear — paths run along an axis,
/// polygons turn only at right angles, placements only reflect or rotate
/// by a right angle — so a rectangle is enough to describe any piece of
/// geometry exactly, and a handful of them describe any polygon. Working
/// in *half* database units makes a path's half-width exact, so an
/// odd-width wire doesn't quietly lose half a nanometre.
///
/// Inclusive bounds matter: two shapes that merely touch along an edge are
/// electrically one wire, which is exactly what [`Rect::touches`] reports.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rect {
    min_x: i64,
    min_y: i64,
    max_x: i64,
    max_y: i64,
}

impl Rect {
    /// Whether these two rectangles share at least one point, edges and
    /// corners included.
    fn touches(self, other: Rect) -> bool {
        self.min_x <= other.max_x
            && other.min_x <= self.max_x
            && self.min_y <= other.max_y
            && other.min_y <= self.max_y
    }
}

/// One flattened piece of geometry, in top-level coordinates.
struct Element {
    /// Which cell instance it came from — an index into
    /// [`Flattened::cell_names`].
    cell_id: usize,
    layer: Layer,
    /// A label's text, `None` for drawn geometry.
    text: Option<String>,
    /// The rectangles covering it. A polygon takes several; a label is one
    /// degenerate rectangle at its origin, which still `touches` whatever
    /// wire it sits on.
    rects: Vec<Rect>,
}

/// A whole layout reduced to elements in one coordinate space.
struct Flattened {
    elements: Vec<Element>,
    /// Cell instance name per `cell_id`, the top-level structure first.
    cell_names: Vec<String>,
    /// Each instance's outline centre, in microns — the coordinate a
    /// [`Graph`] cell is drawn at.
    centroids: HashMap<usize, (f64, f64)>,
}

/// The placement transform of one cell instance, as the exact integer map
/// it is for these layouts.
///
/// GDSII allows an arbitrary rotation and magnification, which would turn
/// integer coordinates into irrational ones and rectangles into diamonds.
/// These layouts only ever reflect about the x-axis and rotate by a
/// multiple of 90 degrees at unit magnification, so the transform stays
/// exact and rectilinear — and anything else is refused rather than
/// silently rounded.
struct Transform {
    reflect: bool,
    /// Cosine and sine of the rotation, each exactly -1, 0 or 1.
    cos: i64,
    sin: i64,
    origin: (i64, i64),
}

impl Transform {
    fn new(strans: &Option<GdsStrans>, origin: &GdsPoint) -> Result<Self, Sky130Error> {
        let origin = (origin.x as i64, origin.y as i64);
        let Some(strans) = strans else {
            return Ok(Transform {
                reflect: false,
                cos: 1,
                sin: 0,
                origin,
            });
        };
        let magnification = strans.mag.unwrap_or(1.0);
        if magnification != 1.0 {
            return Err(Sky130Error::Transform(format!(
                "magnification {magnification} would not keep coordinates exact; only 1 is handled"
            )));
        }
        let angle = strans.angle.unwrap_or(0.0).rem_euclid(360.0);
        // Compared as whole degrees so the match is exact; anything that
        // isn't a right angle is refused below rather than rounded to one.
        let quarter_turns = (angle / 90.0) as i64;
        let (cos, sin) = match quarter_turns {
            0 if angle == 0.0 => (1, 0),
            1 if angle == 90.0 => (0, 1),
            2 if angle == 180.0 => (-1, 0),
            3 if angle == 270.0 => (0, -1),
            _ => {
                return Err(Sky130Error::Transform(format!(
                    "rotation of {angle} degrees would not keep geometry rectilinear; only \
                     multiples of 90 are handled"
                )));
            }
        };
        Ok(Transform {
            reflect: strans.reflected,
            cos,
            sin,
            origin,
        })
    }

    /// The identity — for the top-level structure's own geometry, which is
    /// already in top-level coordinates.
    fn identity() -> Self {
        Transform {
            reflect: false,
            cos: 1,
            sin: 0,
            origin: (0, 0),
        }
    }

    /// Maps one point, in GDSII's order: reflect about the x-axis first,
    /// then rotate, then translate.
    fn apply(&self, point: &GdsPoint) -> (i64, i64) {
        let (x, mut y) = (point.x as i64, point.y as i64);
        if self.reflect {
            y = -y;
        }
        (
            x * self.cos - y * self.sin + self.origin.0,
            x * self.sin + y * self.cos + self.origin.1,
        )
    }
}

/// Doubles a transformed coordinate into the half-database units
/// [`Rect`] works in.
fn halves((x, y): (i64, i64)) -> (i64, i64) {
    (x * 2, y * 2)
}

/// How far a path runs past each of its two outer endpoints, in
/// half-database units.
///
/// GDSII path types: 0 leaves the ends flush with the endpoints, 2 extends
/// each end by half the width (so the wire ends square, past its
/// endpoint), and 4 takes explicit per-end extensions. Type 1 is a round
/// end, which no rectangle describes; none of these layouts uses it.
fn path_end_extensions(
    width: i64,
    path_type: Option<i16>,
    begin_extn: Option<i32>,
    end_extn: Option<i32>,
) -> Result<(i64, i64), Sky130Error> {
    match path_type.unwrap_or(0) {
        0 => Ok((0, 0)),
        2 => Ok((width / 2, width / 2)),
        4 => Ok((
            begin_extn.unwrap_or(0) as i64 * 2,
            end_extn.unwrap_or(0) as i64 * 2,
        )),
        other => Err(Sky130Error::Geometry(format!(
            "path type {other} is not a rectangle; only 0, 2 and 4 are handled"
        ))),
    }
}

/// The rectangle covering one axis-aligned path segment of width `width`,
/// run past its start and end by `begin` and `finish`.
///
/// A segment in the middle of a bent path gets no extension: only the two
/// outer ends of the whole path do, which is what
/// [`path_end_extensions`] describes.
fn path_rect(
    start: (i64, i64),
    end: (i64, i64),
    width: i64,
    begin: i64,
    finish: i64,
) -> Result<Rect, Sky130Error> {
    let half = width / 2;

    if start.1 == end.1 {
        let (left, right) = (start.0.min(end.0), start.0.max(end.0));
        // Extensions belong to the ends as drawn, so a path running right
        // to left has its `begin` extension on the right.
        let (before, after) = if start.0 <= end.0 {
            (begin, finish)
        } else {
            (finish, begin)
        };
        Ok(Rect {
            min_x: left - before,
            max_x: right + after,
            min_y: start.1 - half,
            max_y: start.1 + half,
        })
    } else if start.0 == end.0 {
        let (bottom, top) = (start.1.min(end.1), start.1.max(end.1));
        let (before, after) = if start.1 <= end.1 {
            (begin, finish)
        } else {
            (finish, begin)
        };
        Ok(Rect {
            min_y: bottom - before,
            max_y: top + after,
            min_x: start.0 - half,
            max_x: start.0 + half,
        })
    } else {
        Err(Sky130Error::Geometry(format!(
            "diagonal path from {start:?} to {end:?} is not a rectangle"
        )))
    }
}

/// Covers a closed rectilinear polygon with rectangles, exactly.
///
/// Sweeps vertical slabs: between each pair of neighbouring x coordinates
/// the polygon's outline is a fixed set of horizontal spans, so the slab
/// splits into rectangles. Which spans are inside comes from the
/// even-odd rule — sort the y values where the outline crosses the slab
/// and take alternate pairs — which is also what makes a polygon with a
/// cut-line hole come out right.
///
/// The union of the rectangles is the polygon's closed region, so two
/// polygons that merely touch still produce rectangles that touch.
fn polygon_rects(points: &[(i64, i64)]) -> Result<Vec<Rect>, Sky130Error> {
    // GDSII closes a boundary by repeating its first point.
    let ring: &[(i64, i64)] = match points {
        [first, .., last] if first == last => &points[..points.len() - 1],
        _ => points,
    };
    if ring.len() < 4 {
        return Ok(Vec::new());
    }

    let mut edges: Vec<((i64, i64), (i64, i64))> = Vec::with_capacity(ring.len());
    for index in 0..ring.len() {
        let a = ring[index];
        let b = ring[(index + 1) % ring.len()];
        if a.0 != b.0 && a.1 != b.1 {
            return Err(Sky130Error::Geometry(format!(
                "polygon edge from {a:?} to {b:?} is neither horizontal nor vertical"
            )));
        }
        edges.push((a, b));
    }

    let mut xs: Vec<i64> = ring.iter().map(|&(x, _)| x).collect();
    xs.sort_unstable();
    xs.dedup();

    let mut rects = Vec::new();
    let mut crossings: Vec<(i64, i64)> = Vec::new();
    for slab in xs.windows(2) {
        let (left, right) = (slab[0], slab[1]);
        // Any horizontal edge spanning the whole slab crosses it; a
        // vertical edge runs along a slab boundary and never does.
        crossings.clear();
        for &(a, b) in &edges {
            if a.1 != b.1 {
                continue;
            }
            let (edge_left, edge_right) = (a.0.min(b.0), a.0.max(b.0));
            if edge_left <= left && right <= edge_right {
                // Direction decides whether the outline is entering or
                // leaving, which is what pairs the crossings up.
                crossings.push((a.1, if b.0 > a.0 { 1 } else { -1 }));
            }
        }
        crossings.sort_unstable();

        let mut winding = 0;
        let mut span_start = 0i64;
        for &(y, direction) in crossings.iter() {
            if winding == 0 {
                span_start = y;
            }
            winding += direction;
            if winding == 0 && y > span_start {
                rects.push(Rect {
                    min_x: left,
                    max_x: right,
                    min_y: span_start,
                    max_y: y,
                });
            }
        }
    }
    Ok(rects)
}

/// The area-weighted centre of a closed polygon, in database units.
fn polygon_centroid(points: &[(i64, i64)]) -> Option<(f64, f64)> {
    let mut area = 0.0;
    let mut cx = 0.0;
    let mut cy = 0.0;
    for window in points.windows(2) {
        let (x0, y0) = (window[0].0 as f64, window[0].1 as f64);
        let (x1, y1) = (window[1].0 as f64, window[1].1 as f64);
        let cross = x0 * y1 - x1 * y0;
        area += cross;
        cx += (x0 + x1) * cross;
        cy += (y0 + y1) * cross;
    }
    if area == 0.0 {
        return None;
    }
    Some((cx / (3.0 * area), cy / (3.0 * area)))
}

/// Flattens `library` into elements in the top-level coordinate space.
///
/// Element order matters and is not arbitrary: nets are numbered by the
/// lowest element in each of them (see [`connected_components`]), so
/// `fake_inputs` net ids only mean anything against this exact order. It
/// is the order a GDS reader naturally produces — per cell, the drawn
/// polygons, then the paths, then the labels; and per design, the
/// top-level structure first, then each instance it places, in file order.
fn flatten(library: &GdsLibrary, top_cell: Option<&str>) -> Result<Flattened, Sky130Error> {
    let by_name: HashMap<&str, &GdsStruct> = library
        .structs
        .iter()
        .map(|s| (s.name.as_str(), s))
        .collect();

    let top = match top_cell {
        Some(name) => by_name.get(name).copied().ok_or_else(|| {
            Sky130Error::TopCell(format!("no structure named \"{name}\" in this file"))
        })?,
        None => {
            let referenced: HashSet<&str> = library
                .structs
                .iter()
                .flat_map(|s| s.elems.iter())
                .filter_map(|elem| match elem {
                    GdsElement::GdsStructRef(r) => Some(r.name.as_str()),
                    GdsElement::GdsArrayRef(r) => Some(r.name.as_str()),
                    _ => None,
                })
                .collect();
            let tops: Vec<&GdsStruct> = library
                .structs
                .iter()
                .filter(|s| !referenced.contains(s.name.as_str()))
                .collect();
            match tops.as_slice() {
                [only] => *only,
                [] => {
                    return Err(Sky130Error::TopCell(
                        "every structure is referenced by another, so none is the design"
                            .to_string(),
                    ));
                }
                many => {
                    let names: Vec<&str> = many.iter().map(|s| s.name.as_str()).collect();
                    return Err(Sky130Error::TopCell(format!(
                        "several top-level structures ({}); name one with `top_cell`",
                        names.join(", ")
                    )));
                }
            }
        }
    };

    let mut flat = Flattened {
        elements: Vec::new(),
        cell_names: Vec::new(),
        centroids: HashMap::new(),
    };

    flat.cell_names.push(top.name.clone());
    add_struct(&mut flat, top, &Transform::identity(), 0)?;

    for elem in &top.elems {
        let reference = match elem {
            GdsElement::GdsStructRef(r) => r,
            GdsElement::GdsArrayRef(r) => {
                return Err(Sky130Error::NestedReference(format!(
                    "array reference to \"{}\" is not handled",
                    r.name
                )));
            }
            _ => continue,
        };
        let instance = by_name
            .get(reference.name.as_str())
            .copied()
            .ok_or_else(|| {
                Sky130Error::TopCell(format!(
                    "reference to \"{}\", which the file does not define",
                    reference.name
                ))
            })?;
        if instance
            .elems
            .iter()
            .any(|e| matches!(e, GdsElement::GdsStructRef(_) | GdsElement::GdsArrayRef(_)))
        {
            return Err(Sky130Error::NestedReference(format!(
                "\"{}\" places cells of its own; only one level of hierarchy is handled",
                instance.name
            )));
        }
        let transform = Transform::new(&reference.strans, &reference.xy)?;
        let cell_id = flat.cell_names.len();
        flat.cell_names.push(instance.name.clone());
        add_struct(&mut flat, instance, &transform, cell_id)?;
    }

    Ok(flat)
}

/// Appends one structure's own geometry, transformed, as elements of
/// `cell_id`. Polygons first, then paths, then labels — see [`flatten`]
/// on why the order is load-bearing.
fn add_struct(
    flat: &mut Flattened,
    structure: &GdsStruct,
    transform: &Transform,
    cell_id: usize,
) -> Result<(), Sky130Error> {
    for elem in &structure.elems {
        let GdsElement::GdsBoundary(boundary) = elem else {
            continue;
        };
        let points: Vec<(i64, i64)> = boundary.xy.iter().map(|p| transform.apply(p)).collect();
        let layer = Layer::new(boundary.layer, boundary.datatype);
        if layer == CELL_OUTLINE
            && let Some(centre) = polygon_centroid(&points)
        {
            flat.centroids.insert(cell_id, centre);
        }
        let doubled: Vec<(i64, i64)> = points.into_iter().map(halves).collect();
        flat.elements.push(Element {
            cell_id,
            layer,
            text: None,
            rects: polygon_rects(&doubled)?,
        });
    }

    for elem in &structure.elems {
        let GdsElement::GdsPath(path) = elem else {
            continue;
        };
        let points: Vec<(i64, i64)> = path.xy.iter().map(|p| halves(transform.apply(p))).collect();
        let width = path.width.unwrap_or(0) as i64 * 2;
        let (outer_begin, outer_end) =
            path_end_extensions(width, path.path_type, path.begin_extn, path.end_extn)?;
        let last = points.len().saturating_sub(2);
        let mut rects = Vec::with_capacity(points.len().saturating_sub(1));
        for (index, segment) in points.windows(2).enumerate() {
            // Only the outer ends of the whole path run past their
            // endpoint; a bend between two segments is flush. On a
            // two-point path the one segment is both ends at once.
            let begin = if index == 0 { outer_begin } else { 0 };
            let finish = if index == last { outer_end } else { 0 };
            rects.push(path_rect(segment[0], segment[1], width, begin, finish)?);
        }
        flat.elements.push(Element {
            cell_id,
            layer: Layer::new(path.layer, path.datatype),
            text: None,
            rects,
        });
    }

    for elem in &structure.elems {
        let GdsElement::GdsTextElem(text) = elem else {
            continue;
        };
        let (x, y) = halves(transform.apply(&text.xy));
        flat.elements.push(Element {
            cell_id,
            layer: Layer::new(text.layer, text.texttype),
            text: Some(text.string.clone()),
            rects: vec![Rect {
                min_x: x,
                max_x: x,
                min_y: y,
                max_y: y,
            }],
        });
    }

    Ok(())
}

/// A minimal union-find over `0..n`, for growing nets out of pairs of
/// touching geometry.
struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        DisjointSet {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, a: usize, b: usize) {
        let (a, b) = (self.find(a), self.find(b));
        if a != b {
            self.parent[a] = b;
        }
    }
}

/// How wide a bucket of the broad-phase grid is, in half-database units.
/// One micron: wide enough that a via lands in a single bucket, narrow
/// enough that a bucket holds a handful of shapes rather than a district.
const GRID_PITCH: i64 = 2_000;

/// Assigns every element the net it belongs to.
///
/// Two elements join when a rectangle of one touches a rectangle of the
/// other *and* their layers share a [`CONDUCT_LIST`] group. Candidate
/// pairs come from a uniform grid over the layout rather than from testing
/// all pairs, which would be quadratic in the tens of thousands of shapes
/// a routed design has.
///
/// Nets are numbered by their lowest element, so the numbering depends
/// only on the flattening order and not on how the search happens to run.
/// Every element gets a net, including those on layers that conduct with
/// nothing — they are simply nets of one.
fn connected_components(elements: &[Element]) -> Vec<usize> {
    let mut sets = DisjointSet::new(elements.len());

    for group in CONDUCT_LIST {
        let layers: BTreeSet<Layer> = group.iter().copied().collect();
        let mut buckets: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (index, element) in elements.iter().enumerate() {
            if !layers.contains(&element.layer) {
                continue;
            }
            for rect in &element.rects {
                for bucket_x in
                    rect.min_x.div_euclid(GRID_PITCH)..=rect.max_x.div_euclid(GRID_PITCH)
                {
                    for bucket_y in
                        rect.min_y.div_euclid(GRID_PITCH)..=rect.max_y.div_euclid(GRID_PITCH)
                    {
                        buckets.entry((bucket_x, bucket_y)).or_default().push(index);
                    }
                }
            }
        }

        for mut members in buckets.into_values() {
            members.sort_unstable();
            members.dedup();
            for (offset, &left) in members.iter().enumerate() {
                for &right in &members[offset + 1..] {
                    if sets.find(left) == sets.find(right) {
                        continue;
                    }
                    let touching = elements[left]
                        .rects
                        .iter()
                        .any(|a| elements[right].rects.iter().any(|b| a.touches(*b)));
                    if touching {
                        sets.union(left, right);
                    }
                }
            }
        }
    }

    // Number the nets by the lowest element each one contains.
    let mut net_of_root: HashMap<usize, usize> = HashMap::new();
    let mut nets = Vec::with_capacity(elements.len());
    for index in 0..elements.len() {
        let root = sets.find(index);
        let next = net_of_root.len();
        nets.push(*net_of_root.entry(root).or_insert(next));
    }
    nets
}

/// Whether `name` is a standard cell of the `sky130` library, as opposed
/// to the design's own top-level structure.
fn is_standard_cell(name: &str) -> bool {
    name.starts_with("sky130_")
}

/// Builds the [`Graph`]: one cell per standard cell instance, plus the
/// `Input` and `Output` cells standing for the design's own ports, wired
/// together by the nets recovered from the geometry.
fn build_graph(flat: &Flattened, nets: &[usize], scale: f64) -> Result<Layout, Sky130Error> {
    // Pin labels of each standard cell instance, in element order, grouped
    // by the instance they annotate. `TEXT` labels name the design rather
    // than a pin, so they stay out.
    let mut labels_by_cell: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (index, element) in flat.elements.iter().enumerate() {
        if element.text.is_none() || element.layer == TEXT {
            continue;
        }
        if !is_standard_cell(&flat.cell_names[element.cell_id]) {
            continue;
        }
        labels_by_cell
            .entry(element.cell_id)
            .or_default()
            .push(index);
    }

    // Pins are numbered as they are created, so that the graph's net ids
    // are dense and stable; `net_pins` remembers which recovered net each
    // one sits on so the connections can be drawn at the end.
    let mut next_pin = 0u32;
    let mut cells: Vec<Cell> = Vec::new();
    let mut inputs_on_net: BTreeMap<usize, Vec<u32>> = BTreeMap::new();
    let mut output_on_net: BTreeMap<usize, u32> = BTreeMap::new();
    // What each net actually reaches, for reporting an undriven one.
    let mut sinks_on_net: BTreeMap<usize, Vec<DrivenPin>> = BTreeMap::new();

    for (&cell_id, label_indices) in &labels_by_cell {
        let cell_name = &flat.cell_names[cell_id];
        if IGNORED_CELL_TYPES.contains(&cell_name.as_str()) {
            continue;
        }
        let centroid = flat
            .centroids
            .get(&cell_id)
            .map(|&(x, y)| (x * scale, y * scale))
            .unwrap_or((0.0, 0.0));

        let mut inputs: Vec<Pin> = Vec::new();
        let mut outputs: Vec<Pin> = Vec::new();
        let mut net_of_label: HashMap<&str, usize> = HashMap::new();

        for &label_index in label_indices {
            let element = &flat.elements[label_index];
            let text = element.text.as_deref().unwrap_or_default();
            let net = nets[label_index];

            if POWER_PIN_NAMES.contains(&text) {
                continue;
            }
            // A pin is often labelled on several layers at once. Those
            // repeats must agree; if they don't, the geometry has been
            // read wrong and the netlist would be wrong with it.
            if let Some(&seen) = net_of_label.get(text) {
                if seen != net {
                    return Err(Sky130Error::PinConflict(format!(
                        "pin \"{text}\" of {cell_name}#{cell_id} is labelled on net {seen} and \
                         net {net}"
                    )));
                }
                continue;
            }
            net_of_label.insert(text, net);

            let pin = next_pin;
            next_pin += 1;
            if OUTPUT_PIN_NAMES.contains(&text) {
                if let Some(existing) = output_on_net.insert(net, pin) {
                    return Err(Sky130Error::MultipleDrivers(format!(
                        "net {net} is driven by pin {existing} and by \"{text}\" of \
                         {cell_name}#{cell_id}"
                    )));
                }
                outputs.push((pin, text.to_string()));
            } else {
                inputs_on_net.entry(net).or_default().push(pin);
                sinks_on_net.entry(net).or_default().push(DrivenPin {
                    pin: format!("{cell_name}#{cell_id}.{text}"),
                    centroid,
                });
                inputs.push((pin, text.to_string()));
            }
        }

        cells.push(Cell::Sky130Standard {
            cell_id: cell_id as u64,
            cell_name: cell_name.clone(),
            centroid,
            inputs,
            outputs,
        });
    }

    // The design's own ports. A port label lives on the top-level
    // structure, and which way it faces follows from the net under it:
    // a net some cell already drives is something the design *reports*,
    // and a net nothing drives is something the design is *told*. That is
    // the whole of port detection — no list of port names needed.
    let mut nets_by_port: BTreeMap<&str, BTreeSet<usize>> = BTreeMap::new();
    for (index, element) in flat.elements.iter().enumerate() {
        if element.cell_id != 0 || element.layer == TEXT {
            continue;
        }
        let Some(text) = element.text.as_deref() else {
            continue;
        };
        if POWER_PIN_NAMES.contains(&text) {
            continue;
        }
        let net = nets[index];
        // A labelled net no cell pin touches is a supply rail or a stray
        // annotation, not a port: it would contribute a pin wired to
        // nothing.
        if !output_on_net.contains_key(&net) && !inputs_on_net.contains_key(&net) {
            continue;
        }
        nets_by_port.entry(text).or_default().insert(net);
    }

    let mut port_nets: Vec<(&str, usize)> = Vec::new();
    for (text, nets) in nets_by_port {
        match nets.iter().copied().collect::<Vec<_>>().as_slice() {
            [only] => port_nets.push((text, *only)),
            many => {
                return Err(Sky130Error::Port(format!(
                    "{} top-level labels read \"{text}\" and they sit on different nets, so it \
                     does not name one port",
                    many.len()
                )));
            }
        }
    }

    // An input port drives the design, so on the graph it is an *output*
    // pin of the `Input` cell — and the other way round for outputs.
    let mut input_pins: Vec<Pin> = Vec::new();
    let mut output_pins: Vec<Pin> = Vec::new();
    for (text, net) in port_nets {
        let pin = next_pin;
        next_pin += 1;
        match output_on_net.entry(net) {
            std::collections::btree_map::Entry::Occupied(_) => {
                inputs_on_net.entry(net).or_default().push(pin);
                output_pins.push((pin, text.to_string()));
            }
            std::collections::btree_map::Entry::Vacant(slot) => {
                slot.insert(pin);
                input_pins.push((pin, text.to_string()));
            }
        }
    }
    cells.push(Cell::Input {
        outputs: input_pins,
    });
    cells.push(Cell::Output {
        inputs: output_pins,
    });

    // Whatever is still undriven gets an input invented for it — see
    // [`DummyInput`] on why that is a finding and not a fix.
    let undriven: Vec<usize> = inputs_on_net
        .keys()
        .copied()
        .filter(|net| !output_on_net.contains_key(net))
        .collect();
    let mut dummy_inputs = Vec::with_capacity(undriven.len());
    for (index, net) in undriven.into_iter().enumerate() {
        let label = format!("dummy{}", index + 1);
        let pin = next_pin;
        next_pin += 1;
        output_on_net.insert(net, pin);
        dummy_inputs.push(DummyInput {
            label: label.clone(),
            drives: sinks_on_net.remove(&net).unwrap_or_default(),
        });
        cells.push(Cell::Input {
            outputs: vec![(pin, label)],
        });
    }

    // One connection entry per driver, listing every pin it reaches.
    let connections = output_on_net
        .iter()
        .filter_map(|(net, &driver)| {
            let sinks = inputs_on_net.get(net)?;
            Some(HashMap::from([(driver, sinks.clone())]))
        })
        .collect();

    let mut graph = Graph { cells, connections };
    graph.normalize();
    Ok(Layout {
        graph,
        dummy_inputs,
    })
}

/// Reads a `sky130` GDSII layout and recovers the circuit it draws.
///
/// Takes the file's bytes rather than a path, so it works the same on the
/// `wasm32` build, where a browser hands over a file as bytes and there is
/// no filesystem to open. [`read_gds`] is the convenience wrapper for a
/// path on native builds.
///
/// `options` supplies the one thing geometry cannot: which of the
/// top-level labels are the design's inputs and which are its outputs.
///
/// The result is [`Graph::normalize`]d, exactly as [`crate::graph::parse_graph`]
/// normalizes a netlist read from JSON, so either way in gives a graph of
/// the same shape.
pub fn parse_gds(bytes: &[u8], options: &Sky130Options) -> Result<Layout, Sky130Error> {
    let library = GdsLibrary::from_bytes(bytes.to_vec())
        .map_err(|err| Sky130Error::Parse(format!("{err:?}")))?;
    // GDSII stores coordinates as integers in database units; a cell
    // centroid is reported in microns, which is what the rest of the
    // program draws in.
    let scale = library.units.db_unit() * 1e6;
    let flat = flatten(&library, options.top_cell.as_deref())?;
    let nets = connected_components(&flat.elements);
    build_graph(&flat, &nets, scale)
}

/// [`parse_gds`] for a file on disk. Native builds only — the `wasm32`
/// build has no filesystem, and reads its bytes from the browser instead.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_gds(path: &Path, options: &Sky130Options) -> Result<Layout, Sky130Error> {
    let bytes =
        std::fs::read(path).map_err(|err| Sky130Error::Io(format!("{}: {err}", path.display())))?;
    parse_gds(&bytes, options)
}
