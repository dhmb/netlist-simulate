use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};

use batsat::{Lit, SolverInterface, Var, lbool};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum CellType {
    #[serde(rename = "sky130_standard_cell")]
    Sky130Standard,
    #[serde(rename = "input_cell")]
    Input,
    #[serde(rename = "output_cell")]
    Output,
    MergeCell,
}

/// The coarse role a cell plays, for summarizing what a graph is made of
/// (see [`Graph::cell_type_counts`]). Deliberately a handful of buckets,
/// not a full library taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CellCategory {
    /// A [`Cell::Input`] — where the design's own inputs enter.
    Input,
    /// A [`Cell::Output`].
    Output,
    /// Computes something: its outputs are a boolean function of its
    /// inputs alone. The recognized gate families are the ones
    /// [`gate_output_exprs`] can actually write an expression for, plus
    /// the `"BooleanFunction"` merge cells composed out of them — minus
    /// the clock network, which carries the clock rather than computing
    /// with it (see `Other`).
    Boolean,
    /// Holds state across a clock edge: flip-flops and latches, and the
    /// `"MuxedResetableFlipflop"`/`"ShiftRegister"` merge cells built from
    /// them.
    State,
    /// Everything else: the clock network (`clkbuf`, `clkinv`, ...),
    /// which relays the clock rather than computing with it, and any cell
    /// family none of the above recognizes (a delay cell, a tap/fill, an
    /// unfamiliar merge, ...). Beyond the clock cells this is not a claim
    /// that the cell is exotic, just that this classification doesn't
    /// cover it.
    Other,
}

impl CellCategory {
    /// Every category, in declaration order — the order they're listed
    /// in when nothing else (a count) tells them apart.
    pub const ALL: [CellCategory; 5] = [
        CellCategory::Input,
        CellCategory::Output,
        CellCategory::Boolean,
        CellCategory::State,
        CellCategory::Other,
    ];

    pub fn label(self) -> &'static str {
        match self {
            CellCategory::Input => "Input",
            CellCategory::Output => "Output",
            CellCategory::Boolean => "Boolean",
            CellCategory::State => "State",
            CellCategory::Other => "Other",
        }
    }
}

/// Which [`CellCategory`] a `sky130_fd_sc_hd__*` cell name falls into,
/// from its family alone (the drive-strength suffix carries no role).
///
/// The combinational list is kept deliberately in step with
/// [`gate_output_exprs`]: a family that function can't write an expression
/// for isn't claimed to be `Boolean` here either, and lands in `Other`.
fn sky130_cell_category(cell_name: &str) -> CellCategory {
    let Some(base) = strip_sky130_prefix_and_drive(cell_name) else {
        return CellCategory::Other;
    };

    // Sequential families: `df*` flip-flops and their scan/enable variants
    // (`sdf*`, `edf*`, `sedf*`), `dlx*`/`dlr*` latches and the `dlclkp`
    // clock gate. Note the `dly*` delay cells share the `dl` prefix but
    // hold nothing, so the check is on the longer stems, not `"dl"`.
    const STATE_PREFIXES: [&str; 7] = ["df", "sdf", "edf", "sedf", "dlx", "dlr", "dlclkp"];
    if STATE_PREFIXES.iter().any(|prefix| base.starts_with(prefix)) {
        return CellCategory::State;
    }

    // The clock network (`clkbuf`, `clkinv`, `clkdlybuf4s*`, ...) relays
    // the clock instead of computing with it, so it stays out of
    // `Boolean` even where the cell is a plain buffer or inverter that
    // [`gate_output_exprs`] does write an expression for. Checked before
    // the combinational families below, several of which the names would
    // otherwise match.
    if base.starts_with("clk") {
        return CellCategory::Other;
    }

    if matches!(base, "buf" | "inv" | "conb" | "xor2" | "xnor2")
        || base.starts_with("and")
        || base.starts_with("or")
        || base.starts_with("nand")
        || base.starts_with("nor")
        || base.starts_with("mux2")
    {
        return CellCategory::Boolean;
    }

    // The compound AOI/OAI family: a leading 'a'/'o' followed by a digit
    // (`a31o`, `o21bai`, ...), as `gate_output_exprs` recognizes it.
    let mut chars = base.chars();
    let leading = chars.next();
    if matches!(leading, Some('a') | Some('o')) && chars.next().is_some_and(|c| c.is_ascii_digit())
    {
        return CellCategory::Boolean;
    }

    CellCategory::Other
}

pub type Pin = (u32, String);

/// The part of a pin's name to match against for semantic role detection
/// (`"S"`, `"CLK"`, `"A0"`, an `"_N"` suffix, a leading group letter, ...):
/// `name` itself, minus any `" = <value>"` a prior
/// [`Graph::propagate_pin_values`] may have appended. Every place in this
/// module that recognizes a pin by its name — `gate_output_exprs`, the
/// shift-register/flipflop merges' `S`/`A1`/`A0`/`RESET_B`/`CLK`/`Q`/`X`
/// matching — reads it through here instead of the raw name, so a pin
/// already relabeled with its propagated value still matches; matching the
/// raw name directly would silently stop working the moment any pin gets
/// relabeled.
fn pin_base_name(name: &str) -> &str {
    name.split(" = ").next().unwrap_or(name)
}

/// The other half of [`pin_base_name`]: the `<value>` a prior
/// [`Graph::propagate_pin_values`] appended to `name` as `" = <value>"`,
/// or `None` for a pin nothing was ever propagated onto. This is what
/// [`Graph::without_propagated_edges`] compares across an edge, and since
/// every later action carries the label through unchanged (a merge only
/// ever prefixes it with the ancestor's name), it survives the rest of the
/// action chain where the propagation's own net -> value map does not.
fn propagated_value(name: &str) -> Option<&str> {
    name.split_once(" = ").map(|(_, value)| value)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "cell_type")]
pub enum Cell {
    #[serde(rename = "sky130_standard_cell")]
    Sky130Standard {
        cell_id: u64,
        cell_name: String,
        centroid: (f64, f64),
        inputs: Vec<Pin>,
        outputs: Vec<Pin>,
    },
    #[serde(rename = "input_cell")]
    Input { outputs: Vec<Pin> },
    #[serde(rename = "output_cell")]
    Output { inputs: Vec<Pin> },
    MergeCell {
        cell_name: String,
        centroid: (f64, f64),
        inputs: Vec<Pin>,
        outputs: Vec<Pin>,
        ancestor_cells: Vec<Cell>,
        /// For a `"BooleanFunction"` merge cell (see
        /// [`Graph::merge_boolean_functions`]): each boundary output net's
        /// composed combinational expression, in terms of this cell's own
        /// boundary `inputs`. Empty for every other merge kind.
        #[serde(default)]
        boolean_outputs: Vec<(u32, BoolExpr)>,
    },
}

/// A boolean expression over free variables identified by net id — the
/// composed combinational function of a group of gates, once every net
/// internal to the group has been inlined away. See
/// [`Graph::merge_boolean_functions`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum BoolExpr {
    Var(u32),
    Const(bool),
    Not(Box<BoolExpr>),
    And(Vec<BoolExpr>),
    Or(Vec<BoolExpr>),
    Xor(Box<BoolExpr>, Box<BoolExpr>),
}

impl BoolExpr {
    fn not(e: BoolExpr) -> BoolExpr {
        BoolExpr::Not(Box::new(e))
    }

    fn and(es: Vec<BoolExpr>) -> BoolExpr {
        match es.len() {
            0 => BoolExpr::Const(true),
            1 => es.into_iter().next().unwrap(),
            _ => BoolExpr::And(es),
        }
    }

    fn or(es: Vec<BoolExpr>) -> BoolExpr {
        match es.len() {
            0 => BoolExpr::Const(false),
            1 => es.into_iter().next().unwrap(),
            _ => BoolExpr::Or(es),
        }
    }

    fn xor(a: BoolExpr, b: BoolExpr) -> BoolExpr {
        BoolExpr::Xor(Box::new(a), Box::new(b))
    }

    /// Renders this expression as an SMT-LIB2 s-expression. Each free
    /// `Var(net)` becomes the symbol `|net_<id>|` (bar-quoted so the net id
    /// alone is always a valid SMT-LIB identifier).
    ///
    /// Solving no longer goes through SMT — see [`CnfEncoder`] — but this
    /// is still how the cross-check tests hand a function to `z3`, and it
    /// stays a convenient way to dump a function in a portable form.
    #[allow(dead_code)]
    pub fn to_smtlib(&self) -> String {
        match self {
            BoolExpr::Var(net) => format!("|net_{net}|"),
            BoolExpr::Const(true) => "true".to_string(),
            BoolExpr::Const(false) => "false".to_string(),
            BoolExpr::Not(e) => format!("(not {})", e.to_smtlib()),
            BoolExpr::And(es) => format!(
                "(and {})",
                es.iter()
                    .map(BoolExpr::to_smtlib)
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            BoolExpr::Or(es) => format!(
                "(or {})",
                es.iter()
                    .map(BoolExpr::to_smtlib)
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            BoolExpr::Xor(a, b) => format!("(xor {} {})", a.to_smtlib(), b.to_smtlib()),
        }
    }
}

/// Tseitin-encodes [`BoolExpr`]s over net ids into CNF for a SAT solver.
///
/// Every net id gets a solver variable, and every compound sub-expression
/// gets a fresh auxiliary variable constrained to *equal* that
/// sub-expression — both directions, not just one implication — so that a
/// model assigns consistent values to inputs and outputs alike, the way
/// the `(assert (= |net_n| ...))` equalities of [`BoolExpr::to_smtlib`] do.
/// The encoding is linear in expression size, where expanding to raw CNF
/// would be exponential.
pub struct CnfEncoder<S> {
    solver: S,
    /// Solver variable standing for each net id *at one cycle*, allocated
    /// on first use. The purely combinational callers only ever use cycle
    /// 0 (which is what the un-suffixed methods below address); the
    /// sequential unrolling of [`Graph::solve_system`] is what gives the
    /// same net a separate variable per cycle.
    nets: BTreeMap<(u32, u32), Var>,
    /// A literal asserted true, used to encode [`BoolExpr::Const`] (and its
    /// negation for `false`). Allocated on first use.
    unit_true: Option<Lit>,
}

impl<S: SolverInterface> CnfEncoder<S> {
    pub fn new(solver: S) -> Self {
        CnfEncoder {
            solver,
            nets: BTreeMap::new(),
            unit_true: None,
        }
    }

    /// The solver variable for `net` at cycle 0 — see [`Self::net_var_at`].
    pub fn net_var(&mut self, net: u32) -> Var {
        self.net_var_at(net, 0)
    }

    /// The solver variable for `net` at `cycle`, allocating it if this is
    /// its first mention. Every net named anywhere in the problem —
    /// including ones that only ever appear inside an expression — must
    /// round-trip through here so a model can be read back per net.
    pub fn net_var_at(&mut self, net: u32, cycle: u32) -> Var {
        if let Some(&var) = self.nets.get(&(net, cycle)) {
            return var;
        }
        let var = self.solver.new_var_default();
        self.nets.insert((net, cycle), var);
        var
    }

    /// Positive literal for `net` at `cycle`.
    fn net_lit_at(&mut self, net: u32, cycle: u32) -> Lit {
        let var = self.net_var_at(net, cycle);
        Lit::new(var, true)
    }

    /// A literal that is always true, shared across the whole encoding.
    fn true_lit(&mut self) -> Lit {
        if let Some(lit) = self.unit_true {
            return lit;
        }
        let lit = Lit::new(self.solver.new_var_default(), true);
        self.add_clause(vec![lit]);
        self.unit_true = Some(lit);
        lit
    }

    fn fresh(&mut self) -> Lit {
        Lit::new(self.solver.new_var_default(), true)
    }

    /// A fresh, otherwise unconstrained literal, for use as the guard of
    /// [`Self::assert_net_value_at_if`] clauses and as a `solve_limited`
    /// assumption — the standard way to switch a set of clauses on for one
    /// solve without rebuilding the solver.
    pub fn fresh_guard(&mut self) -> Lit {
        self.fresh()
    }

    fn add_clause(&mut self, mut lits: Vec<Lit>) {
        self.solver.add_clause_reuse(&mut lits);
    }

    /// Encodes `expr`, reading every `BoolExpr::Var(net)` in it as that
    /// net *at `cycle`*, and returns a literal equivalent to it, adding
    /// whatever clauses that takes.
    pub fn encode_at(&mut self, expr: &BoolExpr, cycle: u32) -> Lit {
        match expr {
            BoolExpr::Var(net) => self.net_lit_at(*net, cycle),
            BoolExpr::Const(true) => self.true_lit(),
            BoolExpr::Const(false) => !self.true_lit(),
            BoolExpr::Not(e) => !self.encode_at(e, cycle),
            // An empty `And`/`Or` is the identity of its operator, matching
            // `BoolExpr::and`/`BoolExpr::or`. Normalization there means this
            // should not arise, but a deserialized expression can hold any
            // shape, so handle it rather than emit a degenerate clause.
            BoolExpr::And(es) if es.is_empty() => self.true_lit(),
            BoolExpr::Or(es) if es.is_empty() => !self.true_lit(),
            BoolExpr::And(es) => {
                let lits: Vec<Lit> = es.iter().map(|e| self.encode_at(e, cycle)).collect();
                let y = self.fresh();
                // y -> each operand.
                for &l in &lits {
                    self.add_clause(vec![!y, l]);
                }
                // All operands -> y.
                let mut clause = vec![y];
                clause.extend(lits.iter().map(|&l| !l));
                self.add_clause(clause);
                y
            }
            BoolExpr::Or(es) => {
                let lits: Vec<Lit> = es.iter().map(|e| self.encode_at(e, cycle)).collect();
                let y = self.fresh();
                // Each operand -> y.
                for &l in &lits {
                    self.add_clause(vec![y, !l]);
                }
                // y -> some operand.
                let mut clause = vec![!y];
                clause.extend(lits.iter().copied());
                self.add_clause(clause);
                y
            }
            BoolExpr::Xor(a, b) => {
                let (a, b) = (self.encode_at(a, cycle), self.encode_at(b, cycle));
                let y = self.fresh();
                // y <-> a xor b, all four clauses: agreeing operands force
                // y false, differing ones force it true.
                self.add_clause(vec![!a, !b, !y]);
                self.add_clause(vec![a, b, !y]);
                self.add_clause(vec![a, !b, y]);
                self.add_clause(vec![!a, b, y]);
                y
            }
        }
    }

    /// Asserts `net == expr` over cycle-0 nets, the CNF counterpart of the
    /// `(assert (= |net_n| ...))` an SMT-LIB2 script would carry.
    pub fn assert_net_eq(&mut self, net: u32, expr: &BoolExpr) {
        self.assert_eq_at(net, 0, expr, 0);
    }

    /// Asserts `net`-at-`net_cycle` `== expr`-over-`expr_cycle`-nets. The
    /// two cycles differ for a state element's transition relation, where
    /// the flip-flop's value at one cycle equals its next-state expression
    /// evaluated at the one before.
    pub fn assert_eq_at(&mut self, net: u32, net_cycle: u32, expr: &BoolExpr, expr_cycle: u32) {
        let value = self.encode_at(expr, expr_cycle);
        let n = self.net_lit_at(net, net_cycle);
        self.add_clause(vec![!n, value]);
        self.add_clause(vec![n, !value]);
    }

    /// Asserts `net == value` at cycle 0, pinning a net to a fixed truth
    /// value.
    pub fn assert_net_value(&mut self, net: u32, value: bool) {
        self.assert_net_value_at(net, 0, value);
    }

    /// Asserts `net == value` at `cycle`.
    pub fn assert_net_value_at(&mut self, net: u32, cycle: u32, value: bool) {
        let lit = self.net_lit_at(net, cycle);
        self.add_clause(vec![if value { lit } else { !lit }]);
    }

    /// Asserts `guard -> net@cycle == value`: the constraint only binds on
    /// a solve that assumes `guard` (see [`Self::fresh_guard`]), so several
    /// mutually exclusive conditions can share one incrementally built
    /// solver.
    pub fn assert_net_value_at_if(&mut self, guard: Lit, net: u32, cycle: u32, value: bool) {
        let lit = self.net_lit_at(net, cycle);
        self.add_clause(vec![!guard, if value { lit } else { !lit }]);
    }

    /// Asserts `guard -> (net_1@cycle == value_1 | net_2@cycle == value_2
    /// | ...)`: at least one of `terms` holds. With every term the
    /// complement of a bus bit's target value, this is "the bus differs
    /// from that value".
    pub fn assert_any_net_value_at_if(&mut self, guard: Lit, cycle: u32, terms: &[(u32, bool)]) {
        let mut clause = Vec::with_capacity(terms.len() + 1);
        clause.push(!guard);
        for &(net, value) in terms {
            let lit = self.net_lit_at(net, cycle);
            clause.push(if value { lit } else { !lit });
        }
        self.add_clause(clause);
    }

    /// Every net given a cycle-0 solver variable so far, in net id order.
    pub fn nets(&self) -> impl Iterator<Item = (u32, Var)> + '_ {
        self.nets
            .iter()
            .filter(|((_, cycle), _)| *cycle == 0)
            .map(|((net, _), &var)| (*net, var))
    }

    /// `net`'s value at `cycle` in the model of the last successful solve,
    /// or `None` if that net never got a variable at that cycle.
    pub fn net_value_at(&self, net: u32, cycle: u32) -> Option<bool> {
        let &var = self.nets.get(&(net, cycle))?;
        Some(self.solver.value_lit(Lit::new(var, true)) == lbool::TRUE)
    }

    pub fn solver_mut(&mut self) -> &mut S {
        &mut self.solver
    }
}

impl Cell {
    pub fn cell_type(&self) -> CellType {
        match self {
            Cell::Sky130Standard { .. } => CellType::Sky130Standard,
            Cell::Input { .. } => CellType::Input,
            Cell::Output { .. } => CellType::Output,
            Cell::MergeCell { .. } => CellType::MergeCell,
        }
    }

    /// This cell's coarse role — see [`CellCategory`]. A `MergeCell` is
    /// classified by which merge produced it, a standard cell by its
    /// family name.
    pub fn category(&self) -> CellCategory {
        match self {
            Cell::Input { .. } => CellCategory::Input,
            Cell::Output { .. } => CellCategory::Output,
            Cell::Sky130Standard { cell_name, .. } => sky130_cell_category(cell_name),
            Cell::MergeCell { cell_name, .. } => match cell_name.as_str() {
                "BooleanFunction" => CellCategory::Boolean,
                "MuxedResetableFlipflop" | "ShiftRegister" => CellCategory::State,
                _ => CellCategory::Other,
            },
        }
    }

    pub fn centroid(&self) -> Option<(f64, f64)> {
        match self {
            Cell::Sky130Standard { centroid, .. } | Cell::MergeCell { centroid, .. } => {
                Some(*centroid)
            }
            Cell::Input { .. } | Cell::Output { .. } => None,
        }
    }

    pub fn inputs(&self) -> &[Pin] {
        match self {
            Cell::Sky130Standard { inputs, .. }
            | Cell::MergeCell { inputs, .. }
            | Cell::Output { inputs, .. } => inputs,
            Cell::Input { .. } => &[],
        }
    }

    pub fn outputs(&self) -> &[Pin] {
        match self {
            Cell::Sky130Standard { outputs, .. }
            | Cell::MergeCell { outputs, .. }
            | Cell::Input { outputs, .. } => outputs,
            Cell::Output { .. } => &[],
        }
    }
}

pub type Connection = HashMap<u32, Vec<u32>>;

/// One row of [`Graph::cell_type_counts`]: a cell type, its category, and
/// how many cells of it the graph holds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CellTypeCount {
    pub name: String,
    pub category: CellCategory,
    pub count: usize,
}

/// One row of [`Graph::cell_category_counts`]: a category and how many
/// cells of the graph fall into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellCategoryCount {
    pub category: CellCategory,
    pub count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Graph {
    pub cells: Vec<Cell>,
    pub connections: Vec<Connection>,
}

/// Returned by [`Graph::without_cell_names`] when a cell slated for removal
/// isn't a clean single-input/single-output pass-through, so there's no
/// unambiguous way to splice its connections back together.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AmbiguousRemoval {
    pub cell_id: u64,
    pub cell_name: String,
    pub input_count: usize,
    pub output_count: usize,
}

impl std::fmt::Display for AmbiguousRemoval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "cell {} (id {}) has {} input pin(s) and {} output pin(s); only a \
             single-input/single-output cell can be spliced out unambiguously",
            self.cell_name, self.cell_id, self.input_count, self.output_count
        )
    }
}

impl std::error::Error for AmbiguousRemoval {}

/// How [`Graph::merge_boolean_functions_grouped`] decides which recognized
/// gates end up in one `"BooleanFunction"` merge cell. Both variants
/// partition the gates — no gate is ever duplicated into two cells — and
/// both keep any genuine combinational cycle whole; they differ only in
/// how much they lump together.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum GateGrouping {
    /// One cell per weakly connected cloud of gates: every gate that can
    /// reach another through nothing but recognized gates lands in the
    /// same cell. See [`Graph::merge_boolean_functions`].
    ConnectedGates,
    /// One cell per *set of destination state elements*, grouped into
    /// feedback loops. See
    /// [`Graph::merge_boolean_functions_by_register_scc`].
    RegisterScc,
    /// One cell per *exact set of destination state elements*, with no
    /// feedback-loop grouping in between. See
    /// [`Graph::merge_boolean_functions_by_cone_of_influence`].
    ConeOfInfluence,
}

impl Graph {
    /// Returns a new `Graph` with every `Sky130Standard` cell whose
    /// `cell_name` is in `excluded_cell_names` removed. Connections that
    /// passed through a removed cell are rewired directly between whatever
    /// drove it and whatever it drove, so the rest of the network is left
    /// electrically equivalent rather than just losing those edges.
    ///
    /// Rewiring only makes sense for a removed cell with exactly one input
    /// pin and one output pin (true of all buffer cells); a removed cell
    /// shaped any other way has no unambiguous splice, so this returns
    /// `Err` describing the offending cell instead of silently dropping its
    /// connections.
    pub fn without_cell_names(
        &self,
        excluded_cell_names: &HashSet<&str>,
    ) -> Result<Graph, AmbiguousRemoval> {
        let cells: Vec<Cell> = self
            .cells
            .iter()
            .filter(|cell| match cell {
                Cell::Sky130Standard { cell_name, .. } | Cell::MergeCell { cell_name, .. } => {
                    !excluded_cell_names.contains(cell_name.as_str())
                }
                Cell::Input { .. } | Cell::Output { .. } => true,
            })
            .cloned()
            .collect();

        // input net id -> output net id, for each removed cell that is a
        // simple pass-through (one input pin, one output pin).
        let mut passthroughs: HashMap<u32, u32> = HashMap::new();
        for cell in &self.cells {
            let Cell::Sky130Standard {
                cell_id,
                cell_name,
                inputs,
                outputs,
                ..
            } = cell
            else {
                continue;
            };
            if !excluded_cell_names.contains(cell_name.as_str()) {
                continue;
            }
            match (inputs.as_slice(), outputs.as_slice()) {
                ([(in_net, _)], [(out_net, _)]) => {
                    passthroughs.insert(*in_net, *out_net);
                }
                _ => {
                    return Err(AmbiguousRemoval {
                        cell_id: *cell_id,
                        cell_name: cell_name.clone(),
                        input_count: inputs.len(),
                        output_count: outputs.len(),
                    });
                }
            }
        }

        // Original fanout, flattened to a single net id -> net ids map.
        let mut fanout: HashMap<u32, Vec<u32>> = HashMap::new();
        for conn in &self.connections {
            for (&src, dsts) in conn {
                fanout.entry(src).or_default().extend(dsts);
            }
        }

        let removed_output_nets: HashSet<u32> = passthroughs.values().copied().collect();

        let remaining_outputs: HashSet<u32> = cells
            .iter()
            .flat_map(|cell| cell.outputs())
            .map(|&(net_id, _)| net_id)
            .collect();
        let remaining_inputs: HashSet<u32> = cells
            .iter()
            .flat_map(|cell| cell.inputs())
            .map(|&(net_id, _)| net_id)
            .collect();

        // Resolves a destination net id to the (possibly several) real
        // input pins it ends up reaching, splicing through any chain of
        // removed pass-through cells along the way.
        fn resolve(
            net_id: u32,
            passthroughs: &HashMap<u32, u32>,
            fanout: &HashMap<u32, Vec<u32>>,
            visited: &mut HashSet<u32>,
        ) -> Vec<u32> {
            let Some(&out_net) = passthroughs.get(&net_id) else {
                return vec![net_id];
            };
            // Guards against a malformed netlist with a cycle of removed
            // buffers; a real one can't have this.
            if !visited.insert(net_id) {
                return vec![];
            }
            fanout
                .get(&out_net)
                .into_iter()
                .flatten()
                .flat_map(|&dst| resolve(dst, passthroughs, fanout, visited))
                .collect()
        }

        let mut connections: Vec<Connection> = fanout
            .iter()
            .filter(|(src, _)| !removed_output_nets.contains(src))
            .filter_map(|(&src, dsts)| {
                let spliced: Vec<u32> = dsts
                    .iter()
                    .flat_map(|&dst| resolve(dst, &passthroughs, &fanout, &mut HashSet::new()))
                    .filter(|dst| remaining_inputs.contains(dst))
                    .collect();
                (remaining_outputs.contains(&src) && !spliced.is_empty())
                    .then(|| Connection::from([(src, spliced)]))
            })
            .collect();
        connections.sort_by_key(|conn| *conn.keys().next().unwrap());

        Ok(Graph { cells, connections })
    }

    /// Finds every `sky130_fd_sc_hd__mux2_1` whose `X` output feeds a
    /// `sky130_fd_sc_hd__dfrtp_2`'s `D` input, with that flipflop's `Q`
    /// output looped back into the same mux's `A0` input (the standard
    /// "hold current value or load a new one" idiom), and replaces each
    /// such pair with a single `MergeCell` named `"MuxedResetableFlipflop"`.
    ///
    /// The merge cell's centroid is the flipflop's centroid. Every pin
    /// other than the internal `X`->`D` link and the `A0` input (now driven
    /// internally by `Q`) is repatched straight onto the merge cell,
    /// keeping its original net id so existing connections keep working
    /// unchanged. `Q` itself stays exposed as an output, since besides
    /// feeding `A0` it typically also drives other cells.
    ///
    /// A mux not shaped exactly like this (its `X` driving more than one
    /// destination, or not looping back through `A0`) is left untouched.
    pub fn merge_muxed_resetable_flipflops(&self) -> Graph {
        const MUX_CELL_NAME: &str = "sky130_fd_sc_hd__mux2_1";
        const DFF_CELL_NAME: &str = "sky130_fd_sc_hd__dfrtp_2";

        // Flattened net id -> destination net ids, as in `without_cell_names`.
        let mut fanout: HashMap<u32, Vec<u32>> = HashMap::new();
        for conn in &self.connections {
            for (&src, dsts) in conn {
                fanout.entry(src).or_default().extend(dsts.iter().copied());
            }
        }

        // dfrtp_2 cells keyed by their `D` input's net id, so a mux's `X`
        // output can be matched to the flipflop it drives.
        let dff_by_d_net: HashMap<u32, usize> = self
            .cells
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| match cell {
                Cell::Sky130Standard {
                    cell_name, inputs, ..
                } if cell_name == DFF_CELL_NAME => inputs
                    .iter()
                    .find(|(_, name)| pin_base_name(name) == "D")
                    .map(|&(net, _)| (net, i)),
                _ => None,
            })
            .collect();

        // (mux index, dff index) pairs verified to match the expected
        // mux-X->dff-D, dff-Q->mux-A0 pattern.
        let mut claimed: HashSet<usize> = HashSet::new();
        let mut matched_pairs: Vec<(usize, usize)> = Vec::new();

        for (mux_idx, cell) in self.cells.iter().enumerate() {
            let Cell::Sky130Standard {
                cell_name,
                inputs: mux_inputs,
                outputs: mux_outputs,
                ..
            } = cell
            else {
                continue;
            };
            if cell_name != MUX_CELL_NAME {
                continue;
            }

            let Some(&(mux_x_net, _)) = mux_outputs
                .iter()
                .find(|(_, name)| pin_base_name(name) == "X")
            else {
                continue;
            };
            let Some(&(mux_a0_net, _)) = mux_inputs
                .iter()
                .find(|(_, name)| pin_base_name(name) == "A0")
            else {
                continue;
            };

            // `X` must feed exactly one destination: the flipflop's `D`.
            let Some(&dff_idx) = fanout
                .get(&mux_x_net)
                .filter(|dsts| dsts.len() == 1)
                .and_then(|dsts| dff_by_d_net.get(&dsts[0]))
            else {
                continue;
            };

            let Cell::Sky130Standard {
                outputs: dff_outputs,
                ..
            } = &self.cells[dff_idx]
            else {
                continue;
            };
            let Some(&(dff_q_net, _)) = dff_outputs
                .iter()
                .find(|(_, name)| pin_base_name(name) == "Q")
            else {
                continue;
            };

            // `Q` must loop back into this mux's `A0` (it may drive other
            // things too, which is fine).
            let feeds_back = fanout
                .get(&dff_q_net)
                .is_some_and(|dsts| dsts.contains(&mux_a0_net));
            if !feeds_back || claimed.contains(&mux_idx) || claimed.contains(&dff_idx) {
                continue;
            }

            claimed.insert(mux_idx);
            claimed.insert(dff_idx);
            matched_pairs.push((mux_idx, dff_idx));
        }

        // Net ids fully internalized by a merge: the mux's `X` output, the
        // flipflop's `D` input it drives, and the mux's `A0` input now fed
        // internally by `Q`.
        let mut internalized_nets: HashSet<u32> = HashSet::new();
        let mut merge_cells: HashMap<usize, Cell> = HashMap::new();

        for (mux_idx, dff_idx) in matched_pairs {
            let Cell::Sky130Standard {
                inputs: mux_inputs,
                outputs: mux_outputs,
                ..
            } = self.cells[mux_idx].clone()
            else {
                unreachable!("mux_idx was matched above as a Sky130Standard mux2_1 cell")
            };
            let Cell::Sky130Standard {
                centroid: dff_centroid,
                inputs: dff_inputs,
                outputs: dff_outputs,
                ..
            } = self.cells[dff_idx].clone()
            else {
                unreachable!("dff_idx was matched above as a Sky130Standard dfrtp_2 cell")
            };

            let &(mux_x_net, _) = mux_outputs
                .iter()
                .find(|(_, name)| pin_base_name(name) == "X")
                .unwrap();
            let &(mux_a0_net, _) = mux_inputs
                .iter()
                .find(|(_, name)| pin_base_name(name) == "A0")
                .unwrap();
            let &(dff_d_net, _) = dff_inputs
                .iter()
                .find(|(_, name)| pin_base_name(name) == "D")
                .unwrap();

            internalized_nets.insert(mux_x_net);
            internalized_nets.insert(mux_a0_net);
            internalized_nets.insert(dff_d_net);

            let inputs: Vec<Pin> = mux_inputs
                .iter()
                .chain(dff_inputs.iter())
                .filter(|&&(net, _)| net != mux_a0_net && net != dff_d_net)
                .cloned()
                .collect();
            let outputs: Vec<Pin> = mux_outputs
                .iter()
                .chain(dff_outputs.iter())
                .filter(|&&(net, _)| net != mux_x_net)
                .cloned()
                .collect();

            merge_cells.insert(
                mux_idx,
                Cell::MergeCell {
                    cell_name: "MuxedResetableFlipflop".to_string(),
                    centroid: dff_centroid,
                    inputs,
                    outputs,
                    ancestor_cells: vec![self.cells[mux_idx].clone(), self.cells[dff_idx].clone()],
                    boolean_outputs: Vec::new(),
                },
            );
        }

        let cells: Vec<Cell> = self
            .cells
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| {
                if let Some(merge_cell) = merge_cells.remove(&i) {
                    Some(merge_cell)
                } else if claimed.contains(&i) {
                    // The dff half of a merged pair; already folded into
                    // the merge cell replacing its mux above.
                    None
                } else {
                    Some(cell.clone())
                }
            })
            .collect();

        // Drop connection entries sourced from an internalized net (it no
        // longer exists as a standalone pin) and strip internalized nets
        // out of any remaining destination lists, but otherwise leave the
        // connections exactly as they were.
        let connections: Vec<Connection> = self
            .connections
            .iter()
            .filter_map(|conn| {
                let new_conn: Connection = conn
                    .iter()
                    .filter(|(src, _)| !internalized_nets.contains(src))
                    .map(|(&src, dsts)| {
                        let dsts = dsts
                            .iter()
                            .copied()
                            .filter(|dst| !internalized_nets.contains(dst))
                            .collect();
                        (src, dsts)
                    })
                    .collect();
                (!new_conn.is_empty()).then_some(new_conn)
            })
            .collect();

        Graph { cells, connections }
    }

    /// Finds the longest chains of `"MuxedResetableFlipflop"` `MergeCell`s
    /// (as produced by [`Self::merge_muxed_resetable_flipflops`]) linked in
    /// series — each stage's `Q` output feeding the next stage's `A1`
    /// input — and replaces each whole chain with a single `"ShiftRegister"`
    /// `MergeCell`.
    ///
    /// A stage is only chained to the next if its `Q` feeds exactly one
    /// other stage's `A1` (an ambiguous `Q` feeding more than one registered
    /// `A1` breaks the chain there rather than guessing), so this always
    /// merges the longest unambiguous run through any given stage. A stage
    /// with no such link on either side is left alone.
    ///
    /// The merged cell's centroid is the average of its stages' centroids.
    /// `S`, `RESET_B` and `CLK` are the same control signals re-derived per
    /// stage (each stage's own net id, but electrically the same select,
    /// reset and clock as every other stage), so rather than keeping one
    /// pin per stage they're collapsed onto a single `S`/`RESET_B`/`CLK`
    /// pin each, aliased to the first stage's net; whatever drove the other
    /// stages' copies now drives that one pin instead. The head stage's
    /// `A1` survives as the whole register's serial data input and is
    /// relabeled plain `A`, there being no `A0` left beside it to number
    /// it against. Every stage's `Q` output is kept — besides feeding the
    /// next stage, a bit typically also fans out elsewhere — renumbered
    /// `Q0`..`Q<n-1>` in chain order rather than all sharing the name `Q`.
    pub fn merge_shift_registers(&self) -> Graph {
        const STAGE_CELL_NAME: &str = "MuxedResetableFlipflop";
        const SHARED_CONTROL_PINS: [&str; 3] = ["S", "RESET_B", "CLK"];

        // Flattened net id -> destination net ids, as in `without_cell_names`.
        let mut fanout: HashMap<u32, Vec<u32>> = HashMap::new();
        for conn in &self.connections {
            for (&src, dsts) in conn {
                fanout.entry(src).or_default().extend(dsts.iter().copied());
            }
        }

        let stage_indices: Vec<usize> = self
            .cells
            .iter()
            .enumerate()
            .filter(|(_, cell)| {
                matches!(cell, Cell::MergeCell { cell_name, .. } if cell_name == STAGE_CELL_NAME)
            })
            .map(|(i, _)| i)
            .collect();

        // Stages keyed by their `A1` input's net id, so a stage's `Q` can be
        // matched to the stage it feeds.
        let stage_by_a1_net: HashMap<u32, usize> = stage_indices
            .iter()
            .filter_map(|&i| {
                let Cell::MergeCell { inputs, .. } = &self.cells[i] else {
                    unreachable!("stage_indices only contains MergeCell indices")
                };
                inputs
                    .iter()
                    .find(|(_, name)| pin_base_name(name) == "A1")
                    .map(|&(net, _)| (net, i))
            })
            .collect();

        // next[i] = j when stage i's `Q` feeds stage j's `A1` as the sole
        // match among other stages' `A1` pins among its destinations.
        let mut next: HashMap<usize, usize> = HashMap::new();
        for &i in &stage_indices {
            let Cell::MergeCell { outputs, .. } = &self.cells[i] else {
                unreachable!("stage_indices only contains MergeCell indices")
            };
            let Some(&(q_net, _)) = outputs.iter().find(|(_, name)| pin_base_name(name) == "Q")
            else {
                continue;
            };
            let matches: Vec<usize> = fanout
                .get(&q_net)
                .into_iter()
                .flatten()
                .filter_map(|dst| stage_by_a1_net.get(dst).copied())
                .collect();
            if let [only] = matches[..] {
                next.insert(i, only);
            }
        }

        // Longest chains: walk `next` from every stage that isn't itself
        // some other stage's next, as far as it goes.
        let chain_heads: HashSet<usize> = next.values().copied().collect();
        let mut chains: Vec<Vec<usize>> = Vec::new();
        for &start in &stage_indices {
            if chain_heads.contains(&start) {
                continue;
            }
            let mut chain = vec![start];
            let mut visited = HashSet::from([start]);
            let mut cur = start;
            while let Some(&nxt) = next.get(&cur) {
                if !visited.insert(nxt) {
                    break; // cycle guard; can't happen in a real netlist
                }
                chain.push(nxt);
                cur = nxt;
            }
            if chain.len() > 1 {
                chains.push(chain);
            }
        }

        // Net ids internalized by a chain merge: every non-head stage's
        // `A1` input, now fed internally by the previous stage's `Q`.
        let mut internalized_nets: HashSet<u32> = HashSet::new();
        // A collapsed `S`/`RESET_B`/`CLK` net id -> the first stage's net
        // id it now stands in for, applied to `connections` below.
        let mut alias_net: HashMap<u32, u32> = HashMap::new();
        let mut merge_cells: HashMap<usize, Cell> = HashMap::new();
        let mut absorbed: HashSet<usize> = HashSet::new();

        for chain in &chains {
            let mut inputs: Vec<Pin> = Vec::new();
            let mut outputs: Vec<Pin> = Vec::new();
            let mut ancestor_cells: Vec<Cell> = Vec::new();
            let mut centroid_sum = (0.0, 0.0);

            for (stage_pos, &stage_idx) in chain.iter().enumerate() {
                let Cell::MergeCell {
                    centroid,
                    inputs: stage_inputs,
                    outputs: stage_outputs,
                    ..
                } = &self.cells[stage_idx]
                else {
                    unreachable!("stage_indices only contains MergeCell indices")
                };

                centroid_sum.0 += centroid.0;
                centroid_sum.1 += centroid.1;

                if stage_pos == 0 {
                    // The head stage's `A1` is the merged register's serial
                    // data input — nothing selects between it and an `A0`
                    // any more at this level — so it's relabeled to plain
                    // `A`, keeping any `" = <value>"` already propagated
                    // onto it.
                    inputs.extend(stage_inputs.iter().cloned().map(|(net, name)| {
                        if pin_base_name(&name) == "A1" {
                            // Empty, or the `" = <value>"` a prior
                            // propagation appended.
                            let value_label = &name["A1".len()..];
                            (net, format!("A{value_label}"))
                        } else {
                            (net, name)
                        }
                    }));
                } else {
                    let &(a1_net, _) = stage_inputs
                        .iter()
                        .find(|(_, name)| pin_base_name(name) == "A1")
                        .expect("chained via stage_by_a1_net, so this stage has an A1 input");
                    internalized_nets.insert(a1_net);

                    // Alias this stage's S/RESET_B/CLK onto the first
                    // stage's, rather than keeping a separate pin per stage.
                    for shared_name in SHARED_CONTROL_PINS {
                        let Some(&(net, _)) = stage_inputs
                            .iter()
                            .find(|(_, name)| pin_base_name(name) == shared_name)
                        else {
                            continue;
                        };
                        let Some(&(representative_net, _)) = inputs
                            .iter()
                            .find(|(_, name)| pin_base_name(name) == shared_name)
                        else {
                            continue;
                        };
                        alias_net.insert(net, representative_net);
                    }

                    inputs.extend(
                        stage_inputs
                            .iter()
                            .filter(|(net, name)| {
                                *net != a1_net
                                    && !SHARED_CONTROL_PINS.contains(&pin_base_name(name))
                            })
                            .cloned(),
                    );
                    absorbed.insert(stage_idx);
                }

                // Every stage's `Q` is kept, renumbered in chain order
                // (rather than all sharing the name `Q`); any other output
                // pin a stage might have is kept under its own name as-is.
                outputs.extend(stage_outputs.iter().cloned().map(|(net, name)| {
                    if pin_base_name(&name) == "Q" {
                        (net, format!("Q{stage_pos}"))
                    } else {
                        (net, name)
                    }
                }));
                ancestor_cells.push(self.cells[stage_idx].clone());
            }

            let stage_count = chain.len() as f64;
            merge_cells.insert(
                chain[0],
                Cell::MergeCell {
                    cell_name: "ShiftRegister".to_string(),
                    centroid: (centroid_sum.0 / stage_count, centroid_sum.1 / stage_count),
                    inputs,
                    outputs,
                    ancestor_cells,
                    boolean_outputs: Vec::new(),
                },
            );
        }

        let cells: Vec<Cell> = self
            .cells
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| {
                if let Some(merge_cell) = merge_cells.remove(&i) {
                    Some(merge_cell)
                } else if absorbed.contains(&i) {
                    // A non-head stage; already folded into the merge cell
                    // replacing its chain's head above.
                    None
                } else {
                    Some(cell.clone())
                }
            })
            .collect();

        // Every non-head stage's `A1` is now internal, so it's stripped out
        // of whatever destination list fed it (its previous stage's `Q`).
        // Every collapsed `S`/`RESET_B`/`CLK` net is rewritten to the
        // representative net it now aliases (deduplicating repeats — e.g.
        // one shared clock driver that used to fan out to every stage's own
        // `CLK` now only needs to reach the one surviving pin once).
        // Otherwise connections are left exactly as they were. No source
        // entries need dropping: unlike `merge_muxed_resetable_flipflops`,
        // no output pin disappears here — every stage's `Q` is kept.
        let connections: Vec<Connection> = self
            .connections
            .iter()
            .filter_map(|conn| {
                let new_conn: Connection = conn
                    .iter()
                    .map(|(&src, dsts)| {
                        let mut seen = HashSet::new();
                        let dsts = dsts
                            .iter()
                            .copied()
                            .filter(|dst| !internalized_nets.contains(dst))
                            .map(|dst| alias_net.get(&dst).copied().unwrap_or(dst))
                            .filter(|dst| seen.insert(*dst))
                            .collect();
                        (src, dsts)
                    })
                    .collect();
                (!new_conn.is_empty()).then_some(new_conn)
            })
            .collect();

        Graph { cells, connections }
    }

    /// Groups every recognized stateless-boolean-gate `Sky130Standard` cell
    /// (see [`gate_output_exprs`]) into its connected component — two gates
    /// are connected if one's output net directly drives the other's input
    /// net — and replaces each whole component with a single
    /// `"BooleanFunction"` `MergeCell`, however many gates that reaches.
    ///
    /// Every net produced within a component is inlined away: a component's
    /// boundary `inputs` are exactly the nets driven from outside it (or by
    /// nothing at all), named `"<cell_name>#<cell_id>.<pin_name>"` after
    /// whichever ancestor cell/pin they came from; its boundary `outputs`
    /// are the nets that drive something outside the component (or drive
    /// nothing), named the same way. `boolean_outputs` carries, for each
    /// boundary output net, its composed [`BoolExpr`] purely in terms of
    /// this cell's own boundary input nets. A gate this doesn't recognize
    /// (e.g. still-unmerged sequential cells) is left alone and simply
    /// can't join any component.
    ///
    /// A component is never allowed to have a genuine combinational cycle
    /// among its own recognized gates' direct edges (rare — normally
    /// impossible in a valid netlist, since that would mean a gate's
    /// output eventually feeds its own input with no register anywhere in
    /// the loop): any such cycle's members are merged together as their
    /// own separate `BooleanFunction`, keeping the loop fully internal,
    /// while every other gate re-clusters normally. This check stops at
    /// recognized gates on purpose — it does *not* chase a loop through an
    /// external register (an LFSR-style "next input bit is a function of
    /// this cloud's own past output" is completely normal in synchronous
    /// hardware, and turns out to be inescapable by any grouping choice
    /// besides: the gates next to a real register feedback path show it
    /// whether split apart or merged together, since the register itself
    /// can never join a group — chasing it only produced ever more splits
    /// that never converged).
    pub fn merge_boolean_functions(&self) -> Graph {
        self.merge_boolean_functions_grouped(GateGrouping::ConnectedGates)
    }

    /// The same merge as [`Graph::merge_boolean_functions`] — same boundary
    /// naming, same composed `boolean_outputs`, same guarantee that every
    /// recognized gate joins exactly one cell — but grouped by *what each
    /// gate feeds* instead of by raw connectivity, which is what a
    /// sequential design's one big combinational cloud actually needs.
    ///
    /// In a synchronous design the whole netlist is one loop: the state
    /// elements' outputs fan into a single combinational cloud whose
    /// outputs come straight back to those same elements' inputs. Weak
    /// connectivity therefore can't separate anything — on a real design it
    /// can yield one cell with hundreds of inputs and outputs — even though
    /// that cell is really dozens of small, near-independent functions that
    /// merely share a few control signals.
    ///
    /// This grouping recovers them:
    ///
    /// 1. Each gate is labeled with the set of *state elements* (any cell
    ///    that isn't a recognized gate — a flip-flop, a merged
    ///    `ShiftRegister`, an `Input`/`Output`) its own outputs reach
    ///    through nothing but recognized gates. Seeded at the gates whose
    ///    output leaves the cloud and unioned backwards to a fixpoint, so
    ///    a gate's label is exactly the set of destinations its logic
    ///    contributes to.
    /// 2. Those labels induce a graph over the state elements themselves:
    ///    an edge from the element driving some gate's input to every
    ///    element that gate reaches — i.e. "this register's value
    ///    participates in computing that register's next value". Its
    ///    strongly connected components are the design's feedback loops,
    ///    each one a sub-machine whose state can only be understood as a
    ///    whole (a toggle pair, a counter, ...); an element on no cycle is
    ///    its own singleton.
    /// 3. Gates are then partitioned by the *set of components* they feed,
    ///    and each such class split into its weakly connected pieces (two
    ///    unrelated gates that happen to feed the same registers shouldn't
    ///    share a cell). A class feeding exactly one component is that
    ///    sub-machine's next-state logic; a class feeding many is shared
    ///    control logic, factored out into its own cell that then drives
    ///    the others — which is why this partitions rather than giving
    ///    each component its own private copy of the cone feeding it
    ///    (on a real design those cones can overlap enough that copying
    ///    them would multiply the gate count several times over).
    ///
    /// The register-mediated loop the doc comment on
    /// [`Graph::merge_boolean_functions`] describes is still not something
    /// any grouping can remove — but here it's at least *localized*: it
    /// shows up as one small component instead of being smeared across a
    /// single giant cell. A genuine combinational cycle stays whole for
    /// free: its gates are mutually reachable, so they share a label and a
    /// weakly connected piece by construction.
    pub fn merge_boolean_functions_by_register_scc(&self) -> Graph {
        self.merge_boolean_functions_grouped(GateGrouping::RegisterScc)
    }

    /// The same merge again — same boundary naming, same composed
    /// `boolean_outputs`, same guarantee that every recognized gate joins
    /// exactly one cell — grouped this time by each gate's *cone of
    /// influence*: the set of destinations (flip-flops, merged registers,
    /// `Output` cells — anything that isn't a recognized gate) its logic
    /// actually reaches, through nothing but recognized gates.
    ///
    /// Gates whose reached set is identical are the ones that influence
    /// exactly the same state, so they form one cell (split further into
    /// weakly connected pieces, so two unrelated gates that happen to
    /// influence the same registers don't share a cell). Read the other
    /// way round: one destination's cone of influence — every gate feeding
    /// it — is the union of every resulting cell whose reached set
    /// contains that destination. So a cone is generally made of *several*
    /// `"BooleanFunction"` cells, and a cell that influences several
    /// destinations belongs to each of their cones at once; where a cone
    /// boundary cuts through what connectivity alone would have merged,
    /// the function is split along it rather than duplicated (no gate is
    /// ever copied into two cells).
    ///
    /// This differs from [`Graph::merge_boolean_functions_by_register_scc`]
    /// exactly where feedback exists: that grouping first collapses
    /// mutually dependent state elements into one component and keys gates
    /// by the *components* they feed, deliberately treating a feedback
    /// loop's registers as one unit — so logic feeding only one register
    /// of a toggle pair lands in the same cell as logic feeding only the
    /// other. Keying on the raw destination set instead keeps those apart,
    /// which is what you want when the question is "what does this gate
    /// influence", and produces a finer split (many more, smaller cells
    /// on a real design). Neither can escape the register-mediated
    /// loops themselves — see the note on [`Graph::merge_boolean_functions`].
    ///
    /// A gate that reaches nothing at all — its logic drives no
    /// destination — has the empty cone, and those gates group together.
    /// The normalization every load applies, in place: the shaping a
    /// netlist gets on its way in, independent of the file it came from.
    /// Currently just [`Graph::merge_input_cells`].
    ///
    /// [`parse_graph`] runs this, so both load paths get it. It is also
    /// re-runnable on a graph already in memory — which is what the action
    /// list's refresh button does for a project restored from saved app
    /// state, since that `Graph` is deserialized straight out of the saved
    /// state and never passes through [`parse_graph`] again.
    ///
    /// Idempotent: running it on an already-normalized graph changes
    /// nothing.
    pub fn normalize(&mut self) {
        self.merge_input_cells();
    }

    /// Collapses every `Cell::Input` in this graph into one, carrying all
    /// of their output pins, in the order the cells and their pins appear.
    ///
    /// A netlist can declare its inputs across several input cells —
    /// a design might put one extra port on a second one, apart from
    /// the rest — which then shows up as a separate, near-empty cell in
    /// the viewer and as a second `Input#<net>` label everywhere pins are
    /// named. They all mean the same thing (the
    /// boundary where signals enter the design), so loading folds them
    /// together: the merged cell takes the position of the first input
    /// cell, and the rest are dropped.
    ///
    /// Connections are untouched: they are keyed by net id, never by cell
    /// index, and every net keeps the same id and the same driver.
    ///
    /// Two input cells declaring the *same* net is malformed — nothing can
    /// have two drivers — and would leave one cell with two pins sharing a
    /// net id, which the viewer's net-keyed pin map can't represent
    /// anyway. The first pin on a net wins and any later duplicate is
    /// dropped.
    fn merge_input_cells(&mut self) {
        if self
            .cells
            .iter()
            .filter(|cell| matches!(cell, Cell::Input { .. }))
            .count()
            < 2
        {
            return;
        }

        let mut seen_nets: HashSet<u32> = HashSet::new();
        let merged_outputs: Vec<Pin> = self
            .cells
            .iter()
            .filter_map(|cell| match cell {
                Cell::Input { outputs } => Some(outputs),
                _ => None,
            })
            .flatten()
            .filter(|(net, _)| seen_nets.insert(*net))
            .cloned()
            .collect();

        let mut merged_outputs = Some(merged_outputs);
        let mut cells = Vec::with_capacity(self.cells.len());
        for cell in self.cells.drain(..) {
            match cell {
                // The first input cell becomes the merged one, in place;
                // every later one disappears.
                Cell::Input { .. } => {
                    if let Some(outputs) = merged_outputs.take() {
                        cells.push(Cell::Input { outputs });
                    }
                }
                other => cells.push(other),
            }
        }
        self.cells = cells;
    }

    /// `output_pins` optionally narrows the graph first: when it holds at
    /// least one non-blank name, only the cone of influence of those
    /// `Output` pins — their union — is merged, and every cell outside it
    /// is pruned away (see [`Graph::cone_of_influence_cells`]). An empty
    /// list, or one of only blank names, merges the whole graph as before.
    pub fn merge_boolean_functions_by_cone_of_influence(&self, output_pins: &[String]) -> Graph {
        let wanted: Vec<&str> = output_pins
            .iter()
            .map(|name| name.trim())
            .filter(|name| !name.is_empty())
            .collect();
        if wanted.is_empty() {
            return self.merge_boolean_functions_grouped(GateGrouping::ConeOfInfluence);
        }
        self.retaining_cells(&self.cone_of_influence_cells(&wanted))
            .merge_boolean_functions_grouped(GateGrouping::ConeOfInfluence)
    }

    /// The indices of every cell in the *cone of influence* of the
    /// `Output` cell pins named by `output_pins` — the union of the cones,
    /// plus the `Output` cells carrying those pins.
    ///
    /// Pins are matched on their base name (`"success"`, `"O[3]"`), the
    /// same way [`Graph::propagate_pin_values`] matches its own labels, so
    /// a name still matches after a prior propagation has appended
    /// `" = <value>"` to it. Only `Output` cells are searched: these are
    /// the circuit's outputs, the thing a cone is normally taken of.
    ///
    /// The walk runs backwards from each named pin's net — whatever drives
    /// a net, and everything feeding the cell that produces it — until it
    /// runs out at the `Input` cells and state elements on the far side.
    /// A cell is taken whole: reaching one of a multi-output cell's nets
    /// pulls in everything feeding *any* of its outputs, which is the
    /// usual cell-granular reading of a cone rather than a per-pin one.
    ///
    /// A name matching no `Output` pin contributes nothing, so a list of
    /// only unknown names yields the empty cone — and, through
    /// [`Graph::retaining_cells`], an empty graph.
    fn cone_of_influence_cells(&self, output_pins: &[&str]) -> HashSet<usize> {
        // Destination net -> the net driving it, the reverse of the
        // flattened fanout used elsewhere in this module.
        let mut driven_by: HashMap<u32, u32> = HashMap::new();
        for conn in &self.connections {
            for (&src, dsts) in conn {
                for &dst in dsts {
                    driven_by.insert(dst, src);
                }
            }
        }
        // Output pin net -> the cell that produces it.
        let mut produced_by: HashMap<u32, usize> = HashMap::new();
        for (index, cell) in self.cells.iter().enumerate() {
            for &(net, _) in cell.outputs() {
                produced_by.insert(net, index);
            }
        }

        let mut keep: HashSet<usize> = HashSet::new();
        let mut queued: HashSet<u32> = HashSet::new();
        let mut pending: VecDeque<u32> = VecDeque::new();

        for (index, cell) in self.cells.iter().enumerate() {
            let Cell::Output { inputs } = cell else {
                continue;
            };
            for (net, name) in inputs {
                if output_pins.contains(&pin_base_name(name)) {
                    // The Output cell survives for its named pins only;
                    // its other pins' cones are not seeded here.
                    keep.insert(index);
                    if queued.insert(*net) {
                        pending.push_back(*net);
                    }
                }
            }
        }

        while let Some(net) = pending.pop_front() {
            if let Some(&index) = produced_by.get(&net)
                && keep.insert(index)
            {
                for &(input_net, _) in self.cells[index].inputs() {
                    if queued.insert(input_net) {
                        pending.push_back(input_net);
                    }
                }
            }
            if let Some(&src) = driven_by.get(&net)
                && queued.insert(src)
            {
                pending.push_back(src);
            }
        }

        keep
    }

    /// A copy of this graph holding only the cells at the indices in
    /// `keep`, with `connections` restricted to the edges whose two ends
    /// both survive — so no connection is left pointing at a pin that no
    /// longer exists. Unlike [`Graph::without_cell_names`] nothing is
    /// spliced back together: a pruned cell's edges are dropped, not
    /// rewired, because the cells being dropped here are the ones outside
    /// the cone and are meant to disappear entirely.
    fn retaining_cells(&self, keep: &HashSet<usize>) -> Graph {
        let cells: Vec<Cell> = self
            .cells
            .iter()
            .enumerate()
            .filter(|(index, _)| keep.contains(index))
            .map(|(_, cell)| cell.clone())
            .collect();
        let live_nets: HashSet<u32> = cells
            .iter()
            .flat_map(|cell| {
                cell.inputs()
                    .iter()
                    .chain(cell.outputs())
                    .map(|&(net, _)| net)
            })
            .collect();
        let connections: Vec<Connection> = self
            .connections
            .iter()
            .filter_map(|conn| {
                let kept: Connection = conn
                    .iter()
                    .filter(|(src, _)| live_nets.contains(src))
                    .filter_map(|(&src, dsts)| {
                        let dsts: Vec<u32> = dsts
                            .iter()
                            .copied()
                            .filter(|dst| live_nets.contains(dst))
                            .collect();
                        (!dsts.is_empty()).then_some((src, dsts))
                    })
                    .collect();
                (!kept.is_empty()).then_some(kept)
            })
            .collect();
        Graph { cells, connections }
    }

    fn merge_boolean_functions_grouped(&self, grouping: GateGrouping) -> Graph {
        // Flattened net id -> destination net ids, as in `without_cell_names`.
        let mut fanout: HashMap<u32, Vec<u32>> = HashMap::new();
        for conn in &self.connections {
            for (&src, dsts) in conn {
                fanout.entry(src).or_default().extend(dsts.iter().copied());
            }
        }
        // The reverse: destination net id -> the net id (always some
        // cell's output) driving it. Used below to describe a
        // BooleanFunction's boundary *inputs* by whatever actually drives
        // them (e.g. a ShiftRegister's `Q3`), not by whichever internal
        // gate happens to consume them.
        let mut driven_by: HashMap<u32, u32> = HashMap::new();
        for (&src, dsts) in &fanout {
            for &dst in dsts {
                driven_by.insert(dst, src);
            }
        }

        // Every output pin in the *whole* graph, labeled by its owning
        // cell instance and pin name.
        let output_pin_label: HashMap<u32, String> = self
            .cells
            .iter()
            .flat_map(|cell| {
                let instance = cell_instance_label(cell);
                cell.outputs()
                    .iter()
                    .map(move |(net, name)| (*net, format!("{instance}.{name}")))
            })
            .collect();

        struct Gate {
            cell_index: usize,
            raw_outputs: Vec<(u32, BoolExpr)>,
        }

        let gates: Vec<Gate> = self
            .cells
            .iter()
            .enumerate()
            .filter_map(|(cell_index, cell)| {
                let Cell::Sky130Standard {
                    cell_name,
                    inputs,
                    outputs,
                    ..
                } = cell
                else {
                    return None;
                };
                let raw_outputs = gate_output_exprs(cell_name, inputs, outputs)?;
                Some(Gate {
                    cell_index,
                    raw_outputs,
                })
            })
            .collect();

        // net -> gate positions (indices into `gates`) that consume it as
        // an input. Only recognized gates are indexed here, so "some other
        // recognized gate consumes this net" is exactly membership here.
        let mut consumers: HashMap<u32, Vec<usize>> = HashMap::new();
        for (pos, gate) in gates.iter().enumerate() {
            let Cell::Sky130Standard { inputs, .. } = &self.cells[gate.cell_index] else {
                unreachable!("Gate::cell_index always names a Sky130Standard cell")
            };
            for &(net, _) in inputs {
                consumers.entry(net).or_default().push(pos);
            }
        }

        // Union two gates whenever one's output net feeds the other's
        // input net (which, by the definition of `consumers` above, can
        // only be another recognized gate). Also record that same edge
        // directionally, to re-cluster with below whenever a weakly
        // connected group turns out to need splitting.
        let mut dsu = DisjointSet::new(gates.len());
        let mut directed_adj: Vec<Vec<usize>> = vec![Vec::new(); gates.len()];
        for (pos, gate) in gates.iter().enumerate() {
            for &(net, _) in &gate.raw_outputs {
                for dst in fanout.get(&net).into_iter().flatten() {
                    for &consumer_pos in consumers.get(dst).into_iter().flatten() {
                        dsu.union(pos, consumer_pos);
                        directed_adj[pos].push(consumer_pos);
                    }
                }
            }
        }

        let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
        for pos in 0..gates.len() {
            components.entry(dsu.find(pos)).or_default().push(pos);
        }

        /// Re-clusters `positions` by weak connectivity, restricted to
        /// `directed_adj` edges whose far end is also in `positions` (an
        /// edge to a gate pulled out elsewhere no longer joins the two).
        fn recluster(positions: &[usize], directed_adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
            let members: HashSet<usize> = positions.iter().copied().collect();
            let index_of: HashMap<usize, usize> = positions
                .iter()
                .enumerate()
                .map(|(i, &pos)| (pos, i))
                .collect();
            let mut dsu = DisjointSet::new(positions.len());
            for &pos in positions {
                for &adj in &directed_adj[pos] {
                    if members.contains(&adj) {
                        dsu.union(index_of[&pos], index_of[&adj]);
                    }
                }
            }
            let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
            for (i, &pos) in positions.iter().enumerate() {
                groups.entry(dsu.find(i)).or_default().push(pos);
            }
            groups.into_values().collect()
        }

        /// Tarjan's algorithm: the strongly connected components of
        /// `nodes` under `adj` (edges to a node outside `nodes` are
        /// ignored), each as a `Vec` of member nodes. A node not on any
        /// cycle comes back as its own singleton SCC.
        fn strongly_connected_components(
            nodes: &[usize],
            adj: &HashMap<usize, Vec<usize>>,
        ) -> Vec<Vec<usize>> {
            struct State {
                index_counter: usize,
                stack: Vec<usize>,
                on_stack: HashSet<usize>,
                index: HashMap<usize, usize>,
                lowlink: HashMap<usize, usize>,
                result: Vec<Vec<usize>>,
            }

            fn strongconnect(v: usize, adj: &HashMap<usize, Vec<usize>>, state: &mut State) {
                state.index.insert(v, state.index_counter);
                state.lowlink.insert(v, state.index_counter);
                state.index_counter += 1;
                state.stack.push(v);
                state.on_stack.insert(v);

                for &w in adj.get(&v).into_iter().flatten() {
                    if !state.index.contains_key(&w) {
                        strongconnect(w, adj, state);
                        let lower = state.lowlink[&v].min(state.lowlink[&w]);
                        state.lowlink.insert(v, lower);
                    } else if state.on_stack.contains(&w) {
                        let lower = state.lowlink[&v].min(state.index[&w]);
                        state.lowlink.insert(v, lower);
                    }
                }

                if state.lowlink[&v] == state.index[&v] {
                    let mut scc = Vec::new();
                    loop {
                        let w = state.stack.pop().unwrap();
                        state.on_stack.remove(&w);
                        scc.push(w);
                        if w == v {
                            break;
                        }
                    }
                    state.result.push(scc);
                }
            }

            let mut state = State {
                index_counter: 0,
                stack: Vec::new(),
                on_stack: HashSet::new(),
                index: HashMap::new(),
                lowlink: HashMap::new(),
                result: Vec::new(),
            };
            for &node in nodes {
                if !state.index.contains_key(&node) {
                    strongconnect(node, adj, &mut state);
                }
            }
            state.result
        }

        // Nets produced by a gate whose destinations are *all* internal
        // (consumed by another recognized gate, and therefore, by
        // construction, always in the very same component) are dropped
        // entirely as connection sources below; any internal destination
        // net (regardless of whether its source also has external fanout)
        // is stripped from wherever it's listed as a destination, since
        // that pin no longer exists standalone.
        let mut fully_internal_src_nets: HashSet<u32> = HashSet::new();
        let mut internal_dst_nets: HashSet<u32> = HashSet::new();
        // A duplicate consumer net (two gates in the same group fed by the
        // same external driver) -> the representative consumer net that
        // stayed a real boundary input pin; applied to `connections` below
        // so the driver's edge still finds a pin to land on.
        let mut alias_input_nets: HashMap<u32, u32> = HashMap::new();

        let mut merge_cells: HashMap<usize, Cell> = HashMap::new();
        let mut absorbed: HashSet<usize> = HashSet::new();

        // For `ConnectedGates`: for each weakly connected component above,
        // check for a genuine combinational cycle *among its own
        // recognized gates' direct edges* — no other cell type is ever
        // involved in this check.
        //
        // This deliberately does *not* also chase a loop mediated by an
        // external register (e.g. an LFSR-style "next input bit is a
        // function of this cloud's own past output"): two different
        // attempts at that both made this hang on real, densely
        // interconnected data (a ~700-gate cloud from a real design) —
        // once from a shrinking-exclusion-set bug that never converged,
        // and once, after fixing that, from an apparently-genuine
        // non-terminating oscillation in a single ~84-gate cluster with
        // many overlapping feedback paths. Register-mediated feedback is
        // also completely normal in synchronous hardware (state machines,
        // LFSRs) rather than a defect to remove, and is a property of the
        // *whole circuit's* topology (which cell drives which), not of how
        // this merge groups gates — a `BooleanFunction` that both reads
        // and feeds the same `ShiftRegister`, say, isn't something this
        // function's grouping choice can change.
        //
        // A cycle purely among recognized gates, with no register
        // anywhere in it, is different: it's rare — normally impossible in
        // a valid combinational netlist — and merging its members
        // together *does* make it fully internal, with nothing external
        // involved to keep the loop open. Each such cycle's members become
        // one group; every other gate re-clusters by weak connectivity as
        // before.
        //
        // For `RegisterScc` none of this applies: its classes are built
        // from reachability, which already keeps any mutually reachable
        // gates (i.e. any combinational cycle) together — see the second
        // arm below.
        let mut final_groups: Vec<Vec<usize>> = Vec::new();

        if grouping == GateGrouping::ConnectedGates {
            for positions in components.values() {
                let adj: HashMap<usize, Vec<usize>> = positions
                    .iter()
                    .map(|&pos| (pos, directed_adj[pos].clone()))
                    .collect();
                let sccs = strongly_connected_components(positions, &adj);

                if sccs.iter().all(|scc| scc.len() == 1) {
                    final_groups.push(positions.clone());
                    continue;
                }

                let cyclic: HashSet<usize> = sccs
                    .iter()
                    .filter(|scc| scc.len() > 1)
                    .flatten()
                    .copied()
                    .collect();
                for scc in sccs.into_iter().filter(|scc| scc.len() > 1) {
                    final_groups.push(scc);
                }
                let safe_positions: Vec<usize> = positions
                    .iter()
                    .copied()
                    .filter(|p| !cyclic.contains(p))
                    .collect();
                final_groups.extend(recluster(&safe_positions, &directed_adj));
            }
        } else {
            // `RegisterScc` (the three steps its doc comment describes) and
            // `ConeOfInfluence`, which share step 1 — each gate's set of
            // reached state elements — and differ only in what they key
            // the gates by afterwards.

            // Which cells are recognized gates, so a net's owner can be
            // told apart from a state element; and which cell owns each
            // net at all (as either one of its input or output pins).
            let gate_cell_indices: HashSet<usize> =
                gates.iter().map(|gate| gate.cell_index).collect();
            let owning_cell: HashMap<u32, usize> = self
                .cells
                .iter()
                .enumerate()
                .flat_map(|(index, cell)| {
                    cell.inputs()
                        .iter()
                        .chain(cell.outputs())
                        .map(move |&(net, _)| (net, index))
                })
                .collect();

            // Step 1: each gate's set of reachable state elements. Seeded
            // at every gate whose own output net lands on a non-gate cell,
            // then unioned backwards along `directed_adj` (gate -> gate it
            // feeds) until nothing grows. Monotone — a set only ever gains
            // members, and there are finitely many cells — so this always
            // terminates, cycles among the gates included.
            let mut reverse_adj: Vec<Vec<usize>> = vec![Vec::new(); gates.len()];
            for (pos, dsts) in directed_adj.iter().enumerate() {
                for &dst in dsts {
                    reverse_adj[dst].push(pos);
                }
            }
            let mut reaches: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); gates.len()];
            let mut pending: VecDeque<usize> = VecDeque::new();
            for (pos, gate) in gates.iter().enumerate() {
                for &(net, _) in &gate.raw_outputs {
                    for dst in fanout.get(&net).into_iter().flatten() {
                        match owning_cell.get(dst) {
                            Some(cell) if !gate_cell_indices.contains(cell) => {
                                reaches[pos].insert(*cell);
                            }
                            _ => {}
                        }
                    }
                }
                if !reaches[pos].is_empty() {
                    pending.push_back(pos);
                }
            }
            while let Some(pos) = pending.pop_front() {
                let reached = reaches[pos].clone();
                for &pred in &reverse_adj[pos] {
                    let before = reaches[pred].len();
                    reaches[pred].extend(reached.iter().copied());
                    if reaches[pred].len() != before {
                        pending.push_back(pred);
                    }
                }
            }

            // Step 2: the state-element graph — an edge from the element
            // driving one of a gate's inputs to every element that gate
            // reaches — and its strongly connected components, the
            // design's feedback loops. Nodes are collected in cell-index
            // order so the components (and hence the grouping) don't
            // depend on hash iteration order.
            //
            // `ConeOfInfluence` skips all of this: it keys gates by the
            // reached elements themselves, so there's no loop-collapsing
            // step to do.
            let component_of_register: HashMap<usize, usize> = if grouping
                == GateGrouping::RegisterScc
            {
                let mut register_adj: HashMap<usize, Vec<usize>> = HashMap::new();
                let mut register_nodes: BTreeSet<usize> = BTreeSet::new();
                for (pos, gate) in gates.iter().enumerate() {
                    let Cell::Sky130Standard { inputs, .. } = &self.cells[gate.cell_index] else {
                        unreachable!("Gate::cell_index always names a Sky130Standard cell")
                    };
                    register_nodes.extend(reaches[pos].iter().copied());
                    for &(net, _) in inputs {
                        let driver_net = driven_by.get(&net).copied().unwrap_or(net);
                        let Some(&driver_cell) = owning_cell.get(&driver_net) else {
                            continue;
                        };
                        if gate_cell_indices.contains(&driver_cell) {
                            continue; // internal to the cloud, not a state element
                        }
                        register_nodes.insert(driver_cell);
                        register_adj
                            .entry(driver_cell)
                            .or_default()
                            .extend(reaches[pos].iter().copied());
                    }
                }
                let register_nodes: Vec<usize> = register_nodes.into_iter().collect();
                strongly_connected_components(&register_nodes, &register_adj)
                    .into_iter()
                    .enumerate()
                    .flat_map(|(component, members)| {
                        members.into_iter().map(move |node| (node, component))
                    })
                    .collect()
            } else {
                // `ConeOfInfluence` keys gates by the reached elements
                // themselves, so it has no loops to collapse and never
                // needs this map.
                HashMap::new()
            };

            // Step 3: partition by what each gate feeds — the set of
            // feedback-loop components for `RegisterScc`, the reached
            // elements themselves for `ConeOfInfluence`, which is what
            // makes its cells the intersections of the design's cones
            // rather than one cell per loop (`BTreeMap`/`BTreeSet` again
            // for an order that doesn't depend on hashing). Each class is
            // then split into its weakly connected pieces. A gate that
            // reaches no state element at all — its logic drives nothing —
            // gets the empty set either way, and so those gates form their
            // own class.
            let mut classes: BTreeMap<BTreeSet<usize>, Vec<usize>> = BTreeMap::new();
            for (pos, reached) in reaches.iter().enumerate() {
                let key: BTreeSet<usize> = match grouping {
                    GateGrouping::RegisterScc => reached
                        .iter()
                        .map(|node| component_of_register[node])
                        .collect(),
                    _ => reached.clone(),
                };
                classes.entry(key).or_default().push(pos);
            }
            for members in classes.values() {
                final_groups.extend(recluster(members, &directed_adj));
            }
        }

        for positions in &final_groups {
            let raw: HashMap<u32, BoolExpr> = positions
                .iter()
                .flat_map(|&pos| gates[pos].raw_outputs.iter().cloned())
                .collect();

            // Every input pin net belonging to one of *this* final group's
            // own gates. A destination merely being *some* recognized
            // gate's input isn't enough to call it internal here: after
            // splitting, a direct edge from this group's own gate can land
            // on a gate that ended up in a different final group.
            let component_input_nets: HashSet<u32> = positions
                .iter()
                .flat_map(|&pos| {
                    let Cell::Sky130Standard { inputs, .. } = &self.cells[gates[pos].cell_index]
                    else {
                        unreachable!("Gate::cell_index always names a Sky130Standard cell")
                    };
                    inputs.iter().map(|&(net, _)| net).collect::<Vec<_>>()
                })
                .collect();

            // Every input net fed by one of this component's own gates,
            // mapped to the output net driving it (an output net and the
            // input net(s) it drives are always distinct ids, linked only
            // through `connections` — never equal — so neither this nor
            // `inline_bool_expr` below can read the link off `raw`'s keys
            // directly).
            let driver_of: HashMap<u32, u32> = raw
                .keys()
                .flat_map(|&out_net| {
                    fanout
                        .get(&out_net)
                        .into_iter()
                        .flatten()
                        .filter(|dst| component_input_nets.contains(dst))
                        .map(move |&dst| (dst, out_net))
                })
                .collect();

            let mut sorted_positions = positions.clone();
            sorted_positions.sort_unstable_by_key(|&pos| gates[pos].cell_index);

            // Pass 1: for every non-internal input net in this component,
            // find its driving net and pick one consumer net per distinct
            // driver as that driver's *representative* — the first one
            // encountered, in the same deterministic (cell-index) order as
            // `sorted_positions` below. Two gates fed by the same external
            // net (e.g. both reading the same shift register bit) must
            // become *one* boundary input, not one each, and — critically —
            // that boundary input has to keep a net id `connections`
            // already routes the driver to, or the snarl view has no pin to
            // connect the driver's edge to.
            let mut representative_for_driver: HashMap<u32, u32> = HashMap::new();
            let mut canonical_input: HashMap<u32, u32> = HashMap::new();
            for &pos in &sorted_positions {
                let Cell::Sky130Standard { inputs, .. } = &self.cells[gates[pos].cell_index] else {
                    unreachable!("Gate::cell_index always names a Sky130Standard cell")
                };
                for &(net, _) in inputs {
                    if driver_of.contains_key(&net) {
                        continue; // internal; handled in pass 2 below
                    }
                    let driver_net = driven_by.get(&net).copied().unwrap_or(net);
                    let representative =
                        *representative_for_driver.entry(driver_net).or_insert(net);
                    canonical_input.insert(net, representative);
                    if representative != net {
                        // This duplicate consumer net no longer exists as
                        // its own pin; `connections` needs to route to the
                        // representative instead.
                        alias_input_nets.insert(net, representative);
                    }
                }
            }
            let mut boundary_inputs: Vec<Pin> = representative_for_driver
                .iter()
                .map(|(&driver_net, &representative)| {
                    let label = output_pin_label
                        .get(&driver_net)
                        .cloned()
                        .unwrap_or_else(|| format!("net_{driver_net}"));
                    (representative, label)
                })
                .collect();
            // Ordered by origin: grouped by driving cell instance (e.g.
            // each shift register together), then by pin index within it
            // (`Q0`..`Q7`, compared numerically so `Q10` doesn't sort
            // before `Q2`) rather than left in arbitrary discovery order.
            boundary_inputs.sort_by(|a, b| pin_label_sort_key(&a.1).cmp(&pin_label_sort_key(&b.1)));

            let mut memo: HashMap<u32, BoolExpr> = HashMap::new();
            let mut in_progress: HashSet<u32> = HashSet::new();

            // Pass 2: compose each boundary output (numbered `X0`..`Xn-1`
            // rather than named after whichever internal gate happens to
            // produce it) and collect ancestry/centroid.
            let mut boundary_outputs: Vec<Pin> = Vec::new();
            let mut boolean_outputs: Vec<(u32, BoolExpr)> = Vec::new();
            let mut ancestor_cells: Vec<Cell> = Vec::new();
            let mut centroid_sum = (0.0, 0.0);
            let mut cell_count = 0.0;
            let mut output_index = 0usize;

            for &pos in &sorted_positions {
                let cell_index = gates[pos].cell_index;
                let Cell::Sky130Standard {
                    inputs, centroid, ..
                } = &self.cells[cell_index]
                else {
                    unreachable!("Gate::cell_index always names a Sky130Standard cell")
                };

                centroid_sum.0 += centroid.0;
                centroid_sum.1 += centroid.1;
                cell_count += 1.0;

                for &(net, _) in inputs {
                    if driver_of.contains_key(&net) {
                        internal_dst_nets.insert(net);
                    }
                }

                for &(net, _) in &gates[pos].raw_outputs {
                    let dsts = fanout.get(&net).cloned().unwrap_or_default();
                    let has_external = dsts.iter().any(|d| !component_input_nets.contains(d));
                    if !dsts.is_empty() && !has_external {
                        fully_internal_src_nets.insert(net);
                    } else {
                        let expr = inline_bool_expr(
                            net,
                            &raw,
                            &driver_of,
                            &canonical_input,
                            &mut memo,
                            &mut in_progress,
                        );
                        boundary_outputs.push((net, format!("X{output_index}")));
                        boolean_outputs.push((net, expr));
                        output_index += 1;
                    }
                }

                ancestor_cells.push(self.cells[cell_index].clone());
            }

            merge_cells.insert(
                gates[sorted_positions[0]].cell_index,
                Cell::MergeCell {
                    cell_name: "BooleanFunction".to_string(),
                    centroid: (centroid_sum.0 / cell_count, centroid_sum.1 / cell_count),
                    inputs: boundary_inputs,
                    outputs: boundary_outputs,
                    ancestor_cells,
                    boolean_outputs,
                },
            );
            for &pos in &sorted_positions[1..] {
                absorbed.insert(gates[pos].cell_index);
            }
        }

        let cells: Vec<Cell> = self
            .cells
            .iter()
            .enumerate()
            .filter_map(|(i, cell)| {
                if let Some(merge_cell) = merge_cells.remove(&i) {
                    Some(merge_cell)
                } else if absorbed.contains(&i) {
                    // A non-head gate; already folded into the merge cell
                    // replacing its component's head above.
                    None
                } else {
                    Some(cell.clone())
                }
            })
            .collect();

        let connections: Vec<Connection> = self
            .connections
            .iter()
            .filter_map(|conn| {
                let new_conn: Connection = conn
                    .iter()
                    .filter(|(src, _)| !fully_internal_src_nets.contains(src))
                    .map(|(&src, dsts)| {
                        let mut seen = HashSet::new();
                        let dsts = dsts
                            .iter()
                            .copied()
                            .filter(|dst| !internal_dst_nets.contains(dst))
                            .map(|dst| alias_input_nets.get(&dst).copied().unwrap_or(dst))
                            .filter(|dst| seen.insert(*dst))
                            .collect();
                        (src, dsts)
                    })
                    .collect();
                (!new_conn.is_empty()).then_some(new_conn)
            })
            .collect();

        Graph { cells, connections }
    }

    /// How many `connections` edges the graph holds: every `src -> dst`
    /// net pair, over every connection layer, counted separately — one net
    /// fanning out to three destination pins is three edges, the three the
    /// graph view draws.
    pub fn edge_count(&self) -> usize {
        self.connections
            .iter()
            .flat_map(|conn| conn.values())
            .map(Vec::len)
            .sum()
    }

    /// How many cells of each type the graph holds, ordered by count
    /// descending, ties broken by type name. A standard or merged cell
    /// counts under its own `cell_name` (`"sky130_fd_sc_hd__buf_2"`,
    /// `"ShiftRegister"`, ...), which is what distinguishes cells here;
    /// `Cell::Input`/`Cell::Output` carry no name of their own, so they
    /// count under `"Input"`/`"Output"`. Each type's [`CellCategory`]
    /// follows from that same name, so every cell counted in one row
    /// shares the row's category.
    pub fn cell_type_counts(&self) -> Vec<CellTypeCount> {
        let mut counts: BTreeMap<&str, (CellCategory, usize)> = BTreeMap::new();
        for cell in &self.cells {
            let name = match cell {
                Cell::Sky130Standard { cell_name, .. } | Cell::MergeCell { cell_name, .. } => {
                    cell_name.as_str()
                }
                Cell::Input { .. } => "Input",
                Cell::Output { .. } => "Output",
            };
            let entry = counts.entry(name).or_insert((cell.category(), 0));
            entry.1 += 1;
        }

        // From the `BTreeMap`, so the sort's ties are already name-ordered.
        let mut counts: Vec<CellTypeCount> = counts
            .into_iter()
            .map(|(name, (category, count))| CellTypeCount {
                name: name.to_string(),
                category,
                count,
            })
            .collect();
        counts.sort_by_key(|entry| std::cmp::Reverse(entry.count));
        counts
    }

    /// How many cells fall into each [`CellCategory`], ordered by count
    /// descending, ties broken by the category's declaration order. The
    /// coarser view of [`Graph::cell_type_counts`]: each row here is the
    /// sum of that table's rows sharing the category. A category no cell
    /// falls into is left out rather than shown as zero.
    pub fn cell_category_counts(&self) -> Vec<CellCategoryCount> {
        let mut counts: Vec<CellCategoryCount> = CellCategory::ALL
            .into_iter()
            .map(|category| CellCategoryCount { category, count: 0 })
            .collect();
        for cell in &self.cells {
            let category = cell.category();
            counts
                .iter_mut()
                .find(|entry| entry.category == category)
                .expect("every category is listed in CellCategory::ALL")
                .count += 1;
        }
        counts.retain(|entry| entry.count > 0);
        // Stable, and seeded in declaration order, so ties keep it.
        counts.sort_by_key(|entry| std::cmp::Reverse(entry.count));
        counts
    }

    /// For each `(input_pin_label, target_pin_label)` entry: finds the
    /// `Cell::Input` output pin named `input_pin_label` (its own label is
    /// the symbolic value — e.g. an Input pin named `"A"` seeds the value
    /// `"A"`) and seeds that value at every pin named `target_pin_label`
    /// that Input pin is *directly connected to by a `connections` edge*.
    /// `target_pin_label` is therefore a filter over that input's own
    /// destinations, not a free-floating injection point: a pin named
    /// `target_pin_label` somewhere else in the graph, wired to something
    /// else entirely, is never seeded, and neither is a pin this input
    /// does drive but under a different name (an input wired to a cell's
    /// `A` contributes nothing to an entry targeting `B`, not even on
    /// that same cell). The Input cell's own pin is still labeled with its
    /// value — it does carry it — but the value only *enters* the graph at
    /// the entry's matched target pins, so the input's other, unmatched
    /// destinations stay unlabeled.
    ///
    /// From every seed, the value then floods forward along the netlist's
    /// real edges — but only *through* a cell (from one of its input pins
    /// to its own output pins) when that cell has exactly one input pin
    /// total, so the value crossing it is unambiguous (a `buf`, `inv`,
    /// `clkbuf`, ...). A value landing on one input of a multi-input cell
    /// (a gate's `A`/`B`, a flip-flop's `D`/`CLK`/`RESET_B`, a merged
    /// cell's many boundary inputs, ...) still labels that one pin, but
    /// doesn't get copied onto outputs that just as much depend on the
    /// cell's *other*, unrelated inputs — otherwise a signal shared by
    /// many multi-input cells (a clock, say, reaching every flip-flop)
    /// would flood essentially the entire downstream graph, well past
    /// anything that value actually determines. Fan-out along
    /// `connections` (the same net reaching several destination pins) is
    /// always followed, regardless of pin count, since that's just the
    /// same wire, not a computation. If two different values ever reach
    /// the same net, that net is left unlabeled and propagation stops
    /// there — which value should "win" is ambiguous — though any net
    /// already labeled *before* that conflict was found downstream keeps
    /// its label (this is a simple flood fill, not a full simultaneous
    /// evaluation, so which of several colliding sources reaches a net
    /// first can depend on entry order).
    ///
    /// Every pin a value reaches has it appended to its label as
    /// `" = <value>"`. That relabeling is the *whole* change made here:
    /// `connections` comes through untouched, so the graph this returns is
    /// still the same circuit and stays usable by everything downstream
    /// (the merges, the SAT encoding, the simulator). Decluttering the
    /// view by hiding the edges the labels have made redundant is
    /// [`Graph::without_propagated_edges`]'s job, and its result is
    /// only ever laid out and drawn, never solved over. Unlike the other
    /// actions, this doesn't remove or restructure any *cell* — an entry
    /// naming a pin that doesn't exist is simply ignored rather than
    /// treated as an error.
    pub fn propagate_pin_values(&self, entries: &[(String, String)]) -> Graph {
        // Flattened net id -> destination net ids, as in `without_cell_names`.
        let mut fanout: HashMap<u32, Vec<u32>> = HashMap::new();
        for conn in &self.connections {
            for (&src, dsts) in conn {
                fanout.entry(src).or_default().extend(dsts.iter().copied());
            }
        }

        // Which cell (by index) owns each input net, so a value reaching
        // one can carry on to that same cell's own output nets.
        let mut owner_of_input: HashMap<u32, usize> = HashMap::new();
        for (i, cell) in self.cells.iter().enumerate() {
            for &(net, _) in cell.inputs() {
                owner_of_input.insert(net, i);
            }
        }

        /// Tries to label `net` with `value`, reporting whether that's a
        /// brand new assignment (the caller's cue to flood onward from
        /// it). A repeat of the exact same value is left alone (already
        /// handled, and this also guards against looping forever around a
        /// fanout cycle); two different values meeting here is a conflict
        /// — the net is left unlabeled for good.
        fn assign(
            net: u32,
            value: &str,
            value_of: &mut HashMap<u32, String>,
            conflicted: &mut HashSet<u32>,
        ) -> bool {
            if conflicted.contains(&net) {
                return false;
            }
            match value_of.get(&net) {
                Some(existing) if existing == value => false,
                Some(_) => {
                    conflicted.insert(net);
                    value_of.remove(&net);
                    false
                }
                None => {
                    value_of.insert(net, value.to_string());
                    true
                }
            }
        }

        /// [`assign`], queueing a newly labeled net to flood onward.
        fn seed(
            net: u32,
            value: &str,
            value_of: &mut HashMap<u32, String>,
            conflicted: &mut HashSet<u32>,
            queue: &mut VecDeque<u32>,
        ) {
            if assign(net, value, value_of, conflicted) {
                queue.push_back(net);
            }
        }

        let mut value_of: HashMap<u32, String> = HashMap::new();
        let mut conflicted: HashSet<u32> = HashSet::new();
        let mut queue: VecDeque<u32> = VecDeque::new();

        for (input_pin_label, target_pin_label) in entries {
            let Some(&(input_net, _)) = self.cells.iter().find_map(|cell| match cell {
                Cell::Input { outputs } => outputs
                    .iter()
                    .find(|(_, name)| pin_base_name(name) == input_pin_label),
                _ => None,
            }) else {
                continue;
            };

            // The Input cell's own output pin does carry its own value, so
            // it gets the label — but it's deliberately not queued: the
            // value enters the graph only at the entry's target pins
            // below, so flooding from here would label every other
            // destination this input drives too, whatever its pin name.
            assign(input_net, input_pin_label, &mut value_of, &mut conflicted);
            // Only the pins named by the entry that this input actually
            // reaches over an edge — a same-named pin wired to something
            // else is not this input's value, and never seeded.
            for &net in fanout.get(&input_net).into_iter().flatten() {
                let named_target = self.cells.iter().any(|cell| {
                    cell.inputs()
                        .iter()
                        .any(|(n, name)| *n == net && pin_base_name(name) == target_pin_label)
                });
                if named_target {
                    seed(
                        net,
                        input_pin_label,
                        &mut value_of,
                        &mut conflicted,
                        &mut queue,
                    );
                }
            }
        }

        while let Some(net) = queue.pop_front() {
            let Some(value) = value_of.get(&net).cloned() else {
                continue; // conflicted after being queued; stop here
            };
            // `net` is either an input pin's net (continue through the
            // cell that owns it) or an Input cell's own output net
            // (nothing owns it as an input, so only the fanout below
            // applies) — never both, so both lookups are always safe. Only
            // cross into a cell's outputs when it has exactly one input —
            // otherwise this value would get copied onto outputs that just
            // as much depend on the cell's other, unrelated inputs.
            if let Some(&cell_index) = owner_of_input.get(&net) {
                let cell = &self.cells[cell_index];
                if cell.inputs().len() == 1 {
                    for &(out_net, _) in cell.outputs() {
                        seed(out_net, &value, &mut value_of, &mut conflicted, &mut queue);
                    }
                }
            }
            for &dst in fanout.get(&net).into_iter().flatten() {
                seed(dst, &value, &mut value_of, &mut conflicted, &mut queue);
            }
        }

        fn relabel(pin: &Pin, value_of: &HashMap<u32, String>) -> Pin {
            match value_of.get(&pin.0) {
                Some(value) => (pin.0, format!("{} = {value}", pin.1)),
                None => pin.clone(),
            }
        }

        fn relabel_cell(cell: &Cell, value_of: &HashMap<u32, String>) -> Cell {
            match cell {
                Cell::Sky130Standard {
                    cell_id,
                    cell_name,
                    centroid,
                    inputs,
                    outputs,
                } => Cell::Sky130Standard {
                    cell_id: *cell_id,
                    cell_name: cell_name.clone(),
                    centroid: *centroid,
                    inputs: inputs.iter().map(|p| relabel(p, value_of)).collect(),
                    outputs: outputs.iter().map(|p| relabel(p, value_of)).collect(),
                },
                Cell::Input { outputs } => Cell::Input {
                    outputs: outputs.iter().map(|p| relabel(p, value_of)).collect(),
                },
                Cell::Output { inputs } => Cell::Output {
                    inputs: inputs.iter().map(|p| relabel(p, value_of)).collect(),
                },
                Cell::MergeCell {
                    cell_name,
                    centroid,
                    inputs,
                    outputs,
                    ancestor_cells,
                    boolean_outputs,
                } => Cell::MergeCell {
                    cell_name: cell_name.clone(),
                    centroid: *centroid,
                    inputs: inputs.iter().map(|p| relabel(p, value_of)).collect(),
                    outputs: outputs.iter().map(|p| relabel(p, value_of)).collect(),
                    // Provenance, left as originally recorded.
                    ancestor_cells: ancestor_cells.clone(),
                    boolean_outputs: boolean_outputs.clone(),
                },
            }
        }

        let cells = self
            .cells
            .iter()
            .map(|cell| relabel_cell(cell, &value_of))
            .collect();

        Graph {
            cells,
            connections: self.connections.clone(),
        }
    }

    /// [`Graph::propagate_pin_values`] for the graph *view* only: the same
    /// relabeled cells, minus the edges [`Graph::without_propagated_edges`]
    /// hides. Purely a convenience for the tests; the app lays out every
    /// graph through that method instead, so the hiding carries on to the
    /// actions after this one.
    #[cfg(test)]
    pub fn propagate_pin_values_for_view(&self, entries: &[(String, String)]) -> Graph {
        self.propagate_pin_values(entries)
            .without_propagated_edges()
    }

    /// This graph with a `connections` edge dropped wherever *both* its
    /// ends carry the exact same propagated value (see
    /// [`propagated_value`]) — the wire that carried it, now already shown
    /// by the matching labels at each end. An edge on just one labeled
    /// side (the value didn't cross it — e.g. it stopped at a multi-input
    /// cell, or the two sides disagree after a conflict) is left alone.
    ///
    /// Hiding those edges is purely cosmetic, so the result is for laying
    /// out and drawing only — a graph missing the wires it is actually
    /// built from is not the circuit, and solving or simulating over it
    /// would answer about a machine that isn't there. It is decided from
    /// the pin labels alone rather than from a propagation's own state, so
    /// it applies just as well to the graph of any action *after* a
    /// `PropagatePinValues` — a merge or a removal keeps the labels (and,
    /// for the nets it leaves in place, the edges between them), so the
    /// same edges stay hidden all the way down the chain, while every
    /// action's own `Graph` stays fully wired.
    pub fn without_propagated_edges(&self) -> Graph {
        let value_of: HashMap<u32, &str> = self
            .cells
            .iter()
            .flat_map(|cell| cell.inputs().iter().chain(cell.outputs()))
            .filter_map(|(net, name)| propagated_value(name).map(|value| (*net, value)))
            .collect();
        let connections: Vec<Connection> = self
            .connections
            .iter()
            .map(|conn| {
                conn.iter()
                    .map(|(&src, dsts)| {
                        let dsts = dsts
                            .iter()
                            .copied()
                            .filter(|dst| {
                                let src_value = value_of.get(&src);
                                src_value.is_none() || src_value != value_of.get(dst)
                            })
                            .collect();
                        (src, dsts)
                    })
                    .collect()
            })
            .collect();
        Graph {
            cells: self.cells.clone(),
            connections,
        }
    }
}

/// Recursively substitutes every net produced within this group into
/// `net`'s expression, so the result only references true boundary input
/// nets. `raw` holds each of the group's gates' own output expressions
/// (each written in terms of that gate's *input* net ids); `driver_of` maps
/// each such input net id to the output net id (a `raw` key) that drives
/// it — an output net and the input net(s) it feeds are always distinct
/// ids, linked only through `connections`, never equal. `canonical_input`
/// maps every non-internal input net to whichever consumer net was chosen
/// to represent its driver (see the pass 1 loop above), so that two gates
/// fed by the same external net resolve to one shared free variable.
/// Memoized in `memo`; `in_progress` guards against a cycle in malformed
/// data (falls back to leaving the net as a free variable there).
fn inline_bool_expr(
    net: u32,
    raw: &HashMap<u32, BoolExpr>,
    driver_of: &HashMap<u32, u32>,
    canonical_input: &HashMap<u32, u32>,
    memo: &mut HashMap<u32, BoolExpr>,
    in_progress: &mut HashSet<u32>,
) -> BoolExpr {
    fn inline(
        expr: &BoolExpr,
        raw: &HashMap<u32, BoolExpr>,
        driver_of: &HashMap<u32, u32>,
        canonical_input: &HashMap<u32, u32>,
        memo: &mut HashMap<u32, BoolExpr>,
        in_progress: &mut HashSet<u32>,
    ) -> BoolExpr {
        match expr {
            BoolExpr::Var(net) => resolve(*net, raw, driver_of, canonical_input, memo, in_progress),
            BoolExpr::Const(_) => expr.clone(),
            BoolExpr::Not(e) => BoolExpr::not(inline(
                e,
                raw,
                driver_of,
                canonical_input,
                memo,
                in_progress,
            )),
            BoolExpr::And(es) => BoolExpr::and(
                es.iter()
                    .map(|e| inline(e, raw, driver_of, canonical_input, memo, in_progress))
                    .collect(),
            ),
            BoolExpr::Or(es) => BoolExpr::or(
                es.iter()
                    .map(|e| inline(e, raw, driver_of, canonical_input, memo, in_progress))
                    .collect(),
            ),
            BoolExpr::Xor(a, b) => BoolExpr::xor(
                inline(a, raw, driver_of, canonical_input, memo, in_progress),
                inline(b, raw, driver_of, canonical_input, memo, in_progress),
            ),
        }
    }

    fn resolve(
        net: u32,
        raw: &HashMap<u32, BoolExpr>,
        driver_of: &HashMap<u32, u32>,
        canonical_input: &HashMap<u32, u32>,
        memo: &mut HashMap<u32, BoolExpr>,
        in_progress: &mut HashSet<u32>,
    ) -> BoolExpr {
        if let Some(cached) = memo.get(&net) {
            return cached.clone();
        }
        // `net` is either itself an output net (the top-level call site
        // always passes one) or an input net driven by one — either way,
        // find the output net whose `raw` expression it should expand to.
        let producer_net = if raw.contains_key(&net) {
            net
        } else if let Some(&producer_net) = driver_of.get(&net) {
            producer_net
        } else {
            // A true boundary input: rewrite to its canonical
            // (dedup-representative) net id, so that every gate in the
            // group fed by the same external driver collapses to one
            // shared free variable instead of each keeping its own
            // distinct input-pin net id.
            let canonical = canonical_input.get(&net).copied().unwrap_or(net);
            return BoolExpr::Var(canonical);
        };
        if !in_progress.insert(net) {
            return BoolExpr::Var(net); // cycle guard; can't happen in real combinational logic
        }
        let inlined = inline(
            &raw[&producer_net],
            raw,
            driver_of,
            canonical_input,
            memo,
            in_progress,
        );
        in_progress.remove(&net);
        memo.insert(net, inlined.clone());
        inlined
    }

    resolve(net, raw, driver_of, canonical_input, memo, in_progress)
}

/// Returns `cell_name`'s boolean semantics as `(output_net_id, expr)` pairs
/// — `expr` written purely in terms of `Var(net_id)` for each of its own
/// `inputs` (not yet inlined against any other gate) — or `None` if
/// `cell_name` isn't a recognized stateless boolean gate (a sequential
/// cell, or an unrecognized cell type).
///
/// Single-group cells (`and2`, `nor3b`, `and4bb`, ...) just AND/OR every
/// input pin, inverting a pin wherever its name ends `_N` — which is
/// exactly how SkyWater's PDK marks an active-low pin, so this needs no
/// per-variant (`b`/`bb`) special-casing. Compound AOI/OAI cells (`a221o`,
/// `o21bai`, ...) group input pins by their leading letter (`A1`,`A2` ->
/// group `A`; `B1` -> group `B`; ...) — again exactly how the PDK names
/// grouped pins — AND each group (`a`-prefixed cells) or OR each group
/// (`o`-prefixed), then OR (`a`) or AND (`o`) the groups together in
/// letter order, and invert the whole thing if the (drive-strength
/// stripped) cell name ends in `i`.
fn gate_output_exprs(
    cell_name: &str,
    inputs: &[Pin],
    outputs: &[Pin],
) -> Option<Vec<(u32, BoolExpr)>> {
    let base = strip_sky130_prefix_and_drive(cell_name)?;

    fn var(pin: &Pin) -> BoolExpr {
        let (net, name) = pin;
        let v = BoolExpr::Var(*net);
        if pin_base_name(name).ends_with("_N") {
            BoolExpr::not(v)
        } else {
            v
        }
    }

    let single_output = || outputs.first().map(|&(net, _)| net);

    if base == "buf" || base.starts_with("clkbuf") {
        return Some(vec![(single_output()?, var(inputs.first()?))]);
    }
    if base == "inv" {
        return Some(vec![(
            single_output()?,
            BoolExpr::not(var(inputs.first()?)),
        )]);
    }
    if base.starts_with("and") {
        return Some(vec![(
            single_output()?,
            BoolExpr::and(inputs.iter().map(var).collect()),
        )]);
    }
    if base.starts_with("or") {
        return Some(vec![(
            single_output()?,
            BoolExpr::or(inputs.iter().map(var).collect()),
        )]);
    }
    if base.starts_with("nand") {
        return Some(vec![(
            single_output()?,
            BoolExpr::not(BoolExpr::and(inputs.iter().map(var).collect())),
        )]);
    }
    if base.starts_with("nor") {
        return Some(vec![(
            single_output()?,
            BoolExpr::not(BoolExpr::or(inputs.iter().map(var).collect())),
        )]);
    }
    if base == "xor2" {
        return Some(vec![(
            single_output()?,
            BoolExpr::xor(var(inputs.first()?), var(inputs.get(1)?)),
        )]);
    }
    if base == "xnor2" {
        return Some(vec![(
            single_output()?,
            BoolExpr::not(BoolExpr::xor(var(inputs.first()?), var(inputs.get(1)?))),
        )]);
    }
    if base == "conb" {
        return Some(
            outputs
                .iter()
                .map(|&(net, ref name)| (net, BoolExpr::Const(pin_base_name(name) == "HI")))
                .collect(),
        );
    }
    if base.starts_with("mux2") {
        let s = inputs.iter().find(|(_, name)| pin_base_name(name) == "S")?;
        let a1 = inputs
            .iter()
            .find(|(_, name)| pin_base_name(name) == "A1")?;
        let a0 = inputs
            .iter()
            .find(|(_, name)| pin_base_name(name) == "A0")?;
        return Some(vec![(
            single_output()?,
            BoolExpr::or(vec![
                BoolExpr::and(vec![var(s), var(a1)]),
                BoolExpr::and(vec![BoolExpr::not(var(s)), var(a0)]),
            ]),
        )]);
    }

    // Compound AOI/OAI family: leading 'a' (AND-groups OR'd together) or
    // 'o' (OR-groups AND'd together), grouped by each pin's leading letter.
    let mut chars = base.chars();
    let leading = chars.next()?;
    if (leading == 'a' || leading == 'o') && chars.next().is_some_and(|c| c.is_ascii_digit()) {
        let mut groups: std::collections::BTreeMap<char, Vec<BoolExpr>> =
            std::collections::BTreeMap::new();
        for pin in inputs {
            groups
                .entry(pin.1.chars().next()?)
                .or_default()
                .push(var(pin));
        }
        let combine_group = |group: Vec<BoolExpr>| {
            if leading == 'a' {
                BoolExpr::and(group)
            } else {
                BoolExpr::or(group)
            }
        };
        let combined: Vec<BoolExpr> = groups.into_values().map(combine_group).collect();
        let mut result = if leading == 'a' {
            BoolExpr::or(combined)
        } else {
            BoolExpr::and(combined)
        };
        if base.ends_with('i') {
            result = BoolExpr::not(result);
        }
        return Some(vec![(single_output()?, result)]);
    }

    None
}

/// A stable, human-readable identifier for one cell *instance* (as opposed
/// to its type/name alone), used to label a [`Graph::merge_boolean_functions`]
/// boundary input by whatever actually drives it. `Sky130Standard` cells
/// already carry a `cell_id` for this; the other variants don't, so one of
/// their own pin's net ids stands in instead — still unique and stable per
/// instance, just not as tidy a number.
fn cell_instance_label(cell: &Cell) -> String {
    match cell {
        Cell::Sky130Standard {
            cell_name, cell_id, ..
        } => format!("{cell_name}#{cell_id}"),
        Cell::Input { outputs } => format!("Input#{}", outputs.first().map_or(0, |p| p.0)),
        Cell::Output { inputs } => format!("Output#{}", inputs.first().map_or(0, |p| p.0)),
        Cell::MergeCell {
            cell_name,
            inputs,
            outputs,
            ..
        } => {
            let id = outputs.first().or(inputs.first()).map_or(0, |p| p.0);
            format!("{cell_name}#{id}")
        }
    }
}

/// Sort key for a `"<cell instance>.<pin name>"` label (as built by
/// [`cell_instance_label`]): groups by the driving cell instance first
/// (e.g. each shift register's bits stay together), then by the pin
/// name's trailing digits compared numerically — `Q0`..`Q7`, not the
/// lexicographic order that would put `Q10` before `Q2` — falling back to
/// plain string order for a pin name with no trailing digits.
fn pin_label_sort_key(label: &str) -> (&str, Option<u64>, &str) {
    let (cell, pin) = label.rsplit_once('.').unwrap_or(("", label));
    let digits_at = pin
        .rfind(|c: char| !c.is_ascii_digit())
        .map_or(0, |i| i + 1);
    let pin_number = pin[digits_at..].parse().ok();
    (cell, pin_number, pin)
}

/// Strips the `"sky130_fd_sc_hd__"` prefix and trailing `"_<drive
/// strength>"` suffix off a standard-cell name, e.g.
/// `"sky130_fd_sc_hd__and2_2"` -> `"and2"`. `None` if `cell_name` doesn't
/// carry that prefix (not a `sky130_fd_sc_hd` cell).
fn strip_sky130_prefix_and_drive(cell_name: &str) -> Option<&str> {
    let base = cell_name.strip_prefix("sky130_fd_sc_hd__")?;
    Some(match base.rsplit_once('_') {
        Some((stem, suffix))
            if !suffix.is_empty() && suffix.chars().all(|c| c.is_ascii_digit()) =>
        {
            stem
        }
        _ => base,
    })
}

/// A minimal union-find over `0..n`, used to group gates into connected
/// components in [`Graph::merge_boolean_functions`].
struct DisjointSet {
    parent: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        DisjointSet {
            parent: (0..n).collect(),
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            self.parent[ra] = rb;
        }
    }
}

// --- Whole-system sequential solving ---------------------------------------

/// One value in a flattened definition: a net, possibly negated, or a
/// constant. Every operand of a [`SystemModel::defs`] entry is one of
/// these, which is what keeps those definitions small enough to hash — and
/// therefore to share — in [`Flattener`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Ref {
    Net(u32, bool),
    Const(bool),
}

impl Ref {
    fn negated(self) -> Ref {
        match self {
            Ref::Net(net, positive) => Ref::Net(net, !positive),
            Ref::Const(value) => Ref::Const(!value),
        }
    }

    fn expr(self) -> BoolExpr {
        match self {
            Ref::Net(net, true) => BoolExpr::Var(net),
            Ref::Net(net, false) => BoolExpr::not(BoolExpr::Var(net)),
            Ref::Const(value) => BoolExpr::Const(value),
        }
    }
}

/// One flattened definition's shape, and the key two structurally
/// identical sub-expressions share a net by. Operand order is normalized
/// (sorted, duplicates dropped) for the commutative operators, so `A & B`
/// and `B & A` are the same key.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Shape {
    And(Vec<Ref>),
    Or(Vec<Ref>),
    Xor(Ref, Ref),
}

/// Rewrites [`BoolExpr`] trees into a set of *flat* definitions — one
/// operator each, over [`Ref`] operands — introducing a fresh net per
/// distinct sub-expression shape.
///
/// The point is sharing. [`Graph::merge_boolean_functions`] inlines every
/// internal net away, which turns a DAG of gates back into a tree and
/// duplicates whatever the DAG shared: on a real design a cone-of-influence
/// merge's expressions can come to tens of thousands of nodes for a few
/// hundred gates. A combinational solve encodes that once and doesn't care,
/// but an unrolling encodes it *per cycle*, where the duplication multiplies.
/// Re-sharing here brings each cycle back down to roughly one definition
/// per original gate.
struct Flattener {
    /// The next unused net id. Fresh nets count up from above every net id
    /// the graph itself uses, so they can't collide with a real one.
    next_net: u32,
    /// Shape -> the net already standing for it.
    shared: HashMap<Shape, u32>,
    defs: BTreeMap<u32, BoolExpr>,
    /// Set if `next_net` ran off the end of `u32` — see
    /// [`Graph::system_model`], which refuses to solve rather than let
    /// fresh nets wrap around onto real ones.
    overflowed: bool,
}

impl Flattener {
    fn new(next_net: u32) -> Self {
        Flattener {
            next_net,
            shared: HashMap::new(),
            defs: BTreeMap::new(),
            overflowed: false,
        }
    }

    /// The net standing for `shape`, defining a fresh one the first time
    /// that shape is seen.
    fn net_for(&mut self, shape: Shape) -> Ref {
        if let Some(&net) = self.shared.get(&shape) {
            return Ref::Net(net, true);
        }
        let net = self.next_net;
        match self.next_net.checked_add(1) {
            Some(next) => self.next_net = next,
            None => self.overflowed = true,
        }
        let expr = match &shape {
            Shape::And(operands) => BoolExpr::and(operands.iter().map(|r| r.expr()).collect()),
            Shape::Or(operands) => BoolExpr::or(operands.iter().map(|r| r.expr()).collect()),
            Shape::Xor(a, b) => BoolExpr::xor(a.expr(), b.expr()),
        };
        self.defs.insert(net, expr);
        self.shared.insert(shape, net);
        Ref::Net(net, true)
    }

    /// Flattens `expr`, emitting a definition per distinct sub-expression
    /// shape, and returns the operand its value is reachable through.
    fn flatten(&mut self, expr: &BoolExpr) -> Ref {
        match expr {
            BoolExpr::Var(net) => Ref::Net(*net, true),
            BoolExpr::Const(value) => Ref::Const(*value),
            BoolExpr::Not(inner) => self.flatten(inner).negated(),
            BoolExpr::And(operands) => {
                let operands: Vec<Ref> = operands.iter().map(|e| self.flatten(e)).collect();
                if operands.contains(&Ref::Const(false)) {
                    return Ref::Const(false);
                }
                self.commutative(operands, Ref::Const(true), Shape::And)
            }
            BoolExpr::Or(operands) => {
                let operands: Vec<Ref> = operands.iter().map(|e| self.flatten(e)).collect();
                if operands.contains(&Ref::Const(true)) {
                    return Ref::Const(true);
                }
                self.commutative(operands, Ref::Const(false), Shape::Or)
            }
            BoolExpr::Xor(a, b) => {
                let (a, b) = (self.flatten(a), self.flatten(b));
                match (a, b) {
                    (Ref::Const(value), other) | (other, Ref::Const(value)) => {
                        if value {
                            other.negated()
                        } else {
                            other
                        }
                    }
                    // Normalized both ways an `Xor` can be written the same
                    // — operand order, and which side carries the negation
                    // — so the two forms share one net.
                    _ => {
                        let (a, b) = if a.min(b) == a { (a, b) } else { (b, a) };
                        match (a, b) {
                            (Ref::Net(x, false), Ref::Net(y, false)) => {
                                self.net_for(Shape::Xor(Ref::Net(x, true), Ref::Net(y, true)))
                            }
                            _ => self.net_for(Shape::Xor(a, b)),
                        }
                    }
                }
            }
        }
    }

    /// Finishes a commutative operator: drops the operands that are its
    /// identity, normalizes the rest, and short-circuits the degenerate
    /// arities rather than defining a net for them.
    fn commutative(
        &mut self,
        operands: Vec<Ref>,
        identity: Ref,
        shape: fn(Vec<Ref>) -> Shape,
    ) -> Ref {
        let mut operands: Vec<Ref> = operands.into_iter().filter(|r| *r != identity).collect();
        operands.sort_unstable();
        operands.dedup();
        match operands.len() {
            0 => identity,
            1 => operands[0],
            _ => self.net_for(shape(operands)),
        }
    }
}

/// One bit of state in the unrolled system: a flip-flop's output, or one
/// stage of a merged register.
#[derive(Clone, Debug)]
struct StateElement {
    /// `"<cell instance>.<pin>"`, e.g. `"ShiftRegister#812.Q3"`.
    label: String,
    /// The net carrying this bit's current value — the cell output pin the
    /// rest of the graph reads. It gets its own variable per cycle, tied to
    /// `next` at the cycle *before* rather than to any combinational
    /// driver; that's the one place the unrolling steps forward in time.
    q_net: u32,
    /// This bit's value after the next clock edge, over the current
    /// cycle's nets.
    next: BoolExpr,
    /// The value it powers up holding, taken from its asynchronous
    /// set/reset pin: `false` for a `RESET`/`RESET_B` flip-flop, `true` for
    /// a `SET`/`SET_B` one, `None` for one carrying neither — whose initial
    /// value the solver is then free to choose, there being nothing in the
    /// netlist that says what it is.
    initial: Option<bool>,
    /// The net driving its clock pin, if it has one. Used only to work out
    /// which design inputs are clocks; the unrolling is cycle-based and
    /// never constrains it.
    clock_net: Option<u32>,
}

/// One cell's contribution to the sequential model: the state bits it
/// holds, any extra combinational definition it implies (a `Q_N` output
/// mirroring its `Q`), and the pins whose role this doesn't recognize.
struct StateCellModel {
    elements: Vec<StateElement>,
    defs: Vec<(u32, BoolExpr)>,
    unknown_pins: Vec<String>,
}

/// The `sky130` flip-flop families [`state_cell_model`] recognizes:
/// edge-triggered `df*` and its scan (`sdf*`) and enable (`edf*`,
/// `sedf*`) variants. Latches (`dlx*`, `dlr*`) and the `dlclkp` clock gate
/// are deliberately absent: they are level-sensitive, which one frame per
/// clock edge cannot represent.
const FLIPFLOP_PREFIXES: [&str; 4] = ["df", "sdf", "edf", "sedf"];

/// Every input pin name [`flipflop_model`] knows what to do with. A `df*`
/// cell carrying anything else (a scan input, a clock enable) is reported
/// as unmodeled rather than quietly having that pin ignored.
const FLIPFLOP_INPUT_PINS: [&str; 6] = ["D", "CLK", "RESET_B", "RESET", "SET_B", "SET"];

/// The net behind `pins`' pin named `name`, matched on its base name so a
/// pin already relabeled by [`Graph::propagate_pin_values`] still matches.
fn pin_net(pins: &[Pin], name: &str) -> Option<u32> {
    pins.iter()
        .find(|(_, pin_name)| pin_base_name(pin_name) == name)
        .map(|&(net, _)| net)
}

/// `S ? load : hold` — the select idiom every registered cell here is
/// built around, as a `mux2`-shaped expression.
fn select(s: u32, load: BoolExpr, hold: BoolExpr) -> BoolExpr {
    BoolExpr::or(vec![
        BoolExpr::and(vec![BoolExpr::Var(s), load]),
        BoolExpr::and(vec![BoolExpr::not(BoolExpr::Var(s)), hold]),
    ])
}

/// Wraps `next` in whatever asynchronous set/reset pins `inputs` carries,
/// and reports the value an element with those pins powers up holding.
///
/// A real asynchronous reset acts the instant it is asserted; a
/// cycle-based unrolling has nowhere to put that but the clock edge, so
/// this models it as a synchronous override — the usual approximation for
/// bounded model checking of a synchronous design, exact for every
/// sequence where reset is held across an edge. Where a cell has both, set
/// is applied first and reset outside it, making reset dominant.
fn apply_set_reset(inputs: &[Pin], mut next: BoolExpr) -> (BoolExpr, Option<bool>) {
    let mut initial = None;
    if let Some(net) = pin_net(inputs, "SET_B") {
        next = BoolExpr::or(vec![BoolExpr::not(BoolExpr::Var(net)), next]);
        initial = Some(true);
    }
    if let Some(net) = pin_net(inputs, "SET") {
        next = BoolExpr::or(vec![BoolExpr::Var(net), next]);
        initial = Some(true);
    }
    if let Some(net) = pin_net(inputs, "RESET_B") {
        next = BoolExpr::and(vec![BoolExpr::Var(net), next]);
        initial = Some(false);
    }
    if let Some(net) = pin_net(inputs, "RESET") {
        next = BoolExpr::and(vec![BoolExpr::not(BoolExpr::Var(net)), next]);
        initial = Some(false);
    }
    (next, initial)
}

/// A plain `df*` flip-flop: `D` captured at the edge, overridden by
/// whatever asynchronous set/reset it carries.
fn flipflop_model(instance: &str, inputs: &[Pin], outputs: &[Pin]) -> StateCellModel {
    let mut unknown_pins: Vec<String> = inputs
        .iter()
        .map(|(_, name)| pin_base_name(name))
        .filter(|name| !FLIPFLOP_INPUT_PINS.contains(name))
        .map(|name| format!("{instance}.{name}"))
        .collect();

    let d = match pin_net(inputs, "D") {
        Some(net) => BoolExpr::Var(net),
        None => {
            unknown_pins.push(format!("{instance} (no D input)"));
            BoolExpr::Const(false)
        }
    };
    let (next, initial) = apply_set_reset(inputs, d);
    let clock_net = pin_net(inputs, "CLK");

    let mut elements = Vec::new();
    let mut defs = Vec::new();
    match (pin_net(outputs, "Q"), pin_net(outputs, "Q_N")) {
        (Some(q_net), q_n) => {
            elements.push(StateElement {
                label: format!("{instance}.Q"),
                q_net,
                next,
                initial,
                clock_net,
            });
            if let Some(q_n_net) = q_n {
                defs.push((q_n_net, BoolExpr::not(BoolExpr::Var(q_net))));
            }
        }
        // Only the inverted output is exposed, so that net *is* the state
        // bit — holding the complement of what the flip-flop captured.
        (None, Some(q_n_net)) => elements.push(StateElement {
            label: format!("{instance}.Q_N"),
            q_net: q_n_net,
            next: BoolExpr::not(next),
            initial: initial.map(|value| !value),
            clock_net,
        }),
        (None, None) => unknown_pins.push(format!("{instance} (no Q output)")),
    }

    StateCellModel {
        elements,
        defs,
        unknown_pins,
    }
}

/// A `"MuxedResetableFlipflop"` merge cell: `S` picks between the new
/// value on `A1` and the flip-flop's own output, exactly the mux/flip-flop
/// pair [`Graph::merge_muxed_resetable_flipflops`] folded together.
fn muxed_flipflop_model(instance: &str, inputs: &[Pin], outputs: &[Pin]) -> StateCellModel {
    let Some(q_net) = pin_net(outputs, "Q") else {
        return StateCellModel {
            elements: Vec::new(),
            defs: Vec::new(),
            unknown_pins: vec![format!("{instance} (no Q output)")],
        };
    };

    let mut unknown_pins = Vec::new();
    let load = match pin_net(inputs, "A1") {
        Some(net) => BoolExpr::Var(net),
        None => {
            unknown_pins.push(format!("{instance} (no A1 input)"));
            BoolExpr::Var(q_net)
        }
    };
    let hold = BoolExpr::Var(q_net);
    let next = match pin_net(inputs, "S") {
        Some(s) => select(s, load, hold),
        None => {
            unknown_pins.push(format!("{instance} (no S input)"));
            load
        }
    };
    let (next, initial) = apply_set_reset(inputs, next);

    StateCellModel {
        elements: vec![StateElement {
            label: format!("{instance}.Q"),
            q_net,
            next,
            initial,
            clock_net: pin_net(inputs, "CLK"),
        }],
        defs: Vec::new(),
        unknown_pins,
    }
}

/// A `"ShiftRegister"` merge cell: with `S` selecting a shift, `Q0` takes
/// the serial input `A` and every later bit takes the one before it —
/// the chain [`Graph::merge_shift_registers`] folded together. With `S`
/// low every bit holds, and the shared `RESET_B` clears the whole
/// register.
fn shift_register_model(instance: &str, inputs: &[Pin], outputs: &[Pin]) -> StateCellModel {
    // `Q0`..`Q<n-1>` in chain order, by the trailing index the merge
    // numbered them with rather than by pin order.
    let mut stages: Vec<(u64, u32)> = outputs
        .iter()
        .filter_map(|(net, name)| {
            let index = pin_base_name(name).strip_prefix('Q')?.parse().ok()?;
            Some((index, *net))
        })
        .collect();
    stages.sort_unstable();

    let mut unknown_pins = Vec::new();
    if stages.is_empty() {
        unknown_pins.push(format!("{instance} (no Qn outputs)"));
    }
    let serial = match pin_net(inputs, "A") {
        Some(net) => BoolExpr::Var(net),
        None => {
            unknown_pins.push(format!("{instance} (no A input)"));
            BoolExpr::Const(false)
        }
    };
    let s = pin_net(inputs, "S");
    if s.is_none() && !stages.is_empty() {
        unknown_pins.push(format!("{instance} (no S input)"));
    }
    let clock_net = pin_net(inputs, "CLK");

    let elements = stages
        .iter()
        .enumerate()
        .map(|(position, &(index, q_net))| {
            let load = match position.checked_sub(1) {
                Some(previous) => BoolExpr::Var(stages[previous].1),
                None => serial.clone(),
            };
            let next = match s {
                Some(s) => select(s, load, BoolExpr::Var(q_net)),
                None => load,
            };
            let (next, initial) = apply_set_reset(inputs, next);
            StateElement {
                label: format!("{instance}.Q{index}"),
                q_net,
                next,
                initial,
                clock_net,
            }
        })
        .collect();

    StateCellModel {
        elements,
        defs: Vec::new(),
        unknown_pins,
    }
}

/// `cell`'s sequential model, or `None` if it holds no state — in which
/// case it is combinational (or a boundary), and [`Graph::system_model`]
/// handles it there instead.
fn state_cell_model(cell: &Cell) -> Option<StateCellModel> {
    let instance = cell_instance_label(cell);
    match cell {
        Cell::Sky130Standard {
            cell_name,
            inputs,
            outputs,
            ..
        } => {
            let base = strip_sky130_prefix_and_drive(cell_name)?;
            FLIPFLOP_PREFIXES
                .iter()
                .any(|prefix| base.starts_with(prefix))
                .then(|| flipflop_model(&instance, inputs, outputs))
        }
        Cell::MergeCell {
            cell_name,
            inputs,
            outputs,
            ..
        } => match cell_name.as_str() {
            "MuxedResetableFlipflop" => Some(muxed_flipflop_model(&instance, inputs, outputs)),
            "ShiftRegister" => Some(shift_register_model(&instance, inputs, outputs)),
            _ => None,
        },
        _ => None,
    }
}

/// Everything [`Graph::solve_system`] needs about a graph to unroll it in
/// time: what each net is a function of, what state the design holds, and
/// where its inputs enter.
struct SystemModel {
    /// Net -> the expression driving it, over that *same* cycle's nets.
    /// Flattened by [`Flattener`], so each one is a single operator over
    /// net-or-constant operands and structurally identical
    /// sub-expressions share a net.
    defs: BTreeMap<u32, BoolExpr>,
    /// One entry per bit of state, in cell order.
    state: Vec<StateElement>,
    /// The design's own inputs — the `Input` cell's pins — as
    /// `(pin name, net)`, free in every cycle.
    inputs: Vec<(String, u32)>,
    /// Which of `inputs` reach a state element's clock pin, and so are the
    /// design's clock. The unrolling is cycle-based — one frame per clock
    /// edge — so these are left unconstrained rather than made to carry a
    /// waveform, and whatever value the model gives them means nothing.
    clock_inputs: Vec<String>,
    /// Cells and pins with no transition relation here, named for the
    /// report; their effect is simply absent from the unrolling.
    unmodeled: Vec<String>,
}

impl Graph {
    /// Reads this graph as a synchronous machine: see [`SystemModel`].
    ///
    /// Combinational definitions come from whatever each cell is —
    /// a `"BooleanFunction"` merge cell's composed `boolean_outputs`, or a
    /// plain gate's own [`gate_output_exprs`] — so this works on a graph
    /// whether or not a boolean-function merge has been run over it. Every
    /// `connections` edge then defines its destination pin to carry its
    /// source's value, which is what stitches the cells together. A net
    /// left undefined by all of that — a design input, a dangling pin — is
    /// free, independently per cycle.
    fn system_model(&self) -> Result<SystemModel, String> {
        let mut raw_defs: Vec<(u32, BoolExpr)> = Vec::new();
        let mut state: Vec<StateElement> = Vec::new();
        let mut inputs: Vec<(String, u32)> = Vec::new();
        let mut unmodeled: BTreeSet<String> = BTreeSet::new();
        let mut max_net = 0u32;

        for cell in &self.cells {
            for &(net, _) in cell.inputs().iter().chain(cell.outputs().iter()) {
                max_net = max_net.max(net);
            }

            if let Some(model) = state_cell_model(cell) {
                raw_defs.extend(model.defs);
                state.extend(model.elements);
                unmodeled.extend(model.unknown_pins);
                continue;
            }

            match cell {
                Cell::Input { outputs } => inputs.extend(
                    outputs
                        .iter()
                        .map(|(net, name)| (pin_base_name(name).to_string(), *net)),
                ),
                // An output cell computes nothing; its pins carry whatever
                // the connections into them carry.
                Cell::Output { .. } => {}
                Cell::MergeCell {
                    cell_name,
                    boolean_outputs,
                    ..
                } if cell_name == "BooleanFunction" => {
                    raw_defs.extend(boolean_outputs.iter().cloned());
                }
                Cell::Sky130Standard {
                    cell_name,
                    inputs,
                    outputs,
                    ..
                } => match gate_output_exprs(cell_name, inputs, outputs) {
                    Some(exprs) => raw_defs.extend(exprs),
                    None => {
                        unmodeled.insert(cell_instance_label(cell));
                    }
                },
                _ => {
                    unmodeled.insert(cell_instance_label(cell));
                }
            }
        }

        for conn in &self.connections {
            for (&src, dsts) in conn {
                max_net = max_net.max(src);
                for &dst in dsts {
                    max_net = max_net.max(dst);
                    raw_defs.push((dst, BoolExpr::Var(src)));
                }
            }
        }

        let Some(next_net) = max_net.checked_add(1) else {
            return Err(
                "This graph's net ids leave no room for the extra nets the unrolling needs."
                    .to_string(),
            );
        };
        let mut flattener = Flattener::new(next_net);
        let mut defs = BTreeMap::new();
        for (net, expr) in &raw_defs {
            let value = flattener.flatten(expr);
            defs.insert(*net, value.expr());
        }
        if flattener.overflowed {
            return Err(
                "This graph's net ids leave no room for the extra nets the unrolling needs."
                    .to_string(),
            );
        }
        defs.extend(flattener.defs);

        // A design input that reaches a clock pin is a clock. Walked
        // backwards from every clock pin through the definitions above,
        // collecting the free nets they bottom out at.
        let clock_support =
            free_net_support(state.iter().filter_map(|element| element.clock_net), &defs);
        let clock_inputs = inputs
            .iter()
            .filter(|(_, net)| clock_support.contains(net))
            .map(|(name, _)| name.clone())
            .collect();

        Ok(SystemModel {
            defs,
            state,
            inputs,
            clock_inputs,
            unmodeled: unmodeled.into_iter().collect(),
        })
    }

    /// The nets a target pin name refers to, as `(label, net)`.
    ///
    /// Tried in tiers, the first non-empty one winning: an exact
    /// `"<cell instance>.<pin>"` label; an `Output` cell pin with that base
    /// name (the usual case — `"success"`, `"O[3]"`); any pin anywhere with
    /// that base name. A name can legitimately land on several nets — one
    /// per pin it matches — and every one of them is then constrained.
    fn target_pin_nets(&self, name: &str) -> Vec<(String, u32)> {
        let mut exact = Vec::new();
        let mut outputs = Vec::new();
        let mut anywhere = Vec::new();

        for cell in &self.cells {
            let instance = cell_instance_label(cell);
            let is_output_cell = matches!(cell, Cell::Output { .. });
            for (net, pin_name) in cell.inputs().iter().chain(cell.outputs().iter()) {
                let base = pin_base_name(pin_name);
                let label = format!("{instance}.{base}");
                if label == name {
                    exact.push((label.clone(), *net));
                }
                if base == name {
                    if is_output_cell {
                        outputs.push((label.clone(), *net));
                    }
                    anywhere.push((label, *net));
                }
            }
        }

        if !exact.is_empty() {
            return exact;
        }
        if !outputs.is_empty() {
            return outputs;
        }
        anywhere
    }

    /// Every bus among the `Output` cell's pins — each distinct `name` of
    /// a pin shaped `name[i]`, in order of first appearance — with its
    /// nets least significant first (see [`Graph::bus_nets`]).
    fn output_buses(&self) -> Vec<(String, Vec<u32>)> {
        let mut names: Vec<String> = Vec::new();
        for cell in &self.cells {
            let Cell::Output { inputs } = cell else {
                continue;
            };
            for (_, pin_name) in inputs {
                if let Some((name, _)) = split_bus_pin(pin_base_name(pin_name))
                    && !names.iter().any(|known| known == name)
                {
                    names.push(name.to_string());
                }
            }
        }
        names
            .into_iter()
            .map(|name| {
                let nets = self.bus_nets(&name);
                (name, nets)
            })
            .collect()
    }

    /// The nets of the output bus `name` — the pins `name[0]`, `name[1]`,
    /// ... — least significant first, resolved the way
    /// [`Graph::target_pin_nets`] resolves a pin (so `Output` cell pins
    /// win). Empty if no pin is shaped like that.
    fn bus_nets(&self, name: &str) -> Vec<u32> {
        let mut bits: Vec<(usize, u32)> = Vec::new();
        let mut index = 0;
        loop {
            let matches = self.target_pin_nets(&format!("{name}[{index}]"));
            let Some(&(_, net)) = matches.first() else {
                break;
            };
            bits.push((index, net));
            index += 1;
        }
        bits.into_iter().map(|(_, net)| net).collect()
    }

    /// Finds the shortest input sequence that drives this whole graph — as
    /// a synchronous machine, not one boolean function at a time — into a
    /// state where every `(pin name, value)` in `targets` and every bus
    /// condition in `bus_constraints` holds at once, and reports it as a
    /// waveform. Only lengths of at least `min_cycles` are considered.
    ///
    /// The graph is unrolled in time: one *frame* per clock cycle, each
    /// with its own copy of every net. Frame 0 is the design's power-up
    /// state, every flip-flop holding what its asynchronous set/reset pin
    /// says it holds (see [`apply_set_reset`]); each later frame's state
    /// is the frame before it stepped through the transition relation.
    /// Design inputs are free in every frame — that freedom is what the
    /// answer is made of — except the clock, which is abstracted away by
    /// the one-frame-per-edge unrolling rather than modeled as a signal.
    ///
    /// The search for the length is exponential, then binary: frames
    /// `0, 1, 2, 4, 8, ...` up to `max_cycles` until one is satisfiable,
    /// then a binary search back over the gap the doubling jumped. That is
    /// `O(log n)` solver calls instead of `n`, and it finds the *shortest*
    /// sequence whenever reaching the target one cycle later stays possible
    /// — true of any target a design can hold, though a condition that
    /// flashes true for exactly one cycle and can never be reached again
    /// could in principle hide a shorter answer inside the gap.
    ///
    /// One solver holds the whole search: frames are added to it as the
    /// search reaches them, and each candidate length's target constraint
    /// is guarded by its own assumption literal (see
    /// [`CnfEncoder::assert_net_value_at_if`]), so nothing is ever encoded
    /// twice and every probe keeps what the ones before it learned.
    pub fn solve_system(
        &self,
        targets: &[(String, bool)],
        bus_constraints: &[BusConstraint],
        min_cycles: usize,
        max_cycles: usize,
    ) -> Result<SystemSolution, String> {
        let model = self.system_model()?;

        // A target on a net nothing drives — a design input, or a pin
        // whose logic a cone-of-influence merge pruned away — is not a
        // condition on the machine at all: the solver would just pick
        // the value. Refused, naming the cause, rather than answered.
        let driven: HashSet<u32> = model
            .state
            .iter()
            .map(|element| element.q_net)
            .chain(model.defs.keys().copied())
            .collect();
        let check_driven = |label: &str, net: u32| -> Result<(), String> {
            if driven.contains(&net) {
                Ok(())
            } else {
                Err(format!(
                    "{label} is not driven by anything in this graph — if this is a cone-of-\
                     influence merge, add it to the cone pin list."
                ))
            }
        };

        let mut resolved: Vec<(String, u32, bool)> = Vec::new();
        for (name, value) in targets {
            let name = name.trim();
            if name.is_empty() {
                continue;
            }
            let matches = self.target_pin_nets(name);
            if matches.is_empty() {
                return Err(format!("No pin named \"{name}\" in this graph."));
            }
            for (label, net) in &matches {
                check_driven(label, *net)?;
            }
            resolved.extend(matches.into_iter().map(|(label, net)| (label, net, *value)));
        }
        let mut buses: Vec<ResolvedBus> = Vec::new();
        for constraint in bus_constraints {
            let name = constraint.bus.trim();
            if name.is_empty() {
                continue;
            }
            let nets = self.bus_nets(name);
            if nets.is_empty() {
                return Err(format!(
                    "No bus named \"{name}\" in this graph (no pin \"{name}[0]\")."
                ));
            }
            for (bit, &net) in nets.iter().enumerate() {
                check_driven(&format!("{name}[{bit}]"), net)?;
            }
            if nets.len() >= 64 {
                return Err(format!(
                    "{name} is {} bits wide; bus conditions go up to 63 bits.",
                    nets.len()
                ));
            }
            if constraint.value >> nets.len() != 0 {
                return Err(format!(
                    "{name} is {} bits wide, which cannot hold {:#x}.",
                    nets.len(),
                    constraint.value
                ));
            }
            let cycle = match constraint.cycle {
                Some(cycle) => Some(u32::try_from(cycle).map_err(|_| {
                    format!("{name}: frame {cycle} is beyond what can be unrolled.")
                })?),
                None => None,
            };
            buses.push(ResolvedBus {
                label: format!(
                    "{name} {} {}{}",
                    if constraint.equal { "=" } else { "≠" },
                    constraint.describe_value(),
                    match cycle {
                        Some(cycle) => format!(" @ {cycle}"),
                        None => String::new(),
                    }
                ),
                bits: nets
                    .iter()
                    .enumerate()
                    .map(|(bit, &net)| (net, constraint.value >> bit & 1 == 1))
                    .collect(),
                equal: constraint.equal,
                cycle,
            });
        }
        if resolved.is_empty() && buses.is_empty() {
            return Err("Name at least one pin and the value to reach for it.".to_string());
        }
        let condition = Condition {
            pins: resolved,
            buses,
        };

        let max = u32::try_from(max_cycles).unwrap_or(u32::MAX);
        let min = u32::try_from(min_cycles).unwrap_or(u32::MAX);
        if min > max {
            return Err(format!(
                "The minimum of {min} cycles is more than the maximum of {max}."
            ));
        }
        // A condition pinned to a frame needs the sequence to reach that
        // frame, so it raises the floor of the search.
        if let Some(bus) = condition
            .buses
            .iter()
            .find(|bus| bus.cycle.is_some_and(|cycle| cycle > max))
        {
            return Err(format!(
                "{} asks about a frame past the maximum of {max} cycles.",
                bus.label
            ));
        }
        let min = condition
            .buses
            .iter()
            .filter_map(|bus| bus.cycle)
            .fold(min, u32::max);
        let mut unrolling = Unrolling::new();
        let mut probes: Vec<(usize, bool)> = Vec::new();

        // Exponential search from the minimum: min, min+1, min+2, min+4,
        // ... capped at `max`, stopping at the first length that works.
        let mut satisfied = None;
        let mut longest_failed = min;
        let mut cycles = min;
        loop {
            let sat = unrolling.probe(&model, &condition, cycles);
            probes.push((cycles as usize, sat));
            if sat {
                satisfied = Some(cycles);
                break;
            }
            longest_failed = cycles;
            if cycles >= max {
                break;
            }
            let step = cycles - min;
            cycles = min
                .saturating_add(if step == 0 { 1 } else { step.saturating_mul(2) })
                .min(max);
        }

        let Some(mut shortest) = satisfied else {
            return Err(format!(
                "No input sequence of {} clock cycle{} reaches that condition.",
                if min == 0 {
                    format!("up to {max}")
                } else {
                    format!("{min} to {max}")
                },
                if max == 1 { "" } else { "s" }
            ));
        };

        // Binary search back over the gap the doubling jumped: everything
        // at or below `longest_failed` is known not to work (when the
        // minimum itself worked, there is no gap), `shortest` is known to.
        while shortest > longest_failed + 1 {
            let middle = longest_failed + (shortest - longest_failed) / 2;
            let sat = unrolling.probe(&model, &condition, middle);
            probes.push((middle as usize, sat));
            if sat {
                shortest = middle;
            } else {
                longest_failed = middle;
            }
        }

        // The model in the solver is the last probe's, which the binary
        // search above may have left on a length that didn't work.
        if unrolling.last_satisfied != Some(shortest) {
            unrolling.probe(&model, &condition, shortest);
        }
        let resolved = &condition.pins;

        let frames = (0..=shortest)
            .map(|frame| {
                model
                    .inputs
                    .iter()
                    .map(|&(_, net)| unrolling.value(net, frame))
                    .collect()
            })
            .collect();
        let target_frames = (0..=shortest)
            .map(|frame| {
                resolved
                    .iter()
                    .map(|&(_, net, _)| unrolling.value(net, frame))
                    .collect()
            })
            .collect();
        let states = (0..=shortest)
            .map(|frame| {
                model
                    .state
                    .iter()
                    .map(|element| unrolling.value(element.q_net, frame))
                    .collect()
            })
            .collect();
        let bus_frames = (0..=shortest)
            .map(|frame| {
                condition
                    .buses
                    .iter()
                    .map(|bus| {
                        bus.bits
                            .iter()
                            .enumerate()
                            .fold(0u64, |value, (bit, &(net, _))| {
                                value | (u64::from(unrolling.value(net, frame)) << bit)
                            })
                    })
                    .collect()
            })
            .collect();

        let mut notes = Vec::new();
        if !model.clock_inputs.is_empty() {
            notes.push(format!(
                "One frame per clock edge, so the clock itself is abstracted away: {} carries no \
                 meaningful value below.",
                model.clock_inputs.join(", ")
            ));
        }

        // Every output bus, frame by frame — whether or not a condition
        // named it. One nothing drives is left out rather than shown with
        // whatever the solver happened to pick for it.
        let mut output_bus_labels = Vec::new();
        let mut output_bus_widths = Vec::new();
        let mut shown_buses: Vec<Vec<u32>> = Vec::new();
        for (name, nets) in self.output_buses() {
            if nets.is_empty() || nets.len() >= 64 {
                continue;
            }
            if let Some(bit) = nets.iter().position(|net| !driven.contains(net)) {
                notes.push(format!(
                    "{name} is not shown: {name}[{bit}] is not driven by anything in this graph \
                     (if this is a cone-of-influence merge, add the bus to the cone pin list)."
                ));
                continue;
            }
            output_bus_labels.push(name);
            output_bus_widths.push(nets.len());
            shown_buses.push(nets);
        }
        let output_bus_frames = (0..=shortest)
            .map(|frame| {
                shown_buses
                    .iter()
                    .map(|nets| {
                        nets.iter().enumerate().fold(0u64, |value, (bit, &net)| {
                            value | (u64::from(unrolling.value(net, frame)) << bit)
                        })
                    })
                    .collect()
            })
            .collect();
        let free_initial = model
            .state
            .iter()
            .filter(|element| element.initial.is_none())
            .count();
        if free_initial > 0 {
            notes.push(format!(
                "{free_initial} of {} state bits have no set/reset pin, so the solver was free to \
                 choose what they power up holding.",
                model.state.len()
            ));
        }
        if !model.unmodeled.is_empty() {
            notes.push(format!(
                "Not modeled, and so absent from the unrolling: {}.",
                model.unmodeled.join(", ")
            ));
        }

        Ok(SystemSolution {
            cycles: shortest as usize,
            input_labels: model.inputs.iter().map(|(name, _)| name.clone()).collect(),
            clock_inputs: model.clock_inputs.clone(),
            frames,
            target_labels: resolved
                .iter()
                .map(|(label, _, value)| format!("{label} = {}", u8::from(*value)))
                .collect(),
            target_frames,
            bus_labels: condition
                .buses
                .iter()
                .map(|bus| bus.label.clone())
                .collect(),
            bus_widths: condition.buses.iter().map(|bus| bus.bits.len()).collect(),
            bus_frames,
            output_bus_labels,
            output_bus_widths,
            output_bus_frames,
            state_labels: model
                .state
                .iter()
                .map(|element| element.label.clone())
                .collect(),
            states,
            probes,
            notes,
        })
    }
}

/// The free nets — the ones no definition drives — that `roots` ultimately
/// depend on, walked backwards through `defs`.
fn free_net_support(
    roots: impl IntoIterator<Item = u32>,
    defs: &BTreeMap<u32, BoolExpr>,
) -> HashSet<u32> {
    let mut pending: Vec<u32> = roots.into_iter().collect();
    let mut seen: HashSet<u32> = pending.iter().copied().collect();
    let mut free = HashSet::new();
    let mut operands = Vec::new();

    while let Some(net) = pending.pop() {
        let Some(expr) = defs.get(&net) else {
            free.insert(net);
            continue;
        };
        operands.clear();
        collect_expr_vars(expr, &mut operands);
        for &operand in &operands {
            if seen.insert(operand) {
                pending.push(operand);
            }
        }
    }
    free
}

/// Appends every net `expr` reads to `out`, duplicates and all.
fn collect_expr_vars(expr: &BoolExpr, out: &mut Vec<u32>) {
    match expr {
        BoolExpr::Var(net) => out.push(*net),
        BoolExpr::Const(_) => {}
        BoolExpr::Not(inner) => collect_expr_vars(inner, out),
        BoolExpr::And(operands) | BoolExpr::Or(operands) => {
            for operand in operands {
                collect_expr_vars(operand, out);
            }
        }
        BoolExpr::Xor(a, b) => {
            collect_expr_vars(a, out);
            collect_expr_vars(b, out);
        }
    }
}

/// Splits `"O[3]"` into `("O", 3)`; `None` for a name not shaped like
/// that.
pub fn split_bus_pin(label: &str) -> Option<(&str, usize)> {
    let (name, index) = label.strip_suffix(']')?.rsplit_once('[')?;
    Some((name, index.parse().ok()?))
}

/// A condition on an output bus for [`Graph::solve_system`] to reach:
/// the bus `name[0]`, `name[1]`, ... read as an unsigned number, least
/// significant bit first, equal or not equal to `value`.
///
/// `u64`, not `u128`: [`SystemSolution`] carries the bus's value per frame
/// and is saved inside a `#[serde(flatten)]`ed struct, whose buffering
/// has no 128-bit representation — a `u128` there fails to load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BusConstraint {
    pub bus: String,
    pub equal: bool,
    pub value: u64,
    /// The frame the condition applies in: a fixed one, or `None` for
    /// the last frame of the sequence — the frame the pin targets hold
    /// in. A fixed frame is a floor on the sequence's length, and lets
    /// one bus be pinned differently in different frames.
    pub cycle: Option<usize>,
}

impl BusConstraint {
    /// `value` as the label shows it: the character, quoted, when it is
    /// a printable ASCII one, otherwise hex.
    pub fn describe_value(&self) -> String {
        match u8::try_from(self.value) {
            Ok(byte) if byte.is_ascii_graphic() || byte == b' ' => {
                format!("'{}'", char::from(byte))
            }
            _ => format!("{:#x}", self.value),
        }
    }
}

/// What a probe of [`Graph::solve_system`] asks to hold in its last
/// frame, resolved to nets.
struct Condition {
    /// `(label, net, value)` per target pin.
    pins: Vec<(String, u32, bool)>,
    buses: Vec<ResolvedBus>,
}

/// A [`BusConstraint`] resolved to nets: `(net, target bit)` least
/// significant first.
struct ResolvedBus {
    label: String,
    bits: Vec<(u32, bool)>,
    equal: bool,
    /// See [`BusConstraint::cycle`].
    cycle: Option<u32>,
}

/// The one growing SAT instance every probe of [`Graph::solve_system`]'s
/// length search shares: frames encoded once, in order, and one guard
/// literal per candidate length.
struct Unrolling {
    encoder: CnfEncoder<batsat::BasicSolver>,
    /// How many frames are encoded: `0..encoded`.
    encoded: u32,
    /// Candidate length -> the literal guarding its target constraint.
    guards: BTreeMap<u32, Lit>,
    /// The length of the last probe, if it came back satisfiable — i.e.
    /// which frame count the model now in the solver belongs to. Cleared
    /// by an unsatisfiable probe, which leaves no model behind at all.
    last_satisfied: Option<u32>,
}

impl Unrolling {
    fn new() -> Self {
        Unrolling {
            encoder: CnfEncoder::new(batsat::BasicSolver::default()),
            encoded: 0,
            guards: BTreeMap::new(),
            last_satisfied: None,
        }
    }

    /// Encodes frames up to and including `frame`, if they aren't already.
    fn encode_through(&mut self, model: &SystemModel, frame: u32) {
        while self.encoded <= frame {
            let current = self.encoded;
            for (&net, expr) in &model.defs {
                self.encoder.assert_eq_at(net, current, expr, current);
            }
            for element in &model.state {
                match current.checked_sub(1) {
                    // Every later frame's state is the frame before it,
                    // stepped once through the transition relation.
                    Some(previous) => {
                        self.encoder
                            .assert_eq_at(element.q_net, current, &element.next, previous)
                    }
                    None => match element.initial {
                        Some(value) => self.encoder.assert_net_value_at(element.q_net, 0, value),
                        // Nothing says what it powers up holding; leave it
                        // to the solver, but give it a variable so it can
                        // still be read back.
                        None => {
                            self.encoder.net_var_at(element.q_net, 0);
                        }
                    },
                }
            }
            // Every input gets a variable per frame even where nothing
            // reads it, so the reported waveform has a row for it.
            for &(_, net) in &model.inputs {
                self.encoder.net_var_at(net, current);
            }
            self.encoded += 1;
        }
    }

    /// Whether `condition` can hold in frame `cycles`. Leaves the solver
    /// holding that frame count's model when it can.
    fn probe(&mut self, model: &SystemModel, condition: &Condition, cycles: u32) -> bool {
        self.encode_through(model, cycles);
        let guard = match self.guards.get(&cycles) {
            Some(&guard) => guard,
            None => {
                let guard = self.encoder.fresh_guard();
                for &(_, net, value) in &condition.pins {
                    self.encoder
                        .assert_net_value_at_if(guard, net, cycles, value);
                }
                for bus in &condition.buses {
                    // A pinned frame is always at or below `cycles` — the
                    // search floor saw to that — so it's already encoded.
                    let frame = bus.cycle.unwrap_or(cycles);
                    if bus.equal {
                        for &(net, value) in &bus.bits {
                            self.encoder
                                .assert_net_value_at_if(guard, net, frame, value);
                        }
                    } else {
                        // Differs in at least one bit.
                        let differing: Vec<(u32, bool)> =
                            bus.bits.iter().map(|&(net, value)| (net, !value)).collect();
                        self.encoder
                            .assert_any_net_value_at_if(guard, frame, &differing);
                    }
                }
                self.guards.insert(cycles, guard);
                guard
            }
        };
        let satisfied = self.encoder.solver_mut().solve_limited(&[guard]) == lbool::TRUE;
        self.last_satisfied = satisfied.then_some(cycles);
        satisfied
    }

    fn value(&self, net: u32, frame: u32) -> bool {
        self.encoder.net_value_at(net, frame).unwrap_or(false)
    }
}

/// What a successful [`Graph::solve_system`] found: the shortest input
/// sequence that reaches the target condition, frame by frame.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemSolution {
    /// How many clock edges the sequence takes. The condition holds in
    /// frame `cycles`, the last of the `cycles + 1` frames below.
    pub cycles: usize,
    /// One entry per design input pin, in `Input` cell pin order — the
    /// columns of `frames`.
    pub input_labels: Vec<String>,
    /// Which of `input_labels` are clocks, and so carry no meaningful
    /// value: the unrolling is one frame per clock edge.
    pub clock_inputs: Vec<String>,
    /// `frames[t][i]`: what input `input_labels[i]` must be in frame `t`.
    pub frames: Vec<Vec<bool>>,
    /// One entry per target pin the condition named, already rendered as
    /// `"<pin> = <value>"` — the columns of `target_frames`.
    pub target_labels: Vec<String>,
    /// `target_frames[t][i]`: what target pin `i` actually carries in
    /// frame `t`, which is the asked-for value only in the last frame.
    pub target_frames: Vec<Vec<bool>>,
    /// One entry per bus constraint, rendered as `"<bus> ≠ 'A'"` — the
    /// columns of `bus_frames`.
    #[serde(default)]
    pub bus_labels: Vec<String>,
    /// How many bits each bus in `bus_labels` has.
    #[serde(default)]
    pub bus_widths: Vec<usize>,
    /// `bus_frames[t][i]`: what bus `i` reads in frame `t`.
    #[serde(default)]
    pub bus_frames: Vec<Vec<u64>>,
    /// Every output bus of the graph (`O` for `O[0]`..`O[7]`), condition
    /// or not, that the graph actually drives — the columns of
    /// `output_bus_frames`.
    #[serde(default)]
    pub output_bus_labels: Vec<String>,
    /// How many bits each bus in `output_bus_labels` has.
    #[serde(default)]
    pub output_bus_widths: Vec<usize>,
    /// `output_bus_frames[t][i]`: what output bus `i` reads in frame `t`.
    #[serde(default)]
    pub output_bus_frames: Vec<Vec<u64>>,
    /// One entry per state bit, in cell order — the columns of `states`.
    pub state_labels: Vec<String>,
    /// `states[t][i]`: what state bit `i` holds at the start of frame `t`.
    pub states: Vec<Vec<bool>>,
    /// Every length the search tried and whether it worked, in order.
    pub probes: Vec<(usize, bool)>,
    /// What the model could not account for, and where it approximated —
    /// worth reading before trusting a sequence.
    pub notes: Vec<String>,
}

/// Runs a graph forward as a synchronous machine, one clock cycle at a
/// time, with the inputs held at values the caller gives — the concrete
/// counterpart of [`Graph::solve_system`], which reads the same
/// [`SystemModel`] but asks the solver what the inputs must be.
///
/// Built once from a graph and reused across steps; what it carries
/// between them is only the [`Simulation`] trace the caller keeps, so a
/// trace can be saved and resumed against a rebuilt simulator.
pub struct Simulator {
    model: SystemModel,
    /// The `Output` cell's pins as `(pin name, net)` — what a frame
    /// reports, in pin order.
    outputs: Vec<(String, u32)>,
    /// Every defined net in an order that puts each one after everything
    /// it reads, so a frame is a single pass. A combinational loop has no
    /// such order; the nets on it come in DFS post-order and whichever is
    /// evaluated first reads its loop predecessor as `false`.
    order: Vec<u32>,
    /// Whether any combinational loop was found while ordering.
    looped: bool,
    /// One more than the highest net id anything here touches: the size
    /// of a frame's value table.
    net_count: usize,
}

/// One clock cycle of a [`Simulation`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SimulationFrame {
    /// What each input held during this cycle — and so at the clock edge
    /// that ends it — in [`Simulation::input_labels`] order. The same
    /// convention as [`SystemSolution::frames`], so a solved sequence
    /// replays frame for frame.
    pub inputs: Vec<bool>,
    /// What each output carried during this cycle, in
    /// [`Simulation::output_labels`] order.
    pub outputs: Vec<bool>,
    /// What each state bit held at the start of this cycle, in
    /// [`Simulation::state_labels`] order — what the next frame is
    /// computed from.
    pub state: Vec<bool>,
}

/// What a [`Simulator::run`] produced: frame 0 is power-up, each later
/// frame one clock edge after the one before.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Simulation {
    /// One entry per design input pin, in `Input` cell pin order.
    pub input_labels: Vec<String>,
    /// Which of `input_labels` are clocks. Each frame *is* one edge of
    /// them, so the value recorded for one is meaningless.
    pub clock_inputs: Vec<String>,
    /// One entry per `Output` cell pin, in pin order.
    pub output_labels: Vec<String>,
    /// One entry per state bit, in cell order.
    pub state_labels: Vec<String>,
    pub frames: Vec<SimulationFrame>,
    /// What the model could not account for, and where it approximated.
    pub notes: Vec<String>,
}

impl Simulator {
    pub fn new(graph: &Graph) -> Result<Simulator, String> {
        let model = graph.system_model()?;

        let outputs: Vec<(String, u32)> = graph
            .cells
            .iter()
            .filter_map(|cell| match cell {
                Cell::Output { inputs } => Some(inputs),
                _ => None,
            })
            .flatten()
            .map(|(net, name)| (pin_base_name(name).to_string(), *net))
            .collect();

        // Dependency order over the definitions, by iterative DFS
        // post-order: a net is emitted once every net it reads has been.
        // Iterative rather than recursive because a chain of gates can be
        // thousands deep.
        const VISITING: u8 = 1;
        const DONE: u8 = 2;
        let operands: HashMap<u32, Vec<u32>> = model
            .defs
            .iter()
            .map(|(&net, expr)| {
                let mut vars = Vec::new();
                collect_expr_vars(expr, &mut vars);
                (net, vars)
            })
            .collect();
        let mut mark: HashMap<u32, u8> = HashMap::new();
        let mut order = Vec::with_capacity(model.defs.len());
        let mut looped = false;
        let mut stack: Vec<(u32, usize)> = Vec::new();
        for &root in model.defs.keys() {
            if mark.contains_key(&root) {
                continue;
            }
            mark.insert(root, VISITING);
            stack.push((root, 0));
            while let Some((net, next_operand)) = stack.last_mut() {
                let net = *net;
                let ops = &operands[&net];
                if *next_operand < ops.len() {
                    let operand = ops[*next_operand];
                    *next_operand += 1;
                    match mark.get(&operand) {
                        Some(&VISITING) => looped = true,
                        Some(_) => {}
                        // A free net — an input, a state bit, a dangling
                        // pin — has nothing to order.
                        None if !model.defs.contains_key(&operand) => {}
                        None => {
                            mark.insert(operand, VISITING);
                            stack.push((operand, 0));
                        }
                    }
                } else {
                    mark.insert(net, DONE);
                    order.push(net);
                    stack.pop();
                }
            }
        }

        let mut max_net = 0u32;
        for (&net, ops) in &operands {
            max_net = max_net.max(net);
            max_net = max_net.max(ops.iter().copied().max().unwrap_or(0));
        }
        for &(_, net) in model.inputs.iter().chain(outputs.iter()) {
            max_net = max_net.max(net);
        }
        let mut vars = Vec::new();
        for element in &model.state {
            max_net = max_net.max(element.q_net);
            vars.clear();
            collect_expr_vars(&element.next, &mut vars);
            max_net = max_net.max(vars.iter().copied().max().unwrap_or(0));
        }

        Ok(Simulator {
            model,
            outputs,
            order,
            looped,
            net_count: max_net as usize + 1,
        })
    }

    /// The design's input pins, in the order [`Simulation::input_labels`]
    /// and every `inputs` argument here use.
    pub fn input_labels(&self) -> impl Iterator<Item = &str> {
        self.model.inputs.iter().map(|(name, _)| name.as_str())
    }

    /// Which of [`Simulator::input_labels`] are clocks — see
    /// [`Simulation::clock_inputs`].
    pub fn clock_inputs(&self) -> &[String] {
        &self.model.clock_inputs
    }

    /// Runs the design from power-up — every flip-flop holding what its
    /// set/reset pin says (see [`apply_set_reset`]) and `false` where it
    /// has neither — through one frame per entry of `inputs_per_frame`,
    /// each entry the value of every input (one per
    /// [`Simulator::input_labels`]) during that frame.
    pub fn run<'a>(&self, inputs_per_frame: impl IntoIterator<Item = &'a [bool]>) -> Simulation {
        let mut state: Vec<bool> = self
            .model
            .state
            .iter()
            .map(|element| element.initial.unwrap_or(false))
            .collect();

        let mut notes = Vec::new();
        if !self.model.clock_inputs.is_empty() {
            notes.push(format!(
                "One step per clock edge, so the clock itself is abstracted away: {} has no \
                 value to set.",
                self.model.clock_inputs.join(", ")
            ));
        }
        let free_initial = self
            .model
            .state
            .iter()
            .filter(|element| element.initial.is_none())
            .count();
        if free_initial > 0 {
            notes.push(format!(
                "{free_initial} of {} state bits have no set/reset pin, and so nothing says what \
                 they power up holding; they start at 0 here.",
                self.model.state.len()
            ));
        }
        if self.looped {
            notes.push(
                "This graph has a combinational loop; the nets on it are evaluated once per step \
                 in a fixed order, the first reading the last's previous value."
                    .to_string(),
            );
        }
        if !self.model.unmodeled.is_empty() {
            notes.push(format!(
                "Not modeled, and so absent from the simulation: {}.",
                self.model.unmodeled.join(", ")
            ));
        }

        let mut simulation = Simulation {
            input_labels: self.input_labels().map(str::to_string).collect(),
            clock_inputs: self.model.clock_inputs.clone(),
            output_labels: self.outputs.iter().map(|(name, _)| name.clone()).collect(),
            state_labels: self
                .model
                .state
                .iter()
                .map(|element| element.label.clone())
                .collect(),
            frames: Vec::new(),
            notes,
        };
        for inputs in inputs_per_frame {
            state = self.push_frame(&mut simulation, state, inputs);
        }
        simulation
    }

    /// Records the frame `state` and `inputs` make, and returns the state
    /// it steps to.
    fn push_frame(
        &self,
        simulation: &mut Simulation,
        state: Vec<bool>,
        inputs: &[bool],
    ) -> Vec<bool> {
        let inputs: Vec<bool> = (0..self.model.inputs.len())
            .map(|index| inputs.get(index).copied().unwrap_or(false))
            .collect();
        let values = self.evaluate(&state, &inputs);
        let outputs = self
            .outputs
            .iter()
            .map(|&(_, net)| values[net as usize])
            .collect();
        let next = self
            .model
            .state
            .iter()
            .map(|element| eval_expr(&element.next, &values))
            .collect();
        simulation.frames.push(SimulationFrame {
            inputs,
            outputs,
            state,
        });
        next
    }

    /// Every net's value in one cycle, given the state bits and inputs
    /// it starts from. Undefined nets — a dangling pin, an unmodeled
    /// cell's output — read as `false`.
    fn evaluate(&self, state: &[bool], inputs: &[bool]) -> Vec<bool> {
        let mut values = vec![false; self.net_count];
        for (&(_, net), &value) in self.model.inputs.iter().zip(inputs) {
            values[net as usize] = value;
        }
        for (element, &value) in self.model.state.iter().zip(state) {
            values[element.q_net as usize] = value;
        }
        for &net in &self.order {
            values[net as usize] = eval_expr(&self.model.defs[&net], &values);
        }
        values
    }
}

/// `expr` over the net values in `values`, indexed by net id.
fn eval_expr(expr: &BoolExpr, values: &[bool]) -> bool {
    match expr {
        BoolExpr::Var(net) => values.get(*net as usize).copied().unwrap_or(false),
        BoolExpr::Const(value) => *value,
        BoolExpr::Not(inner) => !eval_expr(inner, values),
        BoolExpr::And(operands) => operands.iter().all(|operand| eval_expr(operand, values)),
        BoolExpr::Or(operands) => operands.iter().any(|operand| eval_expr(operand, values)),
        BoolExpr::Xor(a, b) => eval_expr(a, values) != eval_expr(b, values),
    }
}

/// Parses the raw bytes of a netlist JSON file into a [`Graph`] — for a
/// file that isn't read from a path at all, such as one the browser's file
/// picker hands over as bytes on the wasm build.
///
/// Normalizes the parsed graph with [`Graph::normalize`], so every load
/// presents the design's inputs as a single `Input` cell.
///
/// Test-only in practice: the app reads layouts.
#[cfg_attr(not(test), allow(dead_code))]
pub fn parse_graph(bytes: &[u8]) -> Result<Graph, serde_json::Error> {
    let mut graph: Graph = serde_json::from_slice(bytes)?;
    graph.normalize();
    Ok(graph)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_cell(cell_name: &str, inputs: Vec<Pin>, outputs: Vec<Pin>) -> Cell {
        standard_cell_at(cell_name, (0.0, 0.0), inputs, outputs)
    }

    fn standard_cell_at(
        cell_name: &str,
        centroid: (f64, f64),
        inputs: Vec<Pin>,
        outputs: Vec<Pin>,
    ) -> Cell {
        Cell::Sky130Standard {
            cell_id: 0,
            cell_name: cell_name.to_string(),
            centroid,
            inputs,
            outputs,
        }
    }

    fn pin(net_id: u32) -> Pin {
        (net_id, String::new())
    }

    /// Directly evaluates `expr` under `assignment`, as the ground truth
    /// the CNF encoding is checked against.
    fn eval(expr: &BoolExpr, assignment: &BTreeMap<u32, bool>) -> bool {
        match expr {
            BoolExpr::Var(net) => assignment[net],
            BoolExpr::Const(b) => *b,
            BoolExpr::Not(e) => !eval(e, assignment),
            BoolExpr::And(es) => es.iter().all(|e| eval(e, assignment)),
            BoolExpr::Or(es) => es.iter().any(|e| eval(e, assignment)),
            BoolExpr::Xor(a, b) => eval(a, assignment) ^ eval(b, assignment),
        }
    }

    /// Encodes `out == expr`, pins every net in `fixed`, solves, and
    /// returns the model over `read_back` — or `None` for unsat.
    fn solve_expr(
        out: u32,
        expr: &BoolExpr,
        fixed: &BTreeMap<u32, bool>,
        read_back: &[u32],
    ) -> Option<BTreeMap<u32, bool>> {
        let mut encoder = CnfEncoder::new(batsat::BasicSolver::default());
        encoder.assert_net_eq(out, expr);
        for (&net, &value) in fixed {
            encoder.assert_net_value(net, value);
        }
        let vars: Vec<(u32, Var)> = read_back
            .iter()
            .map(|&net| (net, encoder.net_var(net)))
            .collect();
        if encoder.solver_mut().solve_limited(&[]) != batsat::lbool::TRUE {
            return None;
        }
        let solver = encoder.solver_mut();
        Some(
            vars.into_iter()
                .map(|(net, var)| {
                    (
                        net,
                        solver.value_lit(Lit::new(var, true)) == batsat::lbool::TRUE,
                    )
                })
                .collect(),
        )
    }

    /// Every gate shape, checked over every assignment of its inputs: with
    /// the inputs pinned, the solver's value for the output net must equal
    /// what [`eval`] computes. A one-directional Tseitin encoding, or a
    /// flipped polarity, would let the solver pick the other value here.
    #[test]
    fn cnf_encoding_matches_evaluation_for_every_gate_and_assignment() {
        let v = |n| BoolExpr::Var(n);
        let exprs = vec![
            v(1),
            BoolExpr::Const(true),
            BoolExpr::Const(false),
            BoolExpr::not(v(1)),
            BoolExpr::not(BoolExpr::not(v(1))),
            BoolExpr::And(vec![v(1), v(2)]),
            BoolExpr::And(vec![v(1), v(2), v(3)]),
            BoolExpr::Or(vec![v(1), v(2)]),
            BoolExpr::Or(vec![v(1), v(2), v(3)]),
            BoolExpr::xor(v(1), v(2)),
            BoolExpr::xor(BoolExpr::xor(v(1), v(2)), v(3)),
            // Mixed nesting, negations under every operator, and constants
            // folded in where they change the result.
            BoolExpr::And(vec![
                BoolExpr::Or(vec![v(1), BoolExpr::not(v(2))]),
                BoolExpr::xor(v(2), v(3)),
                BoolExpr::not(BoolExpr::And(vec![v(1), v(3)])),
            ]),
            BoolExpr::Or(vec![
                BoolExpr::And(vec![v(1), BoolExpr::Const(false)]),
                BoolExpr::xor(BoolExpr::not(v(2)), v(3)),
            ]),
            // Degenerate vecs: identity of each operator.
            BoolExpr::And(vec![]),
            BoolExpr::Or(vec![]),
        ];

        let inputs = [1u32, 2, 3];
        let out = 99;
        for expr in &exprs {
            for bits in 0..(1u32 << inputs.len()) {
                let assignment: BTreeMap<u32, bool> = inputs
                    .iter()
                    .enumerate()
                    .map(|(i, &net)| (net, bits & (1 << i) != 0))
                    .collect();
                let expected = eval(expr, &assignment);

                let model = solve_expr(out, expr, &assignment, &[out])
                    .unwrap_or_else(|| panic!("unexpected unsat for {expr:?} at {assignment:?}"));
                assert_eq!(
                    model[&out], expected,
                    "{expr:?} under {assignment:?}: solver said {}, eval says {expected}",
                    model[&out]
                );

                // The opposite output value must be unsatisfiable — this is
                // what catches an encoding that only constrains one
                // direction and so leaves the output free.
                let mut contradiction = assignment.clone();
                contradiction.insert(out, !expected);
                assert!(
                    solve_expr(out, expr, &contradiction, &[out]).is_none(),
                    "{expr:?} under {assignment:?} should force out={expected}, \
                     but out={} was also satisfiable",
                    !expected
                );
            }
        }
    }

    /// Solving backwards — the actual use in the panel: pin the *output*
    /// and let the solver choose inputs. Every model it returns must
    /// really evaluate to the pinned value.
    #[test]
    fn cnf_encoding_solves_for_inputs_from_a_pinned_output() {
        // out = (a AND b) XOR c
        let expr = BoolExpr::xor(
            BoolExpr::And(vec![BoolExpr::Var(1), BoolExpr::Var(2)]),
            BoolExpr::Var(3),
        );
        let out = 99;
        for wanted in [false, true] {
            let fixed = BTreeMap::from([(out, wanted)]);
            let model = solve_expr(out, &expr, &fixed, &[1, 2, 3, out]).expect("satisfiable");
            assert_eq!(model[&out], wanted);
            assert_eq!(
                eval(&expr, &model),
                wanted,
                "model {model:?} does not evaluate to {wanted}"
            );
        }
    }

    /// What the action list's refresh button relies on: a `Graph` already
    /// in memory — never passed through [`parse_graph`], as one restored
    /// from saved app state is — can be normalized in place afterwards.
    #[test]
    fn normalize_merges_input_cells_on_a_graph_built_without_parsing() {
        let mut graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![named_pin(1, "A")],
                },
                standard_cell("sky130_fd_sc_hd__inv_2", vec![pin(10)], vec![pin(11)]),
                Cell::Input {
                    outputs: vec![named_pin(2, "fake_pin")],
                },
            ],
            connections: vec![conn(1, vec![10])],
        };

        graph.normalize();

        assert_eq!(graph.cells.len(), 2);
        let Cell::Input { outputs } = &graph.cells[0] else {
            panic!("the merged input cell should hold the first one's slot")
        };
        assert_eq!(
            outputs,
            &vec![(1, "A".to_string()), (2, "fake_pin".to_string())]
        );
    }

    /// A netlist splitting its inputs over several `input_cell`s must come
    /// back from a load as one `Input` cell carrying all of them, in the
    /// order the cells and their pins appeared.
    #[test]
    fn loading_collects_every_input_pin_onto_one_input_cell() {
        let json = br#"{
            "cells": [
                {"cell_type": "input_cell", "outputs": [[1, "A"], [2, "clk"]]},
                {"cell_type": "sky130_standard_cell", "cell_id": 7,
                 "cell_name": "sky130_fd_sc_hd__inv_2", "centroid": [0.0, 0.0],
                 "inputs": [[10, "A"]], "outputs": [[11, "Y"]]},
                {"cell_type": "input_cell", "outputs": [[3, "fake_pin"]]},
                {"cell_type": "output_cell", "inputs": [[20, "S"]]}
            ],
            "connections": [{"1": [10]}, {"11": [20]}]
        }"#;

        let graph = parse_graph(json).expect("fixture should parse");

        let inputs: Vec<&Vec<Pin>> = graph
            .cells
            .iter()
            .filter_map(|cell| match cell {
                Cell::Input { outputs } => Some(outputs),
                _ => None,
            })
            .collect();
        assert_eq!(inputs.len(), 1, "the two input cells should have merged");
        assert_eq!(
            inputs[0],
            &vec![
                (1, "A".to_string()),
                (2, "clk".to_string()),
                (3, "fake_pin".to_string()),
            ]
        );

        // The merged cell keeps the first input cell's slot, and no other
        // cell is disturbed.
        assert!(matches!(graph.cells[0], Cell::Input { .. }));
        assert_eq!(graph.cells.len(), 3);
        assert!(matches!(graph.cells[2], Cell::Output { .. }));
        // Connections are keyed by net, so they survive untouched.
        assert_eq!(dsts_of(&graph.connections, 1), Some(&vec![10]));
        assert_eq!(dsts_of(&graph.connections, 11), Some(&vec![20]));
    }

    /// A graph that already has one input cell is left exactly as it is —
    /// the merge must not reorder or relocate anything on the common path.
    #[test]
    fn loading_leaves_a_single_input_cell_alone() {
        let json = br#"{
            "cells": [
                {"cell_type": "sky130_standard_cell", "cell_id": 7,
                 "cell_name": "sky130_fd_sc_hd__inv_2", "centroid": [0.0, 0.0],
                 "inputs": [[10, "A"]], "outputs": [[11, "Y"]]},
                {"cell_type": "input_cell", "outputs": [[1, "A"]]}
            ],
            "connections": [{"1": [10]}]
        }"#;

        let graph = parse_graph(json).expect("fixture should parse");
        assert_eq!(graph.cells.len(), 2);
        // Still in its original, second slot.
        assert!(matches!(graph.cells[0], Cell::Sky130Standard { .. }));
        assert!(matches!(&graph.cells[1], Cell::Input { outputs } if outputs
            == &vec![(1, "A".to_string())]));
    }

    /// Two input cells declaring the same net is malformed; the first pin
    /// on that net wins rather than leaving one cell with two pins sharing
    /// a net id.
    #[test]
    fn loading_drops_a_duplicate_net_across_input_cells() {
        let json = br#"{
            "cells": [
                {"cell_type": "input_cell", "outputs": [[1, "A"]]},
                {"cell_type": "input_cell", "outputs": [[1, "A_again"], [2, "B"]]}
            ],
            "connections": []
        }"#;

        let graph = parse_graph(json).expect("fixture should parse");
        let Cell::Input { outputs } = &graph.cells[0] else {
            panic!("expected a single merged input cell")
        };
        assert_eq!(
            outputs,
            &vec![(1, "A".to_string()), (2, "B".to_string())],
            "the duplicate net 1 should be dropped, keeping the first pin"
        );
    }

    /// Two independent chains, one per named `Output` pin. Narrowing to a
    /// single pin must keep that pin's chain and prune the other outright
    /// — cells *and* the connections that fed them.
    #[test]
    fn cone_of_influence_narrowed_to_one_output_pin_prunes_the_other_chain() {
        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![named_pin(1, "A"), named_pin(2, "B")],
                },
                standard_cell("sky130_fd_sc_hd__inv_2", vec![pin(10)], vec![pin(11)]),
                standard_cell("sky130_fd_sc_hd__inv_2", vec![pin(20)], vec![pin(21)]),
                Cell::Output {
                    inputs: vec![named_pin(30, "outA"), named_pin(31, "outB")],
                },
            ],
            connections: vec![
                conn(1, vec![10]),
                conn(11, vec![30]),
                conn(2, vec![20]),
                conn(21, vec![31]),
            ],
        };

        // Unnarrowed: both chains merge, one BooleanFunction each.
        let whole = graph.merge_boolean_functions_by_cone_of_influence(&[]);
        assert_eq!(
            whole
                .cells
                .iter()
                .filter(|c| matches!(c, Cell::MergeCell { cell_name, .. } if cell_name == "BooleanFunction"))
                .count(),
            2
        );

        let narrowed = graph.merge_boolean_functions_by_cone_of_influence(&["outA".to_string()]);
        let functions: Vec<&Cell> = narrowed
            .cells
            .iter()
            .filter(|c| matches!(c, Cell::MergeCell { cell_name, .. } if cell_name == "BooleanFunction"))
            .collect();
        assert_eq!(functions.len(), 1, "only outA's chain should survive");

        // The surviving function is the one driving net 11, not net 21.
        let Cell::MergeCell {
            boolean_outputs, ..
        } = functions[0]
        else {
            unreachable!()
        };
        assert_eq!(boolean_outputs.len(), 1);
        assert_eq!(boolean_outputs[0].0, 11);

        // Net 21's gate is gone, and so is the edge that fed net 31.
        assert!(dsts_of(&narrowed.connections, 21).is_none());
        assert!(dsts_of(&narrowed.connections, 2).is_none());
        // outA's own edges survive.
        assert_eq!(dsts_of(&narrowed.connections, 1), Some(&vec![10]));
    }

    /// Naming both pins is the union of their cones — the same graph the
    /// unnarrowed merge produces, here where the two chains are
    /// everything there is.
    #[test]
    fn cone_of_influence_narrowed_to_every_output_pin_keeps_both_cones() {
        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![named_pin(1, "A"), named_pin(2, "B")],
                },
                standard_cell("sky130_fd_sc_hd__inv_2", vec![pin(10)], vec![pin(11)]),
                standard_cell("sky130_fd_sc_hd__inv_2", vec![pin(20)], vec![pin(21)]),
                Cell::Output {
                    inputs: vec![named_pin(30, "outA"), named_pin(31, "outB")],
                },
            ],
            connections: vec![
                conn(1, vec![10]),
                conn(11, vec![30]),
                conn(2, vec![20]),
                conn(21, vec![31]),
            ],
        };

        let both = graph.merge_boolean_functions_by_cone_of_influence(&[
            "outA".to_string(),
            "outB".to_string(),
        ]);
        assert_eq!(both.cells.len(), 4);
        assert_eq!(
            both.cells
                .iter()
                .filter(|c| matches!(c, Cell::MergeCell { cell_name, .. } if cell_name == "BooleanFunction"))
                .count(),
            2
        );
    }

    /// An over-constrained problem has to come back unsat, not with a
    /// model that quietly ignores one of the constraints.
    #[test]
    fn cnf_encoding_reports_unsat_for_contradictory_pins() {
        // out = a AND b, with out pinned true but a pinned false.
        let expr = BoolExpr::And(vec![BoolExpr::Var(1), BoolExpr::Var(2)]);
        let fixed = BTreeMap::from([(99, true), (1, false)]);
        assert!(solve_expr(99, &expr, &fixed, &[99]).is_none());
    }

    fn named_pin(net_id: u32, name: &str) -> Pin {
        (net_id, name.to_string())
    }

    fn conn(src: u32, dsts: Vec<u32>) -> Connection {
        Connection::from([(src, dsts)])
    }

    fn dsts_of<'a>(connections: &'a [Connection], src: u32) -> Option<&'a Vec<u32>> {
        connections.iter().find_map(|c| c.get(&src))
    }

    /// A driver fans out into a chain/tree of buffers:
    /// `driver -> buf1 -> { buf2 -> gate2, gate1 }`.
    /// Removing the buffers should leave `driver` wired directly to both
    /// `gate1` and `gate2`'s input pins.
    #[test]
    fn without_cell_names_splices_through_buffer_chains() {
        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![pin(1)],
                },
                standard_cell("sky130_fd_sc_hd__buf_2", vec![pin(2)], vec![pin(3)]),
                standard_cell("sky130_fd_sc_hd__buf_2", vec![pin(4)], vec![pin(6)]),
                standard_cell("sky130_fd_sc_hd__inv_1", vec![pin(5)], vec![pin(9)]),
                standard_cell("sky130_fd_sc_hd__inv_1", vec![pin(7)], vec![pin(10)]),
                Cell::Output {
                    inputs: vec![pin(9), pin(10)],
                },
            ],
            connections: vec![
                conn(1, vec![2]),    // driver -> buf1
                conn(3, vec![4, 5]), // buf1 -> buf2, gate1
                conn(6, vec![7]),    // buf2 -> gate2
                conn(9, vec![]),
                conn(10, vec![]),
            ],
        };

        let excluded = HashSet::from(["sky130_fd_sc_hd__buf_2"]);
        let filtered = graph
            .without_cell_names(&excluded)
            .expect("buffers are clean pass-throughs");

        assert!(filtered.cells.iter().all(|cell| !matches!(
            cell,
            Cell::Sky130Standard { cell_name, .. } if cell_name == "sky130_fd_sc_hd__buf_2"
        )));

        let mut spliced = dsts_of(&filtered.connections, 1)
            .expect("driver net should still drive something")
            .clone();
        spliced.sort_unstable();
        assert_eq!(spliced, vec![5, 7]);

        // The now-nonexistent buffer nets shouldn't appear anywhere.
        for net in [2, 3, 4, 6] {
            assert!(dsts_of(&filtered.connections, net).is_none());
        }
    }

    /// A removed cell that isn't a clean single-input/single-output
    /// pass-through has no unambiguous splice, so the whole call errors out
    /// instead of silently dropping or mis-wiring its connections.
    #[test]
    fn without_cell_names_errors_on_non_passthrough_shaped_removed_cells() {
        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![pin(1), pin(2)],
                },
                standard_cell(
                    "sky130_fd_sc_hd__weird_3",
                    vec![pin(10), pin(11)],
                    vec![pin(12)],
                ),
                standard_cell("sky130_fd_sc_hd__inv_1", vec![pin(20)], vec![pin(21)]),
                Cell::Output {
                    inputs: vec![pin(21)],
                },
            ],
            connections: vec![
                conn(1, vec![10]),  // driver -> weird's first input
                conn(2, vec![11]),  // driver -> weird's second input
                conn(12, vec![20]), // weird -> kept gate
            ],
        };

        let excluded = HashSet::from(["sky130_fd_sc_hd__weird_3"]);
        let err = graph
            .without_cell_names(&excluded)
            .expect_err("weird_3 has two inputs, so its removal is ambiguous");

        assert_eq!(err.cell_name, "sky130_fd_sc_hd__weird_3");
        assert_eq!(err.input_count, 2);
        assert_eq!(err.output_count, 1);
    }

    /// A `mux2_1` whose `X` feeds a `dfrtp_2`'s `D`, whose `Q` loops back
    /// into that same mux's `A0` (and also drives an external gate), gets
    /// folded into one `MuxedResetableFlipflop` `MergeCell` at the
    /// flipflop's centroid, with every other pin repatched onto it under
    /// its original net id and the internal loop gone.
    #[test]
    fn merge_muxed_resetable_flipflops_merges_matching_pairs() {
        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![pin(1), pin(2), pin(5), pin(7)],
                },
                standard_cell(
                    "sky130_fd_sc_hd__mux2_1",
                    vec![named_pin(1, "S"), named_pin(2, "A1"), named_pin(3, "A0")],
                    vec![named_pin(4, "X")],
                ),
                standard_cell_at(
                    "sky130_fd_sc_hd__dfrtp_2",
                    (9.0, 9.0),
                    vec![
                        named_pin(5, "RESET_B"),
                        named_pin(6, "D"),
                        named_pin(7, "CLK"),
                    ],
                    vec![named_pin(8, "Q")],
                ),
                standard_cell("sky130_fd_sc_hd__inv_1", vec![pin(20)], vec![pin(21)]),
                Cell::Output {
                    inputs: vec![pin(21)],
                },
            ],
            connections: vec![
                conn(1, vec![]),
                conn(2, vec![]),
                conn(5, vec![]),
                conn(7, vec![]),
                conn(4, vec![6]),     // mux.X -> dff.D
                conn(8, vec![3, 20]), // dff.Q -> mux.A0 (loop) and an external gate
                conn(21, vec![]),
            ],
        };

        let merged = graph.merge_muxed_resetable_flipflops();

        let merge_cell = merged
            .cells
            .iter()
            .find(|cell| matches!(cell, Cell::MergeCell { .. }))
            .expect("mux/dff pair should have been merged");

        let Cell::MergeCell {
            cell_name,
            centroid,
            inputs,
            outputs,
            ancestor_cells,
            ..
        } = merge_cell
        else {
            unreachable!()
        };

        assert_eq!(cell_name, "MuxedResetableFlipflop");
        // Centroid comes from the flipflop, not the mux.
        assert_eq!(*centroid, (9.0, 9.0));
        assert_eq!(ancestor_cells.len(), 2);

        let mut input_nets: Vec<u32> = inputs.iter().map(|&(net, _)| net).collect();
        input_nets.sort_unstable();
        // S, A1 (mux) + RESET_B, CLK (dff); A0 and D are now internal.
        assert_eq!(input_nets, vec![1, 2, 5, 7]);

        let output_nets: Vec<u32> = outputs.iter().map(|&(net, _)| net).collect();
        // Q stays exposed; X is now internal.
        assert_eq!(output_nets, vec![8]);

        // The standalone mux and dff cells are gone.
        assert!(!merged.cells.iter().any(|cell| matches!(
            cell,
            Cell::Sky130Standard { cell_name, .. }
                if cell_name == "sky130_fd_sc_hd__mux2_1" || cell_name == "sky130_fd_sc_hd__dfrtp_2"
        )));

        // The internal X->D link is gone entirely...
        assert!(dsts_of(&merged.connections, 4).is_none());
        // ...and Q's connection lost the now-internal A0 destination but
        // kept its real external fanout.
        assert_eq!(dsts_of(&merged.connections, 8), Some(&vec![20]));
    }

    fn stage_cell(centroid: (f64, f64), s: u32, a1: u32, reset_b: u32, clk: u32, q: u32) -> Cell {
        Cell::MergeCell {
            cell_name: "MuxedResetableFlipflop".to_string(),
            centroid,
            inputs: vec![
                named_pin(s, "S"),
                named_pin(a1, "A1"),
                named_pin(reset_b, "RESET_B"),
                named_pin(clk, "CLK"),
            ],
            outputs: vec![named_pin(q, "Q")],
            ancestor_cells: Vec::new(),
            boolean_outputs: Vec::new(),
        }
    }

    /// Three `MuxedResetableFlipflop` stages chained `Q`->`A1`, each also
    /// tapped externally, plus an unrelated fourth stage with no chain
    /// link. The chain should fold into one `ShiftRegister` with all three
    /// stages' `Q` outputs kept (in order) and every `A1` but the first
    /// stage's internalized; the unrelated stage is untouched.
    #[test]
    fn merge_shift_registers_merges_the_longest_chain() {
        let graph = Graph {
            cells: vec![
                stage_cell((0.0, 0.0), 1, 2, 3, 4, 5),
                stage_cell((2.0, 0.0), 6, 7, 8, 9, 10),
                stage_cell((4.0, 0.0), 11, 12, 13, 14, 15),
                stage_cell((10.0, 10.0), 16, 17, 18, 19, 20),
            ],
            connections: vec![
                conn(5, vec![7, 100]),     // stage1.Q -> stage2.A1, plus a tap
                conn(10, vec![12, 101]),   // stage2.Q -> stage3.A1, plus a tap
                conn(15, vec![102]),       // stage3.Q -> external tap only
                conn(20, vec![103]),       // unrelated stage's own tap
                conn(200, vec![1, 6, 11]), // shared S driver, feeds all 3 stages
                conn(201, vec![3, 8, 13]), // shared RESET_B driver
                conn(202, vec![4, 9, 14]), // shared CLK driver
            ],
        };

        let merged = graph.merge_shift_registers();

        let shift_registers: Vec<&Cell> = merged
            .cells
            .iter()
            .filter(|cell| {
                matches!(cell, Cell::MergeCell { cell_name, .. } if cell_name == "ShiftRegister")
            })
            .collect();
        assert_eq!(shift_registers.len(), 1, "exactly one chain should merge");

        let Cell::MergeCell {
            centroid,
            inputs,
            outputs,
            ancestor_cells,
            ..
        } = shift_registers[0]
        else {
            unreachable!()
        };

        assert_eq!(*centroid, (2.0, 0.0)); // average of (0,0), (2,0), (4,0)
        assert_eq!(ancestor_cells.len(), 3);

        let mut input_nets: Vec<u32> = inputs.iter().map(|&(net, _)| net).collect();
        input_nets.sort_unstable();
        // Just stage 1's S/A1/RESET_B/CLK: stages 2 & 3's A1 is now internal
        // (fed by the previous stage's Q) and their S/RESET_B/CLK collapsed
        // onto stage 1's.
        assert_eq!(input_nets, vec![1, 2, 3, 4]);

        // Stage 1's `A1` is the register's serial data input, exposed as `A`.
        let input_names: Vec<&str> = inputs.iter().map(|(_, name)| name.as_str()).collect();
        assert_eq!(input_names, vec!["S", "A", "RESET_B", "CLK"]);

        // Every stage's Q is kept, in order, renumbered Q0..Q2.
        assert_eq!(
            outputs,
            &vec![
                (5, "Q0".to_string()),
                (10, "Q1".to_string()),
                (15, "Q2".to_string()),
            ]
        );

        // The unrelated fourth stage is untouched.
        let untouched: Vec<&Cell> = merged
            .cells
            .iter()
            .filter(|cell| {
                matches!(cell, Cell::MergeCell { cell_name, .. } if cell_name == "MuxedResetableFlipflop")
            })
            .collect();
        assert_eq!(untouched.len(), 1);

        // Chain links (7, 12) are internalized; each Q kept its real
        // external tap.
        assert_eq!(dsts_of(&merged.connections, 5), Some(&vec![100]));
        assert_eq!(dsts_of(&merged.connections, 10), Some(&vec![101]));
        assert_eq!(dsts_of(&merged.connections, 15), Some(&vec![102]));
        assert_eq!(dsts_of(&merged.connections, 20), Some(&vec![103]));

        // The shared control-signal drivers now each fan out to just the
        // one surviving (stage 1's) pin, deduplicated rather than repeated
        // three times.
        assert_eq!(dsts_of(&merged.connections, 200), Some(&vec![1]));
        assert_eq!(dsts_of(&merged.connections, 201), Some(&vec![3]));
        assert_eq!(dsts_of(&merged.connections, 202), Some(&vec![4]));
    }

    /// Two AND2 gates chained (`g1.X` -> `g2.A`, `g2.X` -> `g3.A`... see
    /// below) should compose into one `BooleanFunction` whose single output
    /// is expressed purely in terms of the boundary inputs, with every
    /// internal net gone from `connections`.
    #[test]
    fn merge_boolean_functions_composes_a_connected_chain() {
        fn gate(cell_id: u64, cell_name: &str, inputs: Vec<Pin>, outputs: Vec<Pin>) -> Cell {
            Cell::Sky130Standard {
                cell_id,
                cell_name: cell_name.to_string(),
                centroid: (0.0, 0.0),
                inputs,
                outputs,
            }
        }

        let graph = Graph {
            cells: vec![
                // g1: X1 = A1 & B1
                gate(
                    1,
                    "sky130_fd_sc_hd__and2_2",
                    vec![named_pin(1, "A"), named_pin(2, "B")],
                    vec![named_pin(3, "X")],
                ),
                // g2: Y2 = !(A2 | B2), A2 fed by g1's X1
                gate(
                    2,
                    "sky130_fd_sc_hd__nor2_2",
                    vec![named_pin(4, "A"), named_pin(5, "B")],
                    vec![named_pin(6, "Y")],
                ),
                // An unrelated inverter, its own separate component.
                gate(
                    3,
                    "sky130_fd_sc_hd__inv_2",
                    vec![named_pin(7, "A")],
                    vec![named_pin(8, "Y")],
                ),
            ],
            connections: vec![
                conn(3, vec![4]), // g1.X -> g2.A
                conn(1, vec![]),
                conn(2, vec![]),
                conn(5, vec![]),
                conn(6, vec![100]), // g2.Y -> external tap
                conn(7, vec![]),
                conn(8, vec![101]),
            ],
        };

        let merged = graph.merge_boolean_functions();

        let functions: Vec<&Cell> = merged
            .cells
            .iter()
            .filter(|cell| {
                matches!(cell, Cell::MergeCell { cell_name, .. } if cell_name == "BooleanFunction")
            })
            .collect();
        assert_eq!(functions.len(), 2, "the chain and the lone inverter");

        let chain = functions
            .iter()
            .find(|cell| matches!(cell, Cell::MergeCell { ancestor_cells, .. } if ancestor_cells.len() == 2))
            .expect("g1+g2 should have merged into one BooleanFunction");

        let Cell::MergeCell {
            inputs,
            outputs,
            boolean_outputs,
            ..
        } = chain
        else {
            unreachable!()
        };

        let mut input_nets: Vec<u32> = inputs.iter().map(|&(net, _)| net).collect();
        input_nets.sort_unstable();
        assert_eq!(input_nets, vec![1, 2, 5]); // g1.A, g1.B, g2.B; g2.A (4) is internal

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0], (6, "X0".to_string())); // g2.Y, numbered rather than named after it

        assert_eq!(boolean_outputs.len(), 1);
        assert_eq!(boolean_outputs[0].0, 6);
        // Y = !((A1 & B1) | B2), fully inlined in terms of boundary inputs.
        assert_eq!(
            boolean_outputs[0].1.to_smtlib(),
            "(not (or (and |net_1| |net_2|) |net_5|))"
        );

        // The internal A2 (net 4) link is gone entirely; g2.Y kept its
        // external tap.
        assert!(dsts_of(&merged.connections, 3).is_none());
        assert_eq!(dsts_of(&merged.connections, 6), Some(&vec![100]));
    }

    /// Two gates in the *same* connected component both reading the same
    /// external driver (net 100, fanning out to both `g2.B` and `g3.B`)
    /// must collapse to one shared boundary input, not two — and
    /// `connections` must still route the driver to whichever net id
    /// survived as that boundary input's own pin, or the graph view has no
    /// pin left for the driver's edge to land on.
    #[test]
    fn merge_boolean_functions_dedupes_a_shared_external_driver() {
        fn gate(cell_id: u64, cell_name: &str, inputs: Vec<Pin>, outputs: Vec<Pin>) -> Cell {
            Cell::Sky130Standard {
                cell_id,
                cell_name: cell_name.to_string(),
                centroid: (0.0, 0.0),
                inputs,
                outputs,
            }
        }

        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![named_pin(100, "OUT")],
                },
                // g1: X = A
                gate(
                    1,
                    "sky130_fd_sc_hd__buf_2",
                    vec![named_pin(1, "A")],
                    vec![named_pin(2, "X")],
                ),
                // g2: X = A & B, A fed by g1, B fed by the shared driver
                gate(
                    2,
                    "sky130_fd_sc_hd__and2_2",
                    vec![named_pin(3, "A"), named_pin(4, "B")],
                    vec![named_pin(7, "X")],
                ),
                // g3: X = A | B, A fed by g2, B fed by the *same* shared driver
                gate(
                    3,
                    "sky130_fd_sc_hd__or2_2",
                    vec![named_pin(5, "A"), named_pin(6, "B")],
                    vec![named_pin(8, "X")],
                ),
            ],
            connections: vec![
                conn(2, vec![3]),      // g1.X -> g2.A
                conn(100, vec![4, 6]), // shared driver -> g2.B AND g3.B
                conn(7, vec![5, 300]), // g2.X -> g3.A, plus an external tap
                conn(8, vec![301]),    // g3.X -> external tap
            ],
        };

        let merged = graph.merge_boolean_functions();

        let function = merged
            .cells
            .iter()
            .find(|cell| {
                matches!(cell, Cell::MergeCell { cell_name, .. } if cell_name == "BooleanFunction")
            })
            .expect("g1+g2+g3 should have merged into one BooleanFunction");

        let Cell::MergeCell {
            inputs,
            outputs,
            boolean_outputs,
            ancestor_cells,
            ..
        } = function
        else {
            unreachable!()
        };

        assert_eq!(ancestor_cells.len(), 3);

        // Exactly one pin for the shared driver, not one per consuming gate.
        assert_eq!(inputs.len(), 2, "got: {inputs:?}");
        let shared_input = inputs
            .iter()
            .find(|(_, name)| name == "Input#100.OUT")
            .expect("the shared driver should appear exactly once");
        // g2 (which reaches net 4 first, since it's processed before g3 in
        // cell-index order) wins as the representative.
        assert_eq!(shared_input.0, 4);

        // `connections` must route the driver straight to that surviving
        // pin's net id, deduplicated (not to the other, now-nonexistent
        // consumer net 6).
        assert_eq!(dsts_of(&merged.connections, 100), Some(&vec![4]));

        // Both outputs stay numbered X0/X1, and their composed expressions
        // reference the shared input (net 4) — never the discarded
        // duplicate net 6, nor any internal net (3 or 5).
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].1, "X0");
        assert_eq!(outputs[1].1, "X1");
        for (_, expr) in boolean_outputs {
            let smt = expr.to_smtlib();
            assert!(smt.contains("|net_4|"));
            for stale in ["|net_3|", "|net_5|", "|net_6|"] {
                assert!(!smt.contains(stale), "{smt} should not reference {stale}");
            }
        }
    }

    /// The `sky130` name families each land in the category their role
    /// implies — including the ones the name alone makes easy to confuse:
    /// a `dly*` delay cell is combinational despite sharing its `dl`
    /// prefix with a latch, a `clk*` buffer is clock network rather than
    /// the `Boolean` its buffer/inverter function would suggest, and a
    /// family this classification doesn't cover is `Other` rather than
    /// guessed at.
    #[test]
    fn cell_categories_follow_the_sky130_name_families() {
        for name in [
            "sky130_fd_sc_hd__dfrtp_2",
            "sky130_fd_sc_hd__dfxtp_1",
            "sky130_fd_sc_hd__sdfrtp_1",
            "sky130_fd_sc_hd__edfxtp_1",
            "sky130_fd_sc_hd__dlxtp_1",
            "sky130_fd_sc_hd__dlrtp_1",
        ] {
            assert_eq!(sky130_cell_category(name), CellCategory::State, "{name}");
        }

        for name in [
            "sky130_fd_sc_hd__buf_2",
            "sky130_fd_sc_hd__inv_2",
            "sky130_fd_sc_hd__nand2_2",
            "sky130_fd_sc_hd__nor2_2",
            "sky130_fd_sc_hd__xor2_2",
            "sky130_fd_sc_hd__mux2_1",
            "sky130_fd_sc_hd__a21boi_2",
            "sky130_fd_sc_hd__conb_1",
        ] {
            assert_eq!(sky130_cell_category(name), CellCategory::Boolean, "{name}");
        }

        for name in [
            // The clock network: buffers and inverters like any other by
            // function, but carrying the clock, not computing with it.
            "sky130_fd_sc_hd__clkbuf_16",
            "sky130_fd_sc_hd__clkinv_1",
            "sky130_fd_sc_hd__clkdlybuf4s25_1",
            // Combinational, but not a family `gate_output_exprs` writes
            // an expression for, so not claimed as `Boolean` either.
            "sky130_fd_sc_hd__dlygate4sd3_1",
            "sky130_fd_sc_hd__tapvpwrvgnd_1",
            "not_a_sky130_cell",
        ] {
            assert_eq!(sky130_cell_category(name), CellCategory::Other, "{name}");
        }
    }

    /// Seeding `"A"` at `g1`'s input floods forward through a `g1 -> g2 ->
    /// Output` chain, appending `" = A"` to every pin it reaches along the
    /// way (including the originating Input pin itself), leaving unrelated
    /// pins untouched.
    #[test]
    fn propagate_pin_values_floods_forward_along_a_chain() {
        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![named_pin(1, "A")],
                },
                standard_cell(
                    "sky130_fd_sc_hd__buf_2",
                    vec![named_pin(10, "IN")],
                    vec![named_pin(11, "X")],
                ),
                standard_cell(
                    "sky130_fd_sc_hd__buf_2",
                    vec![named_pin(12, "IN2")],
                    vec![named_pin(13, "X2")],
                ),
                Cell::Output {
                    inputs: vec![named_pin(14, "OUT")],
                },
            ],
            connections: vec![
                conn(1, vec![10]),  // Input.A -> g1.IN, the entry's target
                conn(11, vec![12]), // g1.X -> g2.IN2
                conn(13, vec![14]), // g2.X2 -> Output
            ],
        };

        let entries = vec![("A".to_string(), "IN".to_string())];
        let propagated = graph.propagate_pin_values(&entries);

        let pin_name = |net: u32| -> String {
            propagated
                .cells
                .iter()
                .flat_map(|cell| cell.inputs().iter().chain(cell.outputs()))
                .find(|(n, _)| *n == net)
                .map(|(_, name)| name.clone())
                .unwrap_or_else(|| panic!("no pin for net {net}"))
        };

        assert_eq!(pin_name(1), "A = A"); // the Input's own pin
        assert_eq!(pin_name(10), "IN = A");
        assert_eq!(pin_name(11), "X = A");
        assert_eq!(pin_name(12), "IN2 = A");
        assert_eq!(pin_name(13), "X2 = A");
        assert_eq!(pin_name(14), "OUT = A");

        // Cell count/types are unchanged — this action only relabels pins
        // — and so is every edge: the graph handed downstream is still the
        // whole circuit, whatever the view hides.
        assert_eq!(propagated.cells.len(), graph.cells.len());
        assert_eq!(propagated.connections, graph.connections);

        // In the *view*, though, all three edges carried the value end to
        // end (both sides labeled the same), so all three are hidden: the
        // matching labels at each pin already show it, decluttering what
        // would otherwise be a lot of redundant wires.
        let viewed = graph.propagate_pin_values_for_view(&entries);
        assert!(dsts_of(&viewed.connections, 1).unwrap().is_empty());
        assert!(dsts_of(&viewed.connections, 11).unwrap().is_empty());
        assert!(dsts_of(&viewed.connections, 13).unwrap().is_empty());
    }

    /// An edge whose value didn't cross it — here, because propagation
    /// stopped at a multi-input cell before ever reaching the far side —
    /// is left alone rather than removed.
    #[test]
    fn propagate_pin_values_keeps_edges_the_value_did_not_cross() {
        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![named_pin(1, "A")],
                },
                Cell::Input {
                    outputs: vec![named_pin(2, "B")],
                },
                // Multi-input: IN1 gets labeled, but X does not (and so
                // neither does IN2, since nothing ever seeds it).
                standard_cell(
                    "sky130_fd_sc_hd__and2_2",
                    vec![named_pin(10, "IN1"), named_pin(11, "IN2")],
                    vec![named_pin(12, "X")],
                ),
                // A downstream buffer, wired to the gate's (unlabeled)
                // output — just here so there's a real edge to check.
                standard_cell(
                    "sky130_fd_sc_hd__buf_2",
                    vec![named_pin(13, "IN")],
                    vec![named_pin(14, "X")],
                ),
            ],
            connections: vec![
                conn(1, vec![10]),  // Input.A -> gate.IN1, the entry's target
                conn(12, vec![13]), // gate.X -> buf.IN, but X is never labeled
            ],
        };

        let entries = vec![("A".to_string(), "IN1".to_string())];
        let viewed = graph.propagate_pin_values_for_view(&entries);

        // net 12 (gate.X) never got a value (multi-input, only IN1 was
        // seeded), so the edge 12 -> 13 has nothing to declutter: even the
        // view keeps it exactly as it was.
        assert_eq!(dsts_of(&viewed.connections, 12), Some(&vec![13]));
    }

    /// Two different values reaching the same net (fed by two different
    /// injection points into the same downstream gate) leaves that net —
    /// and anything past it — unlabeled, rather than picking one
    /// arbitrarily.
    #[test]
    fn propagate_pin_values_stops_at_a_conflict() {
        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![named_pin(1, "A")],
                },
                Cell::Input {
                    outputs: vec![named_pin(2, "B")],
                },
                // g: X = IN1 & IN2, one fed by each Input cell.
                standard_cell(
                    "sky130_fd_sc_hd__and2_2",
                    vec![named_pin(10, "IN1"), named_pin(11, "IN2")],
                    vec![named_pin(12, "X")],
                ),
            ],
            connections: vec![
                conn(1, vec![10]), // Input.A -> g.IN1
                conn(2, vec![11]), // Input.B -> g.IN2
            ],
        };

        let entries = vec![
            ("A".to_string(), "IN1".to_string()),
            ("B".to_string(), "IN2".to_string()),
        ];
        let propagated = graph.propagate_pin_values(&entries);

        let gate = propagated
            .cells
            .iter()
            .find(|cell| matches!(cell, Cell::Sky130Standard { .. }))
            .unwrap();
        assert_eq!(gate.inputs()[0].1, "IN1 = A");
        assert_eq!(gate.inputs()[1].1, "IN2 = B");
        // IN1 and IN2 carry different values into the same gate, so its
        // output is ambiguous and stays unlabeled.
        assert_eq!(gate.outputs()[0].1, "X");
    }

    /// Regression for the reported bug: an entry's target pin label only
    /// matches pins the named Input is *wired to*. A cell the input
    /// reaches on its `A` pin must not get its unrelated `B` pin labeled
    /// by an entry targeting `B`, and neither must a same-named `B` pin on
    /// a cell the input doesn't reach at all.
    #[test]
    fn propagate_pin_values_only_seeds_pins_wired_to_the_input() {
        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![named_pin(1, "I")],
                },
                // The input lands on this gate's A pin, not its B pin.
                standard_cell(
                    "sky130_fd_sc_hd__and2_2",
                    vec![named_pin(10, "A"), named_pin(11, "B")],
                    vec![named_pin(12, "X")],
                ),
                // A B pin the input never reaches.
                standard_cell(
                    "sky130_fd_sc_hd__and2_2",
                    vec![named_pin(20, "A"), named_pin(21, "B")],
                    vec![named_pin(22, "X")],
                ),
            ],
            connections: vec![conn(1, vec![10])],
        };

        let entries = vec![("I".to_string(), "B".to_string())];
        let propagated = graph.propagate_pin_values(&entries);

        // The Input's own pin carries its value, but nothing else does:
        // no pin named `B` is wired to it, and the `A` pin it does drive
        // isn't what the entry asked for.
        assert_eq!(propagated.cells[0].outputs()[0].1, "I = I");
        for cell in &propagated.cells[1..] {
            for (_, name) in cell.inputs().iter().chain(cell.outputs()) {
                assert!(!name.contains(" = "), "unexpectedly labeled pin: {name:?}");
            }
        }
        // Nothing crossed the input's edge, so even the view keeps it.
        let viewed = graph.propagate_pin_values_for_view(&entries);
        assert_eq!(dsts_of(&viewed.connections, 1), Some(&vec![10]));

        // Targeting the pin it really is wired to does seed it (and, this
        // being a multi-input gate, stops there).
        let entries = vec![("I".to_string(), "A".to_string())];
        let propagated = graph.propagate_pin_values(&entries);
        let gate = &propagated.cells[1];
        assert_eq!(gate.inputs()[0].1, "A = I");
        assert_eq!(gate.inputs()[1].1, "B");
        assert_eq!(gate.outputs()[0].1, "X");
    }

    /// An entry naming a pin that doesn't exist anywhere is silently
    /// ignored — this action never errors.
    #[test]
    fn propagate_pin_values_ignores_unmatched_entries() {
        let graph = Graph {
            cells: vec![Cell::Input {
                outputs: vec![named_pin(1, "A")],
            }],
            connections: vec![],
        };

        let entries = vec![("A".to_string(), "nonexistent".to_string())];
        let propagated = graph.propagate_pin_values(&entries);

        // The Input's own pin still gets seeded; nothing else changes.
        assert_eq!(propagated.cells[0].outputs()[0].1, "A = A");
    }

    /// An LFSR-style loop: `g1` (AND2) feeds `g2` (BUF) directly, and
    /// `g2`'s output drives an external register (`dfrtp_2`, not a
    /// recognized gate) whose `Q` feeds *back* into `g1`'s other input.
    /// This kind of register-mediated feedback is completely normal in
    /// synchronous hardware, and — since the register itself can never be
    /// part of a `BooleanFunction` — is not something any split could ever
    /// resolve anyway (see the doc comment on `merge_boolean_functions`):
    /// `g1` and `g2` should merge together exactly as any other directly
    /// connected pair of gates would.
    #[test]
    fn merge_boolean_functions_keeps_a_register_mediated_loop_merged() {
        fn gate(cell_id: u64, cell_name: &str, inputs: Vec<Pin>, outputs: Vec<Pin>) -> Cell {
            Cell::Sky130Standard {
                cell_id,
                cell_name: cell_name.to_string(),
                centroid: (0.0, 0.0),
                inputs,
                outputs,
            }
        }

        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![named_pin(1, "A")],
                },
                // g1: X = A & B
                gate(
                    1,
                    "sky130_fd_sc_hd__and2_2",
                    vec![named_pin(1, "A"), named_pin(2, "B")],
                    vec![named_pin(3, "X")],
                ),
                // g2: X2 = IN (buf), directly fed by g1 — a normal edge.
                gate(
                    2,
                    "sky130_fd_sc_hd__buf_2",
                    vec![named_pin(4, "IN")],
                    vec![named_pin(5, "X2")],
                ),
                // The register: not a recognized gate, so it never gets
                // unioned with anything — but it does structurally close
                // the loop back into g1.B.
                gate(
                    3,
                    "sky130_fd_sc_hd__dfrtp_2",
                    vec![named_pin(6, "D")],
                    vec![named_pin(7, "Q")],
                ),
            ],
            connections: vec![
                conn(3, vec![4]), // g1.X -> g2.IN
                conn(5, vec![6]), // g2.X2 -> reg.D
                conn(7, vec![2]), // reg.Q -> g1.B  (closes the loop)
            ],
        };

        let merged = graph.merge_boolean_functions();

        let functions: Vec<&Cell> = merged
            .cells
            .iter()
            .filter(|cell| {
                matches!(cell, Cell::MergeCell { cell_name, .. } if cell_name == "BooleanFunction")
            })
            .collect();

        assert_eq!(functions.len(), 1, "g1 and g2 should merge as usual");
        let Cell::MergeCell { ancestor_cells, .. } = functions[0] else {
            unreachable!()
        };
        assert_eq!(ancestor_cells.len(), 2);
    }

    /// A genuine combinational cycle with *no* register anywhere in it —
    /// `g1`'s output feeds `g2`'s input and `g2`'s output feeds back into
    /// `g1`'s input directly. This is invalid hardware (a real
    /// synthesizable netlist can't have it), but merging both gates
    /// together — rather than either gate individually, or refusing to
    /// merge at all — is the one grouping that actually makes the loop
    /// fully internal, with nothing external left to keep it open.
    #[test]
    fn merge_boolean_functions_merges_a_genuine_combinational_cycle() {
        fn gate(cell_id: u64, cell_name: &str, inputs: Vec<Pin>, outputs: Vec<Pin>) -> Cell {
            Cell::Sky130Standard {
                cell_id,
                cell_name: cell_name.to_string(),
                centroid: (0.0, 0.0),
                inputs,
                outputs,
            }
        }

        let graph = Graph {
            cells: vec![
                gate(
                    1,
                    "sky130_fd_sc_hd__buf_2",
                    vec![named_pin(1, "A")],
                    vec![named_pin(2, "X")],
                ),
                gate(
                    2,
                    "sky130_fd_sc_hd__buf_2",
                    vec![named_pin(3, "A")],
                    vec![named_pin(4, "X")],
                ),
            ],
            connections: vec![
                conn(2, vec![3]), // g1.X -> g2.A
                conn(4, vec![1]), // g2.X -> g1.A  (closes the loop, no register)
            ],
        };

        // Must terminate promptly and merge the pair into one function,
        // rather than hang or refuse to compose either gate's expression.
        let merged = graph.merge_boolean_functions();

        let functions: Vec<&Cell> = merged
            .cells
            .iter()
            .filter(|cell| {
                matches!(cell, Cell::MergeCell { cell_name, .. } if cell_name == "BooleanFunction")
            })
            .collect();
        assert_eq!(functions.len(), 1);
        let Cell::MergeCell { ancestor_cells, .. } = functions[0] else {
            unreachable!()
        };
        assert_eq!(ancestor_cells.len(), 2);
    }

    /// A pure combinational loop has no state element on it, so both its
    /// gates reach exactly the same destinations and are mutually
    /// reachable — the register-SCC grouping has to keep them in one cell
    /// just as [`Graph::merge_boolean_functions`] does, or neither gate's
    /// expression could be composed at all.
    #[test]
    fn merge_boolean_functions_by_register_scc_keeps_a_combinational_cycle_whole() {
        fn gate(cell_id: u64, cell_name: &str, inputs: Vec<Pin>, outputs: Vec<Pin>) -> Cell {
            Cell::Sky130Standard {
                cell_id,
                cell_name: cell_name.to_string(),
                centroid: (0.0, 0.0),
                inputs,
                outputs,
            }
        }

        let graph = Graph {
            cells: vec![
                gate(
                    1,
                    "sky130_fd_sc_hd__buf_2",
                    vec![named_pin(1, "A")],
                    vec![named_pin(2, "X")],
                ),
                gate(
                    2,
                    "sky130_fd_sc_hd__buf_2",
                    vec![named_pin(3, "A")],
                    vec![named_pin(4, "X")],
                ),
            ],
            connections: vec![conn(2, vec![3]), conn(4, vec![1])],
        };

        let merged = graph.merge_boolean_functions_by_register_scc();

        let functions: Vec<&Cell> = merged
            .cells
            .iter()
            .filter(|cell| {
                matches!(cell, Cell::MergeCell { cell_name, .. } if cell_name == "BooleanFunction")
            })
            .collect();
        assert_eq!(functions.len(), 1);
        let Cell::MergeCell { ancestor_cells, .. } = functions[0] else {
            unreachable!()
        };
        assert_eq!(ancestor_cells.len(), 2);
    }

    /// Two independent chains, each feeding its own flip-flop from that
    /// same flip-flop's output, are two separate feedback loops — one cell
    /// each — even though [`Graph::merge_boolean_functions`] would also
    /// keep them apart here (they aren't gate-connected). What this pins
    /// down is the extra split: a third gate feeding *both* chains is
    /// shared control logic and becomes its own third cell rather than
    /// dragging the two loops into one.
    #[test]
    fn merge_boolean_functions_by_register_scc_splits_shared_logic_into_its_own_cell() {
        fn gate(cell_id: u64, cell_name: &str, inputs: Vec<Pin>, outputs: Vec<Pin>) -> Cell {
            Cell::Sky130Standard {
                cell_id,
                cell_name: cell_name.to_string(),
                centroid: (0.0, 0.0),
                inputs,
                outputs,
            }
        }
        fn flipflop(cell_id: u64, d: u32, q: u32) -> Cell {
            Cell::Sky130Standard {
                cell_id,
                cell_name: "sky130_fd_sc_hd__dfxtp_2".to_string(),
                centroid: (0.0, 0.0),
                inputs: vec![named_pin(d, "D"), named_pin(d + 100, "CLK")],
                outputs: vec![named_pin(q, "Q")],
            }
        }

        // shared: an inverter off the primary input, feeding both loops.
        // loop A: ff1.Q -> and(ff1.Q, shared) -> ff1.D
        // loop B: ff2.Q -> and(ff2.Q, shared) -> ff2.D
        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![named_pin(1, "IN")],
                },
                gate(
                    10,
                    "sky130_fd_sc_hd__inv_2",
                    vec![named_pin(2, "A")],
                    vec![named_pin(3, "Y")],
                ),
                gate(
                    11,
                    "sky130_fd_sc_hd__and2_2",
                    vec![named_pin(4, "A"), named_pin(5, "B")],
                    vec![named_pin(6, "X")],
                ),
                gate(
                    12,
                    "sky130_fd_sc_hd__and2_2",
                    vec![named_pin(7, "A"), named_pin(8, "B")],
                    vec![named_pin(9, "X")],
                ),
                flipflop(20, 10, 12),
                flipflop(21, 11, 13),
            ],
            connections: vec![
                conn(1, vec![2]),    // IN -> inv.A
                conn(3, vec![5, 8]), // inv.Y -> both ands' B
                conn(12, vec![4]),   // ff1.Q -> and1.A
                conn(13, vec![7]),   // ff2.Q -> and2.A
                conn(6, vec![10]),   // and1.X -> ff1.D
                conn(9, vec![11]),   // and2.X -> ff2.D
            ],
        };

        // Gate-connectivity alone puts all three gates in one cell, since
        // the inverter touches both.
        let by_connectivity = graph.merge_boolean_functions();
        assert_eq!(
            by_connectivity
                .cells
                .iter()
                .filter(|cell| matches!(cell, Cell::MergeCell { cell_name, .. } if cell_name == "BooleanFunction"))
                .count(),
            1
        );

        let merged = graph.merge_boolean_functions_by_register_scc();
        let mut sizes: Vec<usize> = merged
            .cells
            .iter()
            .filter_map(|cell| match cell {
                Cell::MergeCell {
                    cell_name,
                    ancestor_cells,
                    ..
                } if cell_name == "BooleanFunction" => Some(ancestor_cells.len()),
                _ => None,
            })
            .collect();
        sizes.sort_unstable();
        // One cell per loop, plus the shared inverter on its own.
        assert_eq!(sizes, vec![1, 1, 1]);
    }

    /// Cone-of-influence grouping splits where register-SCC grouping
    /// can't: two cross-coupled flip-flops (each feeding logic that feeds
    /// the other) are one feedback component, so `RegisterScc` puts all
    /// three gates feeding them in one cell. Keyed on the *reached
    /// flip-flops themselves*, the shared gate (which influences both),
    /// the one feeding only `ff1` and the one feeding only `ff2` are three
    /// different cones, so they become three cells — with the shared one
    /// belonging to both flip-flops' cones at once.
    #[test]
    fn cone_of_influence_splits_what_register_scc_keeps_together() {
        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![named_pin(1, "a"), named_pin(2, "b")],
                },
                // Shared: feeds both of the gates below, so it influences
                // both flip-flops.
                standard_cell(
                    "sky130_fd_sc_hd__and2_2",
                    vec![named_pin(10, "A"), named_pin(11, "B")],
                    vec![named_pin(12, "X")],
                ),
                // ff1's next-state logic, also fed by ff2's output.
                standard_cell(
                    "sky130_fd_sc_hd__and2_2",
                    vec![named_pin(20, "A"), named_pin(21, "B")],
                    vec![named_pin(22, "X")],
                ),
                // ff2's next-state logic, also fed by ff1's output.
                standard_cell(
                    "sky130_fd_sc_hd__and2_2",
                    vec![named_pin(30, "A"), named_pin(31, "B")],
                    vec![named_pin(32, "X")],
                ),
                standard_cell(
                    "sky130_fd_sc_hd__dfrtp_2",
                    vec![named_pin(40, "D"), named_pin(41, "CLK")],
                    vec![named_pin(42, "Q")],
                ),
                standard_cell(
                    "sky130_fd_sc_hd__dfrtp_2",
                    vec![named_pin(50, "D"), named_pin(51, "CLK")],
                    vec![named_pin(52, "Q")],
                ),
            ],
            connections: vec![
                conn(1, vec![10]),
                conn(2, vec![11]),
                conn(12, vec![20, 30]), // the shared gate feeds both
                conn(22, vec![40]),     // -> ff1.D
                conn(32, vec![50]),     // -> ff2.D
                conn(42, vec![31]),     // ff1.Q -> ff2's logic
                conn(52, vec![21]),     // ff2.Q -> ff1's logic
            ],
        };

        let function_sizes = |graph: &Graph| -> Vec<usize> {
            let mut sizes: Vec<usize> = graph
                .cells
                .iter()
                .filter_map(|cell| match cell {
                    Cell::MergeCell {
                        cell_name,
                        ancestor_cells,
                        ..
                    } if cell_name == "BooleanFunction" => Some(ancestor_cells.len()),
                    _ => None,
                })
                .collect();
            sizes.sort_unstable();
            sizes
        };

        // Plain connectivity sees one cloud; the two flip-flops are
        // mutually dependent, so register-SCC grouping sees one component
        // and keeps that same cloud whole.
        assert_eq!(function_sizes(&graph.merge_boolean_functions()), vec![3]);
        assert_eq!(
            function_sizes(&graph.merge_boolean_functions_by_register_scc()),
            vec![3]
        );

        // By cone of influence: `{ff1}`, `{ff2}` and `{ff1, ff2}` are three
        // distinct cones, so the cloud splits into a cell per cone — every
        // gate still in exactly one cell, none duplicated.
        let by_cone = graph.merge_boolean_functions_by_cone_of_influence(&[]);
        assert_eq!(function_sizes(&by_cone), vec![1, 1, 1]);

        // Each flip-flop's own cone of influence — every cell whose logic
        // reaches it — is two of those three cells: its own, plus the
        // shared one it has in common with the other flip-flop.
        let reaches_ff = |ff_input_net: u32| -> usize {
            // Walk back from the flip-flop's `D` pin over the merged
            // graph, counting the `"BooleanFunction"` cells that feed it
            // directly or indirectly.
            let driver_of: HashMap<u32, u32> = by_cone
                .connections
                .iter()
                .flat_map(|conn| conn.iter())
                .flat_map(|(&src, dsts)| dsts.iter().map(move |&dst| (dst, src)))
                .collect();
            let owner_of_output: HashMap<u32, usize> = by_cone
                .cells
                .iter()
                .enumerate()
                .flat_map(|(index, cell)| cell.outputs().iter().map(move |&(net, _)| (net, index)))
                .collect();

            let mut seen_cells: HashSet<usize> = HashSet::new();
            let mut queue: VecDeque<u32> = VecDeque::from([ff_input_net]);
            let mut visited: HashSet<u32> = HashSet::new();
            while let Some(net) = queue.pop_front() {
                let Some(&driver) = driver_of.get(&net) else {
                    continue;
                };
                let Some(&cell_index) = owner_of_output.get(&driver) else {
                    continue;
                };
                let Cell::MergeCell { cell_name, .. } = &by_cone.cells[cell_index] else {
                    continue; // a flip-flop's `Q`; the cone stops there
                };
                if cell_name != "BooleanFunction" {
                    continue;
                }
                seen_cells.insert(cell_index);
                for &(net, _) in by_cone.cells[cell_index].inputs() {
                    if visited.insert(net) {
                        queue.push_back(net);
                    }
                }
            }
            seen_cells.len()
        };

        assert_eq!(reaches_ff(40), 2, "ff1's cone");
        assert_eq!(reaches_ff(50), 2, "ff2's cone");
    }

    /// A `dfrtp` flip-flop (asynchronous active-low reset) with the given
    /// nets on `RESET_B`, `D`, `CLK` and `Q`.
    fn flipflop(reset_b: u32, d: u32, clk: u32, q: u32) -> Cell {
        standard_cell(
            "sky130_fd_sc_hd__dfrtp_2",
            vec![
                (reset_b, "RESET_B".to_string()),
                (d, "D".to_string()),
                (clk, "CLK".to_string()),
            ],
            vec![(q, "Q".to_string())],
        )
    }

    /// Three `dfrtp` flip-flops in a row, the design's `I` input feeding
    /// the first and the last driving the `out` output — so a 1 typed on
    /// `I` reaches `out` exactly three clock edges later, and no sooner.
    fn three_stage_pipeline() -> Graph {
        Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![
                        (1, "I".to_string()),
                        (2, "clk".to_string()),
                        (3, "rst_n".to_string()),
                    ],
                },
                flipflop(30, 10, 20, 11),
                flipflop(31, 12, 21, 13),
                flipflop(32, 14, 22, 15),
                Cell::Output {
                    inputs: vec![(16, "out".to_string())],
                },
            ],
            connections: vec![
                Connection::from([(1, vec![10])]),
                Connection::from([(2, vec![20, 21, 22])]),
                Connection::from([(3, vec![30, 31, 32])]),
                Connection::from([(11, vec![12])]),
                Connection::from([(13, vec![14])]),
                Connection::from([(15, vec![16])]),
            ],
        }
    }

    /// Three flip-flops between `I` and `out`, all reset to 0: reaching
    /// `out = 1` takes three clock edges, and the search has to report
    /// exactly that rather than the 4 its doubling first lands on.
    #[test]
    fn solve_system_finds_the_shortest_sequence_through_a_pipeline() {
        let solution = three_stage_pipeline()
            .solve_system(&[("out".to_string(), true)], &[], 0, 32)
            .expect("out = 1 is reachable");

        assert_eq!(solution.cycles, 3);
        assert_eq!(solution.frames.len(), 4);
        // Doubled up to the first length that works, then bisected the gap
        // it jumped over — never a length below the answer coming back
        // satisfiable.
        assert_eq!(
            solution.probes,
            vec![(0, false), (1, false), (2, false), (4, true), (3, true)]
        );
        assert!(
            solution
                .probes
                .iter()
                .all(|&(cycles, sat)| !sat || cycles >= 3),
            "no length shorter than the answer should have worked: {:?}",
            solution.probes
        );

        // `I` has to carry the 1 three edges before the end; the frames
        // after that no longer matter to `out`.
        let input_index = solution
            .input_labels
            .iter()
            .position(|label| label == "I")
            .expect("the design's I input is reported");
        assert!(solution.frames[0][input_index], "I must be 1 in frame 0");

        assert_eq!(solution.target_labels.len(), 1);
        assert!(
            solution.target_frames[3][0],
            "the target pin must actually carry its asked-for value in the last frame"
        );
    }

    /// A condition the search can't reach comes back as an error naming
    /// the bound it searched to — whether it's out of reach only within
    /// that bound (the pipeline needs 3 edges, searched to 2) or out of
    /// reach at any length at all (an output tied to the `conb` cell's
    /// constant-low pin).
    #[test]
    fn solve_system_reports_a_condition_it_cannot_reach() {
        let too_few = three_stage_pipeline()
            .solve_system(&[("out".to_string(), true)], &[], 0, 2)
            .expect_err("the pipeline needs three edges, not two");
        assert!(
            too_few.contains("2 clock cycles"),
            "the error should name the bound it searched to, got: {too_few}"
        );

        let grounded = Graph {
            cells: vec![
                standard_cell(
                    "sky130_fd_sc_hd__conb_1",
                    Vec::new(),
                    vec![(1, "HI".to_string()), (2, "LO".to_string())],
                ),
                Cell::Output {
                    inputs: vec![(3, "low".to_string())],
                },
            ],
            connections: vec![Connection::from([(2, vec![3])])],
        };
        assert!(
            grounded
                .solve_system(&[("low".to_string(), true)], &[], 0, 8)
                .is_err(),
            "a pin wired to a constant 0 can never read 1"
        );
    }

    /// A flip-flop's asynchronous set/reset pin says what it powers up
    /// holding: a `RESET_B` one starts at 0, a `SET_B` one at 1. So
    /// `q_set = 1` needs no clock edge at all, while `q_reset = 1` needs
    /// one.
    #[test]
    fn solve_system_initializes_flipflops_from_their_set_and_reset_pins() {
        let graph = Graph {
            cells: vec![
                Cell::Input {
                    outputs: vec![(1, "I".to_string()), (3, "rst_n".to_string())],
                },
                flipflop(30, 10, 20, 11),
                standard_cell(
                    "sky130_fd_sc_hd__dfstp_2",
                    vec![
                        (40, "SET_B".to_string()),
                        (41, "D".to_string()),
                        (42, "CLK".to_string()),
                    ],
                    vec![(43, "Q".to_string())],
                ),
                Cell::Output {
                    inputs: vec![(50, "q_reset".to_string()), (51, "q_set".to_string())],
                },
            ],
            connections: vec![
                Connection::from([(1, vec![10, 41])]),
                Connection::from([(3, vec![30, 40])]),
                Connection::from([(11, vec![50])]),
                Connection::from([(43, vec![51])]),
            ],
        };

        let set = graph
            .solve_system(&[("q_set".to_string(), true)], &[], 0, 8)
            .expect("a SET_B flip-flop powers up at 1");
        assert_eq!(set.cycles, 0, "no clock edge needed: it starts at 1");

        let reset = graph
            .solve_system(&[("q_reset".to_string(), true)], &[], 0, 8)
            .expect("a RESET_B flip-flop can be loaded with a 1");
        assert_eq!(reset.cycles, 1, "one edge to clock a 1 in past the reset");

        // And the other way round, each in the state it powers up in.
        assert_eq!(
            graph
                .solve_system(&[("q_reset".to_string(), false)], &[], 0, 8)
                .expect("a RESET_B flip-flop powers up at 0")
                .cycles,
            0
        );
        assert_eq!(
            graph
                .solve_system(&[("q_set".to_string(), false)], &[], 0, 8)
                .expect("a SET_B flip-flop can be loaded with a 0")
                .cycles,
            1
        );
    }
}
