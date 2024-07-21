// This code is part of Qiskit.
//
// (C) Copyright IBM 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

use ahash::{HashMap, HashSet};
use pyo3::prelude::*;
use qiskit_circuit::{
    equivalence::{CircuitRep, EdgeData, Equivalence, Key, NodeData},
    operations::{Operation, Param},
};
use rustworkx_core::petgraph::{
    graph::NodeIndex,
    stable_graph::{StableDiGraph, StableGraph},
    visit::{Control, EdgeRef},
};

pub struct BasisSearchVisitor<'a> {
    graph: &'a StableDiGraph<NodeData, Option<EdgeData>>,
    target_basis: HashSet<Key>,
    source_gates_remain: HashSet<Key>,
    num_gates_remain_for_rule: HashMap<usize, usize>,
    basis_transforms: Vec<(&'a str, u32, &'a [Param], &'a CircuitRep)>,
    predecessors: HashMap<&'a Key, &'a Equivalence>,
    opt_cost_map: HashMap<&'a Key, u32>,
}

impl<'a> BasisSearchVisitor<'a> {
    pub fn new(
        graph: &'a StableGraph<NodeData, Option<EdgeData>>,
        source_basis: HashSet<Key>,
        target_basis: HashSet<Key>,
    ) -> Self {
        let mut save_index = usize::MAX;
        let mut num_gates_remain_for_rule = HashMap::default();
        for edge_data in graph.edge_weights().flatten() {
            if save_index == edge_data.index {
                continue;
            }
            num_gates_remain_for_rule.insert(edge_data.index, edge_data.num_gates);
            save_index = edge_data.index;
        }
        Self {
            graph,
            target_basis,
            source_gates_remain: source_basis,
            num_gates_remain_for_rule,
            basis_transforms: vec![],
            predecessors: HashMap::default(),
            opt_cost_map: HashMap::default(),
        }
    }

    pub fn discover_vertex(&mut self, v: NodeIndex, score: u32) -> Control<()> {
        let gate = &self.graph[v].key;
        self.source_gates_remain.remove(gate);
        if let Some(cost_ref) = self.opt_cost_map.get_mut(gate) {
            *cost_ref = score;
        } else {
            self.opt_cost_map.insert(gate, score);
        }
        if let Some(rule) = self.predecessors.get(gate) {
            // TODO: Logger
            self.basis_transforms.push((
                gate.name.as_str(),
                gate.num_qubits,
                &rule.params,
                &rule.circuit,
            ));
        }
        if self.source_gates_remain.is_empty() {
            self.basis_transforms.reverse();
            return Control::Break(());
        }
        Control::Continue
    }

    pub fn examine_edge(&mut self, target: NodeIndex, edata: Option<&'a EdgeData>) -> Control<()> {
        if edata.is_none() {
            return Control::Continue;
        }
        let edata = edata.as_ref().unwrap();
        self.num_gates_remain_for_rule
            .entry(edata.index)
            .and_modify(|val| *val -= 1)
            .or_default();
        let target = &self.graph[target].key;

        if self.num_gates_remain_for_rule[&edata.index] > 0 || self.target_basis.contains(target) {
            return Control::Prune;
        }
        Control::Continue
    }

    pub fn edge_relaxed(&mut self, target: NodeIndex, edata: Option<&'a EdgeData>) -> Control<()> {
        if let Some(edata) = edata {
            let gate = &self.graph[target].key;
            self.predecessors.insert(gate, &edata.rule);
        }
        Control::Continue
    }
    /// Returns the cost of an edge.
    ///
    /// This function computes the cost of this edge rule by summing
    /// the costs of all gates in the rule equivalence circuit. In the
    /// end, we need to subtract the cost of the source since `dijkstra`
    /// will later add it.
    pub fn edge_cost(&self, edge_data: Option<&'a EdgeData>) -> u32 {
        if edge_data.is_none() {
            return 1;
        }
        let edge_data = edge_data.as_ref().unwrap();
        let mut cost_tot = 0;
        for instruction in edge_data.rule.circuit.data.iter() {
            let key = Key {
                name: instruction.op.name().to_string(),
                num_qubits: instruction.op.num_qubits(),
            };
            cost_tot += self.opt_cost_map[&key]
        }
        cost_tot - self.opt_cost_map[&edge_data.source]
    }
}
