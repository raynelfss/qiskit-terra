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

use ahash::HashSet;
use faer::utils::vec;
use pyo3::prelude::*;
use qiskit_circuit::equivalence::{CircuitRep, EquivalenceLibrary, Key, NodeData};
use qiskit_circuit::operations::Param;
use smallvec::SmallVec;

use crate::basis::basis_translator::basis_search_visitor::BasisSearchVisitor;

#[pyfunction]
pub(crate) fn basis_search(py: Python) -> Vec<(String, u32, SmallVec<[Param; 3]>)> {
    todo!()
}

pub(crate) fn _basis_search<'a>(
    py: Python,
    equiv_lib: &mut EquivalenceLibrary,
    source_basis: HashSet<(String, u32)>,
    target_basis: HashSet<Key>,
) -> Vec<(&'a str, u32, &'a [Param], &'a CircuitRep)> {
    // TODO: Logs
    let source_basis: HashSet<Key> = source_basis
        .iter()
        .map(|(gate_name, gate_num_qubits)| Key {
            name: gate_name.to_string(),
            num_qubits: *gate_num_qubits,
        })
        .collect();

    // If source_basis is empty, no work needs to be done.
    if source_basis.is_empty() {
        return vec![];
    }

    // This is only necessary since gates in target basis are currently reported by
    // their names and we need to have in addition the number of qubits they act on.
    let target_basis_keys: Vec<Key> = equiv_lib
        .keys()
        .cloned()
        .filter(|key| target_basis.contains(key))
        .collect();

    let graph_ref = &equiv_lib.graph;
    let vis = BasisSearchVisitor::new(&graph_ref, source_basis, target_basis);

    let dummy = graph_ref.add_node(
        NodeData {
            equivs: vec![],
            key: Key { name: "key".to_string(), num_qubits: 0 },
        }
    );
    // TODO: Finish the function.
    vec![]
}
