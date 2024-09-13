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

use basis_search::basis_search;
use compose_transforms::compose_transforms;
use hashbrown::{HashMap, HashSet};
use itertools::Itertools;
use pyo3::intern;
use pyo3::prelude::*;

pub mod basis_search;
pub mod compose_transforms;

use pyo3::types::IntoPyDict;
use pyo3::types::PyDict;
use pyo3::types::PyTuple;
use qiskit_circuit::circuit_instruction::OperationFromPython;
use qiskit_circuit::imports::CIRCUIT_TO_DAG;
use qiskit_circuit::imports::DAG_TO_CIRCUIT;
use qiskit_circuit::operations::Param;
use qiskit_circuit::packed_instruction::PackedInstruction;
use qiskit_circuit::packed_instruction::PackedOperation;
use qiskit_circuit::{
    circuit_data::CircuitData,
    dag_circuit::{DAGCircuit, NodeType},
    operations::{Operation, OperationRef},
};
use smallvec::SmallVec;

use crate::equivalence::CircuitRep;
use crate::equivalence::EquivalenceLibrary;
use crate::nlayout::PhysicalQubit;
use crate::target_transpiler::exceptions::TranspilerError;
use crate::target_transpiler::{Qargs, Target};

#[pyclass(name = "CoreBasisTranslator")]
pub struct BasisTranslator {
    #[pyo3(get)]
    equiv_lib: EquivalenceLibrary,
    #[pyo3(get)]
    target_basis: HashSet<String>,
    #[pyo3(get)]
    target: Option<Target>,
    #[pyo3(get)]
    non_global_operations: Option<HashSet<String>>,
    #[pyo3(get)]
    qargs_with_non_global_operation: HashMap<Option<Qargs>, HashSet<String>>,
    #[pyo3(get)]
    min_qubits: usize,
}

type ExtraInstructionMap<'a> = HashMap<&'a Option<Qargs>, HashMap<(String, u32), (SmallVec<[Param; 3]>, DAGCircuit)>>;

#[pymethods]
impl BasisTranslator {
    #[new]
    #[pyo3(signature = (equiv_lib, target_basis, min_qubits = 0, target = None))]
    fn new(
        equiv_lib: EquivalenceLibrary,
        target_basis: HashSet<String>,
        min_qubits: usize,
        mut target: Option<Target>,
    ) -> Self {
        let mut non_global_operations = None;
        let mut qargs_with_non_global_operation: HashMap<Option<Qargs>, HashSet<String>> =
            HashMap::default();

        if let Some(target) = target.as_mut() {
            let non_global_from_target: HashSet<String> = target
                .get_non_global_operation_names(false)
                .unwrap_or_default()
                .iter()
                .cloned()
                .collect();
            for gate in &non_global_from_target {
                for qarg in target[gate].keys() {
                    qargs_with_non_global_operation
                        .entry(qarg.cloned())
                        .and_modify(|set| {
                            set.insert(gate.clone());
                        })
                        .or_insert(HashSet::from_iter([gate.clone()]));
                }
            }
            non_global_operations = Some(non_global_from_target);
        }
        Self {
            equiv_lib,
            target_basis,
            target,
            non_global_operations,
            qargs_with_non_global_operation,
            min_qubits,
        }
    }

    fn pre_compose<'py>(&mut self, py: Python<'py>, dag: PyRefMut<'py, DAGCircuit>) -> PyResult<PyRefMut<'py,DAGCircuit>> {
        if self.target_basis.is_empty() && self.target.is_none() {
            return Ok(dag);
        }
        let basic_instrs: HashSet<String>;

        let source_basis: HashSet<(String, u32)>;
        let mut target_basis: HashSet<String>;
        let qargs_local_source_basis: HashMap<Option<Qargs>, HashSet<(String, u32)>>;
        if let Some(target) = self.target.as_ref() {
            basic_instrs = ["barrier", "snapshot", "store"]
                .into_iter()
                .map(|x| x.to_string())
                .collect();
            let non_global_str: HashSet<&str> =
                if let Some(operations) = self.non_global_operations.as_ref() {
                    operations.iter().map(|x| x.as_str()).collect()
                } else {
                    HashSet::default()
                };
            let target_keys = target.keys().collect::<HashSet<_>>();
            target_basis = target_keys
                .difference(&non_global_str)
                .map(|x| x.to_string())
                .collect();
            (source_basis, qargs_local_source_basis) =
                self.extract_basis_target(py, &dag, None, None)?;
        } else {
            basic_instrs = ["measure", "reset", "barrier", "snapshot", "delay", "store"]
                .into_iter()
                .map(|x| x.to_string())
                .collect();
            source_basis = self.extract_basis(py, &dag)?;
            qargs_local_source_basis = HashMap::default();
            target_basis = self.target_basis.clone();
        }
        target_basis = target_basis
            .union(&basic_instrs)
            .map(|x| x.to_string())
            .collect();
        // If the source basis is a subset of the target basis and we have no circuit
        // instructions on qargs that have non-global operations there is nothing to
        // translate and we can exit early.
        let source_basis_names: HashSet<String> =
            source_basis.iter().map(|x| x.0.clone()).collect();
        if source_basis_names.is_subset(&target_basis) && qargs_local_source_basis.is_empty() {
            return Ok(dag);
        }
        let basis_transforms = basis_search(
            &mut self.equiv_lib,
            HashSet::from_iter(source_basis.iter().map(|(x, y)| (x.as_str(), *y))),
            HashSet::from_iter(target_basis.iter().map(|x| x.as_str())),
        );
        let mut qarg_local_basis_transforms: HashMap<
            Option<Qargs>,
            Vec<(String, u32, SmallVec<[Param; 3]>, CircuitRep)>,
        > = HashMap::default();
        for (qarg, local_source_basis) in qargs_local_source_basis.iter() {
            // For any multiqubit operation that contains a subset of qubits that
            // has a non-local operation, include that non-local operation in the
            // search. This matches with the check we did above to include those
            // subset non-local operations in the check here.
            let mut expanded_target = target_basis.clone();
            if let Some(qarg) = qarg {
                let qarg_as_set: HashSet<PhysicalQubit> = HashSet::from_iter(qarg.iter().copied());
                if qarg.len() > 1 {
                    for (non_local_qarg, local_basis) in self.qargs_with_non_global_operation.iter()
                    {
                        if let Some(non_local_qarg) = non_local_qarg {
                            let non_local_qarg_as_set =
                                HashSet::from_iter(non_local_qarg.iter().copied());
                            if qarg_as_set.is_superset(&non_local_qarg_as_set) {
                                expanded_target =
                                    expanded_target.union(local_basis).cloned().collect();
                            }
                        }
                    }
                }
            } else {
                expanded_target = expanded_target
                    .union(&self.qargs_with_non_global_operation[qarg])
                    .cloned()
                    .collect();
            }
            let local_basis_transforms = basis_search(
                &mut self.equiv_lib,
                HashSet::from_iter(local_source_basis.iter().map(|(x, y)| (x.as_str(), *y))),
                HashSet::from_iter(expanded_target.iter().map(|x| x.as_str())),
            );
            if let Some(local_basis_transforms) = local_basis_transforms {
                qarg_local_basis_transforms.insert(qarg.clone(), local_basis_transforms);
            } else {
                return Err(TranspilerError::new_err(format!(
                    "Unable to translate the operations in the circuit: \
                {:?} to the backend's (or manually specified) target \
                basis: {:?}. This likely means the target basis is not universal \
                or there are additional equivalence rules needed in the EquivalenceLibrary being \
                used. For more details on this error see: \
                https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes.\
                BasisTranslator#translation-errors",
                    local_source_basis
                        .iter()
                        .map(|x| x.0.as_str())
                        .collect_vec(),
                    &expanded_target
                )));
            }
        }

        let Some(basis_transforms) = basis_transforms else {
            return Err(TranspilerError::new_err(format!(
                "Unable to translate the operations in the circuit: \
            {:?} to the backend's (or manually specified) target \
            basis: {:?}. This likely means the target basis is not universal \
            or there are additional equivalence rules needed in the EquivalenceLibrary being \
            used. For more details on this error see: \
            https://docs.quantum.ibm.com/api/qiskit/qiskit.transpiler.passes. \
            BasisTranslator#translation-errors"
            , source_basis.iter().map(|x| x.0.as_str()).collect_vec(), &target_basis)));
        };

        let instr_map: HashMap<(String, u32), (SmallVec<[Param; 3]>, DAGCircuit)> = compose_transforms(py, &basis_transforms, &source_basis, &dag)?;
        let extra_inst_map: ExtraInstructionMap = qarg_local_basis_transforms.iter().map(|(qarg, transform)| -> PyResult<_> {Ok((qarg, compose_transforms(py, transform, &qargs_local_source_basis[qarg], &dag)?))}).collect::<PyResult<_>>()?;

        Ok(dag)
    }
}

impl BasisTranslator {
    fn extract_basis(&self, py: Python, circuit: &DAGCircuit) -> PyResult<HashSet<(String, u32)>> {
        let mut basis = HashSet::default();
        // Recurse for DAGCircuit
        fn recurse_dag(
            py: Python,
            circuit: &DAGCircuit,
            basis: &mut HashSet<(String, u32)>,
            min_qubits: usize,
        ) -> PyResult<()> {
            for node in circuit.op_nodes(true) {
                let Some(NodeType::Operation(operation)) = circuit.dag().node_weight(node) else {
                    continue;
                };
                if !circuit.has_calibration_for_index(py, node)?
                    && circuit.get_qargs(operation.qubits).len() >= min_qubits
                {
                    basis.insert((operation.op.name().to_string(), operation.op.num_qubits()));
                }
                if operation.op.control_flow() {
                    let OperationRef::Instruction(inst) = operation.op.view() else {
                        continue;
                    };
                    let inst_bound = inst.instruction.bind(py);
                    for block in inst_bound.getattr("blocks")?.iter()? {
                        recurse_circuit(py, block?, basis, min_qubits)?;
                    }
                }
            }
            Ok(())
        }

        // Recurse for QuantumCircuit
        fn recurse_circuit(
            py: Python,
            circuit: Bound<PyAny>,
            basis: &mut HashSet<(String, u32)>,
            min_qubits: usize,
        ) -> PyResult<()> {
            let circuit_data: PyRef<CircuitData> = circuit
                .getattr(intern!(py, "_data"))?
                .downcast_into()?
                .borrow();
            for (index, inst) in circuit_data.iter().enumerate() {
                let instruction_object = circuit.get_item(index)?;
                let has_calibration = circuit
                    .call_method1(intern!(py, "has_calibration_for"), (&instruction_object,))?;
                if !has_calibration.is_truthy()?
                    && circuit_data.get_qargs(inst.qubits).len() >= min_qubits
                {
                    basis.insert((inst.op.name().to_string(), inst.op.num_qubits()));
                }
                if inst.op.control_flow() {
                    let operation_ob = instruction_object.getattr(intern!(py, "operation"))?;
                    let blocks = operation_ob.getattr("blocks")?;
                    for block in blocks.iter()? {
                        recurse_circuit(py, block?, basis, min_qubits)?;
                    }
                }
            }
            Ok(())
        }

        recurse_dag(py, circuit, &mut basis, self.min_qubits)?;
        Ok(basis)
    }

    fn extract_basis_target(
        &self,
        py: Python,
        dag: &DAGCircuit,
        source_basis: Option<&HashSet<(String, u32)>>,
        qargs_local_source_basis: Option<&HashMap<Option<Qargs>, HashSet<(String, u32)>>>,
    ) -> PyResult<(
        HashSet<(String, u32)>,
        HashMap<Option<Qargs>, HashSet<(String, u32)>>,
    )> {
        let mut source_basis: HashSet<(String, u32)> = source_basis.cloned().unwrap_or_default();
        let mut qargs_local_source_basis: HashMap<Option<Qargs>, HashSet<(String, u32)>> =
            qargs_local_source_basis.cloned().unwrap_or_default();

        for node in dag.op_nodes(true) {
            let node_obj = match dag.dag().node_weight(node).unwrap() {
                NodeType::Operation(op) => op,
                _ => unreachable!("This was supposed to be an op_node."),
            };
            let qargs = dag.get_qargs(node_obj.qubits);
            if dag.has_calibration_for_index(py, node)? || qargs.len() < self.min_qubits {
                continue;
            }
            // Treat the instruction as on an incomplete basis if the qargs are in the
            // qargs_with_non_global_operation dictionary or if any of the qubits in qargs
            // are a superset for a non-local operation. For example, if the qargs
            // are (0, 1) and that's a global (ie no non-local operations on (0, 1)
            // operation but there is a non-local operation on (1,) we need to
            // do an extra non-local search for this op to ensure we include any
            // single qubit operation for (1,) as valid. This pattern also holds
            // true for > 2q ops too (so for 4q operations we need to check for 3q, 2q,
            // and 1q operations in the same manner)
            let physical_qargs: SmallVec<[PhysicalQubit; 2]> =
                qargs.iter().map(|x| PhysicalQubit(x.0)).collect();
            let physical_qargs_as_set: HashSet<PhysicalQubit> =
                HashSet::from_iter(physical_qargs.iter().copied());
            let qargs_is_superset =
                self.qargs_with_non_global_operation
                    .keys()
                    .any(|incomplete_qargs| {
                        if let Some(incomplete_qargs) = incomplete_qargs {
                            let incomplete_qargs =
                                HashSet::from_iter(incomplete_qargs.iter().copied());
                            physical_qargs_as_set.is_superset(&incomplete_qargs)
                        } else {
                            false
                        }
                    });
            if self
                .qargs_with_non_global_operation
                .contains_key(&Some(physical_qargs.iter().copied().collect()))
                || qargs_is_superset
            {
                let _ = &qargs_local_source_basis
                    .entry(Some(physical_qargs.iter().copied().collect()))
                    .and_modify(|set| {
                        set.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
                    })
                    .or_insert(HashSet::from_iter([(
                        node_obj.op.name().to_string(),
                        node_obj.op.num_qubits(),
                    )]));
            } else {
                source_basis.insert((node_obj.op.name().to_string(), node_obj.op.num_qubits()));
            }
            if node_obj.op.control_flow() {
                let OperationRef::Instruction(op) = node_obj.op.view() else {
                    unreachable!("Control flow op is not a control flow op. But control_flow is `true`")
                };
                let bound_inst = op.instruction.bind(py);
                let blocks = bound_inst.getattr("blocks")?.iter()?;
                for block in blocks {
                    let block: PyRef<DAGCircuit> = CIRCUIT_TO_DAG
                        .get_bound(py)
                        .call1((block?,))?
                        .downcast_into()?
                        .borrow();
                    (source_basis, qargs_local_source_basis) = self.extract_basis_target(
                        py,
                        &block,
                        Some(&source_basis),
                        Some(&qargs_local_source_basis),
                    )?;
                }
            }
        }
        Ok((source_basis, qargs_local_source_basis))
    }
    fn apply_translation(&self, py: Python, dag: &DAGCircuit, target_basis: &HashSet<String>, extra_inst_map: &ExtraInstructionMap) -> PyResult<(DAGCircuit, bool)> {
        let is_updated = false;
        let mut out_dag = dag.copy_empty_like(py, "alike")?;
        for node in dag.topological_op_nodes()? {
            let Some(NodeType::Operation(node_obj)) = dag.dag().node_weight(node).cloned() else {
                unreachable!("Node {:?} was in the output of topological_op_nodes, but doesn't seem to be an op_node", node)
 };
            let node_qarg = dag.get_qargs(node_obj.qubits);
            let qubit_set = HashSet::from_iter(node_qarg);
            if target_basis.contains(node_obj.op.name()) || node_qarg.len() < self.min_qubits {
                if node_obj.op.control_flow() {
                    let OperationRef::Instruction(control_op) = node_obj.op.view() else {
                        unreachable!("This instruction says it is of control flow type, but is not an Instruction instance")
                    };
                    let flow_blocks = vec![];
                    let bound_obj = control_op.instruction.bind(py);
                    let blocks = bound_obj.getattr("blocks")?;
                    for block in blocks.iter()? {
                        let dag_block = CIRCUIT_TO_DAG.get_bound(py).call1((block?,))?.downcast_into::<DAGCircuit>()?.borrow();
                        let (updated_dag, is_updated) = self.apply_translation(py, &dag_block, target_basis, &extra_inst_map)?;
                        let flow_circ_block = 
                        if is_updated {
                            DAG_TO_CIRCUIT.get_bound(py).call1((updated_dag,))?
                        } else {
                            block?
                        };
                        flow_blocks.push(flow_circ_block);
                    }
                    let replaced_blocks = bound_obj.call_method1("replace_blocks", (flow_blocks,))?;
                    let new_op: OperationFromPython = replaced_blocks.extract()?;
                    node_obj.op = new_op.operation;
                    node_obj.params = Some(Box::new(new_op.params));
                    node_obj.extra_attrs = new_op.extra_attrs;
                }
                out_dag.push_back(py, node_obj.clone())?;
                continue;
            }
            let node_qarg_as_physical = node_qarg.iter().map(|x| PhysicalQubit(x.0)).collect();
            if self.qargs_with_non_global_operation.contains_key(&Some(node_qarg_as_physical)) && self.qargs_with_non_global_operation[&Some(node_qarg_as_physical)].contains(node_obj.op.name()) {
                out_dag.push_back(py, node_obj)?;
                continue;
            }

            if dag.has_calibration_for_index(py, node)? {
                out_dag.push_back(py, node_obj)?;
                continue;
            }
            let unique_qargs: Qargs = qubit_set.iter().map(|x| PhysicalQubit(x.0)).collect();
            if extra_inst_map.contains_key(&Some(unique_qargs)) {
                todo!()
            }
        }
    
        is_updated
    }

    fn replace_node(&mut self, dag: &mut DAGCircuit, node: PackedInstruction, instr_map: &HashMap<(String, u32), (SmallVec<[Param; 3]>, DAGCircuit)>) -> PyResult<()> {
        let (target_params, target_dag) = &instr_map[&(node.op.name().to_string(), node.op.num_qubits())];
        if node.params_view().len() != target_params.len() {
            return Err(TranspilerError::new_err(format!(
                "Translation num_params not equal to op num_params. \
                Op: {:?} {} Translation: {:?}\n{:?}",
                node.params_view(),
                node.op.name(),
                &target_params,
                &target_dag
            )));
        }
        if !node.params_view().is_empty() {
            let parameter_map = target_params.iter().zip(node.params_view());
            for inner_index in target_dag.topological_op_nodes()? {
                let NodeType::Operation(inner_node) = &target_dag.dag()[inner_index] else {
                    unreachable!("Node returned by topological_op_nodes was not an Operation node.")
                };
                let new_qubits = dag.push_back(py, instr)?;

            }
        }

        Ok(())
    }

}



#[pymodule]
pub fn basis_translator(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<BasisTranslator>()?;
    Ok(())
}
