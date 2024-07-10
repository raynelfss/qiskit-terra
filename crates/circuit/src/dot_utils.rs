// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// This module is forked from rustworkx at:
// https://github.com/Qiskit/rustworkx/blob/c4256daf96fc3c08c392450ed33bc0987cdb15ff/src/dot_utils.rs
// and has been modified to generate a dot file from a Rust DAGCircuit instead
// of a rustworkx PyGraph object

use std::collections::BTreeMap;
use std::io::prelude::*;

use crate::dag_circuit::{DAGCircuit, Wire};
use pyo3::prelude::*;
use rustworkx_core::petgraph::visit::{
    Data, EdgeRef, GraphBase, GraphProp, IntoEdgeReferences, IntoNodeReferences, NodeIndexable,
    NodeRef,
};

static TYPE: [&str; 2] = ["graph", "digraph"];
static EDGE: [&str; 2] = ["--", "->"];

pub fn build_dot<T>(
    py: Python,
    dag: &DAGCircuit,
    file: &mut T,
    graph_attrs: Option<BTreeMap<String, String>>,
    node_attrs: Option<PyObject>,
    edge_attrs: Option<PyObject>,
) -> PyResult<()>
where
    T: Write,
{
    let graph = &dag.dag;
    writeln!(file, "{} {{", TYPE[graph.is_directed() as usize])?;
    if let Some(graph_attr_map) = graph_attrs {
        for (key, value) in graph_attr_map.iter() {
            writeln!(file, "{}={} ;", key, value)?;
        }
    }

    for node in graph.node_references() {
        let node_weight = dag.get_node(py, node.id())?;
        writeln!(
            file,
            "{} {};",
            graph.to_index(node.id()),
            attr_map_to_string(py, node_attrs.as_ref(), node_weight)?
        )?;
    }
    for edge in graph.edge_references() {
        let edge_weight = match edge.weight() {
            Wire::Qubit(qubit) => dag.qubits.get(*qubit).unwrap(),
            Wire::Clbit(clbit) => dag.clbits.get(*clbit).unwrap(),
        };
        writeln!(
            file,
            "{} {} {} {};",
            graph.to_index(edge.source()),
            EDGE[graph.is_directed() as usize],
            graph.to_index(edge.target()),
            attr_map_to_string(py, edge_attrs.as_ref(), edge_weight)?
        )?;
    }
    writeln!(file, "}}")?;
    Ok(())
}

static ATTRS_TO_ESCAPE: [&str; 2] = ["label", "tooltip"];

/// Convert an attr map to an output string
fn attr_map_to_string<'a, T: ToPyObject>(
    py: Python,
    attrs: Option<&'a PyObject>,
    weight: T,
) -> PyResult<String> {
    if attrs.is_none() {
        return Ok("".to_string());
    }
    let attr_callable = |node: T| -> PyResult<BTreeMap<String, String>> {
        let res = attrs.unwrap().call1(py, (node.to_object(py),))?;
        res.extract(py)
    };

    let attrs = attr_callable(weight)?;
    if attrs.is_empty() {
        return Ok("".to_string());
    }
    let attr_string = attrs
        .iter()
        .map(|(key, value)| {
            if ATTRS_TO_ESCAPE.contains(&key.as_str()) {
                format!("{}=\"{}\"", key, value)
            } else {
                format!("{}={}", key, value)
            }
        })
        .collect::<Vec<String>>()
        .join(", ");
    Ok(format!("[{}]", attr_string))
}
