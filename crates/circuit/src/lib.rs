// This code is part of Qiskit.
//
// (C) Copyright IBM 2023, 2024
//
// This code is licensed under the Apache License, Version 2.0. You may
// obtain a copy of this license in the LICENSE.txt file in the root directory
// of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
//
// Any modifications or derivative works of this code must retain this
// copyright notice, and modified files need to carry a notice indicating
// that they have been altered from the originals.

pub mod circuit_data;
pub mod circuit_instruction;

mod bit_data;
mod interner;

mod bit_data;
mod dag_node;
mod error;
mod packed_instruction;

use pyo3::prelude::*;
use pyo3::types::iter::BoundTupleIterator;
use pyo3::types::{PySequence, PySlice, PyTuple};
use std::ops::Deref;

/// A private enumeration type used to extract arguments to pymethod
/// that may be either an index or a slice
#[derive(FromPyObject)]
pub enum SliceOrInt<'a> {
    // The order here defines the order the variants are tried in the FromPyObject` derivation.
    // `Int` is _much_ more common, so that should be first.
    Int(isize),
    Slice(Bound<'a, PySlice>),
}

pub type BitType = u32;
#[derive(Copy, Clone, Debug, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct Qubit(BitType);
#[derive(Copy, Clone, Debug, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct Clbit(BitType);

pub struct TupleLikeArg<'py> {
    value: Bound<'py, PyTuple>,
}

impl<'py> FromPyObject<'py> for TupleLikeArg<'py> {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        Ok(TupleLikeArg {
            value: ob.downcast::<PySequence>()?.to_tuple()?,
        })
    }
}

impl From<BitType> for Qubit {
    fn from(value: BitType) -> Self {
        Qubit(value)
    }
}

impl From<Qubit> for BitType {
    fn from(value: Qubit) -> Self {
        value.0
    }
}

impl From<BitType> for Clbit {
    fn from(value: BitType) -> Self {
        Clbit(value)
    }
}

impl From<Clbit> for BitType {
    fn from(value: Clbit) -> Self {
        value.0
    }
}

/// A trait implemented by containers that support interning
/// semantics for `T`.
trait Interner<T> {
    /// The error type returned when interning.
    type Error;
    /// The type of the interned value.
    type InternedType;
    /// Interns the provided value, returning the interned value.
    fn intern(&mut self, item: T) -> Result<Self::InternedType, Self::Error>;
    /// Looks up the original value given the interned value.
    fn get_interned(&self, interned: &Self::InternedType) -> T;
}

#[pymodule]
pub fn circuit(m: Bound<PyModule>) -> PyResult<()> {
    m.add_class::<circuit_data::CircuitData>()?;
    m.add_class::<circuit_instruction::CircuitInstruction>()?;
    m.add_class::<dag_circuit::DAGCircuit>()?;
    m.add_class::<dag_node::DAGNode>()?;
    m.add_class::<dag_node::DAGInNode>()?;
    m.add_class::<dag_node::DAGOutNode>()?;
    m.add_class::<dag_node::DAGOpNode>()?;
    Ok(())
}
