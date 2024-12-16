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

use std::{
    fmt::Display,
    hash::{DefaultHasher, Hash, Hasher},
    ops::{Deref, DerefMut},
};

use exceptions::CircuitError;
use pyo3::{intern, prelude::*, types::PyDict};

use crate::imports::DEEPCOPY;
pub(crate) mod exceptions {
    use pyo3::import_exception_bound;
    import_exception_bound! {qiskit.circuit.exceptions, CircuitError}
}

pub type RegisterKey = (String, u32);
/// Opaque struct representing a bit instance which can be stored in any circuit or register.
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bit();

impl Display for Bit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} at {:p}", stringify!(Bit), self)
    }
}

macro_rules! create_bit {
    ($name:ident) => {
        #[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name(Bit);

        impl Deref for $name {
            type Target = Bit;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }

        impl DerefMut for $name {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{} at {:p}", stringify!($name), self)
            }
        }
    };
}

#[pyclass(name = "Bit", subclass)]
#[derive(Debug, Clone)]
pub struct PyBit {
    inner_bit: Bit,
    #[pyo3(get)]
    index: Option<u32>,
    #[pyo3(get)]
    register: Option<PyObject>,
}

#[pymethods]
impl PyBit {
    #[new]
    #[pyo3(signature = (register=None, index=None,))]
    pub fn py_new(py: Python, register: Option<PyObject>, index: Option<i32>) -> PyResult<Self> {
        let inner_bit = Bit();
        match (register, index) {
            (None, None) => Ok(Self {
                inner_bit,
                index: None,
                register: None,
            }),
            (Some(reg), Some(idx)) => {
                let size: u32 = reg.getattr(py, intern!(py, "size"))?.extract(py)?;
                if idx >= size.try_into().unwrap() {
                    return Err(CircuitError::new_err(format!(
                        "index must be under the size of the register: {idx} was provided"
                    )));
                }
                let index: Option<u32> = index.map(|index| {
                    if index.is_negative() {
                        size - index as u32
                    } else {
                        index as u32
                    }
                });
                Ok(Self {
                    inner_bit: Bit(),
                    index,
                    register: Some(reg),
                })
            }
            _ => Err(CircuitError::new_err(
                "You should provide both an index and a register, not just one of them.",
            )),
        }
    }

    fn __eq__<'py>(slf: Bound<'py, Self>, other: Bound<'py, PyBit>) -> PyResult<bool> {
        let bit = slf.borrow();
        let other_borrow = other.borrow();
        if bit.register.is_none() && bit.index.is_none() {
            return Ok(std::ptr::eq(&bit.inner_bit, &other_borrow.inner_bit));
        }

        Ok(slf.repr()?.to_string() == other.repr()?.to_string())
    }

    fn __repr__(slf: Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let bit = slf.borrow();
        match (bit.register.as_ref(), bit.index) {
            (Some(reg), Some(idx)) => Ok(format!(
                "{}({}, {})",
                slf.get_type().name()?,
                reg.bind(py).repr()?,
                idx
            )),
            _ => Ok(bit.inner_bit.to_string()),
        }
    }

    fn __copy__(slf: Bound<Self>) -> Bound<Self> {
        slf
    }

    #[pyo3(signature = (memo=None))]
    fn __deepcopy__<'py>(
        slf: Bound<'py, Self>,
        memo: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let borrowed: PyRef<Self> = slf.borrow();
        if borrowed.index.is_none() && borrowed.register.is_none() {
            return Ok(slf.into_any());
        }

        let copied_reg = DEEPCOPY
            .get_bound(py)
            .call1((slf.getattr("register")?, memo))?;
        println!("{}", copied_reg.repr()?);
        slf.get_type().call1((copied_reg, borrowed.index))
    }

    fn __hash__(slf: Bound<'_, Self>, py: Python<'_>) -> PyResult<isize> {
        let borrowed = slf.borrow();
        if let (Some(reg), Some(idx)) = (borrowed.register.as_ref(), borrowed.index) {
            return (reg.bind(py), idx).to_object(py).bind(py).hash();
        }

        let mut hasher = DefaultHasher::new();
        borrowed.inner_bit.hash(&mut hasher);
        Ok(hasher.finish() as isize)
    }
}

macro_rules! create_py_bit {
    ($name:ident, $pyname:literal) => {
        #[pyclass(name=$pyname, extends=PyBit, subclass)]
        #[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
        pub struct $name;

        #[pymethods]
        impl $name {
            #[new]
            #[pyo3(signature = (register=None, index=None))]
            pub fn py_new(
                py: Python,
                register: Option<PyObject>,
                index: Option<i32>,
            ) -> PyResult<(Self, PyBit)> {
                Ok((Self {}, PyBit::py_new(py, register, index)?))
            }
        }
    };
}

// Create rust instances
create_bit! {QubitObject}
create_bit! {ClbitObject}

// Create python instances
create_py_bit! {PyQubit, "Qubit"}
create_py_bit! {PyClbit, "Clbit"}
