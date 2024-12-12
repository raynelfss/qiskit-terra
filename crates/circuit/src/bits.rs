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
    ops::{Deref, DerefMut},
};

use exceptions::CircuitError;
use pyo3::{intern, prelude::*, types::PyDict};
pub(crate) mod exceptions {
    use pyo3::import_exception_bound;
    import_exception_bound! {qiskit.circuit.exceptions, CircuitError}
}

pub type RegisterKey = (String, u32);
/// Opaque struct representing a bit instance which can be stored in any circuit or register.
#[derive(Debug, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Bit(Option<RegisterKey>, Option<u32>);

impl Bit {
    pub fn new(&self, register: Option<RegisterKey>, index: Option<u32>) -> Self {
        match (register, index) {
            (None, None) => todo!(),
            (Some(_), Some(_)) => todo!(),
            _ => panic!("You should provide both an index and a register, not just one of them."),
        }
    }

    pub fn index(&self) -> Option<u32> {
        self.1
    }

    pub fn register_key(&self) -> Option<&RegisterKey> {
        self.0.as_ref()
    }
}

macro_rules! create_bit {
    ($name:ident) => {
        #[pyclass]
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
                let form = match (&self.0 .0, &self.0 .1) {
                    (None, None) => format!("{} at {:p}", stringify!($name), self),
                    _ => format!(
                        "{}({}Register({}, {}), {})",
                        stringify!($name),
                        stringify!($name),
                        self.register_key().unwrap().0,
                        self.register_key().unwrap().1,
                        self.index().unwrap()
                    ),
                };
                write!(f, "{}", form,)
            }
        }
    };
}

#[pyclass(name = "Bit")]
pub struct PyBit {
    inner_bit: Bit,
    register: Option<PyObject>,
}

#[pymethods]
impl PyBit {
    #[new]
    #[pyo3(signature = (index=None, register=None))]
    pub fn new(py: Python, index: Option<i32>, register: Option<PyObject>) -> PyResult<Self> {
        let inner_bit = Bit(None, None);
        match (register, index) {
            (None, None) => Ok(Self {
                inner_bit,
                register: None,
            }),
            (Some(reg), Some(idx)) => {
                let size: u32 = reg.getattr(py, intern!(py, "size"))?.extract(py)?;
                let name: String = reg.getattr(py, intern!(py, "name"))?.extract(py)?;
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
                    inner_bit: Bit(Some((name, size)), index),
                    register: Some(reg),
                })
            }
            _ => Err(CircuitError::new_err(
                "You should provide both an index and a register, not just one of them.",
            )),
        }
    }

    #[getter]
    fn get_index(&self) -> Option<u32> {
        self.inner_bit.index()
    }

    fn __eq__(&self, other: &PyBit) -> bool {
        if self.register.is_none() && self.get_index().is_none() {
            return std::ptr::eq(&self.inner_bit, &other.inner_bit);
        }

        self.inner_bit == other.inner_bit
    }

    fn __copy__(slf: Bound<Self>) -> Bound<Self> {
        slf
    }

    fn __deepcopy__<'py>(slf: Bound<'py, Self>, memo: Bound<'py, PyDict>) -> PyResult<Bound<'py, PyAny>> {
        let borrowed: PyRef<Self> = slf.borrow();
        if borrowed.get_index().is_none() && borrowed.register.is_none() {
            return Ok(slf.into_any())
        }

        slf.get_type().call0()
    }
}

create_bit! {Qubit}
create_bit! {Clbit}
