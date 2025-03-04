use core::fmt::{Debug, Display};

mod bezier_curve;
mod control_point;
mod vector_space;

pub use bezier_curve::BezierCurve;
pub use control_point::ControlPoint;

pub use vector_space::ProjectiveEmbedding;
pub use vector_space::Vec2D;
pub use vector_space::Vec3D;
pub use vector_space::Vec4D;
pub use vector_space::VectorSpace;
use vector_space::interp;

#[cfg(test)]
pub mod test_util;

/// Alias for `num_traits` Float that adds `Debug` and `Display`.
pub trait Float: num_traits::Float + Debug + Display {}
impl<T: num_traits::Float + Debug + Display> Float for T {}
