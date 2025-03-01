mod control_point;
mod rational_bezier_curve;
mod vector_space;

pub use control_point::ControlPoint;
pub use rational_bezier_curve::RationalBezierCurve;

pub use vector_space::ProjectiveEmbedding;
pub use vector_space::Vec2D;
pub use vector_space::Vec3D;
pub use vector_space::VectorSpace;
use vector_space::interp;

#[cfg(test)]
pub mod test_util;
