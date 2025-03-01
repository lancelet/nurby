use crate::{ProjectiveEmbedding, Vec2D, Vec3D, VectorSpace};
use core::fmt::{Debug, Display};
use num_traits::Float;

/// Rational control point.
#[derive(Debug, Clone)]
pub struct ControlPoint<F, V>
where
    F: Float + Debug + Display,
    V: VectorSpace<F> + ProjectiveEmbedding<F, V>,
{
    location: V,
    weight: F,
}
impl<F, V> ControlPoint<F, V>
where
    F: Float + Debug + Display,
    V: VectorSpace<F> + ProjectiveEmbedding<F, V>,
{
    /// Creates a new `ControlPoint`.
    ///
    /// # Parameters
    ///
    /// - `location`: Location in vector space of the control point.
    /// - `weight`: weight of the control point.
    ///
    /// # Returns
    ///
    /// A new `ControlPoint`.
    pub fn new(location: V, weight: F) -> Self {
        Self { location, weight }
    }

    /// Creates a `ControlPoint` in 2D Cartesian space.
    ///
    /// # Parameters
    ///
    /// - `x`: x location.
    /// - `y`: y location.
    /// - `w`: weight for the control point.
    ///
    /// # Returns
    ///
    /// A 2D vector control point.
    pub fn in_2d(x: F, y: F, w: F) -> ControlPoint<F, Vec2D<F>> {
        ControlPoint::new(Vec2D::new(x, y), w)
    }

    /// Creates a `ControlPoint` in 3D Cartesian space.
    ///
    /// # Parameters
    ///
    /// - `x`: x location.
    /// - `y`: y location.
    /// - `z`: z location.
    /// - `w`: weight for the control point.
    ///
    /// # Returns
    ///
    /// A 3D vector control point.
    pub fn in_3d(x: F, y: F, z: F, w: F) -> ControlPoint<F, Vec3D<F>> {
        ControlPoint::new(Vec3D::new(x, y, z), w)
    }

    /// Returns the location of the `ControlPoint`.
    pub fn location(&self) -> &V {
        &self.location
    }

    /// Returns the weight of the `ControlPoint`.
    pub fn weight(&self) -> &F {
        &self.weight
    }

    /// Returns a homogeneous vector containing the control point coordinates
    /// and its weight.
    ///
    /// This uses a projective embedding, which is a trait that associates the
    /// correct vector spaces with each other.
    pub fn to_homogeneous(&self) -> V::Homogeneous {
        V::embed(self.location, self.weight)
    }
}
