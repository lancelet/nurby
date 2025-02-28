use crate::algebra::{Field, ProjectiveEmbedding, VectorSpace};

/// Rational control point.
pub struct ControlPoint<F: Field, V: VectorSpace<F>> {
    location: V,
    weight: F,
}
impl<F: Field, V: VectorSpace<F>> ControlPoint<F, V> {
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

    /// Creates a new `ControlPoint` with `weight = 1` from a location.
    ///
    /// # Parameters
    ///
    /// - `location`: Location in vector space of the control point.
    ///
    /// # Returns
    ///
    /// A new `ControlPoint`.
    pub fn from_location(location: V) -> Self {
        Self::new(location, F::ONE)
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
    pub fn to_homogeneous<E>(&self) -> E::Homogeneous
    where
        E: ProjectiveEmbedding<F, V>,
    {
        E::embed(&self.location, &self.weight)
    }
}
