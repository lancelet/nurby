use crate::{ControlPoint, ProjectiveEmbedding, VectorSpace, interp};
use core::fmt::{Debug, Display};
use num_traits::Float;

/// Rational Bézier curve.
pub struct RationalBezierCurve<F, V>
where
    F: Float + Debug + Display,
    V: VectorSpace<F> + ProjectiveEmbedding<F, V>,
{
    control_points: Vec<ControlPoint<F, V>>,
}
impl<F, V> RationalBezierCurve<F, V>
where
    F: Float + Debug + Display,
    V: VectorSpace<F> + ProjectiveEmbedding<F, V>,
{
    /// Creates a new `RationalBezierCurve`.
    ///
    /// # Parameters
    ///
    /// - `control_points`: Vector of rational control points.
    ///
    /// # Returns
    ///
    /// A new `RationalBezierCurve`.
    pub fn new(control_points: Vec<ControlPoint<F, V>>) -> Self {
        assert!(
            control_points.len() >= 2,
            "A RationalBezierCurve needs at least 2 control points."
        );
        Self { control_points }
    }

    /// Returns the degree of the curve.
    ///
    /// The degree is given by `number_of_control_points - 1`.
    pub fn degree(&self) -> usize {
        self.control_points.len() - 1
    }

    /// Returns an iterator over the control points of the curve as
    /// homogeneous vectors.
    pub fn homogeneous_control_points_iter(
        &self,
    ) -> impl Iterator<Item = &V::Homogeneous> {
        self.control_points.iter().map(ControlPoint::homogeneous)
    }

    /// Returns a vector of the control points of the curve as homogeneous
    /// vectors.
    pub fn homogeneous_control_points(&self) -> Vec<V::Homogeneous> {
        self.homogeneous_control_points_iter().copied().collect()
    }

    /// Evaluates the curve at a parameter value using the de Casteljau
    /// recursive algorithm.
    ///
    /// # Parameters
    ///
    /// - `u`: The parameter value. For points inside the curve, this should
    ///   be in the range `[0, 1]`, but values outside that range can be used
    ///   for extrapolation.
    ///
    /// # Returns
    ///
    /// The vector value which results from evaluating the curve at `u`.
    pub fn decasteljau_eval(&self, u: F) -> V {
        // Copy control points in homogeneous form.
        let mut pts: Vec<V::Homogeneous> = self.homogeneous_control_points();

        // Perform deCasteljau recursive evaluation. This version is expressed
        // non-recursively.
        for j in (0..self.degree()).rev() {
            for i in 0..j {
                pts[i] = interp(pts[i], pts[i + 1], u);
            }
        }

        // Project final point.
        V::project(pts[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /*
    proptest! {
        #[test]
        fn prop_circle(u in )
    }
    */
}
