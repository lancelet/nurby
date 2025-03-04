use crate::{ControlPoint, ProjectiveEmbedding, VectorSpace, interp};
use core::fmt::{Debug, Display};
use num_traits::Float;

/// Rational Bézier curve.
pub struct BezierCurve<F, V>
where
    F: Float + Debug + Display,
    V: VectorSpace<F> + ProjectiveEmbedding<F, V>,
{
    control_points: Vec<ControlPoint<F, V>>,
}
impl<F, V> BezierCurve<F, V>
where
    F: Float + Debug + Display,
    V: VectorSpace<F> + ProjectiveEmbedding<F, V>,
{
    /// Creates a new `BezierCurve`.
    ///
    /// # Parameters
    ///
    /// - `control_points`: Vector of rational control points.
    ///
    /// # Returns
    ///
    /// A new `BezierCurve`.
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
    /// This computes both the position on the curve and its first derivative.
    ///
    /// # Parameters
    ///
    /// - `u`: The parameter value. For points inside the curve, this should
    ///   be in the range `[0, 1]`, but values outside that range can be used
    ///   for extrapolation.
    ///
    /// # Returns
    ///
    /// The [CurvePt] corresponding to the point on the curve, along with its
    /// gradient.
    pub fn decasteljau_eval(&self, u: F) -> V {
        // Copy control points in homogeneous form.
        let mut pts: Vec<V::Homogeneous> = self.homogeneous_control_points();

        // Perform deCasteljau recursive evaluation. This version is expressed
        // non-recursively.
        for j in (0..=self.degree()).rev() {
            // Interpolate points.
            for i in 0..j {
                pts[i] = interp(pts[i], pts[i + 1], u);
            }
        }

        V::project(pts[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Vec2D, assert_approx_eq, test_util::approx_eq};
    use proptest::prelude::*;

    proptest! {
        /// Points evaluated on a rational Bézier quarter circle must have a
        /// radius of 1.
        #[test]
        fn prop_circle(u in 0f64..1f64) {
            // Quarter circle
            let q_circle = BezierCurve::new(
                vec![
                    ControlPoint::in_2d(1.0, 0.0, 1.0),
                    ControlPoint::in_2d(1.0, 1.0, 1.0 / 2.0.sqrt()),
                    ControlPoint::in_2d(0.0, 1.0, 1.0)
                ]
            );

            // Evaluate a point on the circle.
            let p: Vec2D<f64> = q_circle.decasteljau_eval(u);
            let r = (p.x * p.x + p.y * p.y).sqrt();
            assert_approx_eq!(1f64, r, |a, b| approx_eq(a, b, 1e-12, 1e-12));
        }
    }
}
