use crate::Float;
use core::fmt::{Debug, Display};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Vector space.
///
/// See: <https://en.wikipedia.org/wiki/Vector_space>
pub trait VectorSpace<F: Float>:
    Sized
    + Debug
    + Display
    + Copy
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<F, Output = Self>
    + Div<F, Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign<F>
    + DivAssign<F>
{
    /// Returns the zero (additive identity) of the vector space.
    fn zero() -> Self;

    /// Returns an iterator over the elements of the vector space.
    fn elements(&self) -> impl Iterator<Item = F>;
}

/// Interpolate linearly between vectors.
///
/// This function linearly interpolates between `p0` and `p1`, using the
/// value `u`. To obtain points in the line segment `[p0, p1]`, `u`
/// should be in the range `[0, 1]`. If `u` is outside this range, then
/// the line linearly extrapolating the line `p0` and `p1` is obtained. For
/// a value of `u = 0`, the value `p0` is returned, while for a value
/// of `u = 1`, the value `p1` is returned.
///
/// This is "affine interpolation". For interpolation of homogeneous values,
/// please see the [interp_h] function.
///
/// # Parameters
///
/// - `p0`: Origin point, corresponding to `u = 0`.
/// - `p1`: Destination point, corresponding to `u = 1`.
/// - `u`: Interpolation value; from `[0, 1]` for interpolation, and
///   outside that range for extrapolation.
///
/// # Returns
///
/// Interpolated value.
pub fn interp<F, V>(p0: V, p1: V, u: F) -> V
where
    F: Float,
    V: VectorSpace<F>,
{
    p0 + (p1 - p0) * u
}

/// Embeds a vector space into homogeneous coordinates.
///
/// This is useful for both rational Bézier and NURBS geometry.
///
/// Rational Bézier curves and NURBS require control points with homogeneous
/// weights. This trait converts standard control points into their homogeneous
/// form, allowing for perspective-correct interpolation and exact
/// representations of conics.
///
/// For example, a 2D control point $(x, y)$ with weight $w$ is mapped to:
///
/// $$
/// \mathbf{Q} = (w x, w y, w)
/// $$
///
/// More generally, an $n$-dimensional control point $(x_1, \dots, x_n)$
/// becomes:
///
/// $$
/// \mathbf{Q} = (w x_1, \dots, w x_n, w)
/// $$
///
/// In the nomenclature of homogeneous coordinates, there are three key values:
///
/// - **Euclidean control points**: $\mathbf{P} = (x_1, x_2, \dots, x_n)$.
///   These are the positions of the control points in Euclidean space.
/// - **Weighted control points**: $\mathbf{R} = (w x_1, w x_2, \dots, w x_n)$.
///   These represent the control points in homogeneous space before projective
///   division.
/// - **Weight**: $w$. This is the scalar weight associated with each
///   control point, influencing its contribution to the final curve or surface.
///
/// This trait is distinct from [VectorSpace] because not all vector spaces
/// support projective embeddings required for rational splines.
pub trait ProjectiveEmbedding<F, V>
where
    F: Float,
    V: VectorSpace<F>,
{
    /// The homogeneous representation of this vector space.
    ///
    /// This type should correspond to a vector space with one additional
    /// dimension. For example, for [Vec2D], `Homogeneous` should be [Vec3D].
    type Homogeneous: VectorSpace<F>;

    /// Converts a vector into its homogeneous form by appending a weight.
    ///
    /// This function maps a vector from its current space into the homogeneous
    /// space by adding a weight component:
    ///
    /// $$
    /// \mathrm{embed}(\mathbf{P}, w) \rightarrow \mathbf{Q}
    /// $$
    ///
    /// This operation can be reversed using [`project`][Self::project] and
    /// [`weight`][Self::weight] functions.
    ///
    /// # Parameters
    /// - `v`: The vector to embed.
    /// - `w`: The weight to append.
    ///
    /// # Returns
    /// A homogeneous vector with the additional dimension.
    fn embed(v: V, w: F) -> Self::Homogeneous;

    /// Returns the weighted control point, $\mathbf{R}$.
    fn weighted_control_point(h: Self::Homogeneous) -> V;

    /// Returns the weight, $w$.
    fn weight(h: Self::Homogeneous) -> F;

    /// Converts a homogeneous vector back to the original vector space.
    ///
    /// This function performs a perspective division / dehomogenization by the
    /// homogeneous coordinate, to return the vector in its original space.
    ///
    /// $$
    /// \mathrm{project}(\mathbf{Q}) \rightarrow \mathbf{P}
    /// $$
    ///
    /// # Parameters
    /// - `h`: The homogeneous vector to project.
    ///
    /// # Returns
    /// The original vector in the lower-dimensional space.
    fn project(h: Self::Homogeneous) -> V;
}

/// 2D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec2D<F: Float> {
    pub x: F,
    pub y: F,
}
impl<F: Float> Vec2D<F> {
    pub fn new(x: F, y: F) -> Self {
        Vec2D { x, y }
    }
}
impl<F: Float> Add for Vec2D<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Vec2D::new(self.x + rhs.x, self.y + rhs.y)
    }
}
impl<F: Float> Sub for Vec2D<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2D::new(self.x - rhs.x, self.y - rhs.y)
    }
}
impl<F: Float> Mul<F> for Vec2D<F> {
    type Output = Vec2D<F>;
    fn mul(self, rhs: F) -> Self::Output {
        Vec2D::new(self.x * rhs, self.y * rhs)
    }
}
impl<F: Float> Div<F> for Vec2D<F> {
    type Output = Vec2D<F>;
    fn div(self, rhs: F) -> Self::Output {
        Vec2D::new(self.x / rhs, self.y / rhs)
    }
}
impl<F: Float> Neg for Vec2D<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Vec2D::new(-self.x, -self.y)
    }
}
impl<F: Float> AddAssign for Vec2D<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<F: Float> SubAssign for Vec2D<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<F: Float> MulAssign<F> for Vec2D<F> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}
impl<F: Float> DivAssign<F> for Vec2D<F> {
    fn div_assign(&mut self, rhs: F) {
        *self = *self / rhs;
    }
}
impl<F: Float> Display for Vec2D<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}
impl<F: Float> ProjectiveEmbedding<F, Vec2D<F>> for Vec2D<F> {
    type Homogeneous = Vec3D<F>;
    fn embed(v: Vec2D<F>, w: F) -> Self::Homogeneous {
        Vec3D::new(w * v.x, w * v.y, w)
    }
    fn project(h: Self::Homogeneous) -> Vec2D<F> {
        let w = h.z;
        if w.is_zero() {
            panic!("Cannot project a homogeneous vector with zero weight");
        }
        Vec2D::new(h.x / w, h.y / w)
    }
    fn weighted_control_point(h: Self::Homogeneous) -> Vec2D<F> {
        Vec2D::new(h.x, h.y)
    }
    fn weight(h: Self::Homogeneous) -> F {
        h.z
    }
}
impl<F: Float> VectorSpace<F> for Vec2D<F> {
    fn zero() -> Self {
        let z = F::zero();
        Self::new(z, z)
    }
    fn elements(&self) -> impl Iterator<Item = F> {
        [self.x, self.y].into_iter()
    }
}

/// 3D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3D<F: Float> {
    x: F,
    y: F,
    z: F,
}
impl<F: Float> Vec3D<F> {
    pub fn new(x: F, y: F, z: F) -> Self {
        Vec3D { x, y, z }
    }
}
impl<F: Float> Add for Vec3D<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Vec3D::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl<F: Float> Sub for Vec3D<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec3D::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl<F: Float> AddAssign for Vec3D<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<F: Float> SubAssign for Vec3D<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<F: Float> Mul<F> for Vec3D<F> {
    type Output = Vec3D<F>;
    fn mul(self, rhs: F) -> Self::Output {
        Vec3D::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}
impl<F: Float> Div<F> for Vec3D<F> {
    type Output = Vec3D<F>;
    fn div(self, rhs: F) -> Self::Output {
        Vec3D::new(self.x / rhs, self.y / rhs, self.z / rhs)
    }
}
impl<F: Float> Neg for Vec3D<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Vec3D::new(-self.x, -self.y, -self.z)
    }
}
impl<F: Float> MulAssign<F> for Vec3D<F> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}
impl<F: Float> DivAssign<F> for Vec3D<F> {
    fn div_assign(&mut self, rhs: F) {
        *self = *self / rhs;
    }
}
impl<F: Float> Display for Vec3D<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}
impl<F: Float> ProjectiveEmbedding<F, Vec3D<F>> for Vec3D<F> {
    type Homogeneous = Vec4D<F>;
    fn embed(v: Vec3D<F>, w: F) -> Self::Homogeneous {
        Vec4D::new(w * v.x, w * v.y, w * v.z, w)
    }
    fn project(h: Self::Homogeneous) -> Vec3D<F> {
        if h.w.is_zero() {
            panic!("Cannot project a homogeneous vector with zero weight");
        }
        Vec3D::new(h.x / h.w, h.y / h.w, h.z / h.w)
    }
    fn weighted_control_point(h: Self::Homogeneous) -> Vec3D<F> {
        Vec3D::new(h.x, h.y, h.z)
    }
    fn weight(h: Self::Homogeneous) -> F {
        h.w
    }
}
impl<F: Float> VectorSpace<F> for Vec3D<F> {
    fn zero() -> Self {
        let z = F::zero();
        Self::new(z, z, z)
    }
    fn elements(&self) -> impl Iterator<Item = F> {
        [self.x, self.y, self.z].into_iter()
    }
}

/// 4D vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec4D<F: Float> {
    x: F,
    y: F,
    z: F,
    w: F,
}
impl<F: Float> Vec4D<F> {
    pub fn new(x: F, y: F, z: F, w: F) -> Self {
        Vec4D { x, y, z, w }
    }
}
impl<F: Float> Add for Vec4D<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Vec4D::new(
            self.x + rhs.x,
            self.y + rhs.y,
            self.z + rhs.z,
            self.w + rhs.w,
        )
    }
}
impl<F: Float> Sub for Vec4D<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec4D::new(
            self.x - rhs.x,
            self.y - rhs.y,
            self.z - rhs.z,
            self.w - rhs.w,
        )
    }
}
impl<F: Float> AddAssign for Vec4D<F> {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}
impl<F: Float> SubAssign for Vec4D<F> {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}
impl<F: Float> Mul<F> for Vec4D<F> {
    type Output = Vec4D<F>;
    fn mul(self, rhs: F) -> Self::Output {
        Vec4D::new(self.x * rhs, self.y * rhs, self.z * rhs, self.w * rhs)
    }
}
impl<F: Float> Div<F> for Vec4D<F> {
    type Output = Vec4D<F>;
    fn div(self, rhs: F) -> Self::Output {
        Vec4D::new(self.x / rhs, self.y / rhs, self.z / rhs, self.w / rhs)
    }
}
impl<F: Float> Neg for Vec4D<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Vec4D::new(-self.x, -self.y, -self.z, -self.w)
    }
}
impl<F: Float> MulAssign<F> for Vec4D<F> {
    fn mul_assign(&mut self, rhs: F) {
        *self = *self * rhs;
    }
}
impl<F: Float> DivAssign<F> for Vec4D<F> {
    fn div_assign(&mut self, rhs: F) {
        *self = *self / rhs;
    }
}
impl<F: Float> Display for Vec4D<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {}, {})", self.x, self.y, self.z, self.w)
    }
}
impl<F: Float> VectorSpace<F> for Vec4D<F> {
    fn zero() -> Self {
        let z = F::zero();
        Self::new(z, z, z, z)
    }
    fn elements(&self) -> impl Iterator<Item = F> {
        [self.x, self.y, self.z, self.w].into_iter()
    }
}

#[cfg(test)]
pub mod tests {
    use crate::{
        assert_approx_eq,
        test_util::{approx_eq, reasonable_f32, reasonable_f64},
    };

    use super::*;
    use proptest::{
        prelude::*,
        test_runner::{Config, TestRunner},
    };

    pub fn strategy_vec2d<F, SF>(strategy_f: SF) -> BoxedStrategy<Vec2D<F>>
    where
        F: Float,
        SF: Strategy<Value = F> + Clone + 'static,
    {
        (strategy_f.clone(), strategy_f)
            .prop_map(|(x, y)| Vec2D::new(x, y))
            .boxed()
    }

    pub fn strategy_vec3d<F, SF>(strategy_f: SF) -> BoxedStrategy<Vec3D<F>>
    where
        F: Float,
        SF: Strategy<Value = F> + Clone + 'static,
    {
        (strategy_f.clone(), strategy_f.clone(), strategy_f.clone())
            .prop_map(|(x, y, z)| Vec3D::new(x, y, z))
            .boxed()
    }

    pub fn strategy_vec4d<F, SF>(strategy_f: SF) -> BoxedStrategy<Vec4D<F>>
    where
        F: Float,
        SF: Strategy<Value = F> + Clone + 'static,
    {
        (
            strategy_f.clone(),
            strategy_f.clone(),
            strategy_f.clone(),
            strategy_f.clone(),
        )
            .prop_map(|(x, y, z, w)| Vec4D::new(x, y, z, w))
            .boxed()
    }

    pub fn elementwise_approx_eq<F, V, C>(a: V, b: V, approx_eq: C) -> bool
    where
        F: Float,
        V: VectorSpace<F>,
        C: Fn(F, F) -> bool,
    {
        let a_elems = a.elements().collect::<Vec<F>>();
        let b_elems = b.elements().collect::<Vec<F>>();
        assert!(a_elems.len() == b_elems.len());
        a_elems
            .into_iter()
            .zip(b_elems.into_iter())
            .all(|(ae, be)| approx_eq(ae, be))
    }

    pub fn check_vector_space_axioms<F, V, C>(
        x: V,
        y: V,
        z: V,
        a: F,
        b: F,
        eq: C,
    ) where
        F: Float,
        V: VectorSpace<F>,
        C: Fn(V, V) -> bool,
    {
        // ---- Basic Axioms

        // Associativity of vector addition.
        assert_approx_eq!(x + (y + z), (x + y) + z, eq);

        // Commutativity of vector addition.
        assert_approx_eq!(x + y, y + x, eq);

        // Identity element of vector addition (zero).
        assert_approx_eq!(x + V::zero(), x, eq);

        // Inverse elements of vector addition (neg).
        assert_approx_eq!(x + x.neg(), V::zero(), eq);

        // Compatibility of scalar multiplication with field multiplication
        assert_approx_eq!((x * b) * a, x * (a * b), eq);

        // Identity element of scalar multiplication
        assert_approx_eq!(x * F::one(), x, eq);

        // Distributivity of scalar multiplication with respect to vector
        // addition
        assert_approx_eq!((x + y) * a, x * a + y * a, eq);

        // Distributivity of scalar multiplication with respect to field
        // addition
        assert_approx_eq!(x * (a + b), x * a + x * b, eq);

        // ---- Inverse Operations

        // Addition and subtraction are inverses
        assert_approx_eq!((x + y) - y, x, eq);

        // Multiplication by a scalar and division by a scalar are inverses.
        if a != F::zero() {
            assert_approx_eq!((x * a) / a, x, eq);
        }

        // ---- Compatibility of XxxAssign Operations

        // Add and AddAssign
        assert_approx_eq!(
            x + y,
            {
                let mut xx = x;
                xx += y;
                xx
            },
            eq
        );

        // Sub and SubAssign
        assert_approx_eq!(
            x - y,
            {
                let mut xx = x;
                xx -= y;
                xx
            },
            eq
        );

        // Mul and MulAssign
        assert_approx_eq!(
            x * a,
            {
                let mut xx = x;
                xx *= a;
                xx
            },
            eq
        );

        // Div and DivAssign
        if a != F::zero() {
            assert_approx_eq!(
                x / a,
                {
                    let mut xx = x;
                    xx /= a;
                    xx
                },
                eq
            );
        }
    }

    pub fn prop_vector_space_axioms<F, V, SF, SV, C>(
        strategy_f: SF,
        strategy_v: SV,
        approx_eq: C,
        n_cases: u32,
    ) where
        F: Float,
        V: VectorSpace<F>,
        SF: Strategy<Value = F> + Clone,
        SV: Strategy<Value = V> + Clone,
        C: Fn(V, V) -> bool + Clone,
    {
        let mut runner = TestRunner::new(Config::with_cases(n_cases));
        runner
            .run(
                &(
                    strategy_v.clone(),
                    strategy_v.clone(),
                    strategy_v.clone(),
                    strategy_f.clone(),
                    strategy_f.clone(),
                ),
                |(x, y, z, a, b)| {
                    check_vector_space_axioms(x, y, z, a, b, approx_eq.clone());
                    Ok(())
                },
            )
            .unwrap()
    }

    pub fn check_projective_embedding<F, V, CF, CV>(
        x: V,
        w: F,
        eq_f: CF,
        eq_v: CV,
    ) where
        F: Float,
        V: VectorSpace<F> + ProjectiveEmbedding<F, V> + PartialEq,
        CF: Fn(F, F) -> bool + Clone,
        CV: Fn(V, V) -> bool + Clone,
    {
        // Only work with non-zero weights.
        if !w.is_zero() {
            // Embed in homogeneous coordinates.
            let h = V::embed(x, w);

            // Check Euclidean coordinate and weight.
            assert_approx_eq!(V::project(h), x, eq_v);
            assert_approx_eq!(V::weight(h), w, eq_f);

            // Check homogeneous coordinate.
            x.elements()
                .map(|xe| w * xe)
                .zip(V::weighted_control_point(h).elements())
                .for_each(|(xh, wch)| {
                    assert_approx_eq!(xh, wch, eq_f);
                });
        }
    }

    pub fn prop_projective_embedding<F, V, SF, SV, CF, CV>(
        strategy_f: SF,
        strategy_v: SV,
        approx_eq_f: CF,
        approx_eq_v: CV,
        n_cases: u32,
    ) where
        F: Float,
        V: VectorSpace<F> + ProjectiveEmbedding<F, V> + PartialEq,
        SF: Strategy<Value = F> + Clone,
        SV: Strategy<Value = V> + Clone,
        CF: Fn(F, F) -> bool + Clone,
        CV: Fn(V, V) -> bool + Clone,
    {
        let mut runner = TestRunner::new(Config::with_cases(n_cases));
        runner
            .run(&(strategy_v.clone(), strategy_f.clone()), |(x, w)| {
                check_projective_embedding(
                    x,
                    w,
                    approx_eq_f.clone(),
                    approx_eq_v.clone(),
                );
                Ok(())
            })
            .unwrap()
    }

    pub fn check_interp<F, V, CF, CV>(p0: V, p1: V, u: F, eq_f: CF, eq_v: CV)
    where
        F: Float,
        V: VectorSpace<F>,
        CF: Fn(F, F) -> bool,
        CV: Fn(V, V) -> bool,
    {
        // When u = 0, the result is p0.
        assert_approx_eq!(interp(p0, p1, F::zero()), p0, eq_v);

        // When u = 1, the result is p1.
        assert_approx_eq!(interp(p0, p1, F::one()), p1, eq_v);

        // For the given value of u, we interpolate.
        let r = interp(p0, p1, u);
        assert!(
            p0.elements()
                .into_iter()
                .zip(p1.elements().into_iter())
                .map(|(e0, e1)| e0 * (F::one() - u) + e1 * u)
                .zip(r.elements())
                .all(|(ei, ri)| eq_f(ei, ri))
        );
    }

    pub fn prop_check_interp<F, V, SF, SV, CF, CV>(
        strategy_f: SF,
        strategy_v: SV,
        approx_eq_f: CF,
        approx_eq_v: CV,
        n_cases: u32,
    ) where
        F: Float,
        V: VectorSpace<F>,
        SF: Strategy<Value = F> + Clone,
        SV: Strategy<Value = V> + Clone,
        CF: Fn(F, F) -> bool + Clone,
        CV: Fn(V, V) -> bool + Clone,
    {
        let mut runner = TestRunner::new(Config::with_cases(n_cases));
        runner
            .run(
                &(strategy_v.clone(), strategy_v.clone(), strategy_f.clone()),
                |(p0, p1, u)| {
                    check_interp(
                        p0,
                        p1,
                        u,
                        approx_eq_f.clone(),
                        approx_eq_v.clone(),
                    );
                    Ok(())
                },
            )
            .unwrap()
    }

    #[test]
    pub fn test_vec2d_f32_vector_space_axioms() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_vector_space_axioms(
            reasonable_f32(true),
            strategy_vec2d(reasonable_f32(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec2d_f32_interp() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_check_interp(
            reasonable_f32(true),
            strategy_vec2d(reasonable_f32(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec2d_f32_projective_embedding() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_projective_embedding(
            reasonable_f32(true),
            strategy_vec2d(reasonable_f32(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec3d_f32_vector_space_axioms() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_vector_space_axioms(
            reasonable_f32(true),
            strategy_vec3d(reasonable_f32(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec3d_f32_interp() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_check_interp(
            reasonable_f32(true),
            strategy_vec3d(reasonable_f32(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec3d_f32_projective_embedding() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_projective_embedding(
            reasonable_f32(true),
            strategy_vec3d(reasonable_f32(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec4d_f32_vector_space_axioms() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_vector_space_axioms(
            reasonable_f32(true),
            strategy_vec4d(reasonable_f32(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec4d_f32_interp() {
        let abstol = 1e-3f32;
        let reltol = 1e-1f32;

        prop_check_interp(
            reasonable_f32(true),
            strategy_vec4d(reasonable_f32(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec2d_f64_vector_space_axioms() {
        let abstol = 1e-12f64;
        let reltol = 1e-8f64;

        prop_vector_space_axioms(
            reasonable_f64(true),
            strategy_vec2d(reasonable_f64(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec2d_f64_interp() {
        let abstol = 1e-12f64;
        let reltol = 1e-8f64;

        prop_check_interp(
            reasonable_f64(true),
            strategy_vec4d(reasonable_f64(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec2d_f64_projective_embedding() {
        let abstol = 1e-3f64;
        let reltol = 1e-1f64;

        prop_projective_embedding(
            reasonable_f64(true),
            strategy_vec2d(reasonable_f64(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec3d_f64_vector_space_axioms() {
        let abstol = 1e-12f64;
        let reltol = 1e-8f64;

        prop_vector_space_axioms(
            reasonable_f64(true),
            strategy_vec3d(reasonable_f64(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec3d_f64_interp() {
        let abstol = 1e-12f64;
        let reltol = 1e-8f64;

        prop_check_interp(
            reasonable_f64(true),
            strategy_vec3d(reasonable_f64(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec3d_f64_projective_embedding() {
        let abstol = 1e-3f64;
        let reltol = 1e-1f64;

        prop_projective_embedding(
            reasonable_f64(true),
            strategy_vec3d(reasonable_f64(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec4d_f64_vector_space_axioms() {
        let abstol = 1e-12f64;
        let reltol = 1e-8f64;

        prop_vector_space_axioms(
            reasonable_f64(true),
            strategy_vec4d(reasonable_f64(true)),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec4d_f64_interp() {
        let abstol = 1e-12f64;
        let reltol = 1e-8f64;

        prop_check_interp(
            reasonable_f64(true),
            strategy_vec4d(reasonable_f64(true)),
            |a, b| approx_eq(a, b, abstol, reltol),
            |x, y| {
                elementwise_approx_eq(x, y, |a, b| {
                    approx_eq(a, b, abstol, reltol)
                })
            },
            2048,
        );
    }

    #[test]
    pub fn test_vec2d_f64_display() {
        assert_eq!("(2.2, 4.2)", Vec2D::new(2.2f64, 4.2f64).to_string());
    }

    #[test]
    pub fn test_vec3d_f64_display() {
        assert_eq!(
            "(2.2, 4.2, 6.2)",
            Vec3D::new(2.2f64, 4.2f64, 6.2f64).to_string()
        );
    }

    #[test]
    pub fn test_vec4d_f64_display() {
        assert_eq!(
            "(2.2, 4.2, 6.2, 8.2)",
            Vec4D::new(2.2f64, 4.2f64, 6.2f64, 8.2f64).to_string()
        );
    }

    #[test]
    pub fn test_interp_example() {
        let p0 = Vec2D::new(1f32, 2f32);
        let p1 = Vec2D::new(3f32, 5f32);

        // end points
        assert_eq!(p0, interp(p0, p1, 0.0));
        assert_eq!(p1, interp(p0, p1, 1.0));
        // mid point
        assert_eq!(Vec2D::new(2f32, 3.5f32), interp(p0, p1, 0.5));
        // extrapolation
        assert_eq!(Vec2D::new(5f32, 8f32), interp(p0, p1, 2.0));
    }
}
