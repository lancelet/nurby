use crate::algebra::Field;
use core::fmt::{Debug, Display};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Vector space.
///
/// See: <https://en.wikipedia.org/wiki/Vector_space>
pub trait VectorSpace<F: Field>:
    Sized
    + Debug
    + Display
    + Clone
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
    const ZERO: Self;
    fn elements(&self) -> Vec<F>;
}

/// 2D vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Vec2D<F: Field> {
    x: F,
    y: F,
}
impl<F: Field> Vec2D<F> {
    pub fn new(x: F, y: F) -> Self {
        Vec2D { x, y }
    }
}
impl<F: Field> Add for Vec2D<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Vec2D::new(self.x + rhs.x, self.y + rhs.y)
    }
}
impl<F: Field> Sub for Vec2D<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec2D::new(self.x - rhs.x, self.y - rhs.y)
    }
}
impl<F: Field> Mul<F> for Vec2D<F> {
    type Output = Vec2D<F>;
    fn mul(self, rhs: F) -> Self::Output {
        Vec2D::new(self.x * rhs.clone(), self.y * rhs)
    }
}
impl<F: Field> Div<F> for Vec2D<F> {
    type Output = Vec2D<F>;
    fn div(self, rhs: F) -> Self::Output {
        Vec2D::new(self.x / rhs.clone(), self.y / rhs)
    }
}
impl<F: Field> Neg for Vec2D<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Vec2D::new(-self.x, -self.y)
    }
}
impl<F: Field> AddAssign for Vec2D<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}
impl<F: Field> SubAssign for Vec2D<F> {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}
impl<F: Field> MulAssign<F> for Vec2D<F> {
    fn mul_assign(&mut self, rhs: F) {
        self.x *= rhs.clone();
        self.y *= rhs;
    }
}
impl<F: Field> DivAssign<F> for Vec2D<F> {
    fn div_assign(&mut self, rhs: F) {
        self.x /= rhs.clone();
        self.y /= rhs;
    }
}
impl<F: Field> Display for Vec2D<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}
impl<F: Field> VectorSpace<F> for Vec2D<F> {
    const ZERO: Self = Vec2D {
        x: F::ZERO,
        y: F::ZERO,
    };
    fn elements(&self) -> Vec<F> {
        vec![self.x.clone(), self.y.clone()]
    }
}

/// 3D vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Vec3D<F: Field> {
    x: F,
    y: F,
    z: F,
}
impl<F: Field> Vec3D<F> {
    pub fn new(x: F, y: F, z: F) -> Self {
        Vec3D { x, y, z }
    }
}
impl<F: Field> Add for Vec3D<F> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Vec3D::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl<F: Field> Sub for Vec3D<F> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Vec3D::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl<F: Field> AddAssign for Vec3D<F> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
        self.z += rhs.z;
    }
}
impl<F: Field> SubAssign for Vec3D<F> {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
        self.z -= rhs.z;
    }
}
impl<F: Field> Mul<F> for Vec3D<F> {
    type Output = Vec3D<F>;
    fn mul(self, rhs: F) -> Self::Output {
        Vec3D::new(self.x * rhs.clone(), self.y * rhs.clone(), self.z * rhs)
    }
}
impl<F: Field> Div<F> for Vec3D<F> {
    type Output = Vec3D<F>;
    fn div(self, rhs: F) -> Self::Output {
        Vec3D::new(self.x / rhs.clone(), self.y / rhs.clone(), self.z / rhs)
    }
}
impl<F: Field> Neg for Vec3D<F> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Vec3D::new(-self.x, -self.y, -self.z)
    }
}
impl<F: Field> MulAssign<F> for Vec3D<F> {
    fn mul_assign(&mut self, rhs: F) {
        self.x *= rhs.clone();
        self.y *= rhs.clone();
        self.z *= rhs;
    }
}
impl<F: Field> DivAssign<F> for Vec3D<F> {
    fn div_assign(&mut self, rhs: F) {
        self.x /= rhs.clone();
        self.y /= rhs.clone();
        self.z /= rhs;
    }
}
impl<F: Field> Display for Vec3D<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {}, {})", self.x, self.y, self.z)
    }
}
impl<F: Field> VectorSpace<F> for Vec3D<F> {
    const ZERO: Self = Self {
        x: F::ZERO,
        y: F::ZERO,
        z: F::ZERO,
    };
    fn elements(&self) -> Vec<F> {
        vec![self.x.clone(), self.y.clone(), self.z.clone()]
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::algebra::{
        QRational,
        field::tests::{
            check_addassign_add_compatible, check_addition_associative,
            check_addition_commutative, check_additive_identity,
            check_additive_inverse, check_subassign_sub_compatible,
        },
        qrational::tests::strategy_qrational_i32s,
    };
    use crate::{assert_approx_eq, check_approx_eq, cloned};
    use proptest::{
        prelude::*,
        test_runner::{Config, TestRunner},
    };

    pub fn strategy_vec2d<F, SF>(strategy_f: &SF) -> BoxedStrategy<Vec2D<F>>
    where
        F: Field,
        SF: Strategy<Value = F> + Clone + 'static,
    {
        (strategy_f.clone(), strategy_f.clone())
            .prop_map(|(x, y)| Vec2D::new(x, y))
            .boxed()
    }

    pub fn strategy_vec3d<F, SF>(strategy_f: &SF) -> BoxedStrategy<Vec3D<F>>
    where
        F: Field,
        SF: Strategy<Value = F> + Clone + 'static,
    {
        (strategy_f.clone(), strategy_f.clone(), strategy_f.clone())
            .prop_map(|(x, y, z)| Vec3D::new(x, y, z))
            .boxed()
    }

    pub fn elementwise_approx_eq<F, V, C>(a: &V, b: &V, approx_eq: C) -> bool
    where
        F: Field,
        V: VectorSpace<F>,
        C: Fn(&F, &F) -> bool,
    {
        let a_elems = a.elements();
        let b_elems = b.elements();
        assert!(a_elems.len() == b_elems.len());
        a_elems
            .into_iter()
            .zip(b_elems.into_iter())
            .all(|(ae, be)| approx_eq(&ae, &be))
    }

    /// Check that scalar multiplication is compatible with field
    /// multiplication.
    pub fn check_compat_scalar_field_mul<F, V, C>(
        a: &F,
        b: &F,
        v: &V,
        approx_eq: C,
    ) where
        F: Field,
        V: VectorSpace<F>,
        C: Fn(&V, &V) -> bool,
    {
        check_approx_eq!(a b v => (v * b) * a, v * (a * b), approx_eq)
    }

    /// Check distributivity of scalar multiplication with respect to vector
    /// addition.
    pub fn check_distributivity_scalar_mul_wrt_vector_add<F, V, C>(
        x: &V,
        y: &V,
        a: &F,
        approx_eq: C,
    ) where
        F: Field,
        V: VectorSpace<F>,
        C: Fn(&V, &V) -> bool,
    {
        check_approx_eq!(x y a => (x + y) * a, x * a.clone() + y * a, approx_eq)
    }

    /// Check distributivity of scalar multiplication with respect to field
    /// addition.
    pub fn check_distributivity_scalar_mul_wrt_field_add<F, V, C>(
        v: &V,
        a: &F,
        b: &F,
        approx_eq: C,
    ) where
        F: Field,
        V: VectorSpace<F>,
        C: Fn(&V, &V) -> bool,
    {
        check_approx_eq!(v a b => v * (a + b), v.clone() * a + v * b, approx_eq)
    }

    /// Check that the scalar multiplicative identity is compatible with
    /// the vector type.
    pub fn check_vec_scalar_mul_identity<F, V, C>(v: &V, one: &F, approx_eq: C)
    where
        F: Field,
        V: VectorSpace<F>,
        C: Fn(&V, &V) -> bool,
    {
        check_approx_eq!(v one => v * one, v, approx_eq)
    }

    pub fn prop_vector_space_axioms<F, V, SF, SV, C>(
        strategy_f: SF,
        strategy_v: SV,
        approx_eq: C,
        n_cases: u32,
    ) where
        F: Field,
        V: VectorSpace<F>,
        SF: Strategy<Value = F> + Clone,
        SV: Strategy<Value = V> + Clone,
        C: Fn(&V, &V) -> bool + Clone,
    {
        let ae = approx_eq;
        let mut runner = TestRunner::new(Config {
            cases: n_cases,
            ..Config::default()
        });

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
                    check_addition_associative(&x, &y, &z, ae.clone());
                    check_addition_commutative(&x, &y, ae.clone());
                    check_additive_identity(&x, &V::ZERO, ae.clone());
                    check_additive_inverse(&x, &V::ZERO, ae.clone());
                    check_compat_scalar_field_mul(&a, &b, &x, ae.clone());
                    check_vec_scalar_mul_identity(&x, &F::ONE, ae.clone());
                    check_distributivity_scalar_mul_wrt_vector_add(
                        &x,
                        &y,
                        &a,
                        ae.clone(),
                    );
                    check_distributivity_scalar_mul_wrt_field_add(
                        &x,
                        &a,
                        &b,
                        ae.clone(),
                    );
                    // Compatibility of operations
                    check_addassign_add_compatible(&x, &y, ae.clone());
                    check_subassign_sub_compatible(&x, &y, ae.clone());
                    Ok(())
                },
            )
            .unwrap()
    }

    #[test]
    pub fn test_vec2d_qrational_vector_space_axioms() {
        let strategy_f = strategy_qrational_i32s();
        let strategy_v = strategy_vec2d(&strategy_f);
        let approx_eq = |a: &Vec2D<QRational>, b: &Vec2D<QRational>| {
            elementwise_approx_eq(a, b, QRational::eq)
        };

        prop_vector_space_axioms(strategy_f, strategy_v, approx_eq, 256);
    }

    #[test]
    pub fn test_vec3d_qrational_vector_space_axioms() {
        let strategy_f = strategy_qrational_i32s();
        let strategy_v = strategy_vec3d(&strategy_f);
        let approx_eq = |a: &Vec3D<QRational>, b: &Vec3D<QRational>| {
            elementwise_approx_eq(a, b, QRational::eq)
        };

        prop_vector_space_axioms(strategy_f, strategy_v, approx_eq, 256);
    }
}
