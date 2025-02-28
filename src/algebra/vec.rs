use crate::algebra::Field;
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Vector space.
///
/// See: <https://en.wikipedia.org/wiki/Vector_space>
pub trait Vec<F: Field>:
    Sized
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
impl<F: Field> Vec<F> for Vec2D<F> {
    const ZERO: Self = Vec2D {
        x: F::ZERO,
        y: F::ZERO,
    };
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
impl<F: Field> Vec<F> for Vec3D<F> {
    const ZERO: Self = Self {
        x: F::ZERO,
        y: F::ZERO,
        z: F::ZERO,
    };
}

#[cfg(test)]
pub mod tests {
    // TODO: Tests pending.
}
