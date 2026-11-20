# Numerical Discretization & Solver Architecture

## 1. Governing Equations
We solve the 2D incompressible Navier-Stokes equations in non-dimensional convective form:

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\nabla p + \frac{1}{Re}\nabla^2 \mathbf{u} + \mathbf{f}, \qquad \nabla \cdot \mathbf{u} = 0$$

where $Re = \frac{U L}{\nu}$ is the Reynolds number.

## 2. MAC Staggered Grid
To prevent odd-even checkerboard decoupling between pressure and velocity:
- Pressure $p_{i,j}$ and passive scalar $\phi_{i,j}$ are stored at cell centers $(i + \tfrac{1}{2}, j + \tfrac{1}{2})$.
- Horizontal velocity $u_{i,j}$ is stored on vertical cell faces $(i, j + \tfrac{1}{2})$.
- Vertical velocity $v_{i,j}$ is stored on horizontal cell faces $(i + \tfrac{1}{2}, j)$.

## 3. MacCormack (BFECC) Advection
First-order semi-Lagrangian advection introduces excessive artificial viscosity. We use second-order Runge-Kutta (RK2) backtracing $\Phi^{-1}(\mathbf{x})$ combined with a forward error-correction step $\Phi^{+1}$:

$$\hat{\mathbf{q}} = \mathbf{q}(\Phi^{-1}(\mathbf{x})), \qquad \mathbf{q}^{n+1} = \hat{\mathbf{q}} + \frac{1}{2}\left(\mathbf{q}^n(\mathbf{x}) - \hat{\mathbf{q}}(\Phi^{+1}(\Phi^{-1}(\mathbf{x})))\right)$$

Values are clamped to the local 4-point bilinear stencil bounds at $\Phi^{-1}(\mathbf{x})$ to guarantee monotonicity near sharp gradients.

## 4. Geometric Multigrid Pressure Projection
Enforcing $\nabla \cdot \mathbf{u}^{n+1} = 0$ yields the discrete Poisson equation:

$$\nabla_h^2 p = \frac{1}{\Delta t}\nabla_h \cdot \mathbf{u}^*$$

We solve this across a hierarchy of grids from $N \times N$ down to $16 \times 16$ using V-cycles:
1. **Pre-smoothing:** 3 Red-Black Gauss-Seidel sweeps ($\omega = 1.35$).
2. **Residual & Restriction:** Compute $r = \text{rhs} - \nabla_h^2 p$ using the identical discrete Laplacian stencil, then downsample $2\times 2$ onto the coarse grid.
3. **Coarse Solve:** 14 Red-Black sweeps on the $16 \times 16$ grid.
4. **Prolongation & Post-smoothing:** Bilinearly interpolate coarse error corrections (renormalized around solid boundaries) and apply 3 post-smoothing sweeps.
