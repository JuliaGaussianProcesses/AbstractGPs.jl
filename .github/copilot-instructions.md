# AbstractGPs.jl

AbstractGPs.jl is a Julia package that defines a low-level API for working with Gaussian processes (GPs), providing basic functionality for working with them in the simplest cases. It is aimed at developers and researchers who want to use it as a building block for more complex GP implementations.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Bootstrap and Setup
- Install Julia dependencies and instantiate the project:
  ```bash
  julia --project=. -e "using Pkg; Pkg.instantiate()"
  ```
  - Takes ~25 seconds with precompilation on first run
  - NEVER CANCEL: Always wait for completion, even if it appears to hang

### Build and Test the Repository
- Run the main package tests (recommended for most development):
  ```bash
  GROUP="AbstractGPs" julia --project=. -e "using Pkg; Pkg.test()"
  ```
  - Takes ~3 minutes to complete. NEVER CANCEL. Set timeout to 10+ minutes.
  - Tests 791 core functionality tests
  
- Run all tests including PPL (Probabilistic Programming Language) integration:
  ```bash
  julia --project=. -e "using Pkg; Pkg.test()"
  ```
  - Takes ~15 minutes to complete. NEVER CANCEL. Set timeout to 30+ minutes.
  - May have 1 error related to PPL dependencies due to network connectivity - this is expected
  - Includes both AbstractGPs and PPL test groups

### Code Formatting and Quality
- Format all Julia code using the Blue style:
  ```bash
  julia -e "using Pkg; Pkg.add(\"JuliaFormatter\")"
  julia -e "using JuliaFormatter; format(\".\"; verbose=true)"
  ```
  - Takes <1 second. Always run before committing changes.
  - Uses Blue style as configured in `.JuliaFormatter.toml`

### Documentation
- Set up documentation dependencies:
  ```bash
  julia --project=docs -e "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()"
  ```
  - Takes ~45 seconds
  
- Build documentation (may fail due to network dependencies):
  ```bash
  julia --project=docs docs/make.jl
  ```
  - NEVER CANCEL: Takes 5-10 minutes but may fail due to external example dependencies
  - Failure is expected in sandboxed environments due to gitlab.com connectivity issues

### Running Examples
Examples are located in the `examples/` directory and use Literate.jl for notebook-style scripts:

- Navigate to specific example directory:
  ```bash
  cd examples/0-intro-1d  # or any other example
  ```

- Set up example dependencies (takes 2-5 minutes depending on network):
  ```bash
  julia --project=. -e "using Pkg; Pkg.instantiate()"
  ```

- Run example directly:
  ```bash
  julia --project=. script.jl
  ```

- Or run interactively:
  ```bash
  julia --project=.
  julia> include("script.jl")
  ```

- Available examples:
  - `0-intro-1d/` - Basic 1D GP regression introduction
  - `1-mauna-loa/` - Real-world CO2 data modeling
  - `2-deep-kernel-learning/` - Advanced kernel learning
  - `3-parametric-heteroscedastic/` - Heteroscedastic modeling

**⚠️ CRITICAL: Literate.jl Limitation**
When working with code in the `examples/` folder, **NEVER insert comments inside function definitions**. This breaks Literate.jl processing. Only add comments:
- At the top level (outside functions)
- Between function definitions
- As single-line comments before functions

```julia
# ✓ Good: Comment outside function
function my_function()
    # This will break Literate.jl processing
    return 42
end

# ✓ Good: Comment between functions
function another_function()
    return 24  # This will also break
end
```

## Validation

### Core Package Functionality
Always manually validate GP functionality after making changes with this test:
```julia
using AbstractGPs, Random
Random.seed!(42)

# Test basic GP functionality
x = rand(10)
y = sin.(x) 
f = GP(Matern32Kernel())
fx = f(x, 0.01)

println("Created GP with $(length(x)) training points")
println("Log marginal likelihood: $(logpdf(fx, y))")

# Test posterior
p_fx = posterior(fx, y)
println("Created posterior GP")

# Test prediction  
x_test = [0.5]
pred = marginals(p_fx(x_test))
println("Prediction at x=0.5: mean=$(pred[1].μ), std=$(pred[1].σ)")

println("✓ Basic GP workflow successful")
```

For more comprehensive testing of advanced features:
```julia
using AbstractGPs, Random, LinearAlgebra
Random.seed!(123)

# Test different mean functions
x, y = randn(15), randn(15)
for (name, mean_func) in [("Zero", ZeroMean()), ("Const", ConstMean(2.0)), ("Custom", CustomMean(x -> x^2))]
    f = GP(mean_func, SqExponentialKernel())
    fx = f(x, 0.01)
    ll = logpdf(fx, y)
    println("✓ $name mean function: logpdf = $ll")
end

# Test sparse approximations
z = randn(5)  # inducing points
f = GP(SqExponentialKernel())
fx, fz = f(x, 0.01), f(z)
vfe_approx = VFE(fz)
elbo_val = elbo(vfe_approx, fx, y)
println("✓ VFE approximation: ELBO = $elbo_val")

# Test posterior prediction
posterior_f = posterior(vfe_approx, fx, y)
pred_mean, pred_cov = mean_and_cov(posterior_f([0.0, 1.0]))
println("✓ Posterior predictions successful")
```

### Example Validation
To validate changes affecting examples or when working with example code:
```bash
# Navigate to an example (e.g., basic intro)
cd examples/0-intro-1d

# Install example dependencies (may take 2-5 minutes)
julia --project=. -e "using Pkg; Pkg.instantiate()"

# Run the example to ensure it works
julia --project=. script.jl

# Return to repository root
cd ../..
```

Expected output for `0-intro-1d` example includes:
- GP regression plots
- Posterior mean and variance predictions  
- No errors during execution

### Before Committing Changes
Always run these validation steps before committing:
1. Format code: `julia -e "using JuliaFormatter; format(\".\"; verbose=true)"`
2. Run main tests: `GROUP="AbstractGPs" julia --project=. -e "using Pkg; Pkg.test()"`
3. Validate basic GP functionality with the test above
4. If working with examples, validate at least one example runs successfully
5. Check that CI workflows will pass by ensuring formatting and tests succeed

## Repository Structure

### Key Directories
- `src/` - Main package source code
  - `AbstractGPs.jl` - Main module file
  - `abstract_gp.jl` - Abstract GP interface
  - `finite_gp_projection.jl` - Finite GP projections
  - `base_gp.jl` - Base GP implementation
  - `exact_gpr_posterior.jl` - Exact GP regression posteriors
  - `sparse_approximations.jl` - Sparse GP approximations
  - `mean_function.jl` - Mean function implementations
  - `latent_gp.jl` - Latent GP functionality
  - `util/` - Utility functions including plotting and test utilities

- `test/` - Test suite
  - `runtests.jl` - Main test runner with GROUP support
  - `ppl/` - PPL integration tests (Turing.jl)
  - Test files mirror the src/ structure

- `docs/` - Documentation
  - `make.jl` - Documentation build script
  - `src/` - Documentation source files

- `examples/` - Example notebooks/scripts
  - `0-intro-1d/` - 1D introduction example
  - `1-mauna-loa/` - Real-world Mauna Loa CO2 example
  - `2-deep-kernel-learning/` - Deep kernel learning example
  - `3-parametric-heteroscedastic/` - Heteroscedastic GP example

### Configuration Files
- `Project.toml` - Package dependencies and metadata
- `.JuliaFormatter.toml` - Code formatting configuration (Blue style)
- `.github/workflows/` - CI/CD pipelines
  - `CI.yml` - Main test suite
  - `PPL.yml` - PPL integration tests
  - `docs.yml` - Documentation building
  - `format.yml` - Code formatting checks

## Common Development Tasks

### Adding New Functionality
1. Add implementation to appropriate file in `src/`
2. Add tests to corresponding file in `test/`
3. Run formatting and tests as described above
4. Update documentation if adding public API

### Working with GP APIs
The package provides three main API levels:
1. **Primary Public API** - For basic GP operations without covariance matrix computation
2. **Secondary Public API** - When covariance matrix computation is acceptable
3. **Internal AbstractGPs API** - For implementing new GP types

Refer to `docs/src/api.md` for detailed API documentation.

### Testing Specific Components
- Test utilities: Use `AbstractGPs.TestUtils` for consistency testing
- Mean functions: Test with `ZeroMean()`, `ConstMean(c)`, `CustomMean(f)`
- Kernels: Use kernels from KernelFunctions.jl (re-exported)
- Plotting: Test with Plots.jl integration

### Troubleshooting

#### Test Failures
- PPL test failures are often due to network connectivity - focus on AbstractGPs group tests
- Use `GROUP="AbstractGPs"` to skip PPL tests during development

#### Network Issues
- Documentation builds may fail due to external dependencies
- PPL tests may fail due to Turing.jl dependency resolution
- This is expected in sandboxed environments

#### Performance
- Package instantiation: 25 seconds (first time)
- Main tests: 3 minutes  
- Full tests: 15 minutes
- Always set appropriate timeouts (10+ minutes for tests, 30+ for full suite)

## Important Notes

- **Julia Version**: Requires Julia 1.10+ (currently tested on 1.11.6)
- **Style**: Uses Blue code style enforced by JuliaFormatter
- **Dependencies**: Heavy reliance on KernelFunctions.jl, Distributions.jl, LinearAlgebra
- **Testing**: Comprehensive test suite with ~791 tests covering all major functionality
- **Documentation**: Uses Documenter.jl with live examples that may require network access

## Common Commands Reference

Save time by using these frequently needed commands instead of searching:

### Repository Navigation
```bash
# Repository root contents
ls -la  # Shows: src/, test/, docs/, examples/, Project.toml, README.md, .github/, etc.

# Key source files 
find src/ -name "*.jl"  # All source files
ls src/  # Main modules: AbstractGPs.jl, abstract_gp.jl, finite_gp_projection.jl, etc.
ls src/util/  # Utilities: TestUtils.jl, plotting.jl, common_covmat_ops.jl

# Test structure
ls test/  # Mirrors src/ structure plus ppl/ subdirectory
ls test/ppl/  # PPL integration tests (runtests.jl, turing.jl)

# Example structure
ls examples/  # Example directories: 0-intro-1d/, 1-mauna-loa/, etc.
ls examples/0-intro-1d/  # Contains: script.jl, Project.toml
```

### Package Information
```julia
# Check package version and dependencies
julia --project=. -e "using Pkg; Pkg.status()"

# See what's exported
julia --project=. -e "using AbstractGPs; println(names(AbstractGPs))"

# Quick GP kernel check
julia --project=. -e "using AbstractGPs; println(names(KernelFunctions))" 
```

### Quick Status Checks
```bash
# Check if tests would pass before committing
GROUP="AbstractGPs" julia --project=. -e "using Pkg; Pkg.test()"  # 3 min

# Check formatting status (dry run)
julia -e "using JuliaFormatter; format(\".\"; overwrite=false, verbose=true)"

# Get git status
git status --short  # See modified files quickly

# Quick example validation (from repo root)
cd examples/0-intro-1d && julia --project=. -e "include(\"script.jl\")" && cd ../..
```