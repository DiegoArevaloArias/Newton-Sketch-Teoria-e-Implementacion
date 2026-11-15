### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000002
using BenchmarkTools, LinearAlgebra, Random, Statistics, Printf, Plots

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000001
md"""
# Newton Sketch frente a Newton exacto — notebook explicativo y benchmarks

Este notebook construye un problema sintético de regresión logística, presenta **cada función por separado** y compara Newton exacto con cuatro métodos de sketching: Gaussian Sketch, CountSketch, Row Sampling y SRHT.

La primera sección explica la formulación, el papel de cada componente y la construcción de cada matriz. Las secciones siguientes conservan los experimentos de tiempo, memoria, escala y estabilidad entre semillas, e incorporan cinco figuras iniciales (convergencia, tasas, memoria y precisión) y una sección opcional con seis figuras para identificar el punto de equilibrio entre ahorro por iteración y convergencia.

### Cómo usarlo

Abre este archivo con **Pluto.jl**, ejecuta todas las celdas y ajusta los parámetros de la sección 2 si necesitas reducir el costo de los benchmarks. Para usar varios hilos, inicia Julia con `julia -t auto`. Pluto instalará automáticamente `BenchmarkTools` y `Plots` si hacen falta. Se mantienen las interfaces y los experimentos anteriores. Gaussian y SRHT utilizan implementaciones nuevas de memoria acotada; CountSketch y RowSampling permanecen intactos.
"""

# ╔═╡ ab8aa28f-1fb2-4a22-86bb-fdcb164632e9
md"""
### Versión revisada — criterios verificables y modos de ejecución

**Criterio principal:** tiempo para alcanzar la precisión relativa común
`(f(x)-f_ref)/(f(0)-f_ref) <= τ`. Los experimentos por decremento se conservan
solo como diagnósticos de trayectoria, nunca para declarar que un método es más
rápido a precisión equivalente. Se certifica `f_ref` con una cota derivada de
la fuerte convexidad antes de medir.

**Perfil rápido** (`EXPERIMENT_MODE=:quick`): problema base reducido (n=5000, d=100) y 5 repeticiones en barridos exploratorios;
**perfil final** (`:final`): problema base original (n=20000, d=400), 15 repeticiones e intervalos bootstrap. Los grandes
benchmarks (`RUN_COSTLY_BENCHMARKS`) y los cuatro barridos (`BE_RUN`) comienzan
desactivados. Si se cambian los perfiles, Pluto recalcula solo las celdas dependientes.

Los solvers ahora devuelven `reason` (`:target`, `:decrement`, `:maxit`,
`:line_search_failed`), `backtracks` y `last_decrement_sq`. Los archivos y figuras
originales se mantienen, salvo modificaciones necesarias para mostrar los nuevos
criterios y diagnósticos. Se incluyen pruebas numéricas adicionales.
"""

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000003
md"""
## 1. Formulación y funciones del algoritmo

Se presentan las estructuras, las funciones numéricas y las matrices de sketch como celdas de Pluto independientes, no como una biblioteca monolítica.
"""

# ╔═╡ f1697827-3fd3-4dc8-9e01-b419eccb7120
md"""
### Constantes de los métodos

`A_LS` y `B_LS` controlan la búsqueda de línea de Armijo: el primero establece la disminución exigida y el segundo reduce el paso cuando no se cumple. `EPS_SK`, `RHO`, `ETA` y `NU` son parámetros reservados para la versión adaptativa del método, **que todavía no está implementada** en este notebook.
"""

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000004
begin
const A_LS   = 0.1
const B_LS   = 0.5
const EPS_SK = 1 / 8
const RHO    = (1 + EPS_SK) / (1 - EPS_SK)
const ETA    = (1 / 8) * (1 - 0.5 * RHO^2 - A_LS) / RHO^3
const NU     = A_LS * B_LS * ETA^2 / (1 + RHO * ETA)
end

# ╔═╡ c9733114-aa9e-4c6b-8fde-0676a821bb22
md"""
### Función `alpha_tau`

Calcula un coeficiente dependiente de `tau` y de la tolerancia `eps` para los parámetros teóricos del método adaptativo. No participa en los experimentos de Newton Sketch de tamaño fijo.
"""

# ╔═╡ b2c4b3d2-f373-4cbe-ae05-25dade6bc736
alpha_tau(tau; eps = EPS_SK) = (0.57 + 16 * tau / 15) * sqrt(1 + eps) / (1 - eps)^((1 + tau) / 2)

# ╔═╡ 70cfdfa7-f9a4-42f8-a0d2-9759c2f0200f
md"""
## 1.1. Representación de la regresión logística

`LogReg` almacena la matriz de datos `A` (de tamaño `n × d`), las etiquetas binarias `y` (valores `-1` o `+1`) y la regularización `mu`. También se define `size(P)` para consultar directamente el tamaño de la matriz de datos.
"""

# ╔═╡ ccbb5898-f3d4-4150-8b0b-d64c04c50a41
begin
struct LogReg
    A::Matrix{Float64}      # n x d data
    y::Vector{Float64}      # labels in {-1, +1}
    mu::Float64             # ridge parameter
end
Base.size(P::LogReg) = size(P.A)
end

# ╔═╡ e902b24e-e9b4-481d-a59e-5f3005257624
md"""
### Construcción del problema sintético: `make_problem`

Esta función genera un **problema de clasificación logística con estructura espectral controlada**. Su propósito no es simular datos reales en detalle, sino permitir comparar Newton exacto y Newton Sketch bajo condiciones reproducibles.

**1. Matriz de diseño.** Se construye una matriz ortogonal aleatoria `Q` mediante una factorización QR. Después se define una secuencia decreciente:

```math
s_j = j^{-\mathrm{decay}},\qquad j=1,\ldots,d.
```

Con una matriz gaussiana estándar `G`, se produce

```math
A_{\mathrm{full}} = G\,\mathrm{diag}(s_1,\ldots,s_d)\,Q^{\mathsf T}.
```

Así, las columnas de los datos tienen direcciones correlacionadas y distintas escalas. Un `decay` mayor provoca una caída más rápida de los valores singulares **esperados** de la matriz de diseño; esto favorece que gran parte de la información esté concentrada en pocas direcciones. `decay = 0` elimina ese decaimiento intencional. Las singularidades de una muestra finita no son exactamente los valores `s_j`.

**2. Vector verdadero y relación señal-ruido.** Se sortea un vector `xt`, se calculan los márgenes `z = A_full * xt` y se reescala `xt` de forma que la desviación estándar empírica de dichos márgenes sea `snr`. Por tanto, aquí `snr` controla la **escala de los márgenes logísticos**, no una razón de varianzas de ruido aditivo en el sentido tradicional.

**3. Etiquetas binarias.** Para cada fila `aᵢ`, se genera una etiqueta `yᵢ ∈ {-1,+1}` de acuerdo con

```math
\Pr(y_i=+1\mid a_i)=\sigma(a_i^{\mathsf T}x_\star),
\qquad \sigma(t)=\frac{1}{1+e^{-t}}.
```

Esto introduce aleatoriedad en las clases: incluso el vector verdadero no tiene por qué clasificarlas todas correctamente.

**4. Entrenamiento y prueba.** Las primeras `n` filas constituyen el problema `P`; las siguientes `ntest` forman el conjunto `test`. La regularización se fija como `mu = lam * n` y la función devuelve `(P, test, xt)`.

El problema que se resolverá es

```math
\min_{x\in\mathbb R^d}\; f(x)=
\sum_{i=1}^{n}\log\bigl(1+e^{-y_i a_i^{\mathsf T}x}\bigr)
+\frac{\mu}{2}\Vert x\Vert_2^2.
```

**Parámetros:** `n` es el número de ejemplos de entrenamiento, `d` el número de variables, `decay` controla el decaimiento espectral, `lam` la regularización relativa, `snr` la escala de los márgenes, `seed` la reproducibilidad y `ntest` el tamaño de prueba. En las comparaciones al variar `n`, debe recordarse que `mu` también crece si `lam` se mantiene fijo.
"""

# ╔═╡ 0b0dd8ba-418a-4cae-86d0-ea9123113eb3
"""
    make_problem(n, d; decay, lam, snr, seed, ntest)

Rows a_i = Q diag(j^-decay) g_i with g_i ~ N(0, I): fast-decaying spectrum, so the
effective dimension of the Hessian is much smaller than d. Labels follow a logistic
model with margin standard deviation `snr`. Ridge parameter mu = lam * n.
"""
function make_problem(n::Int, d::Int; decay = 1.0, lam = 1e-4, snr = 3.0, seed = 2026, ntest = 0)
    rng = Xoshiro(seed)
    Q = Matrix(qr(randn(rng, d, d)).Q)
    s = [Float64(j)^(-decay) for j in 1:d]
    M = Diagonal(s) * Q'
    Afull = randn(rng, n + ntest, d) * M
    xt = randn(rng, d)
    z = Afull * xt
    xt .*= snr / std(z)
    z = Afull * xt
    y = [rand(rng) < 1 / (1 + exp(-zi)) ? 1.0 : -1.0 for zi in z]
    P = LogReg(Afull[1:n, :], y[1:n], lam * n)
    test = (Afull[n+1:end, :], y[n+1:end])
    return P, test, xt
end

# ╔═╡ 1281a006-687f-454d-8b88-dc63e4dbe80c
md"""
### Evaluación estable: `l1pe`

Calcula `log(1 + exp(t))` evitando desbordamientos numéricos cuando `t` es muy grande. Esta función se usa al evaluar la pérdida logística.
"""

# ╔═╡ 66a94d59-bd13-4248-a15d-2eb8fc657e4f
@inline l1pe(t::Float64) = t > 0 ? t + log1p(exp(-t)) : log1p(exp(t))   # log(1+e^t)

# ╔═╡ 131511a4-f9aa-4a5c-9089-900148ff3b27
md"""
### Espacio de trabajo `WS`

Guarda vectores temporales reutilizables (productos matriz-vector, gradiente, dirección de Newton y puntos candidatos). Así se evitan nuevas asignaciones de memoria en cada iteración.
"""

# ╔═╡ dcaebc11-c273-4aab-a56f-a29c1eb99628
begin
mutable struct WS
    z::Vector{Float64}     # n : A*x
    r::Vector{Float64}     # n : derivative of loss wrt a_i'x
    sw::Vector{Float64}    # n : sqrt Hessian weights at x
    swn::Vector{Float64}   # n : sqrt Hessian weights at the candidate point
    g::Vector{Float64}
    gn::Vector{Float64}
    xn::Vector{Float64}
    v::Vector{Float64}
    vn::Vector{Float64}
end
WS(n, d) = WS(zeros(n), zeros(n), zeros(n), zeros(n), zeros(d), zeros(d), zeros(d), zeros(d), zeros(d))
end

# ╔═╡ 48947f90-d78a-47a3-aa7a-5b648041e8fc
md"""
### Espacio de trabajo `SkWS`

Almacena las matrices y los vectores temporales usados por los sketches y por la resolución del sistema lineal. Por ejemplo, `SB` guarda el producto comprimido, `S` un panel gaussiano, `C` las columnas temporales de Hadamard y `H` la matriz del sistema de Newton.
"""

# ╔═╡ 0c9771c1-8929-4959-8213-827328383ea0
begin
mutable struct SkWS
    SB::Matrix{Float64}    # m x d  sketched Hessian square root
    S::Matrix{Float64}     # m x b  Gaussian panel reutilizable
    C::Matrix{Float64}     # npad x workers SRHT buffers
    B::Matrix{Float64}     # n x d  (only when m >= n: exact Hessian)
    h::Vector{Int}         # CountSketch buckets
    sg::Vector{Float64}    # signs * sqrt weights
    perm::Vector{Int}      # SRHT row subset
    idx::Vector{Int}       # row-sampling indices
    H::Matrix{Float64}     # d x d or m x m system matrix
    t::Vector{Float64}
end
SkWS() = SkWS(zeros(0, 0), zeros(0, 0), zeros(0, 0), zeros(0, 0), Int[], Float64[], Int[], Int[], zeros(0, 0), Float64[])
end

# ╔═╡ daca700e-1e3b-45b8-aeca-485637e43363
md"""
### Valor de la función objetivo: `fval!`

Evalúa la suma de pérdidas logísticas y la penalización cuadrática en `x`, reutilizando el espacio de trabajo `ws`. El signo `!` indica que modifica buffers existentes.
"""

# ╔═╡ 6338d899-53e7-4c79-9a44-a501ba5f342c
function fval!(P::LogReg, ws::WS, x::Vector{Float64})
    mul!(ws.z, P.A, x)
    s = 0.0
    @inbounds for i in eachindex(ws.z)
        s += l1pe(-P.y[i] * ws.z[i])
    end
    return s + 0.5 * P.mu * dot(x, x)
end

# ╔═╡ a39e85cb-5b13-4c12-ae81-328b0be628a8
md"""
### Objetivo, gradiente y pesos: `fgw!`

Calcula de una vez el valor de `f(x)`, el gradiente `g` y el vector `sw` con los pesos de la raíz del Hessiano. Si `pᵢ = σ(-yᵢ aᵢᵀx)`, entonces `sw[i] = √(pᵢ(1-pᵢ))`. Estos pesos permiten escribir el Hessiano como `BᵀB + mu*I`, con `B = diag(sw)*A`.
"""

# ╔═╡ 5db4de9c-21f0-482e-bac8-8edd953f2de4
function fgw!(P::LogReg, ws::WS, x::Vector{Float64}, g::Vector{Float64}, sw::Vector{Float64})
    mul!(ws.z, P.A, x)
    s = 0.0
    @inbounds for i in eachindex(ws.z)
        mi = P.y[i] * ws.z[i]
        s += l1pe(-mi)
        p = 1 / (1 + exp(mi))              # sigma(-m_i)
        ws.r[i] = -P.y[i] * p
        sw[i] = sqrt(p * (1 - p))
    end
    mul!(g, P.A', ws.r)
    g .+= P.mu .* x
    return s + 0.5 * P.mu * dot(x, x)
end

# ╔═╡ 1a7a49a8-1681-4a1c-b221-f777aa2ecdb2
md"""
### Evaluación sencilla: `logreg_obj`

Devuelve el valor de la función objetivo en un punto `x`. Es una interfaz cómoda que crea internamente los buffers necesarios.
"""

# ╔═╡ c37af10d-81e5-4817-96fc-dfd349d7f3d5
logreg_obj(P::LogReg, x) = fval!(P, WS(size(P.A)...), Vector{Float64}(x))

# ╔═╡ e1bba540-73c0-4832-bb58-316c1926747c
md"""
### Exactitud de clasificación: `test_accuracy`

Calcula la proporción de etiquetas correctamente predichas mediante el signo de `A*x` en el conjunto indicado.
"""

# ╔═╡ 8e103f08-6a35-421a-8953-0abcf5d9a04b
test_accuracy(test, x) = mean(sign.(test[1] * x) .== test[2])

# ╔═╡ b983097c-b7c6-49c7-b9c8-3cbd55265d07
md"""
## 1.2. Tipos de matrices de sketching

Todos los métodos usan una matriz de compresión `S` de tamaño `m × n`, pero la construyen de manera diferente. Los tipos `GaussianSketch`, `CountSketch`, `RowSampling` y `SRHTSketch` identifican el procedimiento sin guardar parámetros adicionales.

La matriz relevante es la **raíz ponderada del Hessiano de la pérdida**, `B = diag(sw)*A`, de tamaño `n × d`. Los cuatro métodos buscan construir `SB = S*B` (tamaño `m × d`), para aproximar

```math
\nabla^2 f(x)=B^{\mathsf T}B+\mu I_d
\qquad\text{por}\qquad
\widetilde H(x)=(SB)^{\mathsf T}(SB)+\mu I_d.
```

La regularización `mu*I` **no se proyecta**: se suma después de hacer el sketch.
"""

# ╔═╡ 47689f2a-4851-4104-aa7d-47d61a35a8b4
begin
abstract type Sketch end
struct GaussianSketch <: Sketch end
struct CountSketch    <: Sketch end
struct RowSampling    <: Sketch end
struct SRHTSketch     <: Sketch end
end

# ╔═╡ bcde3567-77bf-425d-b6b7-d0ecb5d2fd1c
md"""
### Nombres de los sketches: `sketchname`

Devuelve una cadena de texto con el nombre de cada método. Se utiliza únicamente para mostrar e identificar resultados.
"""

# ╔═╡ 79a65887-b030-43de-9c37-1440212c7716
begin
sketchname(::GaussianSketch) = "Gaussian"
sketchname(::CountSketch)    = "CountSketch"
sketchname(::RowSampling)    = "RowSampling"
sketchname(::SRHTSketch)     = "SRHT"
end

# ╔═╡ 0c519c48-5733-4b6f-a1e2-c16977142d01
md"""
### Reserva del resultado: `getSB!`

Prepara una matriz `SB` de tamaño `m × d`. Si ya existe una con las dimensiones correctas, la reutiliza en lugar de asignar otra.
"""

# ╔═╡ 2562a36d-33b0-4e7d-bddf-c0854e23ae58
function getSB!(sk::SkWS, m::Int, d::Int)
    if size(sk.SB) != (m, d)
        sk.SB = Matrix{Float64}(undef, m, d)
    end
    return sk.SB
end

# ╔═╡ 18ce628f-3a2d-4ba7-b044-05340d8c2997
md"""
#### Tamaño del bloque gaussiano: `gaussian_panel_cols`

La matriz gaussiana se genera en **paneles de columnas**. Se limita el buffer a un
presupuesto aproximado de **8 MiB**, con un máximo de 2048 columnas por panel.
Esta decisión controla la memoria, sin modificar la distribución de la matriz aleatoria.
Un panel más pequeño ahorra memoria pero produce más llamadas a BLAS; el valor elegido
es un punto de partida razonable y no pretende ser óptimo para todo hardware.
"""

# ╔═╡ d859320e-a73a-4fbf-8efe-9a8feb70cb57
function gaussian_panel_cols(m::Int, n::Int)
    @assert m > 0 && n > 0
    budget = 8 * 1024^2    # bytes destinados a G_panel
    return min(n, 2048, max(1, budget ÷ (sizeof(Float64) * m)))
end

# ╔═╡ 1cd89be1-5046-4e5d-900b-f2da16b40ad3
md"""
### Gaussian Sketch: implementación por bloques con BLAS

**Modelo matemático (inalterado).** Se utiliza una matriz gaussiana
`G ∈ ℝ^(m×n)` con entradas independientes `N(0,1)` y

```math
SB = \frac{1}{\sqrt m}G\operatorname{diag}(sw)A.
```

La identidad `E[SᵀS]=I` sigue siendo válida. El Hessiano aproximado
es `(SB)ᵀ(SB) + μI`, exactamente como antes.

**Cambio computacional.** No se guarda `G` completa. Se divide el conjunto de filas
de `A` en bloques `I₁,I₂,…` de hasta `b = gaussian_panel_cols(m,n)` filas y se acumula

```math
SB = \sum_{q} \frac{1}{\sqrt m}
      G_{:,I_q}\operatorname{diag}(sw_{I_q})A_{I_q,:}.
```

1. Se reutiliza un único panel `G_panel` de tamaño `m×b`.
2. Se generan sus entradas normales y se ponderan sus columnas por `sw[i]/√m`.
3. `mul!` usa BLAS (GEMM) para acumular el producto en `SB`, sin construir `diag(sw)*A`.
4. El último panel se procesa mediante vistas sin crear una copia.

**Memoria extra del sketch:** `O(md + mb)` en lugar de `O(md + mn)`;
`b ≤ 2048` y se controla por un presupuesto de 8 MiB. El costo aritmético
sigue siendo `O(mnd)`; una gaussiana densa no puede convertirse en un sketch
rápido tipo CountSketch solamente por ahorrar memoria.

**Nota:** el orden de los productos matriciales cambia, de modo que pueden aparecer
diferencias del orden del redondeo numérico respecto a calcular un único GEMM.
"""

# ╔═╡ 4320e82a-d8c3-4025-ae47-f23b41b51373
function sketch_gaussian!(::GaussianSketch, sk::SkWS, A::Matrix{Float64}, sw::Vector{Float64}, m::Int, rng)
    n, d = size(A)
    @assert length(sw) == n && m > 0
    SB = getSB!(sk, m, d)
    b = gaussian_panel_cols(m, n)
    if size(sk.S) != (m, b)
        sk.S = Matrix{Float64}(undef, m, b)
    end
    Gpanel = sk.S
    scale = inv(sqrt(Float64(m)))

    for first in 1:b:n
        last = min(n, first + b - 1)
        nb = last - first + 1
        # Las vistas evitan materializar A[first:last, :] y Gpanel[:, 1:nb].
        G = @view Gpanel[:, 1:nb]
        Ab = @view A[first:last, :]
        randn!(rng, G)
        @inbounds for k in 1:nb
            wk = sw[first + k - 1] * scale
            @simd for i in 1:m
                G[i, k] *= wk
            end
        end
        # Primer panel: SB = G*Ab. Siguientes: SB += G*Ab.
        mul!(SB, G, Ab, 1.0, first == 1 ? 0.0 : 1.0)
    end
    return SB
end

# ╔═╡ 2f4a307e-9ef3-4cec-8748-7e0a96aecce2
md"""
### CountSketch

**Idea:** cada columna de `S` tiene **exactamente un elemento no nulo**, de valor `+1` o `-1`. Cada fila original se asigna aleatoriamente a una de las `m` filas comprimidas (un *bucket*), con un signo independiente.

Para cada fila original `i`, se sortean `h(i) ∈ {1,…,m}` y `ξᵢ ∈ {-1,+1}`. La fila `k` del resultado es

```math
(SB)_{k,:}=\sum_{i:\,h(i)=k}\xi_i\,sw_i\,A_{i,:}.
```

**Qué hace el código:** (1) `h` guarda los buckets; (2) `sg` guarda los signos ya multiplicados por `sw`; (3) se pone `SB` en cero; (4) se acumula `sg[i]*A[i,j]` en `SB[h[i],j]`. El bucle externo sobre columnas `j` permite paralelizar con `Threads.@threads` sin que dos hilos escriban la misma entrada de `SB`.

**Por qué hay signos:** reducen el sesgo de las sumas cuando varias filas colisionan en un mismo bucket. En particular, `E[SᵀS]=Iₙ` y `E[||SBu||²]=||Bu||²` para vectores fijos `u`.

**Costo:** un recorrido `O(nd)` para una matriz densa (`O(nnz(A))` de manera conceptual) y memoria adicional `O(n+md)` sin formar la matriz dispersa `S`. Su precisión puede sufrir por colisiones si `m` es demasiado pequeño.
"""

# ╔═╡ 410332ff-c5a5-4ff4-a636-aced6acbb0d8
function sketch_count!(::CountSketch, sk::SkWS, A::Matrix{Float64}, sw::Vector{Float64}, m::Int, rng)
    n, d = size(A)
    SB = getSB!(sk, m, d)
    if length(sk.h) != n
        sk.h = Vector{Int}(undef, n)
    end
    if length(sk.sg) != n
        sk.sg = Vector{Float64}(undef, n)
    end
    h = sk.h
    sg = sk.sg
    rand!(rng, h, 1:m)
    @inbounds for i in 1:n
        sg[i] = ifelse(rand(rng, Bool), sw[i], -sw[i])
    end
    fill!(SB, 0.0)
    Threads.@threads for j in 1:d
        @inbounds for i in 1:n
            SB[h[i], j] += sg[i] * A[i, j]
        end
    end
    return SB
end

# ╔═╡ a298cde7-17e6-4555-89a1-01635ad1a81e
md"""
### Row Sampling: muestreo uniforme de filas

**Idea:** en lugar de combinar filas, se eligen `m` filas de `B` uniformemente **con reemplazo**. Una misma fila puede aparecer varias veces y otras pueden no aparecer.

Si `i₁,…,iₘ` son índices independientes, uniformes en `{1,…,n}`, se construye

```math
(SB)_{k,:}=\sqrt{\frac{n}{m}}\,sw_{i_k}\,A_{i_k,:},
\qquad k=1,\ldots,m.
```

**Qué hace el código:** (1) almacena los índices seleccionados en `idx`; (2) genera índices con `rand!(rng, idx, 1:n)`; (3) copia las filas ponderadas directamente a `SB`, multiplicando por `√(n/m)`. Así no se crea la matriz de selección `S` explícita.

**Por qué se reescala:** garantiza `E[(SB)ᵀ(SB)] = BᵀB`. Sin el factor `√(n/m)`, la curvatura estimada sería demasiado pequeña en promedio.

**Costo:** `O(md)` operaciones para los datos densos, y `O(md+m)` memoria para el resultado y los índices. Es muy eficiente si todas las filas tienen relevancia comparable, pero puede perder direcciones importantes cuando las **leverage scores** están muy concentradas: el muestreo uniforme no favorece esas filas.
"""

# ╔═╡ 0930da91-614a-4a52-a59b-bcf13aedb490
function sketch_sampling!(::RowSampling, sk::SkWS, A::Matrix{Float64}, sw::Vector{Float64}, m::Int, rng)
    n, d = size(A)
    SB = getSB!(sk, m, d)
    if length(sk.idx) != m
        sk.idx = Vector{Int}(undef, m)
    end
    idx = sk.idx
    rand!(rng, idx, 1:n)
    c = sqrt(n / m)
    @inbounds for j in 1:d
        for k in 1:m
            i = idx[k]
            SB[k, j] = c * sw[i] * A[i, j]
        end
    end
    return SB
end

# ╔═╡ a066eb01-4503-4a3a-a800-97ace6392ac3
md"""
### Transformada rápida de Walsh–Hadamard: `fwht!`

Calcula **in situ** la transformada de Hadamard de un vector cuya longitud es una potencia de dos. Utiliza operaciones de suma y resta en etapas de tipo mariposa, con costo `O(n log n)`. En el código esta versión **no está normalizada**; la normalización se aplica después, en el sketch SRHT.
"""

# ╔═╡ f5f5597e-8008-4dbd-8f25-d3775425d161
function fwht!(x::AbstractVector{Float64})
    n = length(x)
    h = 1
    @inbounds while h < n
        for i0 in 0:(2h):(n - 1)
            @simd for j in (i0 + 1):(i0 + h)
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
            end
        end
        h <<= 1
    end
    return x
end

# ╔═╡ 730c6a8e-f118-4333-b5e2-d1282609ebb4
md"""
#### Paralelismo y presupuesto de memoria de SRHT: `srht_worker_count`

Se asigna **una columna temporal de longitud `np` por trabajador**, no por columna
de la matriz de datos. Los trabajadores se limitan por el número de hilos, por `d`
y por un presupuesto aproximado de **16 MiB**. La reducción de trabajadores
puede ahorrar memoria a costa de disminuir el paralelismo si `np` es muy grande.
"""

# ╔═╡ 24893b06-287d-4b73-a8be-d05c5eb1b81e
function srht_worker_count(np::Int, d::Int)
    @assert np > 0 && d > 0
    budget = 16 * 1024^2
    by_memory = max(1, budget ÷ (sizeof(Float64) * np))
    return min(d, Threads.nthreads(), by_memory)
end

# ╔═╡ db95c50c-51dc-410e-adf2-2116751856cb
md"""
### SRHT optimizada: Hadamard en espacio reducido por trabajador

**Modelo matemático (inalterado).** Con `np=nextpow(2,n)`, `H` la matriz
normalizada de Hadamard, `D` una diagonal aleatoria de signos y `R` un selector
uniforme de `mm=min(m,np)` filas sin reemplazo, se usa

```math
SB = \sqrt{\frac{np}{mm}}\;RHD\bar B,
\qquad \bar B=\begin{bmatrix}\operatorname{diag}(sw)A\\0\end{bmatrix}.
```

**Optimización de memoria y paralelismo:**

1. Genera los signos una sola vez y los combina con los pesos `sw`.
2. Usa Fisher–Yates parcial para escoger las filas sin reemplazo (`O(mm)`).
3. Divide las columnas de `A` entre `workers = srht_worker_count(np,d)` tareas.
4. Cada tarea reutiliza una columna de `C` de longitud `np`: copia una columna
   ponderada de `A`, rellena los ceros, ejecuta `fwht!` **in situ** y extrae
   únicamente las filas escogidas.
5. Los trabajadores escriben columnas disjuntas de `SB`; no hay bloqueos
   ni buffers que se compartan al transformar.

La transformada `fwht!` es **no normalizada**, por lo que el factor en código
es `1/√mm` (equivalente a la expresión con `H` normalizada).

**Memoria temporal:** `O(np·workers + np + n + md)` en lugar de
`O(np·d + np + n + md)`. `C` se limita a unos 16 MiB. El costo computacional
permanece `O(np·d·log(np))`, y el padding a potencia de dos sigue siendo
necesario en esta implementación. Este cambio reduce presión sobre memoria y caché,
pero **no garantiza** mayor velocidad en toda máquina o dimensión.
"""

# ╔═╡ fca7fd8f-e112-4161-a267-ee142395760e
function sketch_srht!(::SRHTSketch, sk::SkWS, A::Matrix{Float64}, sw::Vector{Float64}, m::Int, rng)
    n, d = size(A)
    @assert length(sw) == n && m > 0
    np = nextpow(2, n)
    mm = min(m, np)
    workers = srht_worker_count(np, d)
    SB = getSB!(sk, mm, d)
    # Antes: C tenía np*d elementos. Ahora: np*workers (reutilizados).
    if size(sk.C) != (np, workers)
        sk.C = Matrix{Float64}(undef, np, workers)
    end
    if length(sk.perm) != np
        sk.perm = collect(1:np)
    end
    if length(sk.sg) != n
        sk.sg = Vector{Float64}(undef, n)
    end
    C = sk.C
    sg = sk.sg
    perm = sk.perm
    @inbounds for i in 1:n
        sg[i] = ifelse(rand(rng, Bool), sw[i], -sw[i])
    end
    # Uniforme sin reemplazo, sin materializar la matriz S.
    @inbounds for k in 1:mm
        r = rand(rng, k:np)
        perm[k], perm[r] = perm[r], perm[k]
    end
    scale = inv(sqrt(Float64(mm)))

    # Cada tarea posee una columna diferente de C. No depende del threadid()
    # y sigue siendo seguro aunque Julia migre las tareas entre hilos.
    Threads.@threads for worker in 1:workers
        col = @view C[:, worker]
        for j in worker:workers:d
            @inbounds @simd for i in 1:n
                col[i] = sg[i] * A[i, j]
            end
            @inbounds for i in (n + 1):np
                col[i] = 0.0
            end
            fwht!(col)
            @inbounds @simd for k in 1:mm
                SB[k, j] = scale * col[perm[k]]
            end
        end
    end
    return SB
end

# ╔═╡ 4fea99cf-1ee1-42dd-b17a-9837e3965f50
md"""
### Interfaz común `sketch!`

Esta función selecciona automáticamente la implementación según el tipo de sketch. En Pluto, sus cuatro métodos se definen **juntos en una sola celda** para evitar conflictos por definiciones repetidas del mismo nombre. Los algoritmos reales están separados en las celdas anteriores.
"""

# ╔═╡ 8033b036-036f-48e5-818b-5107d685659d
begin
    sketch!(::GaussianSketch, sk::SkWS, A::Matrix{Float64}, sw::Vector{Float64}, m::Int, rng) =
        sketch_gaussian!(GaussianSketch(), sk, A, sw, m, rng)
    sketch!(::CountSketch, sk::SkWS, A::Matrix{Float64}, sw::Vector{Float64}, m::Int, rng) =
        sketch_count!(CountSketch(), sk, A, sw, m, rng)
    sketch!(::RowSampling, sk::SkWS, A::Matrix{Float64}, sw::Vector{Float64}, m::Int, rng) =
        sketch_sampling!(RowSampling(), sk, A, sw, m, rng)
    sketch!(::SRHTSketch, sk::SkWS, A::Matrix{Float64}, sw::Vector{Float64}, m::Int, rng) =
        sketch_srht!(SRHTSketch(), sk, A, sw, m, rng)
end

# ╔═╡ 9eecaba9-8e72-460e-9d32-fda1751d231a
md"""
### Comprobación numérica de las optimizaciones

`validate_optimized_sketches` verifica con matrices pequeñas dos propiedades:

- Gaussian por bloques coincide, hasta error de redondeo, con el producto denso
  construido con **la misma semilla**. Se usa `n>2048` para obligar a trabajar con
  más de un panel; `randn!` recorre los paneles en orden de columnas.
- SRHT con **todas** las `np` filas seleccionadas preserva `BᵀB` a precisión numérica,
  por ortogonalidad de Hadamard y del cambio aleatorio de signos.

También comprueba que los buffers tienen las dimensiones reducidas. Si algo
cambia accidentalmente la matemática, la celda lanza un error. Esta prueba
es pequeña y no reemplaza los benchmarks de rendimiento.
"""

# ╔═╡ ef0de825-4da3-429f-8756-1bdc88bafcce
function validate_optimized_sketches()
    rng = Xoshiro(202603)
    n, d, m = 2300, 5, 48
    A = randn(rng, n, d)
    sw = 0.1 .+ rand(rng, n)
    sk = SkWS()
    result = copy(sketch_gaussian!(GaussianSketch(), sk, A, sw, m, Xoshiro(41)))
    # Referencia densa: genera los mismos paneles con la misma secuencia RNG,
    # pero ensambla G completa y hace un único producto matricial.
    G = Matrix{Float64}(undef, m, n)
    ref_rng = Xoshiro(41)
    panel = gaussian_panel_cols(m, n)
    for first in 1:panel:n
        last = min(n, first + panel - 1)
        randn!(ref_rng, @view G[:, first:last])
    end
    expected = (G * (sw .* A)) / sqrt(m)
    err_gaussian = norm(result - expected) / max(norm(expected), eps(Float64))
    @assert err_gaussian < 1e-10 "Error inesperado en Gaussian por bloques"
    @assert size(sk.S) == (m, gaussian_panel_cols(m, n))

    n2, d2 = 73, 6
    A2 = randn(rng, n2, d2)
    sw2 = 0.2 .+ rand(rng, n2)
    np = nextpow(2, n2)
    sk2 = SkWS()
    full = copy(sketch_srht!(SRHTSketch(), sk2, A2, sw2, np, Xoshiro(72)))
    B2 = sw2 .* A2
    err_srht = norm(full' * full - B2' * B2) / norm(B2' * B2)
    @assert err_srht < 1e-10 "SRHT no preservó B'B en transformada completa"
    @assert size(sk2.C) == (np, srht_worker_count(np, d2))
    return (error_gaussian = err_gaussian, error_srht = err_srht,
            gaussian_panel = size(sk.S), srht_buffer = size(sk2.C), validacion = "OK")
end

# ╔═╡ d19c88cf-72e1-49f8-a932-1e6bd2831b6f
FAST_SKETCH_CHECK = validate_optimized_sketches()

# ╔═╡ ed73bb18-c5be-4713-9452-a1d35c7b02aa
md"""
## 1.3. Dirección de Newton aproximada: `sk_direction!`

Resuelve el sistema de Newton usando la aproximación de curvatura generada por `SB`:

```math
\widetilde H=(SB)^{\mathsf T}(SB)+\mu I_d,
\qquad v=-\widetilde H^{-1}g.
```

Si `m ≥ d`, forma una matriz `d × d` y resuelve mediante Cholesky. Si `m < d`, emplea la identidad de Woodbury para resolver un sistema `m × m`, lo que puede ahorrar trabajo cuando el sketch es muy pequeño. Devuelve también el decremento aproximado `√max(-gᵀv, 0)`.

La regularización positiva hace que los sistemas sean definidos positivos. Esta función reutiliza los buffers del objeto `SkWS`.
"""

# ╔═╡ 1d9aefa9-396c-43d2-88a7-e098cd47bed7
function sk_direction!(v::Vector{Float64}, sk::SkWS, SB::Matrix{Float64}, g::Vector{Float64}, mu::Float64)
    m, d = size(SB)
    if m >= d
        if size(sk.H) != (d, d)
            sk.H = Matrix{Float64}(undef, d, d)
        end
        H = sk.H
        BLAS.syrk!('U', 'T', 1.0, SB, 0.0, H)
        @inbounds for i in 1:d
            H[i, i] += mu
        end
        F = cholesky!(Symmetric(H, :U))
        copyto!(v, g)
        ldiv!(F, v)
        v .*= -1
    else
        if size(sk.H) != (m, m)
            sk.H = Matrix{Float64}(undef, m, m)
        end
        K = sk.H
        BLAS.syrk!('U', 'N', 1.0, SB, 0.0, K)
        @inbounds for i in 1:m
            K[i, i] += mu
        end
        F = cholesky!(Symmetric(K, :U))
        if length(sk.t) != m
            sk.t = Vector{Float64}(undef, m)
        end
        t = sk.t
        mul!(t, SB, g)
        ldiv!(F, t)
        mul!(v, SB', t)
        @inbounds for i in 1:d
            v[i] = -(g[i] - v[i]) / mu
        end
    end
    return sqrt(max(-dot(g, v), 0.0))
end

# ╔═╡ f8aa3dcf-27ac-49d1-aef5-ce459d445e56
"""
    validate_numerical_methods(; trials=320)

Contrasta ambos sistemas de resolución (Cholesky y Woodbury) contra solución
directa y verifica de manera empírica la insesgadez del Gram aproximado para
los cuatro sketches. Las tolerancias estadísticas son amplias y reproducibles:
no sustituyen análisis teórico ni pruebas bajo diferentes condicionamientos.
"""
function validate_numerical_methods(; trials::Int=320)
    @assert trials >= 100
    rng=Xoshiro(20261115)
    A=randn(rng,65,5)
    sw=0.2 .+ 0.7.*rand(rng,65)
    B=sw .* A
    gram=B'*B
    g=randn(rng,5)
    mu=0.7
    solves=NamedTuple[]
    for m in (3,9)
        SB=randn(rng,m,5)
        v=zeros(5)
        val=sk_direction!(v,SkWS(),SB,g,mu)
        vref=-((SB'*SB + mu*I)\g)
        err=norm(v-vref)/max(norm(vref),eps(Float64))
        @assert err < 1e-9 "Error en resolución Cholesky / Woodbury"
        @assert isapprox(val,sqrt(max(-dot(g,v),0.0));rtol=1e-12)
        push!(solves,(m=m,relative_error=err))
    end
    sketches=NamedTuple[]
    for (j,kind) in enumerate((GaussianSketch(),CountSketch(),RowSampling(),SRHTSketch()))
        mean_gram=zeros(5,5)
        sk=SkWS()
        for t in 1:trials
            SB=sketch!(kind,sk,A,sw,45,Xoshiro(100000*j+t))
            mean_gram .+= SB'*SB
        end
        mean_gram ./= trials
        err=norm(mean_gram-gram)/norm(gram)
        @assert err < 0.28 "Fallo de consistencia estadística del sketch $(sketchname(kind))"
        push!(sketches,(method=sketchname(kind),error_gram=err,replicas=trials))
    end
    return (systems=solves,sketches=sketches,validation="OK")
end

# ╔═╡ 0c128160-e0ca-46da-ae47-096c22b84305
md"""
### Preparación de la dirección: `sketched_direction!`

Si `m < n`, construye `SB` con el sketch seleccionado y llama al solver anterior. Si `m ≥ n`, utiliza directamente la matriz ponderada original, de modo que la dirección corresponde al Hessiano exacto en vez de seguir comprimiendo.
"""

# ╔═╡ ce612ca7-7139-420b-b946-7f1ffab6a815
function sketched_direction!(v, kind::Sketch, sk::SkWS, A::Matrix{Float64}, sw::Vector{Float64},
                             g::Vector{Float64}, mu::Float64, m::Int, rng)
    n, d = size(A)
    if m >= n                                     # sketch as large as the data: use exact Hessian
        if size(sk.B) != (n, d)
            sk.B = Matrix{Float64}(undef, n, d)
        end
        sk.B .= sw .* A
        return sk_direction!(v, sk, sk.B, g, mu)
    end
    SB = sketch!(kind, sk, A, sw, m, rng)
    return sk_direction!(v, sk, SB, g, mu)
end

# ╔═╡ 0e0abbd7-45f5-4ba3-96fd-1dfc1e3b6e35
md"""
### Búsqueda de línea: `backtrack!`

Prueba primero un paso completo y lo va reduciendo multiplicándolo por `B_LS` hasta satisfacer la condición de descenso de Armijo. Devuelve el tamaño aceptado y el nuevo valor de la función; si el paso se vuelve demasiado pequeño, devuelve cero.
"""

# ╔═╡ 93b7d8ad-060b-48ce-806e-97b35d75527a
function backtrack!(P::LogReg, ws::WS, x::Vector{Float64}, v::Vector{Float64}, gv::Float64, fx::Float64)
    # Devuelve paso, valor nuevo y número de reducciones de paso.
    if !(isfinite(gv) && gv < 0)
        copyto!(ws.xn, x)
        return 0.0, fx, 0
    end
    s = 1.0
    reductions = 0
    slack = 10 * eps(Float64) * abs(fx)
    while true
        @. ws.xn = x + s * v
        fn = fval!(P, ws, ws.xn)
        if isfinite(fn) && fn <= fx + A_LS * s * gv + slack
            return s, fn, reductions
        end
        s *= B_LS
        reductions += 1
        if s < 1e-12
            copyto!(ws.xn, x)
            return 0.0, fx, reductions
        end
    end
end

# ╔═╡ a5b3934a-00a1-438b-989d-7a08d85a2def
md"""
### Registro de iteraciones: `rec!`

Si se proporciona un objeto `trace`, guarda el número de iteración, el tiempo transcurrido, el valor de la función y el tamaño `m` del sketch.
"""

# ╔═╡ 9cc2f7b2-b142-45bf-9c80-9ff8a408c9f7
@inline function rec!(trace, it, t0, f, m)
    trace === nothing && return nothing
    push!(trace, (Float64(it), (time_ns() - t0) / 1e9, f, Float64(m)))
    return nothing
end

# ╔═╡ 348a3bba-893b-4efd-a5f2-3935ef033ed5
md"""
## 1.4. Método de Newton exacto: `newton_exact`

En cada iteración construye el Hessiano completo `H = Aᵀ diag(sw²) A + mu*I`, resuelve la dirección con Cholesky y aplica búsqueda de línea. Termina cuando la mitad del cuadrado del decremento de Newton es menor que `delta`, o al alcanzar `maxit`. Sirve como referencia para evaluar los sketches. El parámetro opcional `target_f` permite detenerlo al conseguir `f(x) ≤ target_f` (por defecto está desactivado).
"""

# ╔═╡ 54b5e787-1113-4586-9fdf-cb481162ebe5
function newton_exact(P::LogReg; x0 = zeros(size(P.A, 2)), delta = 1e-8,
                      maxit = 100, trace = nothing, target_f = -Inf,
                      stop_on_decrement::Bool = true)
    n, d = size(P.A)
    @assert maxit >= 0
    t0 = time_ns()
    ws = WS(n, d)
    B = Matrix{Float64}(undef, n, d)
    H = Matrix{Float64}(undef, d, d)
    x = copy(x0)
    f = fgw!(P, ws, x, ws.g, ws.sw)
    it, backtracks = 0, 0
    last_lam2 = NaN
    reason = :unknown
    while true
        # Registra también el punto final, sin pagar un Hessiano adicional.
        rec!(trace, it, t0, f, n)
        if f <= target_f
            reason = :target
            break
        elseif it >= maxit
            reason = :maxit
            break
        end
        B .= ws.sw .* P.A
        BLAS.syrk!('U', 'T', 1.0, B, 0.0, H)
        @inbounds for i in 1:d
            H[i, i] += P.mu
        end
        F = cholesky!(Symmetric(H, :U))
        copyto!(ws.v, ws.g)
        ldiv!(F, ws.v)
        ws.v .*= -1
        last_lam2 = max(-dot(ws.g, ws.v), 0.0)
        if stop_on_decrement && last_lam2 / 2 <= delta
            reason = :decrement
            break
        end
        step, _, nred = backtrack!(P, ws, x, ws.v, dot(ws.g, ws.v), f)
        backtracks += nred
        if step == 0.0
            reason = :line_search_failed
            break
        end
        copyto!(x, ws.xn)
        f = fgw!(P, ws, x, ws.g, ws.sw)
        it += 1
    end
    return (x = x, f = f, iters = it, m_final = n,
            rejected = backtracks, backtracks = backtracks,
            last_decrement_sq = last_lam2, reason = reason)
end

# ╔═╡ af18ff88-18c5-46d9-86b8-214c7f1458f0
md"""
### Solución de referencia: `reference_solution`

Ejecuta Newton exacto con una tolerancia de decremento mucho más estricta y hasta 60 iteraciones. Se utiliza para comparar las soluciones y los valores finales de la función objetivo.
"""

# ╔═╡ 275de420-a266-4e85-b093-475fee814919
reference_solution(P::LogReg) = newton_exact(P; delta = 1e-26, maxit = 60)

# ╔═╡ 3a2a72da-e0a7-41bb-96c0-9871bfb82691
"""
    reference_certificate(P, ref; tau_min=1e-4, fraction=0.1, strict=true)

Certifica el error de la referencia usando la fuerte convexidad (μ>0):
0 <= f(ref.x)-f* <= ||grad f(ref.x)||²/(2μ).
Se exige que el error relativo de referencia sea a lo sumo
`fraction*tau_min` del gap inicial. Con `strict=true` detiene el análisis si
la certificación no es suficientemente fuerte.
"""
function reference_certificate(P::LogReg, ref;
                               tau_min::Real = 1e-4, fraction::Real = 0.1,
                               strict::Bool = true)
    @assert P.mu > 0 && 0 < tau_min < 1 && 0 < fraction < 1
    n, d = size(P.A)
    ws = WS(n, d)
    g = zeros(d)
    sw = zeros(n)
    fx = fgw!(P, ws, ref.x, g, sw)
    f0 = logreg_obj(P, zeros(d))
    gap0 = f0 - fx
    gap0 > 100*eps(Float64) * max(abs(f0), 1.0) ||
        error("Gap inicial demasiado pequeño para certificar la comparación.")
    bound = dot(g, g) / (2 * P.mu)
    rel_bound = bound / gap0
    certified = isfinite(rel_bound) && rel_bound <= fraction * tau_min
    if strict && !certified
        error("Referencia no certificada: cota relativa=$(rel_bound), " *
              "límite=$(fraction*tau_min). Ajusta la solución de referencia o τ.")
    end
    return (f0 = f0, f_ref = fx, gap0 = gap0, grad_inf = norm(g, Inf),
            grad_norm = norm(g), bound = bound, relative_bound = rel_bound,
            tau_min = Float64(tau_min), fraction = Float64(fraction),
            certified = certified, reference_reason = ref.reason)
end

# ╔═╡ 9ebdc73a-39a7-452d-b0df-70550dba1a5f
md"""
## 1.5. Newton Sketch: `newton_sketch`

Mantiene el gradiente de la función original, pero sustituye la parte de datos del Hessiano por `SBᵀSB`. En cada iteración genera un nuevo sketch de tamaño `m`, calcula la dirección aproximada y realiza una búsqueda de línea **sobre la función original**. Se detiene usando el decremento aproximado o `maxit`.

El sketch reduce el costo de representar la curvatura, pero su eficacia depende de `m`, de la estructura de los datos y del método de compresión elegido. Esta implementación emplea tamaño de sketch **fijo**, no adaptativo. El parámetro opcional `target_f` detiene el método al alcanzar una precisión especificada en valor objetivo, sin modificar la regla normal de parada cuando no se proporciona.
"""

# ╔═╡ 2f00d7e1-a197-4127-9a63-007b09c193e3
function newton_sketch(P::LogReg, kind::Sketch, m::Int;
                       rng = Xoshiro(1), x0 = zeros(size(P.A, 2)),
                       delta = 1e-8, maxit = 500, trace = nothing,
                       target_f = -Inf, stop_on_decrement::Bool = true)
    n, d = size(P.A)
    @assert m >= 1 && maxit >= 0
    t0 = time_ns()
    m = min(m, n)
    ws = WS(n, d)
    sk = SkWS()
    x = copy(x0)
    f = fgw!(P, ws, x, ws.g, ws.sw)
    it, backtracks = 0, 0
    last_lam2 = NaN
    reason = :unknown
    while true
        rec!(trace, it, t0, f, m)
        if f <= target_f
            reason = :target
            break
        elseif it >= maxit
            reason = :maxit
            break
        end
        lt = sketched_direction!(ws.v, kind, sk, P.A, ws.sw, ws.g, P.mu, m, rng)
        last_lam2 = lt^2
        if stop_on_decrement && last_lam2 / 2 <= delta
            reason = :decrement
            break
        end
        step, _, nred = backtrack!(P, ws, x, ws.v, dot(ws.g, ws.v), f)
        backtracks += nred
        if step == 0.0
            reason = :line_search_failed
            break
        end
        copyto!(x, ws.xn)
        f = fgw!(P, ws, x, ws.g, ws.sw)
        it += 1
    end
    return (x = x, f = f, iters = it, m_final = m,
            rejected = backtracks, backtracks = backtracks,
            last_decrement_sq = last_lam2, reason = reason)
end

# ╔═╡ 80df2ff1-09db-451b-b645-bfb309ce2694
NUMERICAL_VALIDATION = validate_numerical_methods(; trials=320)

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000005
begin
    BLAS.set_num_threads(Threads.nthreads())
    (julia_version = VERSION, threads = Threads.nthreads(),
     blas = BLAS.get_num_threads())
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000006
md"""
## 2. Configuración

Los benchmarks costosos de las secciones 4–7 se controlan desde aquí y están desactivados por defecto. Los parámetros de las figuras se encuentran en la sección 8.

* `N_ROWS`, `N_COLS`: `n` y `d` del problema.
* `DECAY`: `s_j = j^-DECAY`, la caída del espectro del diseño. Es el parámetro que decide
  cuánto puede ayudar un sketch: con `decay = 1` la dimensión efectiva es mucho menor que
  `d`.
* `LAM`: `μ = LAM * n` (así lo define tu `make_problem`). Ojo: al crecer `n`, crece la
  regularización. Si quieres comparar solo el efecto de `n`, usa `lam` constante.
* `SEC`: tiempo máximo por `@benchmark`; siempre se verifica también el número real de muestras.
"""

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000007

begin
    EXPERIMENT_MODE = :quick
    @assert EXPERIMENT_MODE in (:quick, :final)

    # Activar las tablas de las secciones 4–7
    RUN_COSTLY_BENCHMARKS = true

    # Benchmark
    SEC = 20.0
    BENCH_SAMPLES = 10
    MIN_BENCH_SAMPLES = 5
    BENCH_TAU = 1e-4
    REF_FRACTION = 0.1

    # Dimensiones del problema
    N_ROWS = 5000
    N_COLS = 100

    # Características del problema
    DECAY = 1.0
    LAM = 1e-4
    SNR = 3.0
    SEED = 2026
    N_TEST = 1000

    # Relaciones m/d
    RATIOS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)

    # Convergencia
    DELTA = 1e-8
    MAXIT_SK = 300
    MAXIT_EX = 100

    nothing
end


# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000008
begin
    # El presupuesto temporal no garantiza un número de muestras.
    # Todas las tablas verifican además el número efectivamente obtenido.
    BenchmarkTools.DEFAULT_PARAMETERS.seconds = SEC
    BenchmarkTools.DEFAULT_PARAMETERS.samples = BENCH_SAMPLES
    BenchmarkTools.DEFAULT_PARAMETERS.evals = 1
    (mode = EXPERIMENT_MODE, seconds = SEC,
     requested_samples = BENCH_SAMPLES, minimum_samples = MIN_BENCH_SAMPLES,
     costly_benchmarks_enabled = RUN_COSTLY_BENCHMARKS)
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000009
md"""
## 3. El problema artificial y su solución de referencia
"""

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000010
begin
    P, TEST, XTRUE = make_problem(N_ROWS, N_COLS; decay = DECAY, lam = LAM,
                                  snr = SNR, seed = SEED, ntest = N_TEST)
    (n = size(P.A, 1), d = size(P.A, 2), mu = P.mu,
     norm_xtrue = round(norm(XTRUE), digits = 3),
     acc_x_true_en_test = round(test_accuracy(TEST, XTRUE), digits = 4),
     balance = (mean(P.y .== 1), mean(P.y .== -1)))
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000011
md"""
### Estadísticas del Hessiano y coherencia

`describe_problem` es el reemplazo de tu `problem_stats`. Lo importante para lo que sigue:
**cuán concentradas están las leverage scores**. Si son casi uniformes (`max/mean ≈ 1`),
el muestreo uniforme de filas es casi óptimo y `RowSampling` va a brillar. Si están
concentradas (en MNIST, por ejemplo, `max/mean = 29`), deja de serlo. Esa es una propiedad
de `make_problem`, no del algoritmo.
"""

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000012
begin
    """Dimensión efectiva regularizada de la parte de datos del Hessiano."""
    function weighted_effective_dimension(P::LogReg, x::AbstractVector)
        n, d = size(P.A)
        ws = WS(n, d)
        g = zeros(d)
        sw = zeros(n)
        fgw!(P, ws, Vector{Float64}(x), g, sw)
        B = sw .* P.A
        sig2 = max.(eigvals(Symmetric(B' * B)), 0.0)
        return sum(s -> s / (s + P.mu), sig2)
    end

    function describe_problem(P::LogReg; ntop = 3)
        n, d = size(P.A)
        D0 = 0.25 .* (P.A' * P.A)
        ev_data = max.(eigvals(Symmetric(D0)), 0.0)
        ev = sort(ev_data .+ P.mu, rev = true)
        lev = vec(sum(abs2, svd(P.A).U[:, 1:d], dims = 2))
        neff = count(>(1e-2 * ev[1]), ev)
        deff = sum(s -> s / (s + P.mu), ev_data)
        return (n = n, d = d, mu = P.mu,
                cond = ev[1] / ev[end],
                eig_top = round.(ev[1:min(ntop, length(ev))], digits = 2),
                eig_min = ev[end], neff = neff,
                deff = deff,
                tail_fraction = 1 - neff / d,
                lev_mean = sum(lev) / n,
                lev_max = maximum(lev),
                coherence = (n / d) * maximum(lev))
    end
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000013
md"""
### Tablas

`mdtable` convierte filas en una tabla Markdown para que Pluto la renderice, y `fmt_t`
elige la unidad. Nada de esto toca la numerística.
"""

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000014
begin
    function mdtable(hdr, rows)
        s = "| " * join(hdr, " | ") * " |\n|" * join(fill(" --- ", length(hdr)), "|") * "|\n"
        for r in rows
            s *= "| " * join(r, " | ") * " |\n"
        end
        return Markdown.parse(s)
    end
    fmt_t(ns) = ns < 1e3 ? @sprintf("%.0f ns", ns) :
                ns < 1e6 ? @sprintf("%.3f µs", ns / 1e3) :
                ns < 1e9 ? @sprintf("%.2f ms", ns / 1e6) : @sprintf("%.3f s", ns / 1e9)
    fmt_safe_t(t) = isfinite(t) ? fmt_t(t) : "—"
    fmt_mem(b) = b < 2^20 ? @sprintf("%.0f KiB", b / 2^10) :
                 b < 2^30 ? @sprintf("%.1f MiB", b / 2^20) : @sprintf("%.2f GiB", b / 2^30)
    const KINDS = Dict(:Gaussian => GaussianSketch(), :CountSketch => CountSketch(),
                       :RowSampling => RowSampling(), :SRHT => SRHTSketch())
    const KORDER = (:Gaussian, :CountSketch, :RowSampling, :SRHT)
    function bench_summary(tr; min_samples = MIN_BENCH_SAMPLES)
        count = length(tr.times)
        good = count >= min_samples
        return (count = count, good = good,
                time = good ? median(tr).time : NaN,
                q1 = good ? quantile(tr.times, 0.25) : NaN,
                q3 = good ? quantile(tr.times, 0.75) : NaN,
                mem = median(tr).memory)
    end
    nothing
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000015
begin
    ST = describe_problem(P)
    mdtable(["cantidad", "valor"],
        [["n, d", "$(ST.n), $(ST.d)"],
         ["μ = lam·n", string(round(ST.mu, sigdigits = 4))],
         ["cond(Hessiano en 0)", @sprintf("%.3g", ST.cond)],
          ["dimensión efectiva regularizada en x=0", @sprintf("%.2f / %d", ST.deff, ST.d)],
         ["mayores autovalores", join(ST.eig_top, ", ")],
         ["menor autovalor", @sprintf("%.3g", ST.eig_min)],
         ["dim. efectiva (autoval > 1% del mayor)", "$(ST.neff) de $(ST.d) · el $(round(100 * ST.tail_fraction, digits = 1))% del espectro está por debajo del 1%"],
         ["leverage media / máxima", @sprintf("%.4f / %.4f", ST.lev_mean, ST.lev_max)],
         ["coherencia (n/d)·max ℓ", @sprintf("%.3f", ST.coherence)]])
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000016
begin
    REF = reference_solution(P)
    REF_CERT = reference_certificate(P, REF;
                  tau_min = BENCH_TAU, fraction = REF_FRACTION)
    (reference_iters = REF.iters, f_star = REF.f,
     grad_inf = REF_CERT.grad_inf,
     grad_gap_bound = REF_CERT.bound,
     relative_reference_error_bound = REF_CERT.relative_bound,
     reference_certified = REF_CERT.certified,
     reason = REF.reason,
     effective_dimension_at_reference = weighted_effective_dimension(P, REF.x),
     acc_train = round(test_accuracy((P.A, P.y), REF.x), digits = 4),
     acc_test = round(test_accuracy(TEST, REF.x), digits = 4))
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000017
md"""
## 4. ¿Cuánto de una iteración es el *sketch*?

Lo que explica todo lo demás. Como diagnóstico opcional, con `m = 4d` se mide por separado:

* `gradient` — `fgw!`, el gradiente `A'r` y el diagonal `√D`. Es `O(n·d)` y **no** se puede
  sketchear: es el suelo de cualquier aceleración (Amdahl).
* `sketch S·D·A` — `sketch!(kind, sk, P.A, sw, m, rng)`, construir `S diag(sw) A`.
* `solve` — `sk_direction!(...)`, el Cholesky `d×d` (o el sistema `m×m` de Woodbury).

Y, para comparar, el Hessiano exacto tal como lo arma `newton_exact`: `B .= sw .* A` y
`BLAS.syrk!`. Los buffers están preasignados (`SkWS` reutilizado), así que son costes de
régimen, no de asignación.
"""

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000018
function hess_exact!(H, B, A, sw, mu)
    B .= sw .* A
    BLAS.syrk!('U', 'T', 1.0, B, 0.0, H)
    @inbounds for i in 1:size(H, 1)
        H[i, i] += mu
    end
    return H
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000019
function bench_iteration_parts(P, x, m)
    n, d = size(P.A)
    ws = WS(n, d)
    g, sw = zeros(d), zeros(n)
    fgw!(P, ws, x, g, sw)
    rng = Xoshiro(7)
    out = NamedTuple[]
    B = Matrix{Float64}(undef, n, d)
    H = Matrix{Float64}(undef, d, d)
    t_ex = @benchmark hess_exact!($H, $B, $(P.A), $sw, $(P.mu)) seconds=SEC samples=BENCH_SAMPLES evals=1
    t_gr = @benchmark fgw!($P, $ws, $x, $g, $sw) seconds=SEC samples=BENCH_SAMPLES evals=1
    e, gr = bench_summary(t_ex), bench_summary(t_gr)
    push!(out, (method = "exact", part = "gradient + D·A + syrk",
                t = e.time + gr.time, mem = e.mem, samples = min(e.count, gr.count)))
    sk = SkWS()
    v = zeros(d)
    for nm in KORDER
        kind = KINDS[nm]
        sketch!(kind, sk, P.A, sw, m, rng)
        mm = nm === :SRHT ? min(m, nextpow(2, n)) : m
        SB = getSB!(sk, mm, d)
        t_sk = @benchmark sketch!($kind, $sk, $(P.A), $sw, $m, $rng) seconds=SEC samples=BENCH_SAMPLES evals=1
        t_so = @benchmark sk_direction!($v, $sk, $SB, $g, $(P.mu)) seconds=SEC samples=BENCH_SAMPLES evals=1
        a, b = bench_summary(t_sk), bench_summary(t_so)
        push!(out, (method = String(nm), part = "gradient", t = gr.time, mem = 0, samples = gr.count))
        push!(out, (method = String(nm), part = "sketch S·D·A", t = a.time,
                    mem = a.mem, samples = a.count))
        push!(out, (method = String(nm), part = "solve", t = b.time,
                    mem = b.mem, samples = b.count))
    end
    return out
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000020
if !RUN_COSTLY_BENCHMARKS
    md"Sección 4 desactivada. Cambia `RUN_COSTLY_BENCHMARKS=true` para medir las operaciones."
else
    PARTS = bench_iteration_parts(P, REF.x, 4 * size(P.A, 2))
    rows = Vector{Vector{String}}()
    for nm in (:exact, KORDER...)
        sub = filter(p -> p.method == String(nm), PARTS)
        for p in sub
            push!(rows, [p.method, p.part, fmt_safe_t(p.t), fmt_mem(p.mem), string(p.samples)])
        end
        push!(rows, ["**" * String(nm) * "**", "**total por iteración**",
                     "**" * fmt_safe_t(sum(p -> p.t, sub)) * "**", "", ""])
    end
    mdtable(["método", "operación", "mediana", "asignado", "muestras"], rows)
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000021
md"""
`asignado` es el campo `memory` de BenchmarkTools: bytes por evaluación. Vale `0` donde se
reutiliza el buffer preasignado, que es exactamente para lo que sirve `SkWS`.
"""

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000022
md"""
## 5. De punta a punta: tiempo hasta una **misma precisión**

La comparación principal usa el mismo problema, punto inicial y umbral
`f_ref + BENCH_TAU*(f(x0)-f_ref)` para todos los métodos. El valor de referencia
se certifica por la cota de fuerte convexidad, y el experimento se detiene
**solo por alcanzar el objetivo o el máximo de iteraciones**, no por decrementos
exactos/aproximados que no son equivalentes. `delta` queda como diagnóstico
independiente. Un tiempo sin suficientes muestras efectivas se muestra como `—`.
`@benchmark` aquí usa una semilla controlada por configuración; el estudio de
**variabilidad entre semillas y órdenes aleatorizados** se realiza por separado
en las secciones 7 y 10.

`RUN_COSTLY_BENCHMARKS=false` mantiene apagadas las secciones 4–7 y la figura
3 para abrir el cuaderno sin lanzar comparaciones grandes.
"""

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000023
function bench_full(P, ratios; nsamples = BENCH_SAMPLES, seconds = SEC,
                    tau = BENCH_TAU, seed = SEED)
    n, d = size(P.A)
    ref = reference_solution(P)
    cert = reference_certificate(P, ref; tau_min = tau, fraction = REF_FRACTION)
    target = cert.f_ref + tau * cert.gap0
    rows = Vector{Vector{String}}()
    res = NamedTuple[]
    # Calentamiento por método, no se cronometra.
    rex = newton_exact(P; delta = 0.0, maxit = MAXIT_EX,
                      target_f = target, stop_on_decrement = false)
    tr = @benchmark newton_exact($P; delta=0.0, maxit=$MAXIT_EX,
                target_f=$target, stop_on_decrement=false) seconds=seconds samples=nsamples evals=1
    ex = bench_summary(tr)
    exok = rex.reason === :target && ex.good
    tex = exok ? ex.time : NaN
    push!(rows, ["exact", "—", "—", string(rex.iters), fmt_safe_t(tex),
                 ex.good ? "["*fmt_t(ex.q1)*", "*fmt_t(ex.q3)*"]" : "—",
                 string(ex.count), "1.00×", fmt_mem(ex.mem), "0", "0", String(rex.reason)])
    push!(res, (method="exact", ratio=0.0, m=n, iters=rex.iters, t=tex,
                mem=ex.mem, iqr=ex.q3-ex.q1, samples=ex.count, reason=rex.reason))
    for ratio in ratios, nm in KORDER
        kind = KINDS[nm]
        m = round(Int, ratio*d)
        rs = newton_sketch(P, kind, m; rng = Xoshiro(seed+Int(m)+findfirst(==(nm), KORDER)),
              delta=0.0, maxit=MAXIT_SK, target_f=target, stop_on_decrement=false)
        skseed = seed + Int(m) + findfirst(==(nm), KORDER)
        tb = @benchmark newton_sketch($P, $kind, $m; rng=Xoshiro($skseed),
              delta=0.0, maxit=$MAXIT_SK, target_f=$target,
              stop_on_decrement=false) seconds=seconds samples=nsamples evals=1
        sks = bench_summary(tb)
        good = rs.reason === :target && sks.good && exok
        tm = good ? sks.time : NaN
        speed = good && tm > 0 ? @sprintf("%.2f×", tex/tm) : "—"
        rel_gap = max(rs.f - cert.f_ref, 0.0) / cert.gap0
        push!(rows, [String(nm), string(m), string(ratio), string(rs.iters),
            fmt_safe_t(tm), sks.good ? "["*fmt_t(sks.q1)*", "*fmt_t(sks.q3)*"]" : "—",
            string(sks.count), speed, fmt_mem(sks.mem),
            @sprintf("%.2e", norm(rs.x-ref.x)/max(norm(ref.x),eps(Float64))),
            @sprintf("%.2e", rel_gap), String(rs.reason)])
        push!(res, (method=String(nm), ratio=Float64(ratio), m=m, iters=rs.iters,
                    t=tm, mem=sks.mem, iqr=sks.q3-sks.q1,
                    samples=sks.count, reason=rs.reason))
    end
    return rows, res, tex
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000024
begin
    if !RUN_COSTLY_BENCHMARKS
        FULL_ROWS = Vector{Vector{String}}()
        FULL_RES = NamedTuple[]
        T_EXACT = NaN
        md"Sección 5 desactivada. Usa `RUN_COSTLY_BENCHMARKS=true` para comparar tiempos a `BENCH_TAU`."
    else
        FULL_ROWS, FULL_RES, T_EXACT = bench_full(P, RATIOS)
        mdtable(["método", "m", "m/d", "iters", "mediana", "[Q1, Q3]",
                 "muestras efectivas", "vs exacto", "memoria/llamada", "err_x",
                 "error objetivo relativo", "parada"], FULL_ROWS)
    end
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000025
md"""
## 6. Escalado en `n`

`d` fijo en 200 para que quepan varios `n`, `m = 4d`. Cada entrada es
`mediana(exacto) / mediana(sketch)`. Aquí comprobamos empíricamente si el sketch gana al crecer `n`:
el Hessiano exacto cuesta `O(n·d²)` y el sketch tiene un costo de
compresión dependiente del operador más la resolución del sistema reducido.

**Ojo con `μ`:** en tu `make_problem`, `μ = lam·n`. Al crecer `n` no solo crece el problema,
también crece la regularización, y por eso los conteos de iteraciones se quedan planos. Si
cambias `LAM`, la tabla cambia con ella.
"""

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000026
function bench_scaling(ns, d; tau = BENCH_TAU)
    rows = Vector{Vector{String}}()
    for nn in ns
        Pn, _, _ = make_problem(nn, d; decay=DECAY, lam=LAM, snr=SNR, seed=SEED, ntest=0)
        ref = reference_solution(Pn)
        cert = reference_certificate(Pn, ref; tau_min=tau, fraction=REF_FRACTION)
        target = cert.f_ref + tau*cert.gap0
        r_ex = newton_exact(Pn; delta=0.0, maxit=MAXIT_EX,
                            target_f=target, stop_on_decrement=false)
        tr = @benchmark newton_exact($Pn; delta=0.0, maxit=$MAXIT_EX,
                    target_f=$target, stop_on_decrement=false) seconds=SEC samples=BENCH_SAMPLES evals=1
        ex = bench_summary(tr)
        te = r_ex.reason === :target ? ex.time : NaN
        row = [string(nn), fmt_safe_t(te), string(ex.count)]
        for (j,nm) in enumerate(KORDER)
            kind = KINDS[nm]
            m = 4*d
            rng_seed = SEED + j
            rs = newton_sketch(Pn, kind, m; rng=Xoshiro(rng_seed), delta=0.0,
                               maxit=MAXIT_SK, target_f=target, stop_on_decrement=false)
            tb = @benchmark newton_sketch($Pn, $kind, $m; rng=Xoshiro($rng_seed),
                         delta=0.0, maxit=$MAXIT_SK, target_f=$target,
                         stop_on_decrement=false) seconds=SEC samples=BENCH_SAMPLES evals=1
            sm = bench_summary(tb)
            speed = rs.reason === :target && sm.good && isfinite(te) ? te/sm.time : NaN
            push!(row, isfinite(speed) ? @sprintf("%.2f× [%d]", speed, sm.count) : "— [$(sm.count)]")
        end
        push!(rows, row)
    end
    return rows
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000027
if RUN_COSTLY_BENCHMARKS
    mdtable(["n", "exacto", "muestras exactas", "Gaussian", "CountSketch", "RowSampling", "SRHT"],
            bench_scaling((5000, 20000, 80000), 200))
else
    md"Sección 6 desactivada; el barrido usa precisión relativa `BENCH_TAU` cuando se activa."
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000028
md"""
## 7. ¿Es estable entre semillas?

`newton_sketch` usa `rng = Xoshiro(1)` por defecto, así que un `@benchmark` normal
puede medir una secuencia aleatoria reproducible y no la variabilidad entre
realizaciones del sketch. Aquí medimos la convergencia hasta `BENCH_TAU` con
cinco semillas en modo rápido o quince en modo final, en un mismo problema.
"""

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000029
function bench_seeds(nn, dd; nseeds = (EXPERIMENT_MODE === :final ? 15 : 5), ratio = 4,
                     tau = BENCH_TAU)
    Pm, _, _ = make_problem(nn, dd; decay=DECAY, lam=LAM, snr=SNR, seed=SEED, ntest=0)
    cert = reference_certificate(Pm, reference_solution(Pm);
                 tau_min=tau, fraction=REF_FRACTION)
    target = cert.f_ref + tau*cert.gap0
    rows = Vector{Vector{String}}()
    methods = (:Exact, KORDER...)
    for (j, nm) in enumerate(methods)
        times = Float64[]
        its = Int[]
        successes = 0
        for rep in 1:nseeds
            t0 = time_ns()
            rs = nm === :Exact ?
                newton_exact(Pm; delta=0.0, maxit=MAXIT_EX,
                             target_f=target, stop_on_decrement=false) :
                newton_sketch(Pm, KINDS[nm], round(Int, ratio*dd);
                      rng=Xoshiro(SEED+100*rep+j), delta=0.0,
                      maxit=MAXIT_SK, target_f=target, stop_on_decrement=false)
            tm = (time_ns()-t0)/1e6
            if rs.reason === :target
                successes += 1
                push!(times, tm)
                push!(its, rs.iters)
            end
        end
        valid = successes == nseeds
        push!(rows, [String(nm), nm === :Exact ? "—" : string(round(Int,ratio*dd)),
            valid ? @sprintf("%.1f", median(its)) : "—",
            valid ? @sprintf("%.2f [%.2f, %.2f]", median(times),
                              quantile(times,0.25),quantile(times,0.75)) : "—",
            "$successes/$nseeds"])
    end
    return rows
end

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000030
if RUN_COSTLY_BENCHMARKS
    mdtable(["método", "m", "iteraciones", "tiempo ms [Q1,Q3]", "éxitos"],
            bench_seeds(5000, 200))
else
    md"Sección 7 desactivada. La comparación al activarla utiliza el mismo umbral de objetivo."
end

# ╔═╡ a14a330f-5b39-4563-938c-9086966b32a9
md"""
## 8. Experimentos visuales de convergencia

Generaremos dos figuras para comparar **Gaussian Sketch, CountSketch, Row Sampling y SRHT**. Cada método conserva el mismo color en ambas figuras y Newton exacto aparece como referencia en negro en la primera.

Para que la visualización no dependa del gran costo de las secciones de benchmarking, se construye un **problema sintético más pequeño pero fijo**, con el mismo generador `make_problem`. Se usa siempre el mismo problema y el punto inicial `x₀ = 0`; solo cambian la matriz de sketch y su dimensión `m`. Los parámetros se pueden ajustar en la celda siguiente.

Ambos algoritmos registran las iteraciones mediante `trace` y ahora reportan además el motivo de parada.
"""

# ╔═╡ b21aba7d-9a5d-4a78-8fd7-e02dcb3acdbd
md"""
### Parámetros y colores de los experimentos

- `PLOT_N` y `PLOT_D`: tamaño del problema usado **solo para las figuras**.
- `FIG1_RATIO`: relación `m/d` de la primera figura.
- `FIG1_ITERS`: máximo de iteraciones registrado en la primera figura.
- `RATE_RATIOS`: valores de `m/d` explorados en la segunda figura.
- `RATE_ITERS`: horizonte fijo `T` con el que se calcula la tasa empírica.
- `RATE_SEEDS`: número de semillas independientes por punto de la segunda figura; se muestra la mediana.

Usamos `delta=0` en los experimentos para no imponer la tolerancia habitual de parada antes de completar el horizonte solicitado. Si el algoritmo termina antes (por decremento nulo o búsqueda de línea fallida), se conserva su último error durante los pasos restantes al calcular la tasa. En las curvas se muestra el registro disponible.

**Nota:** las implementaciones cambian a Hessiano exacto cuando `m ≥ n`, por eso aquí los valores de `m` deben ser menores que `PLOT_N`.
"""

# ╔═╡ 92d26354-28bb-4c3c-93d1-75d4d27dd4ad
begin
    PLOT_N      = 5000
    PLOT_D      = 100
    PLOT_DECAY  = 0.8
    FIG1_RATIO  = 2.0
    FIG1_ITERS  = 12
    RATE_RATIOS = (0.5, 1.0, 2.0, 3.0, 4.0)
    RATE_ITERS  = 4
    RATE_SEEDS  = 3
    PLOT_COLORS = Dict(
        :Gaussian    => :royalblue,
        :CountSketch => :darkorange,
        :RowSampling => :seagreen,
        :SRHT        => :purple,
        :Exact       => :black,
    )
    nothing
end

# ╔═╡ b2220127-2320-46b3-87b6-6c9fb36ac57c
md"""
### Datos sintéticos y referencia óptima

Construimos un solo problema de regresión logística con parámetros propios para estas figuras. Estimamos el óptimo `f⋆` resolviendo el problema con `reference_solution`, que aplica Newton exacto con una tolerancia estricta. Así calcularemos para cada iteración la diferencia **`f(xₜ) − f⋆`** sin que los sketches conozcan ni utilicen ese valor durante sus pasos.
"""

# ╔═╡ 8ff645fb-0e74-4566-af05-8af2c44c1a1d
begin
    PLOT_PROBLEM = first(make_problem(PLOT_N, PLOT_D;
        decay = PLOT_DECAY, lam = LAM, snr = SNR, seed = SEED, ntest = 0))
    PLOT_REF = reference_solution(PLOT_PROBLEM)
    PLOT_CERT = reference_certificate(PLOT_PROBLEM, PLOT_REF;
                tau_min=1e-8, fraction=REF_FRACTION)
    (n = PLOT_N, d = PLOT_D, f_star = PLOT_REF.f,
     reference_iters = PLOT_REF.iters,
     relative_reference_error_bound = PLOT_CERT.relative_bound,
     certified = PLOT_CERT.certified,
     effective_dimension_at_reference = weighted_effective_dimension(PLOT_PROBLEM,PLOT_REF.x))
end

# ╔═╡ 6e54148e-a78f-48a0-abd1-b7da544a9a1f
md"""
### Función auxiliar `trace_gaps`

Extrae del registro de iteraciones los pares `(t, f(xₜ) − f⋆)`. Para poder usar una escala logarítmica, los errores que llegan a la precisión de máquina se representan con un **piso numérico positivo**; ese piso no significa que el método haya dejado de mejorar matemáticamente.
"""

# ╔═╡ ee65150b-5bcb-41ac-8c8c-8ffa099ab998
function trace_gaps(trace, fstar)
    floor_gap = max(1e-14, 50 * eps(Float64) * max(abs(fstar), 1.0))
    ts = [round(Int, step[1]) for step in trace]
    gaps = [max(step[3] - fstar, floor_gap) for step in trace]
    return ts, gaps
end

# ╔═╡ 2eb73278-4880-4c9b-9f52-8f373395f12a
md"""
### Prueba 1. Trayectorias `f(xₜ) − f⋆` frente a iteraciones

Se ejecutan **Newton exacto y los cuatro sketches** desde el mismo vector inicial. Para los sketches se fija `m = FIG1_RATIO × d`, se extrae el valor de la función en cada paso y se grafica el error en escala logarítmica. Una curva que baja más deprisa alcanza antes valores pequeños de la función objetivo.

Cada sketch usa una semilla fija diferente para que los resultados sean reproducibles. El problema, el punto inicial y el valor de referencia son exactamente los mismos para todos.
"""

# ╔═╡ 442086dc-2354-47c8-b07c-89ca662393b1
FIG1_TRACES = let
    n, d = size(PLOT_PROBLEM.A)
    m = round(Int, FIG1_RATIO * d)
    @assert 1 <= m < n "Para comparar sketches se requiere 1 ≤ m < n."
    paths = Dict{Symbol,Vector{NTuple{4,Float64}}}()

    trace_exact = NTuple{4,Float64}[]
    newton_exact(PLOT_PROBLEM; delta = 0.0, maxit = FIG1_ITERS,
                 trace = trace_exact)
    paths[:Exact] = trace_exact

    for (j, nm) in enumerate(KORDER)
        trace_sk = NTuple{4,Float64}[]
        newton_sketch(PLOT_PROBLEM, KINDS[nm], m;
            rng = Xoshiro(SEED + 10*j), delta = 0.0,
            maxit = FIG1_ITERS, trace = trace_sk)
        paths[nm] = trace_sk
    end
    paths
end

# ╔═╡ 79311892-e37a-4136-bbe7-4f6b41c2dcdc
FIG1 = let
    fig = plot(
        title = "Disminución del error de la función objetivo",
        xlabel = "Iteración t",
        ylabel = "f(xₜ) − f⋆",
        yscale = :log10,
        legend = :topright,
        grid = true,
        minorgrid = true,
        linewidth = 2.5,
        size = (880, 510),
    )
    for nm in KORDER
        its, gaps = trace_gaps(FIG1_TRACES[nm], PLOT_REF.f)
        plot!(fig, its, gaps; label = String(nm), color = PLOT_COLORS[nm],
              marker = :circle, markersize = 4, linewidth = 2.5)
    end
    its_exact, gaps_exact = trace_gaps(FIG1_TRACES[:Exact], PLOT_REF.f)
    plot!(fig, its_exact, gaps_exact; label = "Newton exacto",
          color = PLOT_COLORS[:Exact], linestyle = :dash,
          marker = :diamond, markersize = 4, linewidth = 2.5)
    fig
end

# ╔═╡ 3a35cc6d-70d4-4966-bf63-e9c3e96f4e5f
md"""
### Función auxiliar `empirical_convergence_factor`

Definimos la **tasa empírica media por iteración** durante un horizonte fijo de `T` pasos:

```math
\rho_T=\left(\frac{f(x_T)-f^\star}{f(x_0)-f^\star}\right)^{1/T}.
```

Si `ρ_T` es cercano a cero, el error se redujo rápidamente; si está próximo a uno, el progreso fue lento. Esta es una **medida experimental en un horizonte fijo**, no la tasa asintótica garantizada por un teorema. El método de Newton no tiene por qué converger linealmente.

Si el método termina antes de `T`, se usa su último valor como aproximación de los pasos restantes. Igual que en la primera figura, se aplica un piso numérico para no dividir entre cero al alcanzar la precisión de máquina.
"""

# ╔═╡ 1187b46a-46dc-4c85-862c-b42ecea9e300
function empirical_convergence_factor(trace, fstar, T::Int)
    @assert T >= 1
    @assert !isempty(trace)
    floor_gap = max(1e-14, 50 * eps(Float64) * max(abs(fstar), 1.0))
    gap0 = max(trace[1][3] - fstar, floor_gap)
    # El elemento k+1 corresponde a la iteración k (comenzamos en t = 0).
    gapT = max(trace[min(T + 1, length(trace))][3] - fstar, floor_gap)
    return (gapT / gap0)^(1 / T)
end

# ╔═╡ c8182c92-877c-4438-b66b-ebd8ce9a7e21
md"""
### Función `measure_rates_by_ratio`

Para cada razón `m/d`, ejecuta cada sketch durante **a lo sumo `RATE_ITERS` iteraciones**. Repite el experimento con `RATE_SEEDS` semillas distintas y calcula la **mediana de `ρ_T`**, que reduce la influencia de un sorteo particular de la matriz aleatoria.

En toda la comparación se mantiene fijo `PLOT_PROBLEM`: así el efecto de variar `m/d` no se mezcla con cambios en los datos o en el parámetro de regularización. La misma cantidad de pasos `T` se utiliza para los cuatro sketches.
"""

# ╔═╡ 55a8c10b-a01b-4041-a996-c4a338e5cf5d
function measure_rates_by_ratio(P::LogReg, fstar, ratios;
                                T::Int = 4, nseeds::Int = 3, seed::Int = 2026)
    @assert T >= 1 && nseeds >= 1
    n, d = size(P.A)
    rates = Dict(nm => Float64[] for nm in KORDER)
    for (ir, ratio) in enumerate(ratios)
        m = round(Int, ratio * d)
        1 <= m < n || error("Para m/d=$ratio se requiere 1 ≤ m < n.")
        for (ik, nm) in enumerate(KORDER)
            trials = Float64[]
            for rep in 1:nseeds
                tr = NTuple{4,Float64}[]
                newton_sketch(P, KINDS[nm], m;
                    rng = Xoshiro(seed + 1000*ir + 100*ik + rep),
                    delta = 0.0, maxit = T, trace = tr)
                push!(trials, empirical_convergence_factor(tr, fstar, T))
            end
            push!(rates[nm], median(trials))
        end
    end
    return rates
end

# ╔═╡ 7f14c255-91bd-48d3-873f-c2a3edf442cb
md"""
### Prueba 2. Tasa de convergencia según `m/d`

En el eje horizontal se muestra la **razón de compresión** `m/d` y en el vertical la tasa empírica `ρ_T` para cada tipo de sketch. Cada punto resume varias realizaciones independientes; valores **más pequeños** de `ρ_T` significan mayor reducción promedio del error por iteración.

Para dimensiones pequeñas (`m < d`), la regularización `mu > 0` garantiza que el sistema aproximado sea resoluble aunque `S B` no tenga rango columna completo. Aumentar `m` suele mejorar la aproximación del Hessiano, pero puede costar más tiempo: **esta figura mide calidad de convergencia por iteración, no tiempo de ejecución**.
"""

# ╔═╡ 3cb036a4-9048-4ae5-bf31-05e9e2a16e89
RATE_DATA = measure_rates_by_ratio(
    PLOT_PROBLEM, PLOT_REF.f, RATE_RATIOS;
    T = RATE_ITERS, nseeds = RATE_SEEDS, seed = SEED
)

# ╔═╡ ed0dc2d0-9d42-4ae4-abf7-9c06484e9cc4
FIG2 = let
    fig = plot(
        title = "Tasa empírica de convergencia vs. m/d",
        xlabel = "Relación m/d",
        ylabel = "Factor por iteración ρₜ (menor = mejor)",
        legend = :topright,
        grid = true,
        linewidth = 2.5,
        size = (880, 510),
        xticks = collect(RATE_RATIOS),
    )
    for nm in KORDER
        plot!(fig, collect(RATE_RATIOS), RATE_DATA[nm];
              label = String(nm), color = PLOT_COLORS[nm],
              marker = :circle, markersize = 6, linewidth = 2.5)
    end
    fig
end

# ╔═╡ 83af4448-5924-4a60-b07a-10f07e2aaebf
md"""
**Interpretación y exportación.** La primera figura permite comparar directamente las trayectorias hacia el mismo óptimo. La segunda resume el ritmo medio del error a un número **fijo de iteraciones**, y por ello no puede interpretarse como la tasa asintótica ni como una comparación de rapidez en segundos. Las fluctuaciones entre valores de `m/d` son normales cuando las matrices se generan aleatoriamente.

Para guardar ambas figuras desde una celda adicional de Pluto, puedes ejecutar:

```julia
savefig(FIG1, "error_objetivo_vs_iteraciones.png")
savefig(FIG2, "tasa_convergencia_vs_m_d.png")
```

Los gráficos utilizan exactamente la misma paleta: **azul** (Gaussian), **naranja** (CountSketch), **verde** (RowSampling) y **morado** (SRHT).
"""

# ╔═╡ 6ead77ea-e145-4ca6-b186-a678710db00b
md"""
## 9. Memoria y tiempo necesario para alcanzar una precisión

Además de comparar iteraciones, necesitamos estudiar el **costo de memoria** de cada sketch y el **tiempo hasta una precisión común**. Conservar los colores de las figuras anteriores facilita relacionar sus resultados.

Presentamos dos medidas de memoria distintas:

1. **Bytes asignados durante la ejecución completa**: `BenchmarkTools` reporta el total de nuevas asignaciones en una llamada al solver (mediana de las mediciones). Una misma región de memoria puede asignarse y liberarse múltiples veces, por lo que esto **no es el pico de RAM**.
2. **Tamaño estimado de los buffers simultáneos**: contabilizamos las matrices y los vectores que retienen `WS`, `SkWS` y el Newton exacto. No incluye los datos compartidos `A` y `y`, cabeceras de arrays, estructuras internas de BLAS o sobrecostos del runtime. **Es una aproximación analítica, no una medición de RAM residente.**

La segunda medida es especialmente útil aquí porque los buffers se reutilizan: un método puede reservar mucha memoria de trabajo y asignar pocos bytes en cada iteración.
"""

# ╔═╡ 4e7a2a53-9179-4a2d-9b00-2d001d6201a9
md"""
### Figura 3. Memoria asignada por llamada frente a `m/d`

Reutilizamos `FULL_RES`, calculado con `@benchmark` en la sección 5, para **no volver a ejecutar experimentos costosos**. Cada curva representa los MiB asignados durante una llamada completa a Newton Sketch. La línea negra representa Newton exacto, independiente de `m/d`.

Se usa escala logarítmica en el eje vertical para mostrar tanto las implementaciones que reutilizan la mayoría de sus buffers como aquellas con asignaciones grandes. Los métodos se comparan para el mismo problema y la **misma precisión relativa del objetivo**; pueden realizar cantidades diferentes de iteraciones.
"""

# ╔═╡ eef38949-59be-4f16-a062-7feb5c0f9c45
FIG3 = if isempty(FULL_RES)
    md"Activa `RUN_COSTLY_BENCHMARKS=true` para generar la figura 3 con mediciones reales."
else
    x = Float64.(collect(RATIOS))
    fig = plot(title = "Memoria asignada en una ejecución completa",
               xlabel = "Relación m/d", ylabel = "Asignaciones (MiB, escala log)",
               yscale = :log10, legend = :outertopright, grid = true,
               marker = :circle, linewidth = 2.5, size = (920, 520), xticks = x)
    exact_alloc = only([r.mem for r in FULL_RES if r.method == "exact"]) / 2.0^20
    plot!(fig, x, fill(max(exact_alloc, eps(Float64)),length(x));
          label = "Newton exacto", color = PLOT_COLORS[:Exact],
          linestyle = :dash, marker = :diamond, linewidth = 2.5)
    for nm in KORDER
        ys = [only([r.mem for r in FULL_RES if r.method == String(nm) && r.ratio == ratio]) / 2.0^20
              for ratio in RATIOS]
        plot!(fig, x, max.(ys, eps(Float64)); label=String(nm),
              color=PLOT_COLORS[nm], marker=:circle, linewidth=2.5, markersize=5)
    end
    fig
end

# ╔═╡ bbc9359a-bc1c-454a-836b-4d49b7a555ab
md"""
### Función `estimated_workspace_bytes`: estimación del espacio de trabajo

Para aislar la memoria que **permanece almacenada mientras se calcula una dirección**, contabilizamos únicamente los buffers principales de la implementación actual:

- **Newton exacto:** `B` de tamaño `n×d` y `H` de tamaño `d×d`.
- **Gaussian:** `SB` de tamaño `m×d`, un panel `S` de tamaño `m×b` (`b = gaussian_panel_cols(m,n)`) y matriz del sistema `H`.
- **CountSketch:** `SB`, índice de cubetas `h` (`n` enteros), signos/pesos `sg` (`n` reales) y `H`.
- **RowSampling:** `SB`, índices seleccionados (`m` enteros) y `H`.
- **SRHT:** `SB`, buffer de Hadamard `C` (`nextpow(2,n)×workers`), permutación (`nextpow(2,n)` enteros), vector de signos/pesos (`n` reales) y `H`.

Todos comparten el espacio de trabajo `WS`. Si `m < d`, el sistema se resuelve con Woodbury usando un `H` de tamaño `m×m` y un vector temporal adicional. Como `m < n` para estas gráficas, no se activa la sustitución por el Hessiano exacto.

Las cifras son **estimaciones del almacenamiento de arrays**, no equivalen a memoria máxima del proceso y excluyen los datos de entrada `A` y `y`. Los presupuestos de 8 y 16 MiB corresponden a los buffers de Gaussian y SRHT respectivamente, no a toda la memoria del método.
"""

# ╔═╡ ffb01359-93b2-4645-8a0f-6df7e1821b81
function estimated_workspace_bytes(P::LogReg, nm::Symbol, m::Int)
    n, d = size(P.A)
    @assert n >= 1 && d >= 1 && m >= 1
    floatbytes = sizeof(Float64)
    intbytes = sizeof(Int)
    # WS contiene 4 vectores de longitud n y 5 de longitud d; x añade d.
    common = floatbytes * (4*n + 6*d)
    if nm === :Exact || m >= n
        return common + floatbytes * (n*d + d*d)
    end
    hdim = min(m, d)
    shared_sk = common + floatbytes * (m*d + hdim*hdim + (m < d ? m : 0))
    if nm === :Gaussian
        return shared_sk + floatbytes * (m * gaussian_panel_cols(m, n))
    elseif nm === :CountSketch
        return shared_sk + (intbytes + floatbytes) * n
    elseif nm === :RowSampling
        return shared_sk + intbytes * m
    elseif nm === :SRHT
        np = nextpow(2, n)
        return shared_sk + floatbytes * (np * srht_worker_count(np, d) + n) + intbytes * np
    else
        throw(ArgumentError("Sketch desconocido: $nm"))
    end
end

# ╔═╡ 2f0b3e16-798e-4931-b697-93e4f7d5eab9
md"""
### Figura 4. Memoria estimada de trabajo por método

Fijamos `m/d = MEM_RATIO` y comparamos los buffers simultáneos de cada algoritmo con los del Newton exacto. Esta figura incorpora la **nueva memoria acotada por bloques**: Gaussian ya no conserva una matriz `m×n` y SRHT ya no conserva `np×d` elementos, sino un conjunto de buffers por trabajador.

Se emplea el problema reducido `PLOT_PROBLEM` para separar este experimento de los benchmarks grandes de la sección 5. El parámetro `MEM_RATIO` puede modificarse sin alterar las demás pruebas.
"""

# ╔═╡ 9b546354-eff3-4420-baf7-add5ce5726e5
MEM_RATIO = 2.0

# ╔═╡ e403c78e-02f8-46cf-8eb4-da07fd463ebe
FIG4 = let
    n, d = size(PLOT_PROBLEM.A)
    m = round(Int, MEM_RATIO * d)
    @assert m < n "Escoge MEM_RATIO de modo que m < n."
    names = (:Exact, KORDER...)
    labels = [nm === :Exact ? "Newton exacto" : String(nm) for nm in names]
    values = [estimated_workspace_bytes(PLOT_PROBLEM, nm, m) / 2.0^20 for nm in names]
    bar(labels, values; title = "Buffers simultáneos estimados (m/d=$(MEM_RATIO))",
        ylabel = "Memoria estimada (MiB)", legend = false,
        color = [PLOT_COLORS[nm] for nm in names], bar_width = 0.65,
        xrotation = 20, size = (920, 500), grid = true)
end

# ╔═╡ b5676535-677a-4c7b-b9fe-91ad2c7036aa
md"""
### Precisión objetivo y tiempo para alcanzarla

Ahora preguntamos: **¿cuántos segundos necesita cada método para alcanzar la misma calidad de solución?** En lugar de comparar los decrementos de Newton (exacto y aproximado, que no son idénticos), empleamos un criterio **común y verificable** para todos:

```math
\varepsilon_t = \frac{f(x_t)-f^\star}{f(x_0)-f^\star}
\qquad\text{y buscamos el primer } t \text{ tal que }\varepsilon_t\leq\tau.
```

Aquí `f⋆ = PLOT_REF.f` se calcula con Newton exacto y se utiliza **solo para medir la calidad y establecer la parada de esta prueba**, no para construir direcciones. El valor `XTRUE` no se usa: es el parámetro generador de los datos, no el minimizador empírico regularizado.

Para medir el **tiempo real hasta alcanzar cada umbral `τ`**, se usa el argumento opcional `target_f = f⋆ + τ(f(x₀)−f⋆)` de `newton_exact` y `newton_sketch`. Así los métodos se detienen justo después de evaluar un punto que satisface el criterio, sin formar inútilmente la dirección siguiente. Con `target_f = -Inf` (valor por defecto) los algoritmos conservan su comportamiento original.

Se utilizan las **mismas semillas para todos los umbrales** y se repite cada prueba `TIME_SEEDS` veces. Se muestra la **mediana de los tiempos** solo si todas las repeticiones han alcanzado el umbral. Las no alcanzadas **no se representan como si hubieran convergido**. Se realizan ejecuciones de calentamiento antes de cronometrar para reducir el efecto de la compilación inicial de Julia. El tiempo medido incluye armado de Hessiano/sketch, gradiente y búsqueda de línea, pero no la generación del problema ni el cálculo de `f⋆`.
"""

# ╔═╡ 8a7dcbce-87a5-4d11-b18b-8f5d19677509
md"""
### Parámetros del experimento tiempo–precisión

`TIME_RATIO` fija el tamaño de sketch usado en toda la comparación; `TIME_TOLS` contiene las precisiones relativas requeridas (una más pequeña es más exigente); `TIME_MAXIT` limita las iteraciones; `TIME_SEEDS` determina cuántas repeticiones se realizan por método y precisión. Todas las pruebas parten de `x₀=0` y usan `PLOT_PROBLEM`.
"""

# ╔═╡ d6779ab2-3efc-4dc5-99b1-3370f48e52e4
begin
    TIME_RATIO = 2.0
    TIME_TOLS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
    TIME_MAXIT = 25
    TIME_SEEDS = EXPERIMENT_MODE === :final ? 15 : 5
    nothing
end

# ╔═╡ 784f6622-7451-4783-bf85-672ad02a80a3
md"""
### Función `measure_time_to_precision`

Ejecuta cada método desde el mismo punto inicial hasta alcanzar la precisión solicitada. Usa `@elapsed` para medir el tiempo total de cada ejecución y almacena cuántas repeticiones lograron alcanzar el umbral antes de `TIME_MAXIT` o de un posible estancamiento. El resultado sirve tanto para graficar como para construir una tabla de resultados.
"""

# ╔═╡ d7130607-0fd9-4359-b52b-5e9b74c1d7e0
function measure_time_to_precision(P::LogReg, fstar::Real, tolerances, ratio;
                                   maxit::Int = 25, nseeds::Int = 3,
                                   seed::Int = 2026, cert = nothing)
    @assert maxit >= 1 && nseeds >= 1
    n, d = size(P.A)
    m = round(Int, ratio*d)
    1 <= m < n || error("Se requiere 1 ≤ m < n; se obtuvo m=$m y n=$n.")
    taus = Float64.(collect(tolerances))
    all(t -> 0 < t < 1, taus) || error("Se requiere 0 < τ < 1.")
    cert === nothing && error("Se requiere una referencia certificada (`cert`).")
    cert.certified && cert.relative_bound <= REF_FRACTION * minimum(taus) ||
        error("La referencia no satisface la tolerancia mínima solicitada.")
    isapprox(fstar, cert.f_ref; atol=1e-10, rtol=1e-13) ||
        error("La referencia certificada no corresponde a fstar.")
    gap0 = cert.gap0
    warm_target = fstar+maximum(taus)*gap0
    newton_exact(P; delta=0.0, maxit=maxit, target_f=warm_target,
                 stop_on_decrement=false)
    for nm in KORDER
        newton_sketch(P,KINDS[nm],m; rng=Xoshiro(seed),delta=0.0,
                      maxit=maxit,target_f=warm_target,stop_on_decrement=false)
    end
    methods = (:Exact, KORDER...)
    times_ms = Dict(nm => Float64[] for nm in methods)
    q1_ms = Dict(nm => Float64[] for nm in methods)
    q3_ms = Dict(nm => Float64[] for nm in methods)
    successes = Dict(nm => Int[] for nm in methods)
    for tol in taus
        target = fstar+tol*gap0
        for (k,nm) in enumerate(methods)
            times = Float64[]
            for rep in 1:nseeds
                t0 = time_ns()
                result = nm === :Exact ?
                    newton_exact(P; delta=0.0, maxit=maxit,
                                 target_f=target,stop_on_decrement=false) :
                    newton_sketch(P,KINDS[nm],m;
                       rng=Xoshiro(seed+100*k+rep),delta=0.0,maxit=maxit,
                       target_f=target,stop_on_decrement=false)
                ms = (time_ns()-t0)*1e-6
                if result.reason === :target && isfinite(result.f) && result.f <= target
                    push!(times,ms)
                end
            end
            ok = length(times) == nseeds
            push!(successes[nm],length(times))
            push!(times_ms[nm],ok ? median(times) : NaN)
            push!(q1_ms[nm],ok ? quantile(times,0.25) : NaN)
            push!(q3_ms[nm],ok ? quantile(times,0.75) : NaN)
        end
    end
    return (tolerances=taus,times_ms=times_ms,q1_ms=q1_ms,q3_ms=q3_ms,
            successes=successes,runs_per_point=nseeds,m=m,
            initial_gap=gap0,reference_relative_bound=cert.relative_bound)
end

# ╔═╡ c0b5516d-c078-449c-b059-044018027814
TIME_DATA = measure_time_to_precision(PLOT_PROBLEM, PLOT_REF.f, TIME_TOLS,
                                      TIME_RATIO; maxit = TIME_MAXIT,
                                      nseeds = TIME_SEEDS, seed = SEED, cert = PLOT_CERT)

# ╔═╡ c4dff7fd-a7b5-4c4d-86c6-bbaacbbc6b0a
md"""
### Figura 5. Tiempo necesario vs. precisión requerida

Cada punto indica los **milisegundos necesarios para alcanzar por primera vez** la tolerancia relativa `τ`. Hacia la derecha exigimos mayor precisión (`τ` disminuye). Una curva más baja representa un método más rápido para llegar a la **misma tolerancia**. Las interrupciones de una curva indican que no hubo suficientes ejecuciones exitosas para esa precisión.

Esta gráfica es diferente de la tasa por iteración: mide el rendimiento **de extremo a extremo**, incluyendo el costo de formar el sketch y resolver su sistema lineal. Newton exacto aparece como línea negra discontinua. Los colores de los sketches son idénticos a los de las figuras anteriores.
"""

# ╔═╡ d01ab88f-07af-4f41-ba49-04be10b4796f
FIG5 = let
    fig = plot(title = "Tiempo hasta alcanzar la precisión requerida",
        xlabel = "Tolerancia relativa del objetivo τ (más estricta →)",
        ylabel = "Tiempo hasta el objetivo (ms; mediana)",
        xscale = :log10, xflip = true, xticks = (TIME_DATA.tolerances, string.(TIME_DATA.tolerances)),
        legend = :outertopright, grid = true, linewidth = 2.5, size = (990, 520))
    for nm in (:Exact, KORDER...)
        plot!(fig, TIME_DATA.tolerances, TIME_DATA.times_ms[nm];
              label = nm === :Exact ? "Newton exacto" : String(nm),
              color = PLOT_COLORS[nm], linestyle = nm === :Exact ? :dash : :solid,
              marker = nm === :Exact ? :diamond : :circle,
              markersize = 5, linewidth = 2.5)
    end
    fig
end

# ╔═╡ 67a7e4ba-c9f1-438c-9c5d-5a17ceb8b142
md"""
### Tabla de tiempos y resultados no alcanzados

La tabla permite identificar las precisiones que cada método **no** alcanzó consistentemente. El formato es `mediana [Q1,Q3] ms [éxitos/repeticiones]`; un guion indica alguna ejecución no exitosa, por lo que no es apropiado reportar un tiempo típico para esa configuración.
"""

# ╔═╡ 6a38c7f7-9d3c-4c4c-84ef-2d9ec83add9b
TIME_TABLE = let
    hdr = ["Precisión τ", "Newton exacto", "Gaussian", "CountSketch", "RowSampling", "SRHT"]
    names = (:Exact, KORDER...)
    rows = Vector{Vector{String}}()
    for j in eachindex(TIME_DATA.tolerances)
        row = [@sprintf("%.0e", TIME_DATA.tolerances[j])]
        for nm in names
            t = TIME_DATA.times_ms[nm][j]
            successes = TIME_DATA.successes[nm][j]
            notation = "[$successes/$(TIME_DATA.runs_per_point)]"
            push!(row, isfinite(t) ?
                  @sprintf("%.2f [%.2f, %.2f] ms %s", t,
                           TIME_DATA.q1_ms[nm][j],TIME_DATA.q3_ms[nm][j],notation) :
                  "— " * notation)
        end
        push!(rows, row)
    end
    mdtable(hdr, rows)
end

# ╔═╡ 260697bb-1397-42e7-b54c-01d3ea4f5b4d
md"""
### Guardar las nuevas figuras

Puedes exportar los gráficos desde otra celda de Pluto:

```julia
RUN_COSTLY_BENCHMARKS && savefig(FIG3, "memoria_asignada_vs_m_d.png")
savefig(FIG4, "memoria_buffers_por_metodo.png")
savefig(FIG5, "tiempo_hasta_precision.png")
```

**Interpretación importante:** `FIG3` representa el **volumen de asignaciones**, `FIG4` estima los **buffers retenidos** y `FIG5` mide el **tiempo necesario para cumplir un criterio común de precisión del valor objetivo**. Ninguna de las dos primeras es una medición directa del pico de RSS del proceso Julia.
"""

# ╔═╡ 87e64eb4-4c0d-5d4e-bbab-f0da341ac680
md"""
## 10. Experimento: ¿cuándo compensa construir un sketch?

El objetivo es **localizar un punto de equilibrio medido**, no suponer que Newton Sketch será más rápido. Para un mismo problema, punto inicial y precisión requerida, definimos

```math
\tau=\frac{f(x_t)-f^\star}{f(x_0)-f^\star},\qquad
\mathrm{speedup}=\frac{T_{\mathrm{exacto}}}{T_{\mathrm{sketch}}}.
```

- **Speedup mayor que 1:** el sketch llega antes al objetivo.
- **Speedup igual a 1:** punto de equilibrio en tiempo total.
- **Speedup menor que 1:** gana Newton exacto.
- **Sin dato:** la configuración no terminó con la precisión exigida en todas las semillas o `m ≥ n`.

Conservamos constantes las condiciones no analizadas y variamos **`n`, `d`, `m/d`, `decay` y `lam`**. En cada configuración, Newton exacto y los sketches resuelven *el mismo problema generado*, desde `x₀ = 0`, con la misma tolerancia relativa. `f⋆` se calcula mediante la solución de referencia **fuera de los tiempos medidos**.

### Cómo ejecutar este barrido

La sección comienza **desactivada** (`BE_RUN = false`) para no añadir muchas ejecuciones costosas a los benchmarks anteriores. Cambia a `true` cuando quieras obtener las figuras 6–11. El perfil rápido usa `BE_SEEDS = 5` para exploración, sin confirmar fronteras; el perfil final usa 15 rondas e intervalos bootstrap para evaluar la evidencia. Puedes quitar `:Gaussian` de `BE_METHODS` si buscas un barrido inicial rápido: su multiplicación densa es particularmente costosa.

**Control al variar `n`:** se mantiene `lam` constante, de manera que `μ = n·lam` y la regularización relativa en la pérdida media `f/n` permanece constante. Para estudiar el efecto de la regularización independientemente, variamos explícitamente `lam` en otro experimento.
"""

# ╔═╡ 117eccae-03c0-5a36-913a-33ede81341fa

begin
    BE_RUN = true

    # Precisión común
    BE_TAU = 1e-4

    # Tamaños que se van a investigar
    BE_N_GRID = (
        5000, 7500, 10000, 12500,
        15000, 17500, 20000, 30000, 50000
    )

    BE_D_GRID = (40, 100, 200)

    # Todas las relaciones m/d
    BE_RATIOS = (1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)

    BE_DECAYS = (0.0, 0.7, 1.4)
    BE_LAMS = (1e-6, 1e-4, 1e-2)

    # Problema de referencia del break-even
    BE_BASE_N = 5000
    BE_BASE_D = 100
    BE_BASE_DECAY = 1.0
    BE_BASE_LAM = 1e-4

    BE_RATIO_LINE = 2.0

    # Validación estadística
    BE_SEEDS = 15
    BE_BOOTSTRAPS = 1200
    BE_MIN_EVIDENCE_REPS = 10
    BE_MIN_SUSTAINED = 3

    BE_MAXIT = 45
    BE_SEED = 2026

    BE_METHODS = (
        :Gaussian,
        :CountSketch,
        :RowSampling,
        :SRHT
    )

    BE_MARGIN = 1.10
    BE_CI_LEVEL = 0.95

    nothing
end


# ╔═╡ 927d7f48-e52e-475a-bda5-a6306e3cb7ad
"""
    bootstrap_speedup_ci(tex, tsk; level=0.95, nboot=1200, seed=2026)

IC percentil bootstrap para razón de medianas, remuestreando las mismas rondas
para ambos métodos. Es una aproximación estadística (no una garantía formal),
y su nivel nominal no corrige comparaciones múltiples en el barrido.
"""
function bootstrap_speedup_ci(tex::AbstractVector{<:Real},
                              tsk::AbstractVector{<:Real};
                              level::Real = 0.95, nboot::Int = 1200,
                              seed::Int = 2026)
    k = length(tex)
    @assert k == length(tsk) && k >= 2 && 0 < level < 1 && nboot >= 100
    @assert all(t -> isfinite(t) && t > 0, tex)
    @assert all(t -> isfinite(t) && t > 0, tsk)
    rng = Xoshiro(seed)
    boot = Vector{Float64}(undef, nboot)
    for j in 1:nboot
        ids = rand(rng, 1:k, k)
        boot[j] = median(tex[ids]) / median(tsk[ids])
    end
    alpha = (1 - level) / 2
    return (low=quantile(boot,alpha), high=quantile(boot,1-alpha))
end

# ╔═╡ 51e6b5a3-71e4-5ce5-bcb7-820000fe4951
md"""
### Función `be_measure_case`: medir cada problema sin sesgo de compilación

1. Genera `A`, `y`, `μ` y una solución de referencia.
2. Calcula `f⋆` y fija el umbral común `f⋆ + τ(f(x₀)-f⋆)`.
3. Ejecuta una **pasada de calentamiento**, no medida, para compilar las rutas de Newton y de cada sketch.
4. Mide Newton exacto y los sketches en orden aleatorio por ronda, con varias semillas, mediante `time_ns()`. El criterio de parada utiliza `target_f`, de modo que **no se forma otra dirección después de llegar al objetivo**.
5. Guarda tiempo total (mediana y cuartiles), número de iteraciones, tasa de éxito y razón de tiempos. Solo calcula el `speedup` cuando **todas** las repeticiones de ambos métodos alcanzan el objetivo: no se deben premiar corridas fallidas.

No se incluye la construcción de datos ni el cálculo de `f⋆` en el cronómetro; sí se incluye la inicialización de espacios de trabajo, gradientes, sketches, búsquedas de línea y factorizaciones de **cada llamada al algoritmo**. `[Q1,Q3]` señala cuartiles de los tiempos exitosos; además se calcula un intervalo bootstrap para la razón de medianas. La mediana no elimina la necesidad de evaluar variabilidad entre semillas.
"""

# ╔═╡ 8ca1eb1b-6b7a-5901-954e-2353d7a2a088
function be_measure_case(n::Int, d::Int, decay::Real, lam::Real, ratios;
                         tau::Float64 = 1e-4, reps::Int = 15,
                         maxit::Int = 45, seed::Int = 2026,
                         methods = (:Gaussian,:CountSketch,:RowSampling,:SRHT),
                         bootstraps::Int = BE_BOOTSTRAPS)
    @assert n > d > 0 && lam > 0 && 0 < tau < 1 && reps >= 2 && maxit >= 1
    P, _, _ = make_problem(n,d; decay=Float64(decay),lam=Float64(lam),
                           snr=SNR,seed=seed,ntest=0)
    ref = reference_solution(P)
    cert = reference_certificate(P,ref; tau_min=tau,fraction=REF_FRACTION)
    target = cert.f_ref + tau*cert.gap0
    realratios = Float64.(collect(ratios))
    all(r -> r > 0,realratios) || error("Todas las relaciones m/d deben ser positivas.")

    # La compilación NO se incluye en el tiempo de los métodos.
    newton_exact(P; delta=0.0,maxit=1,target_f=-Inf,stop_on_decrement=false)
    specs = [(nm,ratio,round(Int,ratio*d)) for ratio in realratios for nm in methods]
    valid_specs = filter(s -> 1 <= s[3] < n,specs)
    for (j,(nm,ratio,m)) in enumerate(valid_specs)
        newton_sketch(P,KINDS[nm],m; rng=Xoshiro(seed+31+j),delta=0.0,
                      maxit=1,target_f=-Inf,stop_on_decrement=false)
    end

    # Ejecuciones por bloques con orden aleatorio, para mitigar efectos de orden.
    exact_times = fill(NaN,reps)
    exact_its = fill(0,reps)
    tdict = Dict((nm,ratio) => fill(NaN,reps) for (nm,ratio,m) in valid_specs)
    idict = Dict((nm,ratio) => fill(0,reps) for (nm,ratio,m) in valid_specs)
    reasons = Dict((nm,ratio) => fill(:not_run,reps) for (nm,ratio,m) in valid_specs)
    for rep in 1:reps
        work = vcat([(:Exact,0.0,n)],valid_specs)
        shuffle!(Xoshiro(seed + 10007*rep),work)
        for (nm,ratio,m) in work
            t0=time_ns()
            if nm === :Exact
                r = newton_exact(P; delta=0.0,maxit=maxit,
                                 target_f=target,stop_on_decrement=false)
            else
                j = findfirst(==(nm),methods)
                ratio_id = findfirst(==(ratio),realratios)
                r = newton_sketch(P,KINDS[nm],m;
                    rng=Xoshiro(seed+1000*rep+37*j+101*ratio_id),
                    delta=0.0,maxit=maxit,target_f=target,
                    stop_on_decrement=false)
            end
            tm=(time_ns()-t0)*1e-6
            ok = r.reason === :target && isfinite(r.f) && r.f <= target
            if nm === :Exact
                if ok
                    exact_times[rep]=tm
                    exact_its[rep]=r.iters
                end
            else
                key=(nm,ratio)
                reasons[key][rep]=r.reason
                if ok
                    tdict[key][rep]=tm
                    idict[key][rep]=r.iters
                end
            end
        end
    end
    ex_success=count(isfinite,exact_times)
    exok=ex_success==reps
    tex=exok ? median(exact_times) : NaN
    iex=exok ? median(exact_its) : NaN
    records=NamedTuple[]
    q1_exact=exok ? quantile(exact_times,0.25) : NaN
    q3_exact=exok ? quantile(exact_times,0.75) : NaN
    push!(records,(method=:Exact,n=n,d=d,decay=Float64(decay),lam=Float64(lam),
        ratio=0.0,m=n,time_ms=tex,q1_ms=q1_exact,q3_ms=q3_exact,
        iqr_ms=q3_exact-q1_exact,iters=iex,successes=ex_success,reps=reps,
        speedup=exok ? 1.0 : NaN,ci_low=NaN,ci_high=NaN,
        per_iter_gain=1.0,iteration_factor=1.0,status=exok ? :ok : :not_converged,
        exits=exok ? "target" : "no alcanzado"))

    for ratio in realratios, nm in methods
        m=round(Int,ratio*d)
        key=(nm,ratio)
        if !haskey(tdict,key)
            push!(records,(method=nm,n=n,d=d,decay=Float64(decay),lam=Float64(lam),
                ratio=ratio,m=m,time_ms=NaN,q1_ms=NaN,q3_ms=NaN,iqr_ms=NaN,
                iters=NaN,successes=0,reps=reps,speedup=NaN,ci_low=NaN,ci_high=NaN,
                per_iter_gain=NaN,iteration_factor=NaN,status=:invalid_m,exits="m inválido"))
            continue
        end
        t=tdict[key]
        it=idict[key]
        successes=count(isfinite,t)
        ok=exok && successes==reps
        ts=ok ? median(t) : NaN
        ti=ok ? median(it) : NaN
        q1=ok ? quantile(t,0.25) : NaN
        q3=ok ? quantile(t,0.75) : NaN
        speed=ok && ts>0 ? tex/ts : NaN
        ci=ok ? bootstrap_speedup_ci(exact_times,t; level=BE_CI_LEVEL,
                nboot=bootstraps,seed=seed+100*findfirst(==(nm),methods)+round(Int,ratio*100)) :
                (low=NaN,high=NaN)
        infl=ok && iex>0 ? ti/iex : NaN
        per_iter=ok && ti>0 && iex>0 ? (tex/iex)/(ts/ti) : NaN
        exits=join(sort!(unique(string.(reasons[key]))),", ")
        push!(records,(method=nm,n=n,d=d,decay=Float64(decay),lam=Float64(lam),
            ratio=ratio,m=m,time_ms=ts,q1_ms=q1,q3_ms=q3,iqr_ms=q3-q1,
            iters=ti,successes=successes,reps=reps,speedup=speed,
            ci_low=ci.low,ci_high=ci.high,per_iter_gain=per_iter,
            iteration_factor=infl,status=ok ? :ok : :not_converged,exits=exits))
    end
    info=(n=n,d=d,decay=Float64(decay),lam=Float64(lam),mu=P.mu,
          f_star=cert.f_ref,initial_gap=cert.gap0,
          grad_ref_inf=cert.grad_inf,reference_iters=ref.iters,
          reference_reason=ref.reason,ref_bound=cert.bound,
          ref_relative_bound=cert.relative_bound,ref_certified=cert.certified,
          exact_success=ex_success)
    return (rows=records,info=info)
end

# ╔═╡ 5b7c27f7-3582-5fb0-acf9-e30cde2beb66
md"""
### Función `be_run_sweep`: cambiar una variable y mantener las otras fijas

Recorre los valores de `:n`, `:d`, `:decay` o `:lam`, generando un **nuevo problema y una referencia propia para cada configuración**. Las comparaciones de tiempo se hacen únicamente **dentro de cada problema**. Cuando cambia `d`, el número de filas del sketch se ajusta mediante `m = round(Int, (m/d)·d)`.

El barrido conjunto de `n × m/d` y `d × m/d` permite ver el equilibrio en dos dimensiones; los barridos de `decay` y `lam` explican cómo se desplaza esa frontera cuando cambia la dificultad espectral o la regularización.
"""

# ╔═╡ a1e31148-562e-563a-a2dd-188bd75a48bd
function be_run_sweep(which::Symbol, values, ratios;
                      n0::Int,d0::Int,decay0::Real,lam0::Real,
                      tau::Float64,reps::Int,maxit::Int,seed::Int,methods,
                      bootstraps::Int=BE_BOOTSTRAPS)
    which in (:n,:d,:decay,:lam) || error("Barrido desconocido: $which")
    rows=NamedTuple[]
    cases=NamedTuple[]
    for value in values
        n=which===:n ? Int(value) : n0
        d=which===:d ? Int(value) : d0
        decay=which===:decay ? Float64(value) : Float64(decay0)
        lam=which===:lam ? Float64(value) : Float64(lam0)
        out=be_measure_case(n,d,decay,lam,ratios;
             tau=tau,reps=reps,maxit=maxit,seed=seed,
             methods=methods,bootstraps=bootstraps)
        append!(rows,[(;row...,sweep=which,level=Float64(value)) for row in out.rows])
        push!(cases,(;out.info...,sweep=which,level=Float64(value)))
    end
    return (rows=rows,cases=cases)
end

# ╔═╡ 4d5ac21c-579e-5d8b-a6d6-549aacf514ea
md"""
### Ejecución de los cuatro barridos

`BE_DATA` almacena los resultados para reutilizarlos en **todas las gráficas**, sin repetir el experimento cada vez que cambie una figura. Con `BE_RUN = false`, las gráficas muestran una invitación a activar el barrido, en lugar de datos ficticios. Los tiempos y las fronteras aparecerán solo tras ejecutar Julia en tu máquina.
"""

# ╔═╡ f96b90e7-090e-508e-8a58-255d1f0860a9
BE_DATA = let
    if !BE_RUN
        (n = nothing, d = nothing, decay = nothing, lam = nothing)
    else
        kw = (n0 = BE_BASE_N, d0 = BE_BASE_D,
              decay0 = BE_BASE_DECAY, lam0 = BE_BASE_LAM,
              tau = Float64(BE_TAU), reps = BE_SEEDS, maxit = BE_MAXIT,
              seed = BE_SEED, methods = BE_METHODS, bootstraps = BE_BOOTSTRAPS)
        (n = be_run_sweep(:n, BE_N_GRID, BE_RATIOS; kw...),
         d = be_run_sweep(:d, BE_D_GRID, BE_RATIOS; kw...),
         decay = be_run_sweep(:decay, BE_DECAYS, (BE_RATIO_LINE,); kw...),
         lam = be_run_sweep(:lam, BE_LAMS, (BE_RATIO_LINE,); kw...))
    end
end

# ╔═╡ 5d067867-8c5f-514b-a4db-4fb802a1f588
md"""
### Función `be_matrix`: tabla de aceleraciones para un método

Construye una matriz con filas correspondientes al parámetro del barrido y columnas a `m/d`. La entrada contiene el `speedup` mediano en tiempo total. Las casillas sin éxito completo permanecen como `NaN`, para distinguir **falta de convergencia** de una aceleración baja.
"""

# ╔═╡ e0b8ae67-2349-557d-924b-2324cac50ef7
function be_matrix(data, method::Symbol, levels, ratios)
    M = fill(NaN, length(levels), length(ratios))
    for (i, level) in enumerate(levels), (j, ratio) in enumerate(ratios)
        k = findfirst(r -> r.method === method &&
                           r.level == Float64(level) && r.ratio == Float64(ratio), data.rows)
        if k !== nothing
            M[i, j] = data.rows[k].speedup
        end
    end
    return M
end

# ╔═╡ c99c0389-9738-5b60-950a-3691e903073a
md"""
### Función `be_heatmaps`: mapa del punto de equilibrio

Cada panel corresponde a un sketch. Una celda mayor que `1×` señala una **mediana de tiempo menor** del sketch, tras alcanzar la misma precisión; no constituye por sí sola evidencia estadística de ventaja (véase la tabla de fronteras). Una menor que `1×` favorece a Newton exacto. Las cifras impresas son las aceleraciones y las celdas vacías representan resultados no válidos.

Las tonalidades codifican el *speedup*, mientras que el título de cada panel utiliza el color distintivo del sketch. No se interpolan fronteras entre los tamaños probados.
"""

# ╔═╡ a4275682-c658-533a-bfae-3cadf9ae9458
function be_heatmaps(data, levels, ratios, methods, ylabel_text::AbstractString)
    panels = Any[]
    for nm in methods
        M = be_matrix(data, nm, levels, ratios)
        p = heatmap(Float64.(collect(ratios)), Float64.(collect(levels)), M;
            color = :RdYlGn, clims = (0.0, 3.0),
            xlabel = "m/d", ylabel = ylabel_text,
            xticks = collect(ratios), yticks = collect(levels),
            title = String(nm), titlefontcolor = PLOT_COLORS[nm],
            colorbar_title = "T_exact / T_sketch")
        for (i, lev) in enumerate(levels), (j, ratio) in enumerate(ratios)
            if isfinite(M[i, j])
                annotate!(p, Float64(ratio), Float64(lev),
                          text(@sprintf("%.2f×", M[i, j]), 8, :black))
            end
        end
        push!(panels, p)
    end
    return plot(panels...; layout = (cld(length(panels), 2), 2),
                size = (1200, 420 * cld(length(panels), 2)))
end

# ╔═╡ 433f6a83-aed3-5687-ab7b-07ed1cdf186d
md"""
### Figura 6. ¿Desde qué tamaño `n` compensa cada sketch?

Mapas de `n` contra `m/d`, manteniendo `d`, `decay` y `lam` fijos. Responden a: **¿cuánto debe crecer el número de ejemplos para que el ahorro al aproximar el Hessiano compense las iteraciones adicionales?** Cada celda compara las dos ejecuciones completas, incluyendo la búsqueda de línea y la factorización.
"""

# ╔═╡ e7249340-09dd-5ff2-a822-49f7af4a2e29
FIG6 = BE_DATA.n === nothing ?
    md"Activa `BE_RUN = true` para construir el mapa de **n vs. m/d**." :
    be_heatmaps(BE_DATA.n, BE_N_GRID, BE_RATIOS, BE_METHODS, "Número de filas n")

# ╔═╡ 0245e1cb-75d6-522d-963c-3b838f201a68
md"""
### Figura 7. ¿Cómo cambia el equilibrio con la dimensión `d`?

Ahora se varía el número de incógnitas `d` y se mantiene `n` fijo. Es importante porque formar el Hessiano exacto cuesta aproximadamente `O(nd²)`, pero los sketches y la factorización aproximada tienen otros costos. Una mejora al aumentar `d` no es automática: puede cambiar el condicionamiento y, con ello, el número de pasos necesarios.
"""

# ╔═╡ e2c6e0e3-67de-5a44-ac24-0393dc6dca38
FIG7 = BE_DATA.d === nothing ?
    md"Activa `BE_RUN = true` para construir el mapa de **d vs. m/d**." :
    be_heatmaps(BE_DATA.d, BE_D_GRID, BE_RATIOS, BE_METHODS, "Número de columnas d")

# ╔═╡ e53e3bb8-9317-535f-b934-6e334784b1d7
md"""
### Función `be_speedup_lines`: comparar sketches sin perder su identidad visual

Extrae el `speedup` a una relación `m/d` fija y dibuja una curva para cada método con el **mismo color utilizado en las otras figuras**. La línea horizontal en `1×` indica el equilibrio; los cortes corresponden a configuraciones que no llegaron al objetivo con todas las semillas.
"""

# ╔═╡ 00807c8b-99b1-54b5-8ae2-085cfbff4617
function be_speedup_lines(data, levels, ratio, methods;
                          xlabel_text = "Parámetro", xlog = false)
    p = plot(xlabel = xlabel_text,
        ylabel = "Aceleración total T_exact / T_sketch",
        title = @sprintf("Equilibrio con m/d = %.2g", ratio),
        legend = :outertopright, grid = true, linewidth = 2.5,
        xscale = xlog ? :log10 : :identity, size = (900, 490))
    hline!(p, [1.0]; color = :black, linestyle = :dash, linewidth = 2,
           label = "Equilibrio (1×)")
    for nm in methods
        ys = Float64[]
        for v in levels
            k = findfirst(r -> r.method === nm && r.level == Float64(v) &&
                               r.ratio == Float64(ratio), data.rows)
            push!(ys, k === nothing ? NaN : data.rows[k].speedup)
        end
        plot!(p, Float64.(collect(levels)), ys;
              marker = :circle, markersize = 6, color = PLOT_COLORS[nm],
              label = String(nm), linewidth = 2.5)
    end
    return p
end

# ╔═╡ 9e31471b-7b2b-56af-8247-1e41605ecebf
md"""
### Figura 8. Aceleración frente a `n` y frente a `d`

Las dos curvas condensan las figuras 6 y 7 en el valor fijo `BE_RATIO_LINE`. Aquí se reconoce de forma inmediata cuándo una curva cruza la línea `1×`. Es un **cruce observado en los puntos muestreados**, no una estimación continua del tamaño mínimo.
"""

# ╔═╡ 80faf5e9-4274-5421-8348-79c328eabaec
FIG8 = BE_DATA.n === nothing ?
    md"Activa `BE_RUN = true` para comparar **aceleración vs. n y d**." :
    plot(
        be_speedup_lines(BE_DATA.n, BE_N_GRID, BE_RATIO_LINE, BE_METHODS;
                         xlabel_text = "Número de filas n"),
        be_speedup_lines(BE_DATA.d, BE_D_GRID, BE_RATIO_LINE, BE_METHODS;
                         xlabel_text = "Número de columnas d");
        layout = (1, 2), size = (1350, 475))

# ╔═╡ 885aafd9-2ea2-5d4c-85cf-fe2ab80a2a24
md"""
### Figura 9. Efecto del decaimiento espectral

El generador usa `sⱼ = j^(-decay)`. El barrido modifica **solo** `decay`, manteniendo constantes `n`, `d`, `lam` y `m/d`. Así comprobamos si la concentración espectral facilita o dificulta que un sketch imite la curvatura relevante del Hessiano. Se compara siempre el tiempo necesario para la misma tolerancia relativa de **cada problema**, nunca el valor objetivo absoluto entre problemas distintos.
"""

# ╔═╡ b2c23105-8035-575e-8115-c14eea43af77
FIG9 = BE_DATA.decay === nothing ?
    md"Activa `BE_RUN = true` para estudiar el **decaimiento espectral**." :
    be_speedup_lines(BE_DATA.decay, BE_DECAYS, BE_RATIO_LINE, BE_METHODS;
                     xlabel_text = "Exponente decay de s_j = j^{-decay}")

# ╔═╡ 848193c9-aa4a-5ca6-a1b5-421c4e58be16
md"""
### Figura 10. Efecto de la regularización

En `make_problem`, `μ = lam·n`. Este barrido modifica **solo** `lam`, manteniendo fijos `n`, `d`, `decay` y `m/d`. Una regularización mayor puede estabilizar el Hessiano aproximado, pero también permitir una convergencia muy rápida de Newton exacto: por eso el resultado se debe **medir**, no inferir solo desde el condicionamiento. Se usa escala logarítmica en el eje horizontal.
"""

# ╔═╡ de68241a-a076-50cd-a53c-5f6b75aa271e
FIG10 = BE_DATA.lam === nothing ?
    md"Activa `BE_RUN = true` para estudiar la **regularización**." :
    be_speedup_lines(BE_DATA.lam, BE_LAMS, BE_RATIO_LINE, BE_METHODS;
                     xlabel_text = "lam (μ = n·lam)", xlog = true)

# ╔═╡ e31f7cb6-207b-544a-b5c7-5405cbbe525f
md"""
### Función `be_break_even_table`: primera configuración favorable realmente medida

Para cada sketch y relación `m/d`, buscamos el primer tamaño observado cuyo
**límite inferior del intervalo bootstrap** supera `BE_MARGIN` (1.10× por
omisión), con al menos diez rondas y convergencia en todas. El criterio de
"sostenido" exige tres o más tamaños confirmados hasta el final de la malla.
Una mediana aislada (incluso superior a 1×) no basta para confirmar ventaja.
El intervalo no corrige pruebas múltiples; por tanto, el resultado describe
exclusivamente las configuraciones y la máquina utilizadas.
"""

# ╔═╡ 995ab6c3-8487-5d59-aeeb-6497649d7a85
function be_break_even_table(data,n_grid,ratios,methods,margin;
                             min_sustained::Int=BE_MIN_SUSTAINED,
                             min_reps::Int=BE_MIN_EVIDENCE_REPS)
    @assert min_sustained >= 2 && min_reps >= 2
    rows=Vector{Vector{String}}()
    ns=sort(unique(collect(n_grid)))
    for nm in methods, ratio in ratios
        # Una mediana favorable no basta: IC inferior debe superar el margen,
        # éxito en todas las semillas y número de rondas suficiente.
        valid=filter(r -> r.method==nm && r.ratio==Float64(ratio) &&
                 r.status===:ok && r.reps>=min_reps && r.successes==r.reps &&
                 isfinite(r.ci_low) && r.ci_low>=margin, data.rows)
        by_n=Dict(r.n=>r for r in valid)
        first_n=findfirst(n -> haskey(by_n,n),ns)
        first_value=first_n===nothing ? nothing : ns[first_n]
        sustained=nothing
        for i in eachindex(ns)
            tail=ns[i:end]
            if length(tail)>=min_sustained && all(n -> haskey(by_n,n),tail)
                sustained=ns[i]
                break
            end
        end
        if first_value===nothing
            push!(rows,[String(nm),string(ratio),"No confirmado",
                        "No confirmado","—","—","—","—","—"])
        else
            r=by_n[first_value]
            push!(rows,[String(nm),string(ratio),string(first_value),
                 sustained===nothing ? "No confirmado" : string(sustained),
                 @sprintf("%.2f×",r.speedup),
                 @sprintf("[%.2f×, %.2f×]",r.ci_low,r.ci_high),
                 @sprintf("%.2f ms",r.time_ms),
                 @sprintf("%.1f",r.iters),"$(r.successes)/$(r.reps)"])
        end
    end
    return mdtable(["Sketch","m/d","Primer n confirmado",
         "Primer n sostenido","Speedup","IC bootstrap 95 %",
         "Tiempo","Iteraciones","Éxitos"],rows)
end

# ╔═╡ 9d20b0fa-a0e8-50c9-88e0-18f9a7a2bb5d
md"""
### Frontera de aceleración: evidencia observada, no extrapolación

Un tamaño `n` se considera **confirmado** solo si todas las repeticiones llegaron
al mismo objetivo, hay al menos `BE_MIN_EVIDENCE_REPS` rondas y el límite inferior
del IC bootstrap de 95 % para `mediana(T_exact)/mediana(T_sketch)` supera
`BE_MARGIN` (por defecto, 1.10×). El primer tamaño **sostenido** exige, además,
`BE_MIN_SUSTAINED` o más tamaños desde ese punto hasta el final de la malla,
**todos** confirmados. Un único punto en el extremo derecho nunca es sostenido.

Los IC bootstrap son aproximados, condicionados al conjunto de datos sintético
generado para cada configuración, y no corrigen multiplicidad entre métodos,
tamaños y parámetros. Una frontera observada es descriptiva, no una garantía
para valores mayores de `n`. Con `EXPERIMENT_MODE=:quick` y cinco rondas, los
gráficos pueden usarse para exploración, pero **no se confirma** la frontera.
"""

# ╔═╡ 69eb667a-84a7-5407-971b-5cf5ec6fda11
BE_FRONTIER_TABLE = BE_DATA.n === nothing ?
    md"Activa `BE_RUN = true` para calcular los **primeros n favorables**." :
    be_break_even_table(BE_DATA.n, BE_N_GRID, BE_RATIOS, BE_METHODS, BE_MARGIN;
                        min_sustained=BE_MIN_SUSTAINED, min_reps=BE_MIN_EVIDENCE_REPS)

# ╔═╡ f3eef40d-d6b7-5a1b-9a67-28ef0bab3821
md"""
### Figura 11. ¿El ahorro por iteración compensa las iteraciones adicionales?

Descomponemos *exactamente para las medianas resumidas* la razón de tiempos:

```math
\underbrace{\frac{T_{\mathrm{exacto}}}{T_{\mathrm{sketch}}}}_{\text{aceleración total}}
=
\underbrace{\frac{T_{\mathrm{exacto}}/I_{\mathrm{exacto}}}{T_{\mathrm{sketch}}/I_{\mathrm{sketch}}}}_{\text{ahorro amortizado por iteración}}
\;\Big/\;
\underbrace{\frac{I_{\mathrm{sketch}}}{I_{\mathrm{exacto}}}}_{\text{inflación de iteraciones}}.
```

En el plano, un punto **por encima** de la diagonal `y = x` significa que el ahorro por iteración **supera** el costo de las iteraciones extra. Debajo ocurre lo contrario. Los números junto a los puntos son los distintos valores `m/d`.

**Precisión de la descomposición:** `T/I` incluye costos de inicialización amortizados, no es el tiempo aislado de un paso; por eso este análisis se refiere al **costo efectivo por iteración de la ejecución completa**. Para medir cada operación individual se conserva la sección 4.
"""

# ╔═╡ 3d26ea91-db9f-5f2d-81c9-ef71c71cee21
FIG11 = let
    if BE_DATA.n === nothing
        md"Activa `BE_RUN = true` para ver el **costo por iteración frente a iteraciones extra**."
    else
        subset = filter(r -> r.n == BE_BASE_N && r.method !== :Exact &&
                             isfinite(r.per_iter_gain) && isfinite(r.iteration_factor),
                        BE_DATA.n.rows)
        fig = plot(xlabel = "Iteraciones relativas  I_sketch / I_exacto",
                   ylabel = "Ahorro por iteración  (T_ex/I_ex) / (T_sk/I_sk)",
                   title = "¿El ahorro por iteración compensa las iteraciones adicionales?",
                   legend = :outertopright, grid = true, size = (930, 580))
        vmax = isempty(subset) ? 2.0 : max(2.0, maximum(vcat(
            [r.iteration_factor for r in subset], [r.per_iter_gain for r in subset])) * 1.1)
        plot!(fig, [0, vmax], [0, vmax]; label = "Equilibrio: ahorro = iteraciones",
              color = :black, linestyle = :dash, linewidth = 2)
        for nm in BE_METHODS
            rr = filter(r -> r.method === nm, subset)
            if !isempty(rr)
                scatter!(fig, [r.iteration_factor for r in rr],
                         [r.per_iter_gain for r in rr]; label = String(nm),
                         color = PLOT_COLORS[nm], markersize = 7)
                for r in rr
                    annotate!(fig, r.iteration_factor, r.per_iter_gain,
                              text(@sprintf("  %.1f", r.ratio), 8, PLOT_COLORS[nm]))
                end
            end
        end
        xlims!(fig, 0, vmax)
        ylims!(fig, 0, vmax)
        fig
    end
end

# ╔═╡ 3879ddc6-692f-5179-af2a-e7e1d7c2c3c0
md"""
### Tabla de control: costo, iteraciones e incertidumbre en el problema base

Reporta mediana, cuartiles [Q1,Q3] y el IC bootstrap percentil para la razón de
medianas de tiempos. El IC solo se calcula cuando **todas** las rondas alcanzan
la misma precisión objetivo. El costo amortizado por iteración incluye el
trabajo de inicialización del solver. Los datos del modo rápido son exploratorios.
"""

# ╔═╡ 45e92545-2cc9-5a9b-8e02-b4ef46f90040
BE_BASELINE_TABLE = let
    if BE_DATA.n === nothing
        md"Activa `BE_RUN=true` para obtener la tabla detallada del problema base."
    else
        subset=filter(r -> r.n==BE_BASE_N,BE_DATA.n.rows)
        rows=Vector{Vector{String}}()
        for r in subset
            t=isfinite(r.time_ms) ? @sprintf("%.2f [%.2f, %.2f]",r.time_ms,r.q1_ms,r.q3_ms) : "—"
            its=isfinite(r.iters) ? @sprintf("%.1f",r.iters) : "—"
            gain=isfinite(r.per_iter_gain) ? @sprintf("%.2f×",r.per_iter_gain) : "—"
            extra=isfinite(r.iteration_factor) ? @sprintf("%.2f×",r.iteration_factor) : "—"
            speed=isfinite(r.speedup) ? @sprintf("%.2f×",r.speedup) : "—"
            ci=isfinite(r.ci_low) ? @sprintf("[%.2f×, %.2f×]",r.ci_low,r.ci_high) : "—"
            confirmed=r.method===:Exact ? "Referencia" :
                (r.reps>=BE_MIN_EVIDENCE_REPS && isfinite(r.ci_low) &&
                 r.ci_low>=BE_MARGIN ? "Sí" : "No")
            push!(rows,[String(r.method),r.method===:Exact ? "—" : string(r.ratio),
                        t,its,gain,extra,speed,ci,
                        string(r.successes)*"/"*string(r.reps),r.exits,confirmed])
        end
        mdtable(["Método","m/d","Mediana [Q1,Q3] (ms)","Iteraciones",
                 "Ahorro/iter.","Iter. relativas","Speedup","IC 95 % speedup",
                 "Éxitos","Motivos de parada","Confirmado"],rows)
    end
end

# ╔═╡ 0f52914b-9cdc-52b1-a895-dc77a0c94742
md"""
### Certificación de todas las soluciones de referencia

Para cada problema, se informa la cota absoluta
`||∇f(x_ref)||²/(2μ)` y su cociente respecto a `f(0)-f(x_ref)`.
Antes de comparar tiempos, el código exige que esta cota relativa sea menor
que `REF_FRACTION*BE_TAU`. La mera norma pequeña del gradiente no basta.
Un incumplimiento detiene el barrido, evitando presentar tiempos no certificados.
"""

# ╔═╡ d355cf6a-4c6a-56a8-8d4c-b4d641b1df11
BE_REFERENCE_DIAGNOSTICS = let
    if BE_DATA.n === nothing
        md"Activa `BE_RUN=true` para calcular las certificaciones de los óptimos."
    else
        rows=Vector{Vector{String}}()
        for sweep in (BE_DATA.n,BE_DATA.d,BE_DATA.decay,BE_DATA.lam)
            for r in sweep.cases
                push!(rows,[String(r.sweep),string(r.level),string(r.n),
                    string(r.d),@sprintf("%.2e",r.mu),
                    @sprintf("%.2e",r.grad_ref_inf),
                    @sprintf("%.2e",r.ref_bound),
                    @sprintf("%.2e",r.ref_relative_bound),
                    r.ref_certified ? "Sí" : "No",string(r.reference_iters),
                    string(r.exact_success)*"/"*string(BE_SEEDS)])
            end
        end
        mdtable(["Barrido","Valor","n","d","μ","‖∇f_ref‖∞",
                 "Cota absoluta","Cota relativa","Certificada","Iter. ref.",
                 "Éxitos exacto"],rows)
    end
end

# ╔═╡ 0aec96b0-69eb-5399-917d-dba9c360498f
md"""
### Conclusiones que se pueden obtener (después de ejecutar)

1. **Dependencia de `n` y `d`:** busca en las figuras 6–8 qué combinación logra `speedup > 1` y verifica si también sucede con un margen mínimo de `5 %` (tabla de umbrales).
2. **Costo frente a precisión del Hessiano:** la figura 11 separa *cuánto se ahorra por iteración* de *cuántas iteraciones adicionales hacen falta*. El equilibrio ocurre al cruzar la diagonal.
3. **Estructura espectral y regularización:** compara las figuras 9 y 10 manteniendo los otros parámetros fijos. La frontera puede moverse, desaparecer o incluso no ser monótona.
4. **Robustez:** antes de concluir que un sketch gana, aumenta `BE_SEEDS` y refina la grilla cerca de los cruces; ejecuta en condiciones de carga similares. Un `1.02×` aislado no es evidencia suficiente.
5. **Alcance:** estos resultados describen el **generador logístico y las implementaciones de este notebook**, no una superioridad universal del sketching.

Para guardar las nuevas figuras, ejecuta cuando estén calculadas:

```julia
savefig(FIG6, "equilibrio_n_md.png")
savefig(FIG7, "equilibrio_d_md.png")
savefig(FIG8, "aceleracion_n_d.png")
savefig(FIG9, "aceleracion_decaimiento.png")
savefig(FIG10, "aceleracion_regularizacion.png")
savefig(FIG11, "ahorro_por_iteracion_vs_iteraciones.png")
```

**Nota:** `BE_RUN = false` es el valor predeterminado y evita este barrido adicional. Las seis figuras solo se generan al activar `BE_RUN`.
"""

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000031
md"""
## 11. Cómo leer los resultados

Todo lo que sigue se sostiene sobre la tabla de la sección 4, no sobre la máquina.

**1. El sketch tiene que costar menos que `O(n·d²)`, y solo `CountSketch`, `RowSampling` y
`SRHT` lo consiguen.** El gaussiano cuesta `m·n·d`: es `O(n)` *peor* que armar el Hessiano
exacto en cuanto `m > d`. Con `d = 400` no es candidato a ningún tamaño; la tabla lo muestra
perdiendo por mucho. (El gaussiano es para `n ≫ d` con el Hessiano aplicado sin formar la
matriz, no para una `A` densa en memoria.)

**2. SRHT utiliza una transformada rápida, pero todavía debe rellenar** `n` hasta
`nextpow(2,n)` (para `n = 20000` son 32768, un 64% más de elementos). La versión
optimizada ya no guarda el buffer `npad × d`: lo sustituye por `npad × workers`.
Su tiempo puede seguir siendo mayor que el de CountSketch porque la FWHT requiere
`O(np·d·log(np))` operaciones. Debe medirse de nuevo antes de sacar conclusiones de velocidad.

**3. El scatter de `CountSketch` es barato porque la columna de `SB` cabe en caché.** El
bucle interno escribe `SB[h[i], j]` con `j` fijo: una columna de `m` entradas, 6.4 KB con
`m = 800`. Por eso `O(nnz)` le gana a `O(n·d²)` de BLAS aquí. Con `m` grande ese argumento
desaparece.

**4. La aceleración está topada por lo que no se puede sketchear.** El gradiente, el
diagonal `D`, la pérdida y la búsqueda de línea son `O(n·d)`. Con `n = 2·10⁴`, `d = 400` el
gradiente es ~1 ms contra ~32 ms de Hessiano exacto, así que hay margen; con `d = 20` no lo
habría.

**5. Dos advertencias sobre los datos, no sobre los algoritmos.**
`make_problem` genera filas gaussianas i.i.d., así que las leverage scores son casi
uniformes (`max/mean ≈ 1.3`) y **el muestreo uniforme de filas es casi óptimo**: que
`RowSampling` gane es una propiedad del generador. En una matriz de diseño real con
leverage concentrada (MNIST: `max/mean = 29`, el 4% de las filas con el 26% de la
leverage) ese método se degrada y su garantía pide `m ~ μ·d·log d`, que el muestreo
uniforme no puede dar. Y como `μ = lam·n`, los problemas de la sección 6 no son solo más
grandes: están más regularizados. Mantener `LAM` fijo conserva la regularización de la **pérdida media**;
no conserva μ fijo. La generación independiente de datos tampoco aísla el efecto
computacional puro de n, por lo que se recomienda un experimento adicional con
subconjuntos anidados.

**6. Una comprobación adicional útil:** `newton_sketch` puede detenerse con el
decremento *aproximado* si `stop_on_decrement=true`, que puede diferir del real por un factor
constante. Reportar `‖∇f‖∞/‖∇f₀‖∞` a la salida (como hace la sección 5 con `rel. gap`) es
más informativo que `iters` sola.
"""

# ╔═╡ a3f0c1e2-5b47-4d19-8c6a-000000000032
md"""
## 12. Qué falta por implementar (para la siguiente iteración)

* **`adaptive_newton_sketch`** no está en el archivo, aunque sus constantes (`ETA`, `NU`,
  `alpha_tau`, `RHO`) sí. Sin ella solo se puede medir `m` fijo, y la variante adaptativa
  (subir `m` hasta que el decremento sea fiable) es la que suele ganar en la práctica.
* **`newton_exact` asigna `B` (`n×d`) en cada llamada** — 8·n·d bytes (64 MiB si n=20000 y d=400). No es gran parte de
  los 231 ms (`D·A + syrk` es `O(n·d²)` y la copia `O(n·d)`), pero domina la columna de
  memoria.
* **`RowSampling`**: los bucles son `for j, for k` con `A[idx[k], j]`, que recorre una
  columna de `A` en filas aleatorias. En Julia, las matrices se almacenan por **columnas**; cambiar a `for k, for j`
  no vuelve contiguas las filas. Se debe evaluar con microbenchmarks antes de alterar
  el orden, pues el bucle actual favorece escrituras contiguas en `SB`.
* **`SRHT`**: explorar versiones *pruned* de Hadamard o estrategias alternativas para mitigar el padding. La mejora de buffers no elimina el factor `log(np)`.
* **`EPS_SK`, `RHO`, `ETA`, `NU`** están definidas pero no se usan en ningún sitio: son
  constantes huérfanas mientras no exista el método adaptativo.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[compat]
BenchmarkTools = "~1.8.0"
Plots = "~1.41.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.7"
manifest_format = "2.0"
project_hash = "2738b0e879b82d90ee0ad7f1ef66c666748eec05"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BenchmarkTools]]
deps = ["Compat", "JSON", "Logging", "PrecompileTools", "Printf", "Profile", "Statistics", "UUIDs"]
git-tree-sha1 = "9670d3febc2b6da60a0ae57846ba74670290653f"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.8.0"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "1fa950ebc3e37eccd51c6a8fe1f92f7d86263522"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.7+0"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.ColorVectorSpace.weakdeps]
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.1+2"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "b0bc6d2cad1fed8b7fd59a1551a991cb3d2809e6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.6"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.7.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f4d39eee89f1e58c26bf447f1d4156c0125d6838"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.8.3+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libva_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "7a58e45171b63ed4782f2d36fdee8713a469e6e0"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.1.2+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Random", "Statistics"]
git-tree-sha1 = "59af96b98217c6ef4ae0dfe065ac7c20831d1a84"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.6"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "70329abc09b886fd2c5d94ad2d9527639c421e3e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.14.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "64bbbb7d1499297751b536dd39c58b20750ab1db"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.5.1+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "4d777f73c46b46b8b5276206059cf8a195499314"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.27"

    [deps.GR.extensions]
    IJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f8eb8f7ba13ea75083531647fc8faeda8d541f07"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.27+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "090526e65de8f69648ac156daae153de8b56df62"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.88.3+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "69ffb934a5c5b7e086a0b4fee3427db2556fba6e"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.16+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "c3f99f8e7c98031b8845ece9db86dca713ab9bf8"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "100.14003.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7204148362dafe5fe6a273f855b8ccbe4df8173e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.8.0"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "88352712893ec50bee3680605891eaf0e9ed6368"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.8.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "037babc10853eeb8e585418922246cb97b8e5b74"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.2.0+1"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "17b94ecafcfa45e8360a4fc9ca6b583b049e4e37"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.1.0+0"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b7970cef8ae1c990ba0c09cd8bdc1145e006632f"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "22.1.7+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "f88f3ccef05a6a72a0cf0ed417c8fd68530f4ab2"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.1"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "df7566479bd64f20bd16b09960145e70160ffb3b"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.12"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc3ad4faf30015a3e8094c9b5b7f19e85bdf2386"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.42.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "aebd334d06cee9f24cea70bd19a39749daf73881"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.3+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d620582b1f0cbe2c72dd1d5bd195a9ce73370ab1"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.42.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "bba2d9aa057d8f126415de240573e86a8f39d2a1"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "1.0.1"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.11.4"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "dbd2e8cd2c1c27f0b584f6661b4309609c5a685e"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.4"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e2bb57a313a74b8104064b7efd01406c0a50d2ff"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.6.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05f45c2e0de6259db764adbfd2f1dc6d3f8de13c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "2.0.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1912a9f1b9ca55005b03ba075f8e19993583e237"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.58.2+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools"]
git-tree-sha1 = "663e8b48b789916221e0765393b289ca6c88f24e"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "3.0.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "e4a6721aa89e62e5d4217c0b21bd714263779dda"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.46.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.1"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "83bd514e8ff16b5858ac54c53fa0bcf6002a3b00"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.7"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "edbeefc7a4889f528644251bdb5fc9ab5348bc2c"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.4"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "8b770b60760d4451834fe79dd483e318eee709c4"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.2"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Profile]]
deps = ["StyledStrings"]
uuid = "9abbd945-dff8-562f-b5e8-e1ebf5ef1b79"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "4fbbafbc6251b883f4d2705356f3641f3652a7fe"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.4.0"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "144895f6166994730ee7ff8113b981fc360638f1"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.10.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll", "Qt6Svg_jll"]
git-tree-sha1 = "159d253ab126d5b29230cf53521899bea4ef4648"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.10.2+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "4d85eedf69d875982c46643f6b4f66919d7e157b"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.10.2+1"

[[deps.Qt6Svg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "81587ff5ff25a4e1115ce191e36285ede0334c9d"
uuid = "6de9746b-f93d-5813-b365-ba18ad4a9cf3"
version = "6.10.2+0"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "672c938b4b4e3e0169a07a5f227029d4905456f2"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.10.2+1"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates"]
git-tree-sha1 = "8238217340ad0aaabe11afe39c1098b5bc9f4c8e"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.1.1"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "13cd91cc9be159e3f4d95b857fa2aa383b53772a"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.3"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "178ed29fd5b2a2cfc3bd31c13375ae925623ff36"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.8.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "IrrationalConstants", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "adb9da019510162e67a4493fc235c23203d8b09e"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.13"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "2d0fc55c61321ba245c47be599570d11bac50303"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.8.5"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsStaticArraysCoreExt = ["StaticArraysCore"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b29c22e245d092b8b4e8d3c09ad7baa586d9f573"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "808090ede1d41644447dd5cbafced4731c56bd2f"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.13+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "1a4a26870bf1e5d26cd585e38038d399d7e65706"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.8+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "dcb316b3ce0941f195537dda56bea4517fcd3ff5"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.4+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "0ba01bc7396896a4ace8aab67db31403c71628f4"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.7+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c174ef70c96c76f4c3f4d3cfbe09d018bcd1b53"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.6+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libpciaccess_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "58972370b81423fc546c56a60ed1a009450177c3"
uuid = "a65dc6b1-eb27-53a1-bb3e-dea574b5389e"
version = "0.19.0+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "ed756a03e95fff88d8f738ebc2849431bdd4fd1a"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.2.0+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "2e59214e017a55cb87474a00fa76035c82ac0e17"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.47.0+2"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ef17c47d22224aaecc76e597ab21a072e025cf7b"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.14.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "cb007192783c56d8249db4cf0e3495001edfe414"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.5+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libdrm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libpciaccess_jll"]
git-tree-sha1 = "28e57478e8a160d346a19c28b3fffb9273bcc9c2"
uuid = "8e53e030-5e6c-5a89-a30b-be5b7263a166"
version = "2.4.134+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e51150d5ab85cee6fc36726850f0e627ad2e4aba"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.58+0"

[[deps.libva_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll", "Xorg_libXfixes_jll", "libdrm_jll"]
git-tree-sha1 = "7dbf96baae3310fe2fa0df0ccbb3c6288d5816c9"
uuid = "9a156e7d-b971-5f62-b2c9-67348b8fb97c"
version = "2.23.0+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "a1fc6507a40bf504527d0d4067d718f8e179b2b8"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.13.0+0"
"""

# ╔═╡ Cell order:
# ╟─a3f0c1e2-5b47-4d19-8c6a-000000000001
# ╟─ab8aa28f-1fb2-4a22-86bb-fdcb164632e9
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000002
# ╟─a3f0c1e2-5b47-4d19-8c6a-000000000003
# ╟─f1697827-3fd3-4dc8-9e01-b419eccb7120
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000004
# ╟─c9733114-aa9e-4c6b-8fde-0676a821bb22
# ╠═b2c4b3d2-f373-4cbe-ae05-25dade6bc736
# ╟─70cfdfa7-f9a4-42f8-a0d2-9759c2f0200f
# ╠═ccbb5898-f3d4-4150-8b0b-d64c04c50a41
# ╟─e902b24e-e9b4-481d-a59e-5f3005257624
# ╠═0b0dd8ba-418a-4cae-86d0-ea9123113eb3
# ╟─1281a006-687f-454d-8b88-dc63e4dbe80c
# ╠═66a94d59-bd13-4248-a15d-2eb8fc657e4f
# ╟─131511a4-f9aa-4a5c-9089-900148ff3b27
# ╠═dcaebc11-c273-4aab-a56f-a29c1eb99628
# ╟─48947f90-d78a-47a3-aa7a-5b648041e8fc
# ╠═0c9771c1-8929-4959-8213-827328383ea0
# ╟─daca700e-1e3b-45b8-aeca-485637e43363
# ╠═6338d899-53e7-4c79-9a44-a501ba5f342c
# ╟─a39e85cb-5b13-4c12-ae81-328b0be628a8
# ╠═5db4de9c-21f0-482e-bac8-8edd953f2de4
# ╟─1a7a49a8-1681-4a1c-b221-f777aa2ecdb2
# ╠═c37af10d-81e5-4817-96fc-dfd349d7f3d5
# ╟─e1bba540-73c0-4832-bb58-316c1926747c
# ╠═8e103f08-6a35-421a-8953-0abcf5d9a04b
# ╟─b983097c-b7c6-49c7-b9c8-3cbd55265d07
# ╠═47689f2a-4851-4104-aa7d-47d61a35a8b4
# ╟─bcde3567-77bf-425d-b6b7-d0ecb5d2fd1c
# ╠═79a65887-b030-43de-9c37-1440212c7716
# ╟─0c519c48-5733-4b6f-a1e2-c16977142d01
# ╠═2562a36d-33b0-4e7d-bddf-c0854e23ae58
# ╟─18ce628f-3a2d-4ba7-b044-05340d8c2997
# ╠═d859320e-a73a-4fbf-8efe-9a8feb70cb57
# ╟─1cd89be1-5046-4e5d-900b-f2da16b40ad3
# ╠═4320e82a-d8c3-4025-ae47-f23b41b51373
# ╟─2f4a307e-9ef3-4cec-8748-7e0a96aecce2
# ╠═410332ff-c5a5-4ff4-a636-aced6acbb0d8
# ╟─a298cde7-17e6-4555-89a1-01635ad1a81e
# ╠═0930da91-614a-4a52-a59b-bcf13aedb490
# ╟─a066eb01-4503-4a3a-a800-97ace6392ac3
# ╠═f5f5597e-8008-4dbd-8f25-d3775425d161
# ╟─730c6a8e-f118-4333-b5e2-d1282609ebb4
# ╠═24893b06-287d-4b73-a8be-d05c5eb1b81e
# ╟─db95c50c-51dc-410e-adf2-2116751856cb
# ╠═fca7fd8f-e112-4161-a267-ee142395760e
# ╟─4fea99cf-1ee1-42dd-b17a-9837e3965f50
# ╠═8033b036-036f-48e5-818b-5107d685659d
# ╟─9eecaba9-8e72-460e-9d32-fda1751d231a
# ╠═ef0de825-4da3-429f-8756-1bdc88bafcce
# ╠═d19c88cf-72e1-49f8-a932-1e6bd2831b6f
# ╠═f8aa3dcf-27ac-49d1-aef5-ce459d445e56
# ╟─ed73bb18-c5be-4713-9452-a1d35c7b02aa
# ╠═1d9aefa9-396c-43d2-88a7-e098cd47bed7
# ╟─0c128160-e0ca-46da-ae47-096c22b84305
# ╠═ce612ca7-7139-420b-b946-7f1ffab6a815
# ╟─0e0abbd7-45f5-4ba3-96fd-1dfc1e3b6e35
# ╠═93b7d8ad-060b-48ce-806e-97b35d75527a
# ╟─a5b3934a-00a1-438b-989d-7a08d85a2def
# ╠═9cc2f7b2-b142-45bf-9c80-9ff8a408c9f7
# ╟─348a3bba-893b-4efd-a5f2-3935ef033ed5
# ╠═54b5e787-1113-4586-9fdf-cb481162ebe5
# ╟─af18ff88-18c5-46d9-86b8-214c7f1458f0
# ╠═275de420-a266-4e85-b093-475fee814919
# ╠═3a2a72da-e0a7-41bb-96c0-9871bfb82691
# ╟─9ebdc73a-39a7-452d-b0df-70550dba1a5f
# ╠═2f00d7e1-a197-4127-9a63-007b09c193e3
# ╠═80df2ff1-09db-451b-b645-bfb309ce2694
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000005
# ╟─a3f0c1e2-5b47-4d19-8c6a-000000000006
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000007
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000008
# ╟─a3f0c1e2-5b47-4d19-8c6a-000000000009
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000010
# ╟─a3f0c1e2-5b47-4d19-8c6a-000000000011
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000012
# ╟─a3f0c1e2-5b47-4d19-8c6a-000000000013
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000014
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000015
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000016
# ╟─a3f0c1e2-5b47-4d19-8c6a-000000000017
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000018
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000019
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000020
# ╟─a3f0c1e2-5b47-4d19-8c6a-000000000021
# ╟─a3f0c1e2-5b47-4d19-8c6a-000000000022
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000023
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000024
# ╟─a3f0c1e2-5b47-4d19-8c6a-000000000025
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000026
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000027
# ╟─a3f0c1e2-5b47-4d19-8c6a-000000000028
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000029
# ╠═a3f0c1e2-5b47-4d19-8c6a-000000000030
# ╟─a14a330f-5b39-4563-938c-9086966b32a9
# ╟─b21aba7d-9a5d-4a78-8fd7-e02dcb3acdbd
# ╠═92d26354-28bb-4c3c-93d1-75d4d27dd4ad
# ╟─b2220127-2320-46b3-87b6-6c9fb36ac57c
# ╠═8ff645fb-0e74-4566-af05-8af2c44c1a1d
# ╟─6e54148e-a78f-48a0-abd1-b7da544a9a1f
# ╠═ee65150b-5bcb-41ac-8c8c-8ffa099ab998
# ╟─2eb73278-4880-4c9b-9f52-8f373395f12a
# ╠═442086dc-2354-47c8-b07c-89ca662393b1
# ╠═79311892-e37a-4136-bbe7-4f6b41c2dcdc
# ╟─3a35cc6d-70d4-4966-bf63-e9c3e96f4e5f
# ╠═1187b46a-46dc-4c85-862c-b42ecea9e300
# ╟─c8182c92-877c-4438-b66b-ebd8ce9a7e21
# ╠═55a8c10b-a01b-4041-a996-c4a338e5cf5d
# ╟─7f14c255-91bd-48d3-873f-c2a3edf442cb
# ╠═3cb036a4-9048-4ae5-bf31-05e9e2a16e89
# ╠═ed0dc2d0-9d42-4ae4-abf7-9c06484e9cc4
# ╟─83af4448-5924-4a60-b07a-10f07e2aaebf
# ╟─6ead77ea-e145-4ca6-b186-a678710db00b
# ╟─4e7a2a53-9179-4a2d-9b00-2d001d6201a9
# ╠═eef38949-59be-4f16-a062-7feb5c0f9c45
# ╟─bbc9359a-bc1c-454a-836b-4d49b7a555ab
# ╠═ffb01359-93b2-4645-8a0f-6df7e1821b81
# ╟─2f0b3e16-798e-4931-b697-93e4f7d5eab9
# ╠═9b546354-eff3-4420-baf7-add5ce5726e5
# ╠═e403c78e-02f8-46cf-8eb4-da07fd463ebe
# ╟─b5676535-677a-4c7b-b9fe-91ad2c7036aa
# ╟─8a7dcbce-87a5-4d11-b18b-8f5d19677509
# ╠═d6779ab2-3efc-4dc5-99b1-3370f48e52e4
# ╟─784f6622-7451-4783-bf85-672ad02a80a3
# ╠═d7130607-0fd9-4359-b52b-5e9b74c1d7e0
# ╠═c0b5516d-c078-449c-b059-044018027814
# ╟─c4dff7fd-a7b5-4c4d-86c6-bbaacbbc6b0a
# ╠═d01ab88f-07af-4f41-ba49-04be10b4796f
# ╟─67a7e4ba-c9f1-438c-9c5d-5a17ceb8b142
# ╠═6a38c7f7-9d3c-4c4c-84ef-2d9ec83add9b
# ╟─260697bb-1397-42e7-b54c-01d3ea4f5b4d
# ╟─87e64eb4-4c0d-5d4e-bbab-f0da341ac680
# ╠═117eccae-03c0-5a36-913a-33ede81341fa
# ╠═927d7f48-e52e-475a-bda5-a6306e3cb7ad
# ╟─51e6b5a3-71e4-5ce5-bcb7-820000fe4951
# ╠═8ca1eb1b-6b7a-5901-954e-2353d7a2a088
# ╟─5b7c27f7-3582-5fb0-acf9-e30cde2beb66
# ╠═a1e31148-562e-563a-a2dd-188bd75a48bd
# ╟─4d5ac21c-579e-5d8b-a6d6-549aacf514ea
# ╠═f96b90e7-090e-508e-8a58-255d1f0860a9
# ╟─5d067867-8c5f-514b-a4db-4fb802a1f588
# ╠═e0b8ae67-2349-557d-924b-2324cac50ef7
# ╟─c99c0389-9738-5b60-950a-3691e903073a
# ╠═a4275682-c658-533a-bfae-3cadf9ae9458
# ╟─433f6a83-aed3-5687-ab7b-07ed1cdf186d
# ╠═e7249340-09dd-5ff2-a822-49f7af4a2e29
# ╟─0245e1cb-75d6-522d-963c-3b838f201a68
# ╠═e2c6e0e3-67de-5a44-ac24-0393dc6dca38
# ╟─e53e3bb8-9317-535f-b934-6e334784b1d7
# ╠═00807c8b-99b1-54b5-8ae2-085cfbff4617
# ╟─9e31471b-7b2b-56af-8247-1e41605ecebf
# ╠═80faf5e9-4274-5421-8348-79c328eabaec
# ╟─885aafd9-2ea2-5d4c-85cf-fe2ab80a2a24
# ╠═b2c23105-8035-575e-8115-c14eea43af77
# ╟─848193c9-aa4a-5ca6-a1b5-421c4e58be16
# ╠═de68241a-a076-50cd-a53c-5f6b75aa271e
# ╟─e31f7cb6-207b-544a-b5c7-5405cbbe525f
# ╠═995ab6c3-8487-5d59-aeeb-6497649d7a85
# ╟─9d20b0fa-a0e8-50c9-88e0-18f9a7a2bb5d
# ╠═69eb667a-84a7-5407-971b-5cf5ec6fda11
# ╟─f3eef40d-d6b7-5a1b-9a67-28ef0bab3821
# ╠═3d26ea91-db9f-5f2d-81c9-ef71c71cee21
# ╟─3879ddc6-692f-5179-af2a-e7e1d7c2c3c0
# ╠═45e92545-2cc9-5a9b-8e02-b4ef46f90040
# ╟─0f52914b-9cdc-52b1-a895-dc77a0c94742
# ╠═d355cf6a-4c6a-56a8-8d4c-b4d641b1df11
# ╟─0aec96b0-69eb-5399-917d-dba9c360498f
# ╟─a3f0c1e2-5b47-4d19-8c6a-000000000031
# ╟─a3f0c1e2-5b47-4d19-8c6a-000000000032
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
