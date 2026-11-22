### A Pluto.jl notebook ###
# v1.0.3

using Markdown
using InteractiveUtils

# ╔═╡ 128b25dc-507f-58e5-a221-a5fc442f57ba
using LinearAlgebra, Random, Statistics, Printf

# ╔═╡ 4e017c8d-950f-54a7-924e-895ffa2b36fb
md"""
# Newton Sketch en regresión logística

**¿Cuándo compensa aproximar la curvatura de un problema de optimización?** Newton exacto aprovecha el Hessiano para elegir una dirección de descenso. Si hay muchas más observaciones que variables, formar ese Hessiano puede ser la parte costosa de cada iteración. Newton Sketch reduce el número de filas que intervienen en ese cálculo y conserva el gradiente exacto.

La comparación tiene dos componentes. Una dirección aproximada puede ser más barata de calcular y, a la vez, exigir más iteraciones. Para saber qué efecto domina, mediremos el **tiempo total hasta una misma precisión en el valor objetivo**. También registraremos las iteraciones y la memoria asignada durante la resolución.
"""

# ╔═╡ 3661171d-83d3-5117-b5dc-046235e57f3d
md"""
## 1. Las cuatro matrices de sketch

Sea ``B\in\mathbb R^{n\times d}`` una matriz con ``n`` filas y ``d`` columnas. En la sección 2, cada fila de ``B`` será una observación ponderada de la regresión. Por ahora solo interesa cómo transformar esas filas. Una matriz aleatoria ``S\in\mathbb R^{m\times n}`` produce ``C=SB\in\mathbb R^{m\times d}``. El parámetro ``m`` fija cuántas filas se conservan después de la compresión.

Lo que necesitamos aproximar es la curvatura en cada dirección ``v\in\mathbb R^d``. La identidad

```math
v^\top C^\top C v=\|SBv\|_2^2
```

muestra por qué interesa que ``\|SBv\|_2^2`` sea cercana a ``\|Bv\|_2^2``. Las cuatro construcciones siguientes satisfacen ``\mathbb E[C^\top C]=B^\top B`` para ``B`` fija. En una realización concreta, la aproximación depende de cómo se combinan las filas y de cuántas se usan.

**Una misma aplicación para los cuatro métodos.** En cada apartado, la explicación va seguida de la función que usa Newton Sketch y de una llamada con una matriz de cinco filas, dos columnas y una salida de tres filas. Las funciones reciben `A` y `w` por separado y escriben ``C=S\operatorname{diag}(w)A`` en el arreglo `s.C`, reservado por `workspace`. Devuelven ese mismo arreglo para poder inspeccionarlo.

Cada ejemplo muestra ``S`` junto a ``C``. Para reconstruir la pequeña matriz ``S``, `matriz_del_ejemplo` aplica el mismo método a la identidad con pesos unitarios y la misma semilla. De este modo se puede comprobar explícitamente el producto. Las pruebas de la sección 3 aplican las proyecciones directamente a los datos ponderados.
"""

# ╔═╡ 3632e8f2-10d2-5e61-ada1-e690455a43f1
ejemplo_sketch = (
    A = [1.0 0.0; 0.0 1.0; 1.0 1.0; -1.0 2.0; 2.0 -1.0],
    w = [1.0, 0.5, 1.0, 0.5, 1.0],
    m = 3,
    seed = 2026,
)

# ╔═╡ 66d49d26-1ca5-5fbe-a79c-f31003be5435
function signed_weights!(dest, w, rng)
    @inbounds for i in eachindex(w)
        dest[i] = rand(rng, Bool) ? w[i] : -w[i]
    end
    return dest
end

# ╔═╡ afb9b551-b165-5aba-a9d6-dd59557c76f1
begin
    # Tiempo acumulado por etapa en ns. Sin perfil, las marcas no consultan el reloj.
    tick(::Nothing) = UInt64(0)
    tick(times) = time_ns()
    mark!(::Nothing, stage, start) = UInt64(0)
    function mark!(times, stage, start)
        now = time_ns()
        times[stage] += now - start
        now
    end
end

# ╔═╡ bfc746e6-ce71-5638-8cb6-d0386a2b0bfd
md"""
### 1.1 Proyección gaussiana

Tomamos ``S=G/\sqrt m``, donde las entradas de ``G\in\mathbb R^{m\times n}`` son normales estándar independientes. Si ``b_i^\top`` es la fila ``i`` de ``B``, cada fila proyectada mezcla todas las observaciones con coeficientes distintos,

```math
c_k^\top=\frac{1}{\sqrt m}\sum_{i=1}^n G_{ki}\,b_i^\top .
```

La normalización compensa las ``m`` filas de salida. La mezcla densa requiere generar ``mn`` coeficientes en cada proyección nueva y multiplicarlos por las ``d`` columnas. La función `gaussian!` divide los índices de entrada en paneles y acumula sus productos en ``C``. Así puede reutilizar memoria sin guardar la matriz gaussiana completa. La última prueba mide el costo de generar estos coeficientes frente al de multiplicar los paneles.

En la llamada `gaussian!(s, A, w, rng)` se genera una proyección nueva. Para reutilizar coeficientes, la misma función acepta un banco de matrices y el número de iteración. En ambos casos vuelve a ponderar las filas con los pesos actuales.
"""

# ╔═╡ d48a83c5-3815-5055-85f1-b6a854e51db0
begin
    # Paneles de hasta 8 MiB con los tamaños utilizados en las pruebas.
    panelwidth(m,n) = min(n, 2048, max(1, (8 * 1024^2) ÷ (8m)))
    panels(m,n) = (i:min(i+panelwidth(m,n)-1,n) for i in 1:panelwidth(m,n):n)

    function gaussian_bank(rng, m, n, count)
        bank = [Matrix{Float64}(undef,m,n) for _ in 1:count]
        for G in bank, rows in panels(m,n)
            randn!(rng, view(G,:,rows))
        end
        bank
    end
end

# ╔═╡ 6800aa01-f855-5ab5-9fad-3e1996a8d4bd
function workspace(method, n, d, m, cached=false)
    C = zeros(method == :Exacto ? n : m, d)
    b = method == :Gaussian ? panelwidth(m,n) : 0
    np = method == :SRHT ? nextpow(2,n) : 0
    idxlen = method == :CountSketch ? n : method == :RowSampling ? m : 0
    (; C, weighted=zeros(b,d), panel=zeros(cached ? 0 : m,b),
       idx=zeros(Int,idxlen), perm=collect(1:np), column=zeros(np),
       signs=zeros(method in (:CountSketch,:SRHT) ? n : 0))
end

# ╔═╡ 69034d23-e527-5101-b0f3-48a64f8db913
function matriz_del_ejemplo(aplicar!, metodo, datos)
    n = size(datos.A, 1)
    s = workspace(metodo, n, n, datos.m)
    identidad = Matrix{Float64}(I, n, n)
    return aplicar!(s, identidad, ones(n), Xoshiro(datos.seed))
end

# ╔═╡ a901d763-51e9-56af-8ad6-37dc60c82f60
function gaussian!(s, A, w, rng, bank=(), iter=1, times=nothing)
    m, n = size(s.C, 1), size(A, 1)
    scale = inv(sqrt(m))
    for rows in panels(m, n)
        start = tick(times)
        cols = 1:length(rows)
        if isempty(bank)
            G = view(s.panel, :, cols)
            randn!(rng, G)
        else
            G = view(bank[mod1(iter, length(bank))], :, rows)
        end
        start = mark!(times, 2, start)

        B = view(s.weighted, cols, :)
        @views B .= A[rows, :] .* w[rows]
        start = mark!(times, 3, start)

        # El primer panel sobrescribe la salida. Los siguientes acumulan.
        mul!(s.C, G, B, scale, first(rows) == 1 ? 0.0 : 1.0)
        mark!(times, 4, start)
    end
    return s.C
end

# ╔═╡ c7076c38-aee8-55bd-bd4c-267af69c5144
let
    (; A, w, m, seed) = ejemplo_sketch
    s = workspace(:Gaussian, size(A)..., m)
    C = gaussian!(s, A, w, Xoshiro(seed))
    S = matriz_del_ejemplo(gaussian!, :Gaussian, ejemplo_sketch)
    @assert C ≈ S * (w .* A)
    (; S, C)
end

# ╔═╡ 0842659a-e4ea-53ec-ac7e-cb196020b7af
md"""
### 1.2 CountSketch

A cada índice de entrada ``i`` le asignamos, de manera independiente, un destino uniforme ``h(i)\in\{1,\ldots,m\}`` y un signo equiprobable ``\varepsilon_i\in\{-1,1\}``. La columna ``i`` de ``S`` contiene una sola entrada no nula, ``S_{h(i),i}=\varepsilon_i``. La fila de salida ``k`` acumula todas las filas que comparten destino,

```math
c_k^\top=\sum_{i:\,h(i)=k}\varepsilon_i b_i^\top .
```

Por ejemplo, si ``h=(1,2,1,2)`` y ``\varepsilon=(+1,-1,-1,+1)``, obtenemos ``c_1^\top=b_1^\top-b_3^\top`` y ``c_2^\top=-b_2^\top+b_4^\top``. Los signos hacen que los términos cruzados de esas colisiones se cancelen en esperanza. `countsketch!` guarda los pesos con signo en `s.signs` y los destinos en `s.idx`. Empieza con ``C=0``, recorre cada entrada de ``A`` una vez y la acumula en la fila de destino. En la matriz ``S`` del ejemplo puedes identificar ese destino por la única entrada no nula de cada columna.
"""

# ╔═╡ a28802f4-4ff9-57b2-be1d-4737ccf14953
function countsketch!(s, A, w, rng)
    n, d = size(A)
    C = s.C
    m = size(C, 1)
    signed_weights!(s.signs, w, rng)
    rand!(rng, s.idx, 1:m)
    fill!(C, 0.0)
    @inbounds for j in 1:d, i in 1:n
        C[s.idx[i], j] += s.signs[i] * A[i, j]
    end
    return C
end

# ╔═╡ 0f6e1c61-f536-5928-aeee-eeaf44a57c5b
let
    (; A, w, m, seed) = ejemplo_sketch
    s = workspace(:CountSketch, size(A)..., m)
    C = countsketch!(s, A, w, Xoshiro(seed))
    S = matriz_del_ejemplo(countsketch!, :CountSketch, ejemplo_sketch)
    @assert C ≈ S * (w .* A)
    (; S, C)
end

# ╔═╡ 3b50f7e9-efdb-5131-b484-aebe277769e3
md"""
### 1.3 Muestreo uniforme de filas (RowSampling)

Elegimos ``m`` índices ``i_1,\ldots,i_m`` de forma uniforme e independiente en ``\{1,\ldots,n\}``, **con reemplazo**. La fila ``k`` de ``S`` es ``\sqrt{n/m}\,e_{i_k}^\top``, donde ``e_i`` es el vector canónico. Cada fila de salida copia una sola observación ponderada,

```math
c_k^\top=\sqrt{\frac{n}{m}}\,b_{i_k}^\top .
```

El factor ``\sqrt{n/m}`` corrige la escala de la matriz de Gram. Una observación puede aparecer varias veces y otra puede quedar fuera. Si unas pocas filas aportan gran parte de la curvatura en cierta dirección, el resultado dependerá de cuántas salgan en la muestra. `rowsampling!` guarda en `s.idx` los índices sorteados y solo lee esas ``m`` filas para construir ``C``. En la matriz ``S`` del ejemplo, la posición de la única entrada no nula de cada fila indica qué observación se eligió. El cálculo del gradiente sigue usando las ``n`` observaciones.
"""

# ╔═╡ 98fbde4d-0bd3-5b4c-a268-cd81af177081
function rowsampling!(s, A, w, rng)
    n, d = size(A)
    C = s.C
    m = size(C, 1)
    rand!(rng, s.idx, 1:n)
    scale = sqrt(n / m)
    @inbounds for j in 1:d, k in 1:m
        i = s.idx[k]
        C[k, j] = scale * w[i] * A[i, j]
    end
    return C
end

# ╔═╡ 7b872bd6-ebae-5304-aee4-bfbdfebe27ce
let
    (; A, w, m, seed) = ejemplo_sketch
    s = workspace(:RowSampling, size(A)..., m)
    C = rowsampling!(s, A, w, Xoshiro(seed))
    S = matriz_del_ejemplo(rowsampling!, :RowSampling, ejemplo_sketch)
    @assert C ≈ S * (w .* A)
    (; S, C)
end

# ╔═╡ e4cc3a19-4529-5149-bf12-5b902fb6093c
md"""
### 1.4 Transformada aleatoria de Hadamard con submuestreo (SRHT)

La SRHT mezcla las filas antes de elegir una muestra. Sea ``N`` la menor potencia de dos mayor o igual que ``n``. Denotemos por ``P`` la extensión de ``n`` a ``N`` filas con ceros, por ``D`` una diagonal de ``n`` signos aleatorios independientes y por ``H_N`` la matriz de Hadamard **sin normalizar**, con ``H_N^\top H_N=NI``. Si ``R`` selecciona ``m\le N`` filas uniformemente **sin reemplazo**, el producto implementado es

```math
C=\frac{1}{\sqrt m}\,R H_N P D B .
```

La función `hadamard!` transforma cada columna en etapas de sumas y diferencias, ``(u,v)\mapsto(u+v,u-v)``. Así, una fila seleccionada ya contiene contribuciones de las distintas observaciones. `srht!` reutiliza un vector de longitud ``N`` para todas las columnas y emplea la misma selección de filas en cada columna. La mezcla cuesta del orden de ``dN\log_2 N`` operaciones.

En el ejemplo hay cinco filas de entrada. `srht!` completa cada columna con tres ceros para aplicar la transformada de longitud ocho. Las tres filas seleccionadas se leen en el mismo orden para todas las columnas.
"""

# ╔═╡ 5997e6de-27c9-5166-840f-afc75306328d
function hadamard!(v)
    h = 1
    while h < length(v)
        @inbounds for base in 1:2h:length(v)
            @simd for i in base:base+h-1
                a, b = v[i], v[i+h]
                v[i], v[i+h] = a+b, a-b
            end
        end
        h *= 2
    end
end

# ╔═╡ 1d040a45-c257-58f1-9ab2-5639898628ea
function srht!(s, A, w, rng)
    n, d = size(A)
    C = s.C
    m = size(C, 1)
    v = s.column
    m <= length(v) || error("SRHT requiere m ≤ nextpow(2,n)")

    signed_weights!(s.signs, w, rng)
    shuffle!(rng, s.perm)
    scale = inv(sqrt(m))
    for j in 1:d
        @inbounds for i in 1:n
            v[i] = s.signs[i] * A[i, j]
        end
        fill!(view(v, n+1:length(v)), 0.0)
        hadamard!(v)
        @inbounds for k in 1:m
            C[k, j] = scale * v[s.perm[k]]
        end
    end
    return C
end

# ╔═╡ e4c4b731-00fc-5b5e-89aa-20e11a57dfa8
let
    (; A, w, m, seed) = ejemplo_sketch
    s = workspace(:SRHT, size(A)..., m)
    C = srht!(s, A, w, Xoshiro(seed))
    S = matriz_del_ejemplo(srht!, :SRHT, ejemplo_sketch)
    @assert C ≈ S * (w .* A)
    (; S, C)
end

# ╔═╡ cb63e855-0aef-5b1e-8745-a8dc50ec1ebd
md"""
#### Cómo elige Newton la proyección

En cada iteración, `solve` llama a `project!` con el método elegido y los pesos actuales. Esta función delega en la implementación del apartado correspondiente. Para Newton exacto, copia directamente las filas ponderadas en ``C``. Así, todas las variantes entregan al solucionador la matriz que necesita para formar ``C^\top C+\mu I``.
"""

# ╔═╡ 8bc13cf3-d348-588e-a73c-fb6c2ac832f7
function project!(s, method, A, w, rng, bank, iter, times)
    if method == :Exacto
        s.C .= w .* A
        return s.C
    elseif method == :Gaussian
        return gaussian!(s, A, w, rng, bank, iter, times)
    elseif method == :CountSketch
        return countsketch!(s, A, w, rng)
    elseif method == :RowSampling
        return rowsampling!(s, A, w, rng)
    elseif method == :SRHT
        return srht!(s, A, w, rng)
    end
    error("Método desconocido: $method")
end

# ╔═╡ 516c900b-5289-5a48-9506-fb8f67280879
md"""
## 2. Formulación y construcción del problema

### 2.1 Regresión logística regularizada

Cada observación consta de una fila ``a_i^\top`` de ``A\in\mathbb R^{n\times d}`` y una etiqueta ``y_i\in\{-1,1\}``. Buscamos un vector ``x\in\mathbb R^d`` que minimice la suma de pérdidas logísticas con penalización cuadrática,

```math
f(x)=\sum_{i=1}^{n}\log\!\left(1+\exp(-y_i a_i^\top x)\right)
       +\frac{\mu}{2}\|x\|_2^2,\qquad \mu>0.
```

El producto ``y_i a_i^\top x`` es el margen de la observación. Un margen positivo grande produce una pérdida pequeña. La penalización controla el tamaño de ``x`` y hace que el problema tenga un único minimizador ``x^\star``. Todos los métodos parten de ``x_0=0``, por lo que ``f(x_0)=n\log 2``.
"""

# ╔═╡ d1e8a184-728f-5cf1-bfe7-5415d1310c4d
md"""
### 2.2 ¿Cómo se construye la instancia?

1. **Matriz de datos.** `problem(cfg)` obtiene una matriz ortogonal ``Q`` por factorización QR de una matriz gaussiana de tamaño ``d\times d``. Con otra matriz gaussiana independiente ``G_A\in\mathbb R^{n\times d}``, forma ``A=G_A\,\operatorname{diag}(1^{-p},2^{-p},\ldots,d^{-p})\,Q^\top``. El exponente `p` controla el decaimiento de las escalas que se imponen antes de rotar las variables.
2. **Señal.** Se extrae un vector gaussiano ``u\in\mathbb R^d`` y se calcula ``z=Au``. El código multiplica ``z`` por `cfg.snr` dividido por la desviación estándar muestral de ``z``. Así, `snr` fija la dispersión de los puntajes que generarán las etiquetas.
3. **Etiquetas.** Con los puntajes reescalados se elige ``y_i=+1`` con probabilidad ``\sigma(z_i)=(1+e^{-z_i})^{-1}`` y ``y_i=-1`` en el caso restante. La aleatoriedad de las etiquetas y la regularización hacen que el vector usado para generar la señal pueda diferir del minimizador empírico ``x^\star``.

En la configuración inicial hay 5000 observaciones y 100 variables. La celda `cfg` permite modificar las dimensiones, el decaimiento, la regularización, la dispersión y la semilla, junto con los parámetros de las pruebas. `experiment` construye una sola instancia de ``(A,y)`` para compartirla entre los métodos y las repeticiones.
"""

# ╔═╡ 64b99d4d-1547-5873-9928-bef0fbd8e026
cfg = (n=5000, d=100, μ=0.5, p=1.0, snr=3.0, seed=2026, τ=1e-4, maxit=100,
       ratios=(1, 1.5, 2, 3, 4, 5,6,7, 8,9,10), reps=7, family=4, blas_threads=1)

# ╔═╡ fdf21465-3255-5858-ba44-7885b7b22dc5
function problem(c)
    rng = Xoshiro(c.seed)
    Q = Matrix(qr(randn(rng, c.d, c.d)).Q)
    A = randn(rng, c.n, c.d) * (Diagonal((1:c.d) .^ (-c.p)) * Q')
    z = A * randn(rng, c.d)
    z .*= c.snr / std(z)
    y = ifelse.(rand(rng, c.n) .< 1 ./ (1 .+ exp.(-z)), 1.0, -1.0)
    (; A, y, μ=c.μ)
end

# ╔═╡ 4bfd4e2e-56fc-5d5f-b83a-1b0ad2d8a62c
md"""
### 2.3 Del Hessiano a la dirección de Newton Sketch

Escribamos ``z_i=a_i^\top x`` y ``r_i=-y_i\sigma(-y_i z_i)``. El gradiente de la función objetivo es ``g=A^\top r+\mu x``. Para construir el Hessiano, `gradient_weights!` calcula también un peso por observación,

```math
w_i=\sqrt{\sigma(z_i)\sigma(-z_i)},\qquad
B(x)=\operatorname{diag}(w)A,\qquad
\nabla^2 f(x)=B(x)^\top B(x)+\mu I.
```

Aquí aparece la matriz ``B`` de la sección 1. Sus filas dependen del iterado. Una observación con puntaje cercano a cero tiene mayor peso en la curvatura que otra cuyo puntaje tiene gran valor absoluto.

En cada paso, Newton exacto toma ``C=B(x)`` y Newton Sketch toma ``C=SB(x)``. `solve` usa Cholesky para resolver

```math
(C^\top C+\mu I)\,s=-g.
```

Para datos densos, formar la matriz de Gram cuesta del orden de ``nd^2`` operaciones con Newton exacto y de ``md^2`` una vez disponible el sketch. Ambos resuelven un sistema de tamaño ``d\times d``. El término ``\mu I`` mantiene el sistema definido positivo incluso si ``C`` pierde rango. La búsqueda de Armijo comienza con ``\alpha=1`` y reduce el paso a la mitad hasta aceptar ``f(x+\alpha s)\le f(x)+0.1\,\alpha\,g^\top s``. Todos los métodos usan la misma función, el mismo gradiente y la misma regla de aceptación.
"""

# ╔═╡ 1e779b37-7c48-5cea-b4a6-307198ef8e55
begin
    function value(P, x, z)
        loss = 0.0
        @inbounds for i in eachindex(z)
            t = -P.y[i] * z[i]
            loss += max(t, 0.0) + log1p(exp(-abs(t)))
        end
        loss + P.μ * dot(x, x) / 2
    end

    function gradient_weights!(g, w, r, P, x, z)
        @inbounds for i in eachindex(z)
            t = P.y[i] * z[i]
            e = exp(-abs(t))
            r[i] = -P.y[i] * (t >= 0 ? e : 1.0) / (1 + e)
            w[i] = sqrt(e) / (1 + e)
        end
        mul!(g, P.A', r)
        @. g += P.μ * x
    end
end

# ╔═╡ bd7c5eb9-6d17-5f23-9e7a-b2b523e184fb
function solve(P, method, m, rng, bank, target, maxit; times=nothing, reference=false)
    n, d = size(P.A)
    s = workspace(method,n,d,m,!isempty(bank))
    x, trial, g, step = (zeros(d) for _ in 1:4)
    z, ztrial, dz, w, r = (zeros(n) for _ in 1:5)
    H = zeros(d,d)
    f = value(P,x,z)
    for k in 0:maxit
        !reference && f <= target && return (; f, k, ok=true)
        start = tick(times)
        gradient_weights!(g,w,r,P,x,z)
        mark!(times,1,start)
        if reference
            # Por convexidad fuerte: f(x) - f★ ≤ ‖g‖²/(2μ).
            err = dot(g,g)/(2P.μ)
            err <= 1e-12 * max(1,abs(f)) && return (; f, err, k, ok=true)
        end
        k == maxit && return (; f, k, ok=false)
        project!(s,method,P.A,w,rng,bank,k+1,times)
        start = tick(times)
        BLAS.syrk!('U','T',1.0,s.C,0.0,H)
        @inbounds for j in 1:d
            H[j,j] += P.μ
        end
        start = mark!(times,5,start)
        step .= .-g
        ldiv!(cholesky!(Symmetric(H,:U)),step)
        start = mark!(times,6,start)
        slope = dot(g,step)
        mul!(dz,P.A,step)
        α = 1.0
        accepted = false
        while α >= 2.0^-40
            @. trial = x + α * step
            @. ztrial = z + α * dz
            ftrial = value(P,trial,ztrial)
            if isfinite(ftrial) && ftrial <= f + 0.1 * α * slope
                x, trial = trial, x
                z, ztrial = ztrial, z
                f = ftrial
                accepted = true
                break
            end
            α *= 0.5
        end
        mark!(times,7,start)
        accepted || return (; f, k, ok=false)
    end
end

# ╔═╡ ee370f13-e950-5a8a-8213-000904348e26
md"""
## 3. Pruebas numéricas

### 3.1 Protocolo y precisión común

**Referencia.** Antes de medir, `experiment` resuelve la instancia con Newton exacto hasta obtener ``x_{\rm ref}``. Por convexidad fuerte, el valor óptimo ``f^\star`` queda en el intervalo

```math
L=f(x_{\rm ref})-\frac{\|\nabla f(x_{\rm ref})\|_2^2}{2\mu}
\ \le\ f^\star\ \le\ f(x_{\rm ref}).
```

**Parada.** Una corrida tiene éxito cuando ``f(x)\le L+\tau(f(x_0)-L)``. Como ``L\le f^\star`` y ``0<\tau<1``, esto garantiza ``f(x)-f^\star\le\tau\,(f(x_0)-f^\star)``. El código verifica además que el ancho del intervalo de referencia sea como máximo el 1 % de ``\tau(f(x_0)-L)``.

**Medición.** `sweep` calienta cada combinación de método y estrategia. Después cambia la semilla del sketch en cada ronda y sortea el orden de las corridas sobre los mismos datos. La configuración inicial usa siete rondas y un hilo BLAS. Las tablas comparativas muestran medianas y la fracción de éxitos. Si alguna repetición falla, la tabla omite la mediana del grupo. Los MiB representan bytes acumulados que Julia asignó durante la preparación del sketch y la resolución. La construcción de los datos y el cálculo de la referencia se hacen antes del cronometraje.
"""

# ╔═╡ 0b0d2f7d-2eb7-5449-b8d4-4a7f1b2e1d7d
experiment = let
    BLAS.set_num_threads(cfg.blas_threads)
    P = problem(cfg)
    emptybank = Matrix{Float64}[]
    ref = solve(P,:Exacto,0,Xoshiro(cfg.seed),emptybank,-Inf,cfg.maxit; reference=true)
    @assert ref.ok "La referencia no convergió"
    lower = ref.f - ref.err
    f0 = size(P.A,1) * log(2)
    @assert ref.err <= 0.01 * cfg.τ * (f0 - lower)
    (; P, ref, lower, f0, target=lower + cfg.τ * (f0-lower))
end;

# ╔═╡ df795018-66ad-5d53-800f-8d1df6e96977
begin
    method_names = (:Exacto,:Gaussian,:CountSketch,:RowSampling,:SRHT)
    modes = (:nueva,:fija,:familia)

    function measure(E, c, method, m, mode, seed; profile=false)
        rng = Xoshiro(seed)
        count = mode == :fija ? 1 : mode == :familia ? c.family : 0
        times = profile ? zeros(UInt64,7) : nothing
        prep = @timed gaussian_bank(rng,m,size(E.P.A,1),count)
        run = @timed solve(E.P,method,m,rng,prep.value,E.target,c.maxit; times)
        (; method, m, mode, seed, run.value..., prep=1e3*prep.time, online=1e3*run.time,
           total=1e3*(prep.time+run.time), bytes=prep.bytes+run.bytes,
           stored=8*count*m*size(E.P.A,1), stages=times)
    end

    function sweep(E,c)
        cases = [(method,round(Int,ratio*c.d),:nueva) for method in method_names[2:end] for ratio in unique([c.ratios...; 6])]
        append!(cases,[(:Exacto,0,:nueva),(:Gaussian,6*c.d,:fija),(:Gaussian,6*c.d,:familia)])
        for case in unique(t -> (t[1],t[3]),cases)
            measure(E,c,case...,c.seed)
        end
        order = Xoshiro(c.seed)
        samples = NamedTuple[]
        for rep in 1:c.reps, (method,m,mode) in shuffle(order,cases)
            GC.gc(false)
            push!(samples,measure(E,c,method,m,mode,c.seed+rep))
        end
        samples
    end
end

# ╔═╡ 10acaa3a-58e3-5296-ac80-5d003994b829
begin
    function bench_summary(data, method, m, mode=:nueva)
        a = filter(r -> (r.method,r.m,r.mode)==(method,m,mode),data)
        passed = count(r -> r.ok,a)
        complete = !isempty(a) && passed == length(a)
        med(field) = complete ? median(getproperty.(a,field)) : NaN
        (; t=med(:total), online=med(:online), prep=med(:prep), mib=med(:bytes)/2^20,
           stored=med(:stored)/2^20, it=med(:k), success="$passed/$(length(a))")
    end
    fmtiter(x) = isfinite(x) && isinteger(x) ? string(Int(x)) : fmt(x)
    fmt(x) = isfinite(x) ? @sprintf("%.2f",x) : "—"
    table(header,rows) = Markdown.parse(join([join(header," | "),join(fill("---",length(header))," | "),
                                               [join(row," | ") for row in rows]...],"\n"))
end

# ╔═╡ f4f7bcfa-19ee-5fd8-b406-ce09de34801e
samples = sweep(experiment,cfg);

# ╔═╡ 1a233b2a-0070-59c7-9f4b-055090050ded
md"""
### 3.2 Prueba 1 · Tiempo y memoria con tamaño fijo

**¿Cuál llega antes a la tolerancia común cuando ``m=6d``?** La columna «Aceleración» divide la mediana del tiempo total de Newton exacto por la del método. Por ejemplo, una aceleración de ``1.5`` significa que el sketch empleó dos tercios del tiempo de Newton exacto hasta alcanzar la meta. Las columnas de iteraciones y de memoria asignada ayudan a interpretar ese cociente.
"""

# ╔═╡ 3916ffd5-6dec-58fe-9caa-acd4996a7cc8
main_table = let
    baseline = bench_summary(samples,:Exacto,0).t
    rows = map(method_names) do method
        a = bench_summary(samples,method,method == :Exacto ? 0 : 6*cfg.d)
        (method,fmt(a.t),fmt(a.mib),fmtiter(a.it),fmt(baseline/a.t),a.success)
    end
    table(["Método","Total (ms)","Asignada (MiB)","Iter.","Aceleración (×)","Éxitos"],rows)
end

# ╔═╡ 71535e59-77f0-579e-9be6-d53e99418ce0
md"""
### 3.3 Prueba 2 · Efecto del cociente m/d

**¿Cómo cambia el tiempo total al dar más filas al sketch?** `ratio_table` mantiene fijos los datos y la tolerancia y usa ``m=\operatorname{round}(\rho d)`` para cada cociente ``\rho`` de `cfg.ratios`. Cada celda muestra **ms / MiB asignados / iteraciones**. Cuando hay una corrida fallida, aparece el número de éxitos.

Conviene seguir una columna a medida que crece ``m/d``. Si baja el número de iteraciones pero sube el tiempo total, el ahorro en pasos no compensó el costo de cada proyección y de formar su matriz de Gram. Con los valores iniciales, ``m=600`` y ``n=5000`` dan una matriz ``C`` con el 12 % de las entradas de ``B``. La memoria medida incluye también los arreglos de trabajo y las demás asignaciones.
"""

# ╔═╡ 575807a1-0557-5c30-a6fa-124851418312
ratio_table = let
    rows = map(cfg.ratios) do ratio
        m = round(Int,ratio*cfg.d)
        cells = map(method_names[2:end]) do method
            a = bench_summary(samples,method,m)
            isfinite(a.t) ? join((fmt(a.t),fmt(a.mib),fmtiter(a.it))," / ") : "— ($(a.success))"
        end
        (ratio,m,cells...)
    end
    table(["m/d","m",string.(method_names[2:end])...],rows)
end

# ╔═╡ a2aae31a-4cf6-58f5-b35a-6c2d52569922
md"""
### 3.4 Prueba 3 · ¿Conviene precalcular la proyección gaussiana?

Con ``m=6d`` se comparan tres estrategias. **Nueva** genera los paneles gaussianos en cada paso. **Fija** guarda una matriz y la reutiliza durante toda la corrida. **Familia** prepara ``K`` matrices independientes y las usa de forma cíclica, con ``K`` dado por `cfg.family`. Con la misma semilla, la familia reproduce las primeras ``K`` matrices de la estrategia nueva. A partir de allí, reutilizar las matrices puede modificar también el número de iteraciones.

Cada matriz gaussiana guardada ocupa ``8mn`` bytes en `Float64`. Como los pesos dependen de ``x``, cada iteración vuelve a ponderar las filas y a multiplicar los paneles. «En línea» mide el tiempo de resolución una vez preparado el banco. «Total» suma preparación y resolución dentro de cada corrida y luego toma la mediana. Por ello puede diferir de la suma de las dos medianas de fase.
"""

# ╔═╡ 38ad8634-5335-5bce-b1e0-98d3e70e5fca
gaussian_table = let
    rows = map(modes) do mode
        a = bench_summary(samples,:Gaussian,6*cfg.d,mode)
        (mode,fmt(a.prep),fmt(a.online),fmt(a.t),fmt(a.mib),fmt(a.stored),fmtiter(a.it),a.success)
    end
    table(["Estrategia","Preparar (ms)","En línea (ms)","Total (ms)","Asignada (MiB)",
           "Banco retenido (MiB)","Iter.","Éxitos"],rows)
end

# ╔═╡ bcbcec1c-b427-556f-9818-133f968ace86
md"""
#### Perfil por etapa de la proyección gaussiana

**¿Dónde se gasta el tiempo que el precálculo intenta ahorrar?** Se repiten las tres estrategias con relojes internos. Cada celda muestra el tiempo **medio en ms por corrida** y su porcentaje del tiempo total agregado. Para obtener el costo de proyectar, se suman tres filas de la tabla, las de generar o acceder a los coeficientes, ponderar las filas de ``A`` y multiplicar los paneles. «Otros» recoge el tiempo restante, que incluye inicialización, comprobaciones de parada e instrumentación.

Los encabezados indican cuántas corridas alcanzaron la meta. El perfil promedia todas las corridas instrumentadas de cada estrategia. La comparación anterior usa tiempos sin estos relojes.
"""

# ╔═╡ 29dc6ad1-5599-5d88-b2f4-c4409da1ff94
profile_data = let
    m = 6*cfg.d
    for mode in modes
        measure(experiment,cfg,:Gaussian,m,mode,cfg.seed; profile=true)
    end
    order = Xoshiro(cfg.seed)
    [begin
        GC.gc(false)
        measure(experiment,cfg,:Gaussian,m,mode,cfg.seed+rep; profile=true)
     end for rep in 1:cfg.reps for mode in shuffle(order,collect(modes))]
end;

# ╔═╡ eeb441fb-2c93-5202-8db6-bb71fa20f723
profile_table = let
    labels = ["Preparar banco","Gradiente y pesos","Generar G / acceder al banco",
              "Ponderar filas de A","Multiplicar y acumular GB/√m","Formar Gram y sumar μI",
              "Cholesky y resolver","Armijo y actualización","Otros","Total"]
    columns = map(modes) do mode
        a = filter(r -> r.mode==mode,profile_data)
        # Media por corrida para que las etapas sumen el total; porcentajes ponderados por tiempo.
        parts = reduce(+, (Float64.(r.stages) for r in a)) ./ (1e6*length(a))
        total = mean(r.total for r in a)
        prep = mean(r.prep for r in a)
        values = [prep; parts; total-prep-sum(parts); total]
        [@sprintf("%.3f (%.1f %%)",v,100*v/total) for v in values]
    end
    table(["Etapa",["$(mode) ($(count(r -> r.ok && r.mode==mode,profile_data))/$(cfg.reps))" for mode in modes]...],
          [(labels[i],(col[i] for col in columns)...) for i in eachindex(labels)])
end

# ╔═╡ 4bb4c09e-b178-5e01-8031-bce4cfbf187e
reading = let
    m = 6*cfg.d
    costs = [bench_summary(samples,:Gaussian,m,mode) for mode in modes]
    fresh = filter(r -> r.mode == :nueva,profile_data)
    projection = reduce(+, (Float64.(r.stages[2:4]) for r in fresh))
    shares = fmt.(100 .* projection ./ sum(projection))
    if all(a -> isfinite(a.t),costs)
        best_total = modes[argmin([a.t for a in costs])]
        online = modes[argmin([a.online for a in costs])]
        Markdown.parse("""**Lectura del perfil.** Dentro de la proyección gaussiana nueva,
        generar los coeficientes consume $(shares[1]) %, ponderar las filas consume $(shares[2]) %
        y multiplicar los paneles consume $(shares[3]) %.

        Con preparación incluida, la menor mediana corresponde a **$best_total**.
        Con el banco disponible de antemano, la menor mediana en línea corresponde a **$online**.
        La familia reserva $(fmt(costs[3].stored)) MiB para sus $(cfg.family) matrices.""")
    else
        md"""Alguna estrategia no alcanzó la meta en todas las repeticiones. Consulta la columna de éxitos antes de interpretar sus tiempos."""
    end
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.7"
manifest_format = "2.0"
project_hash = "2b1ec542442a81fd15c8315dafe7a21ea99954c3"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.1+2"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "e2b53ce13a53367e96601081e33d34746b571bad"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.5"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"
"""

# ╔═╡ Cell order:
# ╟─4e017c8d-950f-54a7-924e-895ffa2b36fb
# ╟─3661171d-83d3-5117-b5dc-046235e57f3d
# ╟─128b25dc-507f-58e5-a221-a5fc442f57ba
# ╠═3632e8f2-10d2-5e61-ada1-e690455a43f1
# ╟─6800aa01-f855-5ab5-9fad-3e1996a8d4bd
# ╟─66d49d26-1ca5-5fbe-a79c-f31003be5435
# ╟─afb9b551-b165-5aba-a9d6-dd59557c76f1
# ╟─69034d23-e527-5101-b0f3-48a64f8db913
# ╟─bfc746e6-ce71-5638-8cb6-d0386a2b0bfd
# ╠═a901d763-51e9-56af-8ad6-37dc60c82f60
# ╟─d48a83c5-3815-5055-85f1-b6a854e51db0
# ╠═c7076c38-aee8-55bd-bd4c-267af69c5144
# ╟─0842659a-e4ea-53ec-ac7e-cb196020b7af
# ╠═a28802f4-4ff9-57b2-be1d-4737ccf14953
# ╠═0f6e1c61-f536-5928-aeee-eeaf44a57c5b
# ╟─3b50f7e9-efdb-5131-b484-aebe277769e3
# ╠═98fbde4d-0bd3-5b4c-a268-cd81af177081
# ╠═7b872bd6-ebae-5304-aee4-bfbdfebe27ce
# ╟─e4cc3a19-4529-5149-bf12-5b902fb6093c
# ╠═1d040a45-c257-58f1-9ab2-5639898628ea
# ╠═5997e6de-27c9-5166-840f-afc75306328d
# ╠═e4c4b731-00fc-5b5e-89aa-20e11a57dfa8
# ╟─cb63e855-0aef-5b1e-8745-a8dc50ec1ebd
# ╠═8bc13cf3-d348-588e-a73c-fb6c2ac832f7
# ╟─516c900b-5289-5a48-9506-fb8f67280879
# ╟─d1e8a184-728f-5cf1-bfe7-5415d1310c4d
# ╠═64b99d4d-1547-5873-9928-bef0fbd8e026
# ╟─fdf21465-3255-5858-ba44-7885b7b22dc5
# ╟─4bfd4e2e-56fc-5d5f-b83a-1b0ad2d8a62c
# ╟─1e779b37-7c48-5cea-b4a6-307198ef8e55
# ╟─bd7c5eb9-6d17-5f23-9e7a-b2b523e184fb
# ╟─ee370f13-e950-5a8a-8213-000904348e26
# ╟─0b0d2f7d-2eb7-5449-b8d4-4a7f1b2e1d7d
# ╟─df795018-66ad-5d53-800f-8d1df6e96977
# ╟─10acaa3a-58e3-5296-ac80-5d003994b829
# ╟─f4f7bcfa-19ee-5fd8-b406-ce09de34801e
# ╟─1a233b2a-0070-59c7-9f4b-055090050ded
# ╟─3916ffd5-6dec-58fe-9caa-acd4996a7cc8
# ╟─71535e59-77f0-579e-9be6-d53e99418ce0
# ╟─575807a1-0557-5c30-a6fa-124851418312
# ╟─a2aae31a-4cf6-58f5-b35a-6c2d52569922
# ╟─38ad8634-5335-5bce-b1e0-98d3e70e5fca
# ╟─bcbcec1c-b427-556f-9818-133f968ace86
# ╟─29dc6ad1-5599-5d88-b2f4-c4409da1ff94
# ╟─eeb441fb-2c93-5202-8db6-bb71fa20f723
# ╟─4bb4c09e-b178-5e01-8031-bce4cfbf187e
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
