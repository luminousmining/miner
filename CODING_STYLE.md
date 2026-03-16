# Coding Style

> Enforced automatically by `.clang-format` (clang-format-15) and `.clang-tidy`.
> Run `cmake --build <build_dir> --target format` to auto-format.
> Run `cmake --build <build_dir> --target format-check` to verify without modifying.

---

## General

- Use `camelCase` for variables, functions, parameters and members.
- Use `UpperCase` (PascalCase) for class, struct, enum and type alias names.
- Use `UPPER_CASE` for `constexpr`.
- Use `auto` only in range-for loops or for long/complex types.
- Use suffixes `uint64_t`, `uint32_t`, `uint16_t`, `uint8_t` instead of `int`, `long`, `char`, etc.
- Place the pointer/reference symbol on the **type** side: `char** var`.
- Place `const` on the **right** of the type: `char const* const var`.
- Initialize variables with `{}` instead of `=`.
- Leave **2 blank lines** between function definitions.
- Column limit: **120 characters**.

```cpp
int foo()
{
    int const   var{ 0 };
    int const&  varR{ var };
    int const*  varP{ &var };
}
```
```cpp
constexpr uint32_t MY_MACRO{ 12u };
```


---

## Class / Struct

- Prefer `struct` over `class`.
- Declare scopes in order: `public` → `protected` → `private`.
- Initialize member variables **in the declaration**, not in the constructor.
- Align consecutive member declarations on the same column.

```cpp
struct MyStruct
{
public:
    uint32_t id{ 0u };
    float    value{ 0.f };
    void     foo() {}

protected:
    uint32_t internalId{ 0u };
    void     bar() {}

private:
    uint32_t secret{ 0u };
};
```


---

## Include

- Use `<>` for all `#include` directives.
- Order: **STL → third-party → internal**. Each section separated by 1 blank line.
- Sorted alphabetically within each section (enforced by clang-format).

```cpp
#include <cstdint>
#include <string>
#include <vector>

#include <boost/asio.hpp>
#include <gtest/gtest.h>

#include <algo/fast_mod.hpp>
#include <common/custom.hpp>
```

> For files that cannot use `<>` (e.g. some test files), `""` is accepted for internal headers only.


---

## Condition

- Always use braces, even for single-line bodies.
- Place the **constant/literal on the left** of the comparison (Yoda conditions).
- Always use **explicit comparisons** (no implicit bool conversion).
- For multi-line conditions, align operators (`&&`, `||`) at the start of continuation lines.
- Column limit is 120: prefer keeping the comparison on a single line and only wrapping function arguments.

```cpp
if (true == a)   // OK — explicit
{
    // code
}
if (nullptr == ptr)  // OK — explicit
{
    // code
}
if (a)   // NOK — implicit
{
    // code
}
if (!a)  // NOK — implicit
{
    // code
}
```

```cpp
if (1 == a)
{
    // code
}
else if (   2 == a
         && 3u == c)
{
    // code
}
else if (   true == ptr
         && (true == a || true == b))
{
    // code
}
else
{
    // code
}
```

When a condition contains a function call that exceeds 120 columns,
only the function **arguments** wrap — the comparison stays on the same line:

```cpp
if (false == someObject.longMethodName(
                 firstArg,
                 secondArg))
{
    // code
}
```


---

## Loop

- Define the iterator or index **inside** the loop statement.
- Prefer `for` over `while`.
- Use `while` only when looping on a function returning `bool` or a pointer.

```cpp
for (auto& a : list)
{
    // code
}

for (uint32_t i{ 0u }; i < value; ++i)
{
    // code
}

while (true == isOpen())
{
    // code
}
```


---

## Function

- Place specifiers (`inline`, `static`, etc.) on their own line before the return type.
- When parameters do not fit on 120 columns, break after `(` and put each parameter on its own line indented by 4 spaces. The closing `)` stays on the last parameter line.
- In `.hpp` declarations with short names, parameters may stay on the same line if they fit.

```cpp
inline
void function(
    uint8_t  firstParameter,
    uint32_t secondParameter)
{
    // code
}
```

```cpp
// Short declaration — fits on one line
void foo(uint32_t a, uint32_t b);

// Long definition — break after (
void algo::ns::MyClass::longMethodName(
    std::stringstream& ss,
    uint32_t const     param1,
    uint32_t const     param2)
{
    // code
}
```


---

## Naming

| Element | Convention | Example |
|---|---|---|
| Variable / parameter / member | `camelCase` | `myVar`, `firstParam` |
| Function / method | `camelCase` | `getValue()`, `computeHash()` |
| Class / struct / enum | `UpperCase` | `MyStruct`, `AlgoType` |
| `constexpr` constant | `UPPER_CASE` | `MAX_THREADS` |
| Namespace | `lower_case` | `algo::progpow` |
| Template parameter (type) | `UpperCase` | `template<typename MyType>` |
| Template parameter (value) | `UPPER_CASE` | `template<uint32_t SIZE>` |


---

## Template

No space between `template` and `<>`:

```cpp
template<typename T>
void foo(T value);

template<typename T, uint32_t SIZE>
struct MyBuffer
{
    T data[SIZE]{};
};
```


---

## Automatic Formatting Limitations

The following patterns require `// clang-format off/on` because clang-format cannot
reproduce them automatically:

**Array/struct initializer with brace on its own line:**
```cpp
// clang-format off
constexpr uint64_t ROUND_CONSTANTS[24]
{
    0x0000000000000001, 0x0000000000008082,
    0x800000000000808a, 0x8000000080008000,
};
// clang-format on
```

**Deeply nested test assertions:**
```cpp
// clang-format off
EXPECT_EQ(
    expectedValue,
    MyClass::instance().computeResult(
        arg1,
        arg2
    )
);
// clang-format on
```
