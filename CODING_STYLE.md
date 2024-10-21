# Coding Style

## General
Use `camelCase` except for class/struct name using `UpperCase`.  
Use `UPPER` for constexpr.  
Use `auto` only in loop or long type.  
Use suffixes 64_t, 32_t, 16_t, 8_t instead of basic declarations int, long, char, etc.  
Added pointer symbol on type: `char** var;`.  
Did not add `const` on left of the: `char const* const var;`.  
Assign default variable with `{}` instead of `=`.  
```cpp
int foo()
{
    int const var{ 0 };
    int const& varR{ var };
    int const* varP{ &var };
}
```
```cpp
constexpr uint32_t MY_MACRO{ 12u };
```
```cpp
class MyClass{};
```


# Class/Struct
Use `UpperCase`
Prefere using `struct` instead of `class`..
The declaration of scopes in `struct` or `class` must be done in this order -> public / protected / private.  
Initialize default value of members variables in the declaration instead of constructor.  
```cpp
struct MyStruct
{
public:
    int var{ 0 };
    void foo() {}

protected:
    int var2{ 0 };
    void foo2() {}

private:
    int var3{ 0 };
    void foo3() {}
};
```


## Include
Use `<>` for `#include`.  
Order include: STL, third party, internal.  
Each section must have 1 line separator.  
```cpp
#include <STL/file_a.hpp>
#include <STL/file_b.hpp>

#include <thirdparty/file_a.hpp>
#include <thirdparty/file_a.hpp>

#include <internal/file_a.hpp>
#include <internal/file_a.hpp>
```


## Condition
All time use brackets.  
Condition on multi lines.  
Align on column conditions.  
Use const on first element.  
Use explicit compare.  
```cpp
if (true == a) // OK explicit
{
    // code
}
if (nullptr == a) // OK explicit
{
    // code
}
if (a) // NOK implicit
{
    // code
}
if (!a) // NOK implicit
{
    // code
}
```
```cpp
if (1 == a)
{
    // code
}
else if (   2 === a
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

## Loop
Define iterator or index in the loop not outside.  
Prefered use `for`.  
```cpp
for (auto& a : list)
{
    // code
}
for (int i{ 0 }; i < value; ++i)
{
    // code
}
```
Used `while` for looping by function returning `boo`, `pointer`.  
```cpp
while (true == isOpen())
{
    // code
}
```

## Function
```cpp
inline
void funciton(
    uint8_t firstParameter,
    uint32_t secondParameter)
{
    // code
}
```
